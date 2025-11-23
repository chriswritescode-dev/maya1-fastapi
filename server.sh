#!/bin/bash

#############################################
# Maya1 TTS Server Management Script
# Usage: ./server.sh [start|stop|restart|status]
#############################################

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Configuration
VENV_PATH="$PROJECT_ROOT/.venv"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/server.log"
PID_FILE="$PROJECT_ROOT/.server.pid"
VLLM_PID_FILE="$PROJECT_ROOT/.vllm.pid"
HOST="0.0.0.0"
PORT="8880"

# Model Configuration
MAYA1_MODEL_PATH="/storage/models/maya1"  # Default model path
GPU_MEMORY_UTILIZATION="0.25"
export CUDA_VISIBLE_DEVICES="3"  # Ensure we use only one GPU
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Activate virtual environment
activate_venv() {
    if [ ! -d "$VENV_PATH" ]; then
        log_error "Virtual environment not found at $VENV_PATH"
        log_info "Run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
    source "$VENV_PATH/bin/activate"
}

# Stop server
stop_server() {
    log_info "Stopping Maya1 TTS Server..."
    
    # Kill by PID file
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            kill -9 "$PID" 2>/dev/null || true
            sleep 1
        fi
        rm -f "$PID_FILE"
    fi
    
    # Kill any uvicorn processes
    UVICORN_PIDS=$(pgrep -f "uvicorn.*api" || true)
    if [ ! -z "$UVICORN_PIDS" ]; then
        echo "$UVICORN_PIDS" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
    
    # Kill VLLM process by saved PID
    if [ -f "$VLLM_PID_FILE" ]; then
        VLLM_PID=$(cat "$VLLM_PID_FILE")
        if ps -p "$VLLM_PID" > /dev/null 2>&1; then
            kill -9 "$VLLM_PID" 2>/dev/null || true
            sleep 1
        fi
        rm -f "$VLLM_PID_FILE"
    fi
    
    # Kill ALL VLLM processes on GPU 3 (aggressive cleanup)
    VLLM_PIDS=$(nvidia-smi -i 3 --query-compute-apps=pid,process_name --format=csv,noheader | grep "VLLM::EngineCore" | cut -d',' -f1 | tr -d ' ' || true)
    if [ ! -z "$VLLM_PIDS" ]; then
        log_info "Killing VLLM processes on GPU 3: $VLLM_PIDS"
        for VLLM_PID in $VLLM_PIDS; do
            kill -9 $VLLM_PID 2>/dev/null || true
        done
        sleep 2
    fi
    
    # Also kill any remaining python processes using GPU 3 that might be vLLM related
    GPU3_PIDS=$(nvidia-smi -i 3 --query-compute-apps=pid,process_name --format=csv,noheader | grep -E "(python|uvicorn)" | cut -d',' -f1 | tr -d ' ' || true)
    if [ ! -z "$GPU3_PIDS" ]; then
        log_info "Killing additional GPU 3 processes: $GPU3_PIDS"
        for GPU_PID in $GPU3_PIDS; do
            kill -9 $GPU_PID 2>/dev/null || true
        done
        sleep 1
    fi
    
    # Kill any remaining Maya1 python processes
    MAYA1_PIDS=$(pgrep -f "maya1.*api" || true)
    if [ ! -z "$MAYA1_PIDS" ]; then
        echo "$MAYA1_PIDS" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
    
    log_success "Server stopped"
}

# Start server
start_server() {
    log_info "Starting Maya1 TTS Server..."
    
    # Check if already running
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            log_error "Server already running (PID: $PID)"
            exit 1
        fi
    fi
    
    # Create logs directory
    mkdir -p "$LOG_DIR"
    
    # Activate virtual environment
    activate_venv
    
    # Start server
    log_info "Starting on http://$HOST:$PORT"
    log_info "Model: $MAYA1_MODEL_PATH"
    log_info "GPU Memory: ${GPU_MEMORY_UTILIZATION}x"
    log_info "Tensor Parallel: $TENSOR_PARALLEL_SIZE"
    log_info "Data Type: $DTYPE"
    
    # Set environment variables for model configuration
    export MAYA1_MODEL_PATH="$MAYA1_MODEL_PATH"
    export GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION"
    export TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE"
    export DTYPE="$DTYPE"
    
    # Start server and capture VLLM PID
    nohup python -m uvicorn maya1.api_v2:app \
        --host "$HOST" \
        --port "$PORT" \
        --log-level info \
        > "$LOG_FILE" 2>&1 &
    
    SERVER_PID=$!
    echo "$SERVER_PID" > "$PID_FILE"
    
    # Wait for server to be ready, then get VLLM PID from API
    log_info "Waiting for server to be ready..."
    for i in {1..30}; do
        sleep 1
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            break
        fi
        if [ $((i % 5)) -eq 0 ]; then
            log_info "Still waiting for server... (${i}s)"
        fi
    done
    
    # Get VLLM PID from API
    VLLM_PID=$(curl -s "http://localhost:$PORT/v1/pid" 2>/dev/null | grep -o '"vllm_pid":[0-9]*' | cut -d':' -f2 2>/dev/null || echo "")
    
    if [ ! -z "$VLLM_PID" ]; then
        echo "$VLLM_PID" > "$VLLM_PID_FILE"
        log_success "VLLM process detected (PID: $VLLM_PID)"
    else
        log_info "No VLLM PID available - server may use embedded inference"
    fi
    
    # Wait for startup
    sleep 5
    
    # Verify running
    if ps -p "$SERVER_PID" > /dev/null 2>&1; then
        log_success "Server started (PID: $SERVER_PID)"
        log_info "API: http://localhost:$PORT"
        log_info "Logs: tail -f $LOG_FILE"
    else
        log_error "Failed to start. Check: $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
}

# Check status
check_status() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            log_success "Server running (PID: $PID)"
            log_info "URL: http://localhost:$PORT"
            nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true
            exit 0
        else
            rm -f "$PID_FILE"
            rm -f "$VLLM_PID_FILE"
        fi
    fi
    
    log_error "Server not running"
    exit 1
}

# Main
case "${1:-}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        log_info "Restarting..."
        stop_server
        sleep 2
        start_server
        ;;
    status)
        check_status
        ;;
    *)
        echo "Maya1 TTS Server"
        echo ""
        echo "Usage: ./server.sh [start|stop|restart|status]"
        echo ""
        echo "Environment Variables:"
        echo "  MAYA1_MODEL_PATH        Model path (default: maya-research/maya1)"
        echo "  GPU_MEMORY_UTILIZATION  GPU memory fraction (default: 0.85)"
        echo "  TENSOR_PARALLEL_SIZE    Number of GPUs (default: 1)"
        echo "  DTYPE                   Model precision (default: bfloat16)"
        echo ""
        echo "Commands:"
        echo "  start    Start the server"
        echo "  stop     Stop the server"
        echo "  restart  Restart the server"
        echo "  status   Check server status"
        echo ""
        echo "Examples:"
        echo "  ./server.sh start"
        echo "  MAYA1_MODEL_PATH=/path/to/model ./server.sh start"
        echo "  GPU_MEMORY_UTILIZATION=0.9 ./server.sh start"
        echo "  TENSOR_PARALLEL_SIZE=2 ./server.sh start"
        echo ""
        exit 1
        ;;
esac
