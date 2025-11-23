#!/bin/bash

#############################################
# Maya1 TTS Server Management Script
# Usage: ./server.sh [start|stop|restart|status]
#############################################

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Configuration
# Support both venv and .venv directories
if [ -d "$PROJECT_ROOT/.venv" ]; then
    VENV_PATH="$PROJECT_ROOT/.venv"
elif [ -d "$PROJECT_ROOT/venv" ]; then
    VENV_PATH="$PROJECT_ROOT/venv"
else
    VENV_PATH="$PROJECT_ROOT/venv"  # Default
fi
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/server.log"
PID_FILE="$PROJECT_ROOT/.server.pid"
HOST="0.0.0.0"
PORT="8880"
export CUDA_VISIBLE_DEVICES="3"

# Model Configuration
# Set your model path here - can be local path or HuggingFace repo
MAYA1_MODEL_PATH="/storage/models/maya1"
MAYA1_MODEL_DTYPE="${MAYA1_MODEL_DTYPE:-bfloat16}"
MAYA1_MAX_MODEL_LEN="${MAYA1_MAX_MODEL_LEN:-8192}"
MAYA1_GPU_MEMORY_UTILIZATION="0.5"
MAYA1_TENSOR_PARALLEL_SIZE="${MAYA1_TENSOR_PARALLEL_SIZE:-1}"

# Export model configuration as environment variables
export MAYA1_MODEL_PATH
export MAYA1_MODEL_DTYPE
export MAYA1_MAX_MODEL_LEN
export MAYA1_GPU_MEMORY_UTILIZATION
export MAYA1_TENSOR_PARALLEL_SIZE
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
        log_info "Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        log_info "Or: uv .venv && source .venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
    
    # Check if activation script exists
    if [ ! -f "$VENV_PATH/bin/activate" ]; then
        log_error "Activation script not found at $VENV_PATH/bin/activate"
        exit 1
    fi
    
    log_info "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
    
    # Verify activation worked
    if [ -z "$VIRTUAL_ENV" ]; then
        log_error "Failed to activate virtual environment"
        exit 1
    fi
    
    log_info "Virtual environment activated: $VIRTUAL_ENV"
    
    # Verify python and uvicorn are available
    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found in activated environment"
        exit 1
    fi
    
    if ! python3 -c "import uvicorn" 2>/dev/null; then
        log_error "uvicorn not installed in virtual environment"
        log_info "Install with: pip install uvicorn"
        exit 1
    fi
    
    # Check if Maya1 dependencies are available
    if ! python3 -c "import fastapi, vllm, transformers" 2>/dev/null; then
        log_error "Required dependencies not installed"
        log_info "Install with: pip install -r requirements.txt"
        exit 1
    fi
}

# Stop server
stop_server() {
    log_info "Stopping Maya1 TTS Server..."
    
    # Kill by PID file
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            log_info "Terminating server process (PID: $PID)"
            kill -TERM "$PID" 2>/dev/null || true
            sleep 3
            # Force kill if still running
            if ps -p "$PID" > /dev/null 2>&1; then
                kill -9 "$PID" 2>/dev/null || true
            fi
        fi
        rm -f "$PID_FILE"
    fi
    
    # Kill any orphaned uvicorn processes for this app only
    UVICORN_PIDS=$(pgrep -f "uvicorn.*maya1.*api_v2:app" || true)
    if [ ! -z "$UVICORN_PIDS" ]; then
        log_info "Terminating orphaned uvicorn processes"
        echo "$UVICORN_PIDS" | xargs kill -TERM 2>/dev/null || true
        sleep 2
        echo "$UVICORN_PIDS" | xargs kill -9 2>/dev/null || true
    fi
    
    log_success "Server stopped"
}

# Start server
start_server() {
    log_info "Starting Maya1 TTS Server..."
    log_info "Using virtual environment: $VENV_PATH"
    
    # Display model configuration
    log_info "Model Configuration:"
    log_info "  Model Path: $MAYA1_MODEL_PATH"
    log_info "  Data Type: $MAYA1_MODEL_DTYPE"
    log_info "  Max Model Length: $MAYA1_MAX_MODEL_LEN"
    log_info "  GPU Memory Utilization: $MAYA1_GPU_MEMORY_UTILIZATION"
    log_info "  Tensor Parallel Size: $MAYA1_TENSOR_PARALLEL_SIZE"
    log_info "  CUDA Visible Devices: $CUDA_VISIBLE_DEVICES"
    
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
    nohup python3 -m uvicorn maya1.api_v2:app \
        --host "$HOST" \
        --port "$PORT" \
        --log-level info \
        > "$LOG_FILE" 2>&1 &
    
    SERVER_PID=$!
    echo "$SERVER_PID" > "$PID_FILE"
    
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
        fi
    fi
    
    log_error "Server not running"
    exit 1
}

# Show configuration
show_config() {
    echo "Maya1 TTS Server Configuration"
    echo "=============================="
    echo ""
    echo "Virtual Environment:"
    echo "  Path: $VENV_PATH"
    echo ""
    echo "Server Settings:"
    echo "  Host: $HOST"
    echo "  Port: $PORT"
    echo "  CUDA Visible Devices: $CUDA_VISIBLE_DEVICES"
    echo ""
    echo "Model Configuration:"
    echo "  Model Path: $MAYA1_MODEL_PATH"
    echo "  Data Type: $MAYA1_MODEL_DTYPE"
    echo "  Max Model Length: $MAYA1_MAX_MODEL_LEN"
    echo "  GPU Memory Utilization: $MAYA1_GPU_MEMORY_UTILIZATION"
    echo "  Tensor Parallel Size: $MAYA1_TENSOR_PARALLEL_SIZE"
    echo ""
    echo "To modify settings, set environment variables before running:"
    echo "  export MAYA1_MODEL_PATH=\"your-model-path\""
    echo "  export MAYA1_GPU_MEMORY_UTILIZATION=0.9"
    echo "  ./server.sh start"
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
    config)
        show_config
        ;;
    *)
        echo "Maya1 TTS Server"
        echo ""
        echo "Usage: ./server.sh [start|stop|restart|status|config]"
        echo ""
        echo "Commands:"
        echo "  start    Start the server"
        echo "  stop     Stop the server"
        echo "  restart  Restart the server"
        echo "  status   Check server status"
        echo "  config   Show current configuration"
        echo ""
        echo "Environment Variables (can be set before running):"
        echo "  MAYA1_MODEL_PATH              Model path or HF repo (default: maya-research/maya1)"
        echo "  MAYA1_MODEL_DTYPE             Model precision (default: bfloat16)"
        echo "  MAYA1_MAX_MODEL_LEN           Max sequence length (default: 8192)"
        echo "  MAYA1_GPU_MEMORY_UTILIZATION  GPU memory fraction (default: 0.85)"
        echo "  MAYA1_TENSOR_PARALLEL_SIZE    Number of GPUs (default: 1)"
        echo "  CUDA_VISIBLE_DEVICES          GPU devices (default: 3)"
        echo ""
        echo "Examples:"
        echo "  ./server.sh start"
        echo "  MAYA1_MODEL_PATH=/path/to/local/model ./server.sh start"
        echo "  MAYA1_GPU_MEMORY_UTILIZATION=0.9 ./server.sh start"
        echo "  ./server.sh config"
        echo ""
        exit 1
        ;;
esac
