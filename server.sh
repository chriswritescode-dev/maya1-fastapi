#!/bin/bash

#############################################
# Maya1 TTS Server Management Script
# Usage: ./server.sh [start|stop|restart|status]
#############################################

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Configuration
VENV_PATH="$PROJECT_ROOT/venv"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/server.log"
PID_FILE="$PROJECT_ROOT/.server.pid"
HOST="0.0.0.0"
PORT="8000"

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
    
    # Kill VLLM processes
    VLLM_PIDS=$(pgrep -f "VLLM::EngineCore" || true)
    if [ ! -z "$VLLM_PIDS" ]; then
        pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
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
    nohup python -m uvicorn maya1.api_v2:app \
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
        echo "Commands:"
        echo "  start    Start the server"
        echo "  stop     Stop the server"
        echo "  restart  Restart the server"
        echo "  status   Check server status"
        echo ""
        echo "Examples:"
        echo "  ./server.sh start"
        echo "  ./server.sh status"
        echo "  ./server.sh stop"
        echo ""
        exit 1
        ;;
esac
