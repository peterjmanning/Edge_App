#!/bin/bash

SESSION="edge_flask"
APP_PATH="/home/pmanning/Edge_App/app.py"
PYTHON_BIN="/home/pmanning/Edge_App/.venv/bin/python"
LOG_FILE="/home/pmanning/Edge_App/flask_app.log"

start() {
    echo "Starting Flask app in tmux session '$SESSION'..."
    # Kill previous session if it exists
    tmux kill-session -t $SESSION 2>/dev/null
    # Start new tmux session and redirect stdout+stderr to log file
    tmux new-session -d -s $SESSION "$PYTHON_BIN '$APP_PATH' >> '$LOG_FILE' 2>&1"
    echo "Flask app started. Logs are written to $LOG_FILE"
}

stop() {
    echo "Stopping Flask app..."
    tmux kill-session -t $SESSION 2>/dev/null
    echo "Flask app stopped."
}

status() {
    if tmux has-session -t $SESSION 2>/dev/null; then
        echo "Flask app is running."
    else
        echo "Flask app is not running."
    fi
}

case "$1" in
    start) start ;;
    stop) stop ;;
    restart) stop; start ;;
    status) status ;;
    *) echo "Usage: $0 {start|stop|restart|status}" ;;
esac
