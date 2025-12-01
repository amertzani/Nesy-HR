#!/bin/bash

# Start Both Backend and Frontend Servers (DEV VERSION)
# This script starts both servers in separate terminal windows

set -e  # Exit on error

echo "🚀 Starting NesyX Application (DEV VERSION)..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  This script is optimized for macOS"
    echo "   On other systems, you may need to start servers manually"
fi

# Function to check if a port is in use
check_port() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1
    return $?
}

# Check if ports are available (DEV ports)
BACKEND_PORT=${API_PORT:-8002}
FRONTEND_PORT=${PORT:-5006}

if check_port $BACKEND_PORT; then
    echo "⚠️  Port $BACKEND_PORT is already in use (backend)"
    echo "   Set API_PORT environment variable to use a different port"
fi

if check_port $FRONTEND_PORT; then
    echo "⚠️  Port $FRONTEND_PORT is already in use (frontend)"
    echo "   Set PORT environment variable to use a different port"
fi

# Start backend in a new terminal window (macOS)
echo "🔧 Starting backend server (DEV) on port $BACKEND_PORT..."
osascript -e "tell application \"Terminal\" to do script \"cd '$SCRIPT_DIR' && ./start_backend.sh\""

# Wait a moment for backend to start
sleep 3

# Start frontend in a new terminal window (macOS)
echo "🔧 Starting frontend server (DEV) on port $FRONTEND_PORT..."
osascript -e "tell application \"Terminal\" to do script \"cd '$SCRIPT_DIR' && ./start_frontend.sh\""

echo ""
echo "✅ Both servers are starting in separate terminal windows"
echo ""
echo "📋 Server URLs (DEV VERSION):"
echo "   Backend API:  http://localhost:${BACKEND_PORT}"
echo "   Backend Docs: http://localhost:${BACKEND_PORT}/docs"
echo "   Frontend:     http://localhost:${FRONTEND_PORT}"
echo ""
echo "💡 To stop the servers, close the terminal windows or press Ctrl+C in each"
echo ""
