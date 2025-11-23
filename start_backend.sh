#!/bin/bash

# Start Python FastAPI Backend Server (DEV VERSION)
# This script starts the Python backend API server on port 8002

set -e  # Exit on error

echo "🚀 Starting NesyX Backend Server (DEV VERSION)..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup_macos.sh first"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if api_server.py exists
if [ ! -f "api_server.py" ]; then
    echo "❌ api_server.py not found in current directory"
    exit 1
fi

# Set default port for DEV version (8002)
export API_PORT=${API_PORT:-8002}
export API_HOST=${API_HOST:-0.0.0.0}

echo "🌐 Starting FastAPI server on http://localhost:${API_PORT} (DEV VERSION)"
echo "📚 API documentation will be available at http://localhost:${API_PORT}/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
# source venv/bin/activate
python3 api_server.py
