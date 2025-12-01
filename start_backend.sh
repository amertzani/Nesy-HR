#!/bin/bash

# Start Python FastAPI Backend Server (DEV VERSION)
# This script starts the Python backend API server on port 8001

set -e  # Exit on error

# Kill any existing processes on port 8001
echo "🔧 Checking for existing processes on port 8001..."
if lsof -ti:8001 > /dev/null 2>&1; then
    echo "⚠️  Port 8001 is in use, killing existing process..."
    lsof -ti:8001 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Kill any existing api_server processes
pkill -f "python.*api_server" 2>/dev/null || true
sleep 1

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

# Set default port (8001 to match frontend)
export API_PORT=${API_PORT:-8001}
export API_HOST=${API_HOST:-0.0.0.0}

# LLM Configuration: Groq (primary) -> Ollama (fallback)
# Groq API - Primary (fastest, free tier)
export USE_GROQ="true"
export GROQ_MODEL="llama-3.1-8b-instant"

# Ollama - Fallback (local, free) - Good for 32GB RAM
export USE_OLLAMA="true"
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:13b}"  # Best for 32GB RAM
export OLLAMA_BASE_URL="http://localhost:11434"

# Disable OpenAI (not needed if Groq works)
export USE_OPENAI="false"

echo "🌐 Starting FastAPI server on http://localhost:${API_PORT}"
echo "📚 API documentation will be available at http://localhost:${API_PORT}/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
# source venv/bin/activate
# Start the backend server
python3 api_server.py
