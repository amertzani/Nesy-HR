#!/bin/bash

# Start React Frontend Server (DEV VERSION)
# This script starts the Node.js/Express server with React frontend on port 5006

set -e  # Exit on error

echo "🚀 Starting NesyX Frontend Server (DEV VERSION)..."
echo ""

# Check if we're in the right directory
if [ ! -d "RandDKnowledgeGraph" ]; then
    echo "❌ RandDKnowledgeGraph directory not found"
    echo "   Please run this script from the project root directory"
    exit 1
fi

# Navigate to frontend directory
cd RandDKnowledgeGraph

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Node modules not found. Installing dependencies..."
    npm install
fi

# Set default port for DEV version (5006)
export PORT=${PORT:-5006}
export NODE_ENV=${NODE_ENV:-development}

echo "🌐 Starting frontend server on http://localhost:${PORT} (DEV VERSION)"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
npm run dev
