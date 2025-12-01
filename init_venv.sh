#!/bin/bash

# Quick venv initialization script
# Creates and activates the virtual environment, installs critical packages only

set -e

echo "🔧 Initializing Python virtual environment..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate venv
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Install ONLY critical dependencies (skip spaCy and optional ML libs for now)
echo "📥 Installing critical dependencies only..."
echo "   (Skipping spaCy and optional ML libraries - can be installed later)"
echo ""

pip install gradio>=4.0.0
pip install rdflib>=6.0.0
pip install huggingface_hub
pip install PyPDF2
pip install python-docx
pip install pandas
pip install networkx
pip install matplotlib
pip install plotly
pip install numpy
pip install fastapi
pip install uvicorn[standard]
pip install pydantic
pip install python-multipart
pip install clean-text>=0.6.0

echo ""
echo "✅ Critical dependencies installed!"
echo ""
echo "💡 To install optional ML libraries (transformers, torch, etc.), run:"
echo "   ./setup_macos.sh"
echo ""
echo "💡 To install spaCy later (if needed):"
echo "   source venv/bin/activate"
echo "   pip install spacy"
echo "   python3 -m spacy download en_core_web_sm"
echo ""
echo "✅ Virtual environment is ready!"
echo "   Activate it with: source venv/bin/activate"

