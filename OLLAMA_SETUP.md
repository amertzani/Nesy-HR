# Ollama Setup Guide

## Quick Start

1. **Install Ollama** (if not already installed):
   ```bash
   # macOS
   brew install ollama
   # OR download from https://ollama.ai
   ```

2. **Start Ollama**:
   ```bash
   ollama serve
   ```

3. **Pull a lightweight model** (recommended for MacBook Pro):
   ```bash
   # Option 1: llama3.2:1b (fastest, good quality) - DEFAULT
   ollama pull llama3.2:1b
   
   # Option 2: qwen2.5:0.5b (very fast, good for data analysis)
   ollama pull qwen2.5:0.5b
   
   # Option 3: phi3:mini (balanced)
   ollama pull phi3:mini
   
   # Option 4: gemma2:2b (slightly larger but smarter)
   ollama pull gemma2:2b
   ```

4. **Configure the application** (optional):
   ```bash
   # Use default (llama3.2:1b)
   export USE_OLLAMA=true
   
   # Or use a different model
   export OLLAMA_MODEL=qwen2.5:0.5b
   
   # Or use a custom Ollama URL
   export OLLAMA_BASE_URL=http://localhost:11434
   ```

## Model Recommendations for MacBook Pro

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| **llama3.2:1b** | ~1GB | ⚡⚡⚡ | ⭐⭐⭐ | General use (DEFAULT) |
| **qwen2.5:0.5b** | ~500MB | ⚡⚡⚡⚡ | ⭐⭐ | Fast responses, data analysis |
| **phi3:mini** | ~2GB | ⚡⚡ | ⭐⭐⭐⭐ | Better quality, still fast |
| **gemma2:2b** | ~2GB | ⚡⚡ | ⭐⭐⭐⭐ | Best quality, slightly slower |

## Environment Variables

- `USE_OLLAMA=true` (default) - Enable Ollama
- `OLLAMA_MODEL=llama3.2:1b` (default) - Model to use
- `OLLAMA_BASE_URL=http://localhost:11434` (default) - Ollama server URL

## Troubleshooting

**Ollama not responding:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve
```

**Model not found:**
```bash
# List available models
ollama list

# Pull the model
ollama pull llama3.2:1b
```

**Connection refused:**
- Make sure Ollama is running: `ollama serve`
- Check the port (default: 11434)
- Verify OLLAMA_BASE_URL matches your setup

## Performance Tips

1. **Use the smallest model** that meets your quality needs
2. **Keep Ollama running** in the background
3. **First request is slower** (model loading), subsequent requests are faster
4. **Close other heavy applications** if responses are slow

