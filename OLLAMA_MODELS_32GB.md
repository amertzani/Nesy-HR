# Best Ollama Models for 32GB RAM

## 🎯 Recommended Models (in order of quality)

### 1. **llama3.1:13b** ⭐ RECOMMENDED
- **Quality**: Excellent
- **Speed**: Fast (good for 32GB)
- **RAM Usage**: ~8-10GB
- **Best for**: Best balance of quality and speed
- **Command**: `ollama pull llama3.1:13b`

### 2. **mixtral:8x7b**
- **Quality**: Very powerful, excellent reasoning
- **Speed**: Moderate
- **RAM Usage**: ~12-14GB
- **Best for**: Complex analytical queries
- **Command**: `ollama pull mixtral:8x7b`

### 3. **llama3.1:8b**
- **Quality**: Very good
- **Speed**: Very fast
- **RAM Usage**: ~5-6GB
- **Best for**: Fast responses, good quality
- **Command**: `ollama pull llama3.1:8b`

### 4. **mistral:7b**
- **Quality**: Very good
- **Speed**: Very fast
- **RAM Usage**: ~4-5GB
- **Best for**: Fast, reliable responses
- **Command**: `ollama pull mistral:7b`

### 5. **qwen2.5:7b**
- **Quality**: Excellent for data analysis
- **Speed**: Very fast
- **RAM Usage**: ~4-5GB
- **Best for**: Statistical/data queries
- **Command**: `ollama pull qwen2.5:7b`

## 🚀 Quick Setup

### Recommended Setup (llama3.1:13b):
```bash
# Pull the model
ollama pull llama3.1:13b

# Set in your environment or start_backend.sh
export OLLAMA_MODEL="llama3.1:13b"
export USE_OLLAMA="true"
```

### Alternative (if you want even better quality):
```bash
# Pull Mixtral (more powerful)
ollama pull mixtral:8x7b

# Set in your environment
export OLLAMA_MODEL="mixtral:8x7b"
```

## 📊 Current Configuration

Your system is set up as:
1. **Primary**: Groq API (fastest, free)
2. **Fallback**: Ollama with `llama3.1:13b` (excellent quality, local)

This gives you:
- ⚡ Fast responses when Groq is available
- 🔒 Private fallback when Groq is unavailable
- 🎯 High quality from both options

## 💡 Why llama3.1:13b?

- Perfect for 32GB RAM (uses ~8-10GB)
- Much better than current `gemma2:2b` (2B is too small)
- Not too large (avoids 70B models)
- Excellent quality for HR/data analysis tasks

