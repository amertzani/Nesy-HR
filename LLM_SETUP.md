# LLM Setup Guide

## Current Configuration

The system supports three LLM options (in order of preference):

1. **Ollama** (Default, recommended for Mac) - Free, local, fast
2. **OpenAI API** - Most accurate, requires API key
3. **Local Transformers Model** - Fallback, slow on CPU

## Quick Setup

### Option 1: Ollama (Recommended - Free & Fast)

1. Install Ollama: https://ollama.ai
2. Start Ollama: `ollama serve`
3. Pull the model: `ollama pull gemma2:2b`
4. The system will automatically use Ollama

**To use a different model:**
```bash
export OLLAMA_MODEL="llama3.2:1b"  # Faster
# or
export OLLAMA_MODEL="qwen2.5:0.5b"  # Very fast
```

### Option 2: OpenAI API (Most Accurate)

1. Get an API key from https://platform.openai.com/api-keys
2. Set the environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
export USE_OPENAI="true"
export USE_OLLAMA="false"
```

3. Restart the backend server

**Cost:** ~$0.15 per 1M tokens (very cheap for this use case)

### Option 3: Disable LLM (Fastest, Rule-Based Only)

```bash
export USE_LLM="false"
```

## Troubleshooting

### LLM Timeout Issues

- **CPU Mode**: Timeouts increased to 60s (Ollama) and 180s (API)
- **Large Knowledge Graphs**: System automatically increases timeout
- **If still timing out**: Try using OpenAI API (faster) or disable LLM

### Wrong Answers

- System now has **direct answer fallbacks** for common queries:
  - Top employees by absences
  - Best recruitment source
  - Bottom employees by engagement
- These bypass the LLM for accuracy
- LLM system prompts emphasize using EXACT numbers from context

### Performance

- **Ollama**: Fast on Mac, ~2-5 seconds per query
- **OpenAI**: Very fast, ~1-2 seconds per query
- **Local Model**: Slow on CPU, ~15-30 seconds per query

## Environment Variables

```bash
# LLM Configuration
USE_LLM=true                    # Enable/disable LLM
USE_OLLAMA=true                 # Use Ollama (default)
USE_OPENAI=false                # Use OpenAI API
OPENAI_API_KEY=""               # Your OpenAI API key
OLLAMA_BASE_URL="http://localhost:11434"  # Ollama server URL
OLLAMA_MODEL="gemma2:2b"        # Ollama model to use
```

## Recommendations

- **For Development**: Use Ollama (free, local, fast)
- **For Production**: Use OpenAI API (most accurate, reliable)
- **For Testing**: Disable LLM (`USE_LLM=false`) for instant responses

