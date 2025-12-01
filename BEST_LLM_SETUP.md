# Best Free LLM Setup for HR Assistant

## 🚀 Recommended: Groq API (Fastest & Most Powerful FREE Option)

**Why Groq?**
- ⚡ **Extremely fast** - 10-50x faster than other APIs
- 🆓 **Free tier** - Generous free limits
- 🎯 **High quality** - Uses Llama 3.1 8B (very powerful)
- ✅ **Easy setup** - Just need an API key

**Setup:**
1. Get free API key: https://console.groq.com/
2. Set environment variables:
```bash
export GROQ_API_KEY="your-groq-api-key"
export USE_GROQ="true"
export USE_OLLAMA="false"
```

**Models available:**
- `llama-3.1-8b-instant` (default, recommended)
- `llama-3.1-70b-versatile` (most powerful, slower)
- `mixtral-8x7b-32768` (good for long context)

---

## 🥈 Alternative: Together AI (Powerful & Free)

**Why Together AI?**
- 🆓 **Free tier** - $25 free credits
- 🎯 **High quality** - Access to Llama 3, Mistral, etc.
- 📊 **Good for analytics** - Strong reasoning

**Setup:**
1. Get free API key: https://api.together.xyz/
2. Set environment variables:
```bash
export TOGETHER_API_KEY="your-together-api-key"
export USE_TOGETHER="true"
export USE_OLLAMA="false"
```

**Models available:**
- `meta-llama/Llama-3-8b-chat-hf` (default, recommended)
- `mistralai/Mixtral-8x7B-Instruct-v0.1` (very powerful)
- `Qwen/Qwen2.5-72B-Instruct` (excellent for data analysis)

---

## 🥉 Best Local Option: Ollama with Larger Models

**Why Ollama?**
- 🆓 **100% free** - No API keys needed
- 🔒 **Private** - All data stays local
- ⚡ **Fast** - No network latency

**Upgrade to better models:**

### Recommended Models (in order of quality):

1. **`llama3.1:8b`** ⭐ RECOMMENDED
   - Best balance of quality and speed
   - Great for MacBook Pro
   - Command: `ollama pull llama3.1:8b`
   - Set: `export OLLAMA_MODEL="llama3.1:8b"`

2. **`mistral:7b`**
   - Very good quality, fast
   - Command: `ollama pull mistral:7b`
   - Set: `export OLLAMA_MODEL="mistral:7b"`

3. **`qwen2.5:7b`**
   - Excellent for data analysis
   - Command: `ollama pull qwen2.5:7b`
   - Set: `export OLLAMA_MODEL="qwen2.5:7b"`

4. **`llama3.1:70b`** (Most Powerful)
   - Best quality, requires 16GB+ RAM
   - Command: `ollama pull llama3.1:70b`
   - Set: `export OLLAMA_MODEL="llama3.1:70b"`

**Current model is `gemma2:2b` - too small for good accuracy!**

---

## 📊 Comparison

| Option | Speed | Quality | Cost | Setup Difficulty |
|--------|-------|---------|------|------------------|
| **Groq API** | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 🆓 Free | ⭐ Easy |
| **Together AI** | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 🆓 Free | ⭐ Easy |
| **Ollama 8B** | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 🆓 Free | ⭐⭐ Medium |
| **Ollama 2B** (current) | ⚡⚡⚡⚡⚡ | ⭐⭐ | 🆓 Free | ⭐⭐ Medium |
| **OpenAI** | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 💰 Paid | ⭐ Easy |

---

## 🎯 Quick Start (Choose One)

### Option 1: Groq API (Recommended)
```bash
# 1. Get API key from https://console.groq.com/
# 2. Set environment variables
export GROQ_API_KEY="your-key-here"
export USE_GROQ="true"
export USE_OLLAMA="false"

# 3. Restart backend
```

### Option 2: Upgrade Ollama Model
```bash
# 1. Pull better model
ollama pull llama3.1:8b

# 2. Set environment variable
export OLLAMA_MODEL="llama3.1:8b"

# 3. Restart backend
```

### Option 3: Together AI
```bash
# 1. Get API key from https://api.together.xyz/
# 2. Set environment variables
export TOGETHER_API_KEY="your-key-here"
export USE_TOGETHER="true"
export USE_OLLAMA="false"

# 3. Restart backend
```

---

## 🔧 Update Your start_backend.sh

Add to your `start_backend.sh`:
```bash
# Best free LLM option - Groq API
export GROQ_API_KEY="your-groq-api-key"
export USE_GROQ="true"
export USE_OLLAMA="false"

# OR use better Ollama model
export OLLAMA_MODEL="llama3.1:8b"
```

---

## 💡 Why HR Assistant Wasn't Working Well

1. **Current model too small**: `gemma2:2b` is too small for complex reasoning
2. **Better models available**: 8B+ models are much more accurate
3. **Free APIs faster**: Groq/Together AI are faster AND more powerful than local 2B models

---

## ✅ After Setup

Test with:
```bash
# Test query
"what is the best recruitment source based on the number of hires?"
```

You should get accurate answers with exact numbers from your data!

