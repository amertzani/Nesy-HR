# Free LLM Options for HR Assistant

## 🚀 Option 1: Groq API (RECOMMENDED - Best Free Option)

**Why Groq?**
- ⚡ **Extremely fast** - 10-50x faster than other APIs
- 🆓 **100% FREE** - No credit card required
- 🎯 **High quality** - Uses Llama 3.1 8B (very powerful)
- ✅ **Easy setup** - Just need an API key

### Get Your Free API Key:
1. Go to: **https://console.groq.com/**
2. Click "Sign Up" (free, no credit card needed)
3. After signup, go to: **https://console.groq.com/keys**
4. Click "Create API Key"
5. Copy your key (starts with `gsk_...`)

### Setup:
```bash
export GROQ_API_KEY="gsk_your-key-here"
export USE_GROQ="true"
export USE_OLLAMA="false"
```

### Free Tier Limits:
- ✅ 30 requests per minute
- ✅ 14,400 requests per day
- ✅ Very fast responses (~1-2 seconds)

---

## 🥈 Option 2: Ollama with Better Models (100% Local & Free)

**Why Ollama?**
- 🆓 **100% free** - No API keys needed
- 🔒 **Private** - All data stays local
- ⚡ **Fast** - No network latency
- 📦 **No limits** - Use as much as you want

### Current Problem:
- You're using `gemma2:2b` - **too small for good accuracy**

### Solution: Upgrade to Better Model

**Recommended: `llama3.1:8b`** ⭐
```bash
# Pull the better model
ollama pull llama3.1:8b

# Set environment variable
export OLLAMA_MODEL="llama3.1:8b"
```

**Other good options:**
- `mistral:7b` - Very good quality, fast
- `qwen2.5:7b` - Excellent for data analysis
- `llama3.1:70b` - Most powerful (requires 16GB+ RAM)

### Setup:
```bash
# Make sure Ollama is running
ollama serve

# Pull better model
ollama pull llama3.1:8b

# Set model
export OLLAMA_MODEL="llama3.1:8b"
export USE_OLLAMA="true"
```

---

## 📊 Comparison

| Option | Speed | Quality | Cost | Setup |
|--------|-------|---------|------|-------|
| **Groq API** | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 🆓 Free | ⭐ Easy |
| **Ollama 8B** | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 🆓 Free | ⭐⭐ Medium |
| **Ollama 2B** (current) | ⚡⚡⚡⚡⚡ | ⭐⭐ | 🆓 Free | ⭐⭐ Medium |

---

## 🎯 Quick Start (Choose One)

### Option A: Groq API (Easiest & Fastest)
```bash
# 1. Get API key from https://console.groq.com/keys
# 2. Add to start_backend.sh or export:
export GROQ_API_KEY="gsk_your-key-here"
export USE_GROQ="true"
export USE_OLLAMA="false"

# 3. Restart backend
```

### Option B: Upgrade Ollama Model (Best for Privacy)
```bash
# 1. Pull better model
ollama pull llama3.1:8b

# 2. Set environment variable
export OLLAMA_MODEL="llama3.1:8b"

# 3. Restart backend
```

---

## 💡 Recommendation

**Start with Groq API** - It's:
- ✅ Fastest option
- ✅ Easiest to set up
- ✅ High quality (Llama 3.1 8B)
- ✅ 100% free, no credit card needed
- ✅ Generous free limits

If you prefer local/private, upgrade Ollama to `llama3.1:8b` instead of the current `gemma2:2b`.

---

## 🔧 Update Your start_backend.sh

Add one of these to your `start_backend.sh`:

**For Groq:**
```bash
export GROQ_API_KEY="gsk_your-key-here"
export USE_GROQ="true"
export USE_OLLAMA="false"
```

**For Better Ollama:**
```bash
export OLLAMA_MODEL="llama3.1:8b"
export USE_OLLAMA="true"
```

