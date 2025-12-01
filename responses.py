from knowledge import retrieve_context
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try to import transformers for lightweight LLM
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    LLM_AVAILABLE = True
    LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    LLM_AVAILABLE = False
    LLM_DEVICE = "cpu"
    print("⚠️  Transformers not available. Using rule-based responses.")

# LLM configuration
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"  # Enable by default
USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() == "true"  # Use Ollama by default (best for Mac)
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"  # Use OpenAI API if available
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Free API alternatives (powerful and fast)
USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"  # Groq API - VERY fast, free tier
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # Fast and powerful
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # Default Ollama URL
# Ollama models (fallback when Groq is unavailable)
# For 32GB RAM, recommended models (in order of quality):
# 1. "llama3.1:13b" - Best quality, still fast (RECOMMENDED for 32GB)
# 2. "mixtral:8x7b" - Very powerful, excellent reasoning
# 3. "llama3.1:8b" - Good balance, faster
# 4. "mistral:7b" - Very good quality, fast
# 5. "qwen2.5:7b" - Excellent for data analysis
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:13b")  # Best for 32GB RAM - great quality
# Free API alternatives (if you want cloud-based):
# - Groq API: Very fast, free tier (set USE_GROQ=true, GROQ_API_KEY)
# - Together AI: Free tier available (set USE_TOGETHER=true, TOGETHER_API_KEY)
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" if not USE_OPENAI and not USE_OLLAMA else None
LLM_PIPELINE = None

def load_llm_model():
    """Load lightweight LLM for RAG responses (lazy loading)"""
    global LLM_PIPELINE
    
    # Don't load if Ollama is enabled (we use Ollama instead)
    if USE_OLLAMA:
        return False
    
    # Don't load if model name is None or empty
    if not LLM_MODEL_NAME:
        return False
    
    if not LLM_AVAILABLE or not USE_LLM:
        return False
    
    if LLM_PIPELINE is not None:
        return True
    
    try:
        print("🔄 Loading lightweight LLM for research assistant...")
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float16 if LLM_DEVICE == "cuda" else torch.float32,
            device_map="auto" if LLM_DEVICE == "cuda" else None,
            trust_remote_code=True,
            attn_implementation="eager",  # Use eager attention to avoid flash-attn issues
        )
        
        LLM_PIPELINE = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if LLM_DEVICE == "cuda" else -1,
            torch_dtype=torch.float16 if LLM_DEVICE == "cuda" else torch.float32,
        )
        print(f"✅ LLM loaded successfully on {LLM_DEVICE}")
        return True
    except Exception as e:
        print(f"⚠️  Failed to load LLM: {e}")
        print("   Falling back to rule-based responses")
        import traceback
        traceback.print_exc()
        LLM_PIPELINE = None
        return False

def initialize_ollama():
    """Initialize Ollama - check if running and pull model if needed"""
    try:
        import requests
        # Use module-level variables
        global OLLAMA_BASE_URL, OLLAMA_MODEL
        
        # Check if Ollama is running
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code != 200:
                print(f"⚠️  Ollama is not running at {OLLAMA_BASE_URL}")
                print(f"   Start it with: ollama serve")
                return False
        except requests.exceptions.RequestException:
            print(f"⚠️  Cannot connect to Ollama at {OLLAMA_BASE_URL}")
            print(f"   Make sure Ollama is running: ollama serve")
            return False
        
        # Check if model is available
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                if OLLAMA_MODEL not in model_names:
                    print(f"📥 Model {OLLAMA_MODEL} not found. Pulling it now...")
                    print(f"   This may take a few minutes (model size: ~2GB)...")
                    
                    # Pull the model
                    pull_response = requests.post(
                        f"{OLLAMA_BASE_URL}/api/pull",
                        json={"name": OLLAMA_MODEL},
                        timeout=300  # 5 minute timeout for model download
                    )
                    
                    if pull_response.status_code == 200:
                        print(f"✅ Successfully pulled {OLLAMA_MODEL}")
                        return True
                    else:
                        print(f"⚠️  Failed to pull model: {pull_response.status_code}")
                        print(f"   Try manually: ollama pull {OLLAMA_MODEL}")
                        return False
                else:
                    print(f"✅ Ollama model {OLLAMA_MODEL} is ready")
                    return True
        except Exception as e:
            print(f"⚠️  Error checking/pulling model: {e}")
            return False
            
    except ImportError:
        print("⚠️  requests library not available. Install with: pip install requests")
        return False
    except Exception as e:
        print(f"⚠️  Ollama initialization error: {e}")
        return False

def generate_ollama_response(message, context):
    """Generate response using Ollama (best for MacBook Pro)"""
    try:
        import requests
        
        # Clean context - remove agent ownership info
        clean_context = context
        if "agent" in clean_context.lower() or "agent_id" in clean_context.lower():
            lines = clean_context.split('\n')
            clean_lines = [line for line in lines if 'agent' not in line.lower() or 'agent_id' not in line.lower()]
            clean_context = '\n'.join(clean_lines)
        
        # Build prompt focused on insights and traceability
        system_prompt = """You are an intelligent data analyst assistant that helps users understand their data and make evidence-based decisions.

CRITICAL RULES:
1. ALWAYS use EXACT numbers from the context - NEVER estimate, guess, or make up numbers
2. For questions about "best", "top", "most", "highest", "lowest" - find the EXACT value in the context and use it
3. For recruitment source questions: use the EXACT employee_count/hires number from the context
4. Always provide TRACEABLE answers - cite the specific facts from the knowledge graph that support your answer
5. For structured queries (max/min/filter): provide the exact answer first, then explain the evidence
6. Focus on INSIGHTS and ACTIONABLE INFORMATION, not raw data dumps
7. Synthesize information to provide clear, concise answers
8. For CSV/statistical questions: highlight key patterns, trends, and what they mean for decision-making
9. When presenting statistics: explain what they mean, not just list numbers
10. Group related information together
11. If you see insights or correlations, explain their significance
12. Be helpful and conversational - help the user understand, not just list facts
13. For employee queries: use exact names and values from the knowledge graph
14. Always ground your answers in the provided context facts - if a number isn't in the context, don't use it
15. **FOR SPECIFIC CORRELATION QUESTIONS**: When asked about correlation between two specific columns (e.g., "correlation between X and Y"), you MUST:
    - Find the "Specific Correlation Requested" section in the context
    - Give the EXACT correlation value (e.g., -0.64, 0.75) from that section
    - Explain what the correlation means (strong/moderate/weak, positive/negative)
    - If the specific correlation is shown, prioritize it over other correlations in your answer"""
        
        prompt = f"{system_prompt}\n\nKnowledge Base Context:\n{clean_context}\n\nQuestion: {message}\n\nAnswer:"
        
        # Call Ollama API
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more accurate, factual responses
                    "num_predict": 300,  # Limit tokens for faster response
                    "top_p": 0.9,  # Nucleus sampling for better quality
                }
            },
            timeout=60  # 60 second timeout (increased for better reliability)
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "").strip()
            if answer and len(answer) > 10:
                return answer
        else:
            print(f"⚠️  Ollama API error: {response.status_code} - {response.text}")
            return None
            
    except ImportError:
        print("⚠️  requests library not available. Install with: pip install requests")
        return None
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Ollama connection error: {e}")
        print(f"   Make sure Ollama is running: ollama serve")
        print(f"   And model is available: ollama pull {OLLAMA_MODEL}")
        return None
    except Exception as e:
        print(f"⚠️  Ollama error: {e}")
        return None

def generate_groq_response(message, context):
    """Generate response using Groq API - VERY fast and powerful, free tier available"""
    try:
        import requests
        
        clean_context = context
        if "agent" in clean_context.lower() or "agent_id" in clean_context.lower():
            lines = clean_context.split('\n')
            clean_lines = [line for line in lines if 'agent' not in line.lower() or 'agent_id' not in line.lower()]
            clean_context = '\n'.join(clean_lines)
        
        system_prompt = """You are an intelligent data analyst assistant that helps users understand their data and make evidence-based decisions.

CRITICAL RULES:
1. ALWAYS use EXACT numbers from the context - NEVER estimate, guess, or make up numbers
2. For questions about "best", "top", "most", "highest", "lowest" - find the EXACT value in the context and use it
3. For recruitment source questions: use the EXACT employee_count/hires number from the context
4. Always provide TRACEABLE answers - cite the specific facts from the knowledge graph that support your answer
5. For structured queries (max/min/filter): provide the exact answer first, then explain the evidence
6. Focus on INSIGHTS and ACTIONABLE INFORMATION, not raw data dumps
7. Synthesize information to provide clear, concise answers
8. For CSV/statistical questions: highlight key patterns, trends, and what they mean for decision-making
9. When presenting statistics: explain what they mean, not just list numbers
10. Group related information together
11. If you see insights or correlations, explain their significance
12. Be helpful and conversational - help the user understand, not just list facts
13. For employee queries: use exact names and values from the knowledge graph
14. Always ground your answers in the provided context facts - if a number isn't in the context, don't use it
15. **FOR SPECIFIC CORRELATION QUESTIONS**: When asked about correlation between two specific columns (e.g., "correlation between X and Y"), you MUST:
    - Find the "Specific Correlation Requested" section in the context
    - Give the EXACT correlation value (e.g., -0.64, 0.75) from that section
    - Explain what the correlation means (strong/moderate/weak, positive/negative)
    - If the specific correlation is shown, prioritize it over other correlations in your answer"""
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Knowledge Base Context:\n{clean_context}\n\nQuestion: {message}"}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            if answer and len(answer) > 10:
                return answer
        else:
            print(f"⚠️  Groq API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"⚠️  Groq API error: {e}")
        return None

# Together AI removed - not free anymore
# def generate_together_response(message, context):
#     ...

def generate_llm_response(message, context):
    """
    Generate response using LLM with RAG - supports multiple providers (in order of preference):
    1. Groq API (fastest, free tier) ⭐ PRIMARY
    2. Ollama (local, free fallback) - llama3.1:13b recommended for 32GB RAM
    3. OpenAI API (most accurate, paid)
    4. Local transformers model (last resort)
    """
    if not USE_LLM:
        return None
    
    # Try Groq first (fastest, free tier available) - PRIMARY
    if USE_GROQ and GROQ_API_KEY:
        result = generate_groq_response(message, context)
        if result:
            return result
    
    # Try Ollama as fallback (local, free) - GOOD for 32GB RAM
    if USE_OLLAMA:
        ollama_response = generate_ollama_response(message, context)
        if ollama_response:
            return ollama_response
    
    # Try OpenAI API if configured (paid)
    if USE_OPENAI and OPENAI_API_KEY:
        try:
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
            # Build intelligent prompt focused on insights and traceability
            system_prompt = """You are an intelligent data analyst assistant that helps users understand their data and make evidence-based decisions.

CRITICAL RULES:
1. ALWAYS use EXACT numbers from the context - NEVER estimate, guess, or make up numbers
2. For questions about "best", "top", "most", "highest", "lowest" - find the EXACT value in the context and use it
3. For recruitment source questions: use the EXACT employee_count/hires number from the context
4. Always provide TRACEABLE answers - cite the specific facts from the knowledge graph that support your answer
5. For structured queries (max/min/filter): provide the exact answer first, then explain the evidence
6. Focus on INSIGHTS and ACTIONABLE INFORMATION, not raw data dumps
7. Synthesize information to provide clear, concise answers
8. For CSV/statistical questions: highlight key patterns, trends, and what they mean for decision-making
9. When presenting statistics: explain what they mean, not just list numbers
10. Group related information together
11. If you see insights or correlations, explain their significance
12. Be helpful and conversational - help the user understand, not just list facts
13. If information is missing, clearly state what's missing and what would be helpful
14. For employee queries: use exact names and values from the knowledge graph
15. Always ground your answers in the provided context facts - if a number isn't in the context, don't use it"""
            
            # Clean context - remove agent ownership info
            clean_context = context
            if "agent" in clean_context.lower() or "agent_id" in clean_context.lower():
                # Remove agent-related metadata from context
                lines = clean_context.split('\n')
                clean_lines = [line for line in lines if 'agent' not in line.lower() or 'agent_id' not in line.lower()]
                clean_context = '\n'.join(clean_lines)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Fast and smart, works well on Mac
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Knowledge Base Context:\n{clean_context}\n\nQuestion: {message}"}
                ],
                temperature=0.3,  # Lower temperature for more accurate, factual responses
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            if answer and len(answer) > 10:
                return answer
        except Exception as e:
            print(f"⚠️  OpenAI API error: {e}, falling back to local model")
            # Fall through to local model
    
    # Try OpenAI API if configured
    if USE_OPENAI and OPENAI_API_KEY:
        try:
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
            # Build intelligent prompt focused on insights and traceability
            system_prompt = """You are an intelligent data analyst assistant that helps users understand their data and make evidence-based decisions.

CRITICAL RULES:
1. ALWAYS use EXACT numbers from the context - NEVER estimate, guess, or make up numbers
2. For questions about "best", "top", "most", "highest", "lowest" - find the EXACT value in the context and use it
3. For recruitment source questions: use the EXACT employee_count/hires number from the context
4. Always provide TRACEABLE answers - cite the specific facts from the knowledge graph that support your answer
5. For structured queries (max/min/filter): provide the exact answer first, then explain the evidence
6. Focus on INSIGHTS and ACTIONABLE INFORMATION, not raw data dumps
7. Synthesize information to provide clear, concise answers
8. For CSV/statistical questions: highlight key patterns, trends, and what they mean for decision-making
9. When presenting statistics: explain what they mean, not just list numbers
10. Group related information together
11. If you see insights or correlations, explain their significance
12. Be helpful and conversational - help the user understand, not just list facts
13. If information is missing, clearly state what's missing and what would be helpful
14. For employee queries: use exact names and values from the knowledge graph
15. Always ground your answers in the provided context facts - if a number isn't in the context, don't use it"""
            
            # Clean context - remove agent ownership info
            clean_context = context
            if "agent" in clean_context.lower() or "agent_id" in clean_context.lower():
                # Remove agent-related metadata from context
                lines = clean_context.split('\n')
                clean_lines = [line for line in lines if 'agent' not in line.lower() or 'agent_id' not in line.lower()]
                clean_context = '\n'.join(clean_lines)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Fast and smart, works well on Mac
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Knowledge Base Context:\n{clean_context}\n\nQuestion: {message}"}
                ],
                temperature=0.3,  # Lower temperature for more accurate, factual responses
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            if answer and len(answer) > 10:
                return answer
        except Exception as e:
            print(f"⚠️  OpenAI API error: {e}, falling back to local model")
            # Fall through to local model
    
    # Use local model
    if not LLM_AVAILABLE:
        return None
    
    try:
        if LLM_PIPELINE is None:
            if not load_llm_model():
                return None
        
        # Build prompt with context from knowledge graph - focused on insights and traceability
        system_prompt = """You are an intelligent data analyst assistant that helps users understand their data and make evidence-based decisions.

CRITICAL RULES:
1. Always provide TRACEABLE answers - cite the specific facts from the knowledge graph that support your answer
2. For structured queries (max/min/filter): provide the exact answer first, then explain the evidence
3. Focus on INSIGHTS and ACTIONABLE INFORMATION, not raw data dumps
4. Synthesize information to provide clear, concise answers
5. For CSV/statistical questions: highlight key patterns, trends, and what they mean
6. When presenting statistics: explain what they mean, not just list numbers
7. Group related information together
8. If you see insights or correlations, explain their significance
9. Be helpful and conversational - help the user understand
# 10. NEVER mention agent IDs or technical metadata - 
10. focus on the actual data insights
11. For employee queries: use exact names and values from the knowledge graph
12. Always ground your answers in the provided context facts
13. **FOR SPECIFIC CORRELATION QUESTIONS**: When asked about correlation between two specific columns (e.g., "correlation between X and Y"), you MUST:
    - Find the "Specific Correlation Requested" section in the context
    - Give the EXACT correlation value (e.g., -0.64, 0.75) from that section
    - Explain what the correlation means (strong/moderate/weak, positive/negative)
    - If the specific correlation is shown, prioritize it over other correlations in your answer"""
        
        # Format context from knowledge graph - remove agent ownership info
        if context and "No directly relevant facts found" not in context and "Partially Relevant" not in context:
            # Remove markdown formatting but keep the content
            context_text = context.replace("**Relevant Knowledge from Your Documents:**\n", "").replace("**Relevant Knowledge from Knowledge Base:**\n", "").strip()
            if not context_text:
                context_text = context.strip()
            
            # Remove agent ownership information from context
            lines = context_text.split('\n')
            clean_lines = []
            for line in lines:
                # Skip lines mentioning agent IDs or agent ownership
                if 'agent' in line.lower() and ('id' in line.lower() or 'owner' in line.lower()):
                    continue
                clean_lines.append(line)
            context_text = '\n'.join(clean_lines)
        else:
            context_text = "No specific relevant facts found in the knowledge base for this question."
        
        # Build conversation prompt
        prompt = f"""<|system|>
{system_prompt}

Knowledge Base Context:
{context_text}
<|user|>
{message}
<|assistant|>
"""
        
        # Generate response with fixed parameters
        # Use model.generate directly to have more control
        inputs = LLM_PIPELINE.tokenizer(prompt, return_tensors="pt")
        if LLM_DEVICE == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Adjust parameters based on device (CPU is much slower)
        if LLM_DEVICE == "cpu":
            max_new_tokens = 40  # Reduced for CPU (much slower)
            max_time = 15.0  # Shorter timeout for CPU
            do_sample = False  # Greedy decoding is faster on CPU
            print(f"🐌 Using CPU mode - reduced tokens ({max_new_tokens}), greedy decoding, timeout ({max_time}s)")
        else:
            max_new_tokens = 60  # More tokens for GPU
            max_time = 20.0  # Longer timeout for GPU
            do_sample = True  # Sampling for better quality on GPU
        
        generated_text = None
        try:
            # Disable cache to avoid DynamicCache issues
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": LLM_PIPELINE.tokenizer.eos_token_id,
                "use_cache": False,  # Disable cache to avoid DynamicCache errors
                "max_time": max_time,
            }
            
            if do_sample:
                generate_kwargs.update({
                    "temperature": 0.6,
                    "top_p": 0.85,
                    "do_sample": True,
                })
            else:
                generate_kwargs["do_sample"] = False
            
            outputs = LLM_PIPELINE.model.generate(**inputs, **generate_kwargs)
            generated_text = LLM_PIPELINE.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            # Fallback to pipeline if direct generation fails
            print(f"⚠️  Direct generation failed: {e}, trying pipeline...")
            try:
                # Adjust parameters for fallback pipeline too
                if LLM_DEVICE == "cpu":
                    fallback_max_tokens = 40
                    fallback_max_time = 15.0
                    fallback_do_sample = False
                else:
                    fallback_max_tokens = 60
                    fallback_max_time = 20.0
                    fallback_do_sample = True
                
                pipeline_kwargs = {
                    "max_new_tokens": fallback_max_tokens,
                    "pad_token_id": LLM_PIPELINE.tokenizer.eos_token_id,
                    "max_time": fallback_max_time,
                    "return_full_text": False,
                }
                
                if fallback_do_sample:
                    pipeline_kwargs.update({
                        "temperature": 0.6,
                        "top_p": 0.85,
                        "do_sample": True,
                    })
                else:
                    pipeline_kwargs["do_sample"] = False
                
                response = LLM_PIPELINE(prompt, **pipeline_kwargs)
                generated_text = response[0]['generated_text']
            except Exception as e2:
                print(f"⚠️  Pipeline generation also failed: {e2}")
                import traceback
                traceback.print_exc()
                return None
        
        if not generated_text:
            return None
        
        # Remove the prompt part from generated text
        # The model should generate after <|assistant|>
        if "<|assistant|>" in generated_text:
            answer = generated_text.split("<|assistant|>")[-1].strip()
        elif prompt in generated_text:
            # Remove prompt if it's at the start
            answer = generated_text.split(prompt, 1)[-1].strip()
        else:
            answer = generated_text.strip()
        
        # Clean up the response - remove any remaining special tokens or prompt fragments
        answer = answer.split("<|end|>")[0].strip()
        answer = answer.split("<|user|>")[0].strip()
        answer = answer.split("<|system|>")[0].strip()
        answer = answer.split("<|assistant|>")[0].strip()  # In case it appears again
        
        # Remove any remaining special tokens
        answer = answer.replace("<|endoftext|>", "").strip()
        answer = answer.replace("<|end|>", "").strip()
        
        # If answer is too short or seems to be just the prompt, return None to fallback
        if not answer or len(answer) < 5:
            return None
        
        # Check if answer is just repeating the prompt
        if answer.startswith("You are a helpful") or answer.startswith("Knowledge Base Context"):
            return None
        
        return answer
        
    except Exception as e:
        print(f"⚠️  LLM generation error: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_document_summary(context):
    if not context or "No directly relevant facts found" in context:
        return "I don't have enough information about this document to provide a summary. Please add more knowledge to the knowledge base first."
    facts = []
    for line in context.split('\n'):
        if line.strip() and not line.startswith('**'):
            facts.append(line.strip())
    document_type = "document"
    key_info = []
    for fact in facts:
        fact_lower = fact.lower()
        if 'invoice' in fact_lower or 'bill' in fact_lower:
            document_type = "invoice"
        elif 'contract' in fact_lower or 'agreement' in fact_lower:
            document_type = "contract"
        elif 'report' in fact_lower or 'analysis' in fact_lower:
            document_type = "report"
        elif any(k in fact_lower for k in ['company','organization','name','amount','total','cost','price','date','time','address','location','description','type','id','number','code']):
            key_info.append(fact)
    summary = f"Based on the information in my knowledge base, this appears to be a **{document_type}** document. "
    if key_info:
        summary += "Here are the key details I found:\n\n"
        for info in key_info[:5]:
            summary += f"• {info}\n"
    else:
        summary += "However, I don't have enough specific details to provide a comprehensive summary."
    return summary

def _facts_from_context(context):
    """Extract facts from context, preserving the full formatted fact strings"""
    facts = []
    for line in context.split('\n'):
        line = line.strip()
        # Skip empty lines and section headers
        if line and not line.startswith('**') and not line.startswith('Relevant Knowledge'):
            # Remove numbering if present (e.g., "1. " or "1) ")
            if line and (line[0].isdigit() and (line[1:3] in ['. ', ') '] or line[1:2] == '.')):
                line = line.split('. ', 1)[-1] if '. ' in line else line.split(') ', 1)[-1] if ') ' in line else line[2:].strip()
            facts.append(line)
    return facts

def generate_what_response(message, context):
    facts = _facts_from_context(context)
    if not facts:
        return "I don't have specific information about that in my knowledge base."
    response = f"Based on my knowledge base, here are all {len(facts)} relevant facts:\n\n"
    for i, fact in enumerate(facts, 1):
        response += f"• {fact}\n"
    return response

def generate_who_response(message, context):
    facts = _facts_from_context(context)
    facts = [f for f in facts if any(k in f.lower() for k in ['company','name','person','επωνυμία','εταιρεία'])]
    if not facts:
        return "I don't have specific information about people or companies in my knowledge base."
    return f"Here are all {len(facts)} facts about people/entities:\n\n" + "\n".join(f"• {f}" for f in facts)

def generate_when_response(message, context):
    facts = _facts_from_context(context)
    facts = [f for f in facts if any(k in f.lower() for k in ['date','ημερομηνία','due','προθεσμία'])]
    if not facts:
        return "I don't have specific date information in my knowledge base."
    return f"Here are all {len(facts)} facts with date information:\n\n" + "\n".join(f"• {f}" for f in facts)

def generate_where_response(message, context):
    facts = _facts_from_context(context)
    facts = [f for f in facts if any(k in f.lower() for k in ['address','διεύθυνση','location','place'])]
    if not facts:
        return "I don't have specific location information in my knowledge base."
    return f"Here are all {len(facts)} facts with location information:\n\n" + "\n".join(f"• {f}" for f in facts)

def generate_amount_response(message, context):
    facts = _facts_from_context(context)
    facts = [f for f in facts if any(k in f.lower() for k in ['amount','total','price','cost','σύνολο','φόρος','€','$'])]
    if not facts:
        return "I don't have specific financial information in my knowledge base."
    return f"Here are all {len(facts)} facts with financial information:\n\n" + "\n".join(f"• {f}" for f in facts)

def generate_general_response(message, context):
    facts = _facts_from_context(context)
    if not facts:
        return "I don't have relevant information about that in my knowledge base."
    response = f"Based on my knowledge base, here are all {len(facts)} relevant facts:\n\n"
    for i, fact in enumerate(facts, 1):
        response += f"• {fact}\n"
    return response

def generate_intelligent_response(message, context, system_message):
    """Generate intelligent, insight-focused responses"""
    message_lower = message.lower()
    
    # Check if this is about CSV/statistics - provide insight-focused response
    csv_keywords = ["csv", "statistics", "stat", "data", "analysis", "column", "row", "mean", "median", 
                    "distribution", "outlier", "correlation", "insight", "pattern", "trend"]
    is_csv_query = any(keyword in message_lower for keyword in csv_keywords)
    
    # Extract facts from context (removing agent ownership info)
    facts = _facts_from_context(context)
    
    # Remove agent ownership from facts
    clean_facts = []
    for fact in facts:
        # Skip facts mentioning agent IDs
        if 'agent' in fact.lower() and ('id' in fact.lower() or 'owner' in fact.lower()):
            continue
        clean_facts.append(fact)
    facts = clean_facts
    
    if not facts:
        return "I don't have specific information about that in my knowledge base."
    
    # For CSV/statistical queries, provide insight-focused summary
    if is_csv_query:
        # Group facts by type
        insights = [f for f in facts if 'insight' in f.lower() or 'correlat' in f.lower()]
        statistics = [f for f in facts if any(k in f.lower() for k in ['mean', 'median', 'mode', 'percentage', '%'])]
        data_quality = [f for f in facts if 'quality' in f.lower() or 'outlier' in f.lower() or 'missing' in f.lower()]
        
        response = "Based on the data analysis:\n\n"
        
        if insights:
            response += "🔍 **Key Insights:**\n"
            for insight in insights[:5]:  # Top 5 insights
                response += f"• {insight}\n"
            response += "\n"
        
        if data_quality:
            response += "⚠️ **Data Quality Notes:**\n"
            for dq in data_quality[:3]:  # Top 3 data quality issues
                response += f"• {dq}\n"
            response += "\n"
        
        if statistics:
            response += "📊 **Key Statistics:**\n"
            for stat in statistics[:5]:  # Top 5 statistics
                response += f"• {stat}\n"
        
        # If there are more facts, mention them
        remaining = len(facts) - len(insights) - len(statistics) - len(data_quality)
        if remaining > 0:
            response += f"\n(Plus {remaining} additional data points available)"
        
        return response
    
    # For other queries, use pattern matching
    if any(phrase in message_lower for phrase in [
        'what is the document about', 'whats the document about', 'what is this about', 'whats this about', 
        'describe the document', 'summarize the document', 'what does this contain'
    ]):
        return generate_document_summary(context)
    elif message_lower.startswith('what'):
        # Provide focused answer, not all facts
        if len(facts) > 10:
            return f"Based on the knowledge base, here are the key points:\n\n" + "\n".join(f"• {f}" for f in facts[:8]) + f"\n\n(Plus {len(facts) - 8} more facts available)"
        return f"Based on the knowledge base:\n\n" + "\n".join(f"• {f}" for f in facts)
    elif message_lower.startswith('who'):
        facts = [f for f in facts if any(k in f.lower() for k in ['company','name','person','employee','staff'])]
        if not facts:
            return "I don't have specific information about people or companies in my knowledge base."
        return f"Here are the key people/entities:\n\n" + "\n".join(f"• {f}" for f in facts[:8])
    elif message_lower.startswith('when'):
        facts = [f for f in facts if any(k in f.lower() for k in ['date','time','when'])]
        if not facts:
            return "I don't have specific date information in my knowledge base."
        return f"Here are the key dates/times:\n\n" + "\n".join(f"• {f}" for f in facts[:8])
    elif message_lower.startswith('where'):
        facts = [f for f in facts if any(k in f.lower() for k in ['address','location','place','office'])]
        if not facts:
            return "I don't have specific location information in my knowledge base."
        return f"Here are the key locations:\n\n" + "\n".join(f"• {f}" for f in facts[:8])
    elif any(phrase in message_lower for phrase in ['how much','amount','total','cost','price']):
        facts = [f for f in facts if any(k in f.lower() for k in ['amount','total','price','cost','salary','€','$'])]
        if not facts:
            return "I don't have specific financial information in my knowledge base."
        return f"Here are the key financial details:\n\n" + "\n".join(f"• {f}" for f in facts[:8])
    else:
        # General response - focus on insights, limit facts
        if len(facts) > 10:
            return f"Based on the knowledge base, here are the key points:\n\n" + "\n".join(f"• {f}" for f in facts[:8]) + f"\n\n(Plus {len(facts) - 8} more facts available)"
        return f"Based on the knowledge base:\n\n" + "\n".join(f"• {f}" for f in facts)

def respond(message, history=None, system_message="You are an intelligent assistant that answers questions based on factual information from a knowledge base. You provide clear, accurate, and helpful responses. When you have relevant information, you share it directly. When you don't have enough information, you clearly state this limitation. You always stay grounded in the facts provided and never hallucinate information."):
    # CSV queries are handled by the normal LLM flow with context retrieval
    # The LLM will use the facts extracted from CSV data to answer questions
    
    # Try orchestrator-based query processing first
    try:
        import importlib
        orchestrator_module = importlib.import_module('orchestrator')
        orchestrate_query = orchestrator_module.orchestrate_query
        
        query_processor = importlib.import_module('query_processor')
        detect_query_type = query_processor.detect_query_type
        build_evidence_context = query_processor.build_evidence_context
        
        query_info = detect_query_type(message)
        
        # Try orchestrator for operational queries (keyword-based routing)
        if query_info.get("query_type") == "operational":
            answer, evidence_facts, routing_info = orchestrate_query(message, query_info)
            
            if answer:
                # For operational queries, use the answer directly (it already contains formatted response)
                # Build evidence context if needed
                evidence_context = build_evidence_context(evidence_facts, message)
                
                # Format response with evidence and routing info
                response = answer
                if evidence_context and "Based on operational insights" not in answer:
                    response += f"\n\n{evidence_context}"
                if routing_info.get('reason'):
                    response += f"\n\n*Query processed via operational analysis: {routing_info['reason']}*"
                
                return response
            else:
                # Query detected but no answer found - continue with normal flow
                # The normal respond flow will retrieve context from knowledge graph
                # Facts from "operational_insights" source will be prioritized
                pass  # Continue to normal LLM flow below
        
        # Try orchestrator for strategic queries (keyword-based or pattern-based)
        if query_info.get("query_type") == "strategic":
            strategic_type = query_info.get("strategic_type")
            query_type_label = "operational" if strategic_type in ['O1', 'O2', 'O3', 'O4'] else "strategic"
            answer, evidence_facts, routing_info = orchestrate_query(message, query_info)
            
            if answer:
                # Build evidence context
                evidence_context = build_evidence_context(evidence_facts, message)
                
                # Format response with evidence and routing info
                response = answer
                if evidence_context:
                    response += f"\n\n{evidence_context}"
                if routing_info.get('reason'):
                    response += f"\n\n*Query processed via {query_type_label} analysis: {routing_info['reason']}*"
                
                return response
            else:
                # Query detected but no answer found
                if strategic_type:
                    return f"I detected a {query_type_label} query ({strategic_type}) but couldn't process it. Please ensure a CSV file has been uploaded with the required columns."
                else:
                    return f"I detected a strategic query but couldn't process it. Please ensure a CSV file has been uploaded with the required columns."
        
        # Use orchestrator for structured queries
        if query_info.get("query_type") == "structured":
            answer, evidence_facts, routing_info = orchestrate_query(message, query_info)
            
            if answer:
                # Build evidence context
                evidence_context = build_evidence_context(evidence_facts, message)
                
                # Format response with evidence and routing info
                response = answer
                if evidence_context:
                    response += f"\n\n{evidence_context}"
                if routing_info.get('reason'):
                    response += f"\n\n*Query routed via orchestrator: {routing_info['reason']}*"
                
                return response
        
        # Fallback to direct query processing if orchestrator didn't find answer
        if query_info.get("query_type") == "structured":
            process_structured_query = query_processor.process_structured_query
            answer, evidence_facts = process_structured_query(message, query_info)
            
            if answer:
                evidence_context = build_evidence_context(evidence_facts, message)
                response = answer
                if evidence_context:
                    response += f"\n\n{evidence_context}"
                print(f"✅ Direct query answered: {query_info.get('operation')} - {answer}")
                return response
            else:
                # Structured query detected but no answer found - return helpful error instead of using LLM
                operation = query_info.get('operation', 'query')
                attribute = query_info.get('attribute', 'information')
                entity = query_info.get('entity_name', '')
                
                if operation == "filter" and entity:
                    error_msg = f"I couldn't find {attribute} information for {entity} in the knowledge base. "
                    error_msg += f"Please check if the name is spelled correctly or if this employee exists in the uploaded documents."
                elif operation in ["max", "min"]:
                    error_msg = f"I couldn't find any employees with {attribute} information in the knowledge base. "
                    error_msg += f"Please check if the data was uploaded correctly."
                else:
                    error_msg = f"I couldn't process this structured query. Please try rephrasing your question."
                
                print(f"⚠️  Structured query failed: {error_msg}")
                return error_msg
        
    except ImportError as e:
        print(f"⚠️  Orchestrator/query processor not available: {e}")
        # Fall through to normal processing
    except Exception as e:
        print(f"⚠️  Orchestrator query processing failed: {e}")
        import traceback
        traceback.print_exc()
        # If it was a structured query, don't fall through to LLM - return error instead
        try:
            query_processor = importlib.import_module('query_processor')
            detect_query_type = query_processor.detect_query_type
            query_info = detect_query_type(message)
            if query_info.get("query_type") == "structured":
                return f"I encountered an error processing your structured query. Please try rephrasing your question. Error: {str(e)}"
        except:
            pass
        # Fall through to normal processing for non-structured queries
    
    # CRITICAL: For structured queries, NEVER use LLM - return error instead
    # This prevents timeout issues with large documents
    try:
        query_processor = importlib.import_module('query_processor')
        detect_query_type = query_processor.detect_query_type
        query_info = detect_query_type(message)
        if query_info.get("query_type") == "structured":
            return "I detected this as a structured query but couldn't process it. Please try rephrasing your question or check if the data was uploaded correctly."
    except:
        pass
    
    # Check if this is a statistics query (correlation, distribution, min/max)
    # and get statistics context for LLM
    stats_context = None
    try:
        query_processor = importlib.import_module('query_processor')
        detect_query_type = query_processor.detect_query_type
        query_info = detect_query_type(message)
        
        # Check if this should be a statistics query
        orchestrator_module = importlib.import_module('orchestrator')
        find_agents_for_query = orchestrator_module.find_agents_for_query
        temp_routing = find_agents_for_query(message, query_info.get("query_type", "general"), 
                                            query_info.get("attribute"), query_info)
        if temp_routing.get("strategy") == "statistics_agent":
            # Get statistics context
            try:
                from agent_system import format_statistics_context_for_llm
                from strategic_query_agent import get_all_statistics
                all_stats = get_all_statistics()
                if all_stats:
                    stats_context = format_statistics_context_for_llm(message, all_stats)
            except Exception as e:
                print(f"⚠️  Error getting statistics context: {e}")
    except Exception as e:
        print(f"⚠️  Error checking statistics routing: {e}")
    
    # Retrieve relevant context from knowledge graph (only for non-structured queries)
    context = retrieve_context(message)
    
    # Add statistics context if available
    if stats_context:
        context = f"{stats_context}\n\n---\n\n{context}" if context else stats_context
    
    # Enable LLM if Ollama or OpenAI API is configured
    use_llm = USE_OLLAMA or (USE_OPENAI and OPENAI_API_KEY)  # Use LLM if Ollama or OpenAI is available
    
    if use_llm and USE_LLM:
        try:
            llm_response = generate_llm_response(message, context)
            if llm_response and len(llm_response.strip()) > 5:
                return llm_response
        except Exception as e:
            print(f"⚠️  LLM response generation failed: {e}")
            import traceback
            traceback.print_exc()
            # Fall through to rule-based response
    
    # Use rule-based response (fast and reliable)
    return generate_intelligent_response(message, context, system_message)


