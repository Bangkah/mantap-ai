import asyncio
import httpx
from fastapi import FastAPI, Request, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
from typing import Optional, List
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime
import uuid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MantapAI")

OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_TAGS_API = "http://localhost:11434/api/tags"
DEFAULT_MODEL = "phi3"
FALLBACK_MODELS = ["mistral:7b", "llama2:latest", "gemma:2b"]
TIMEOUT = 45
MAX_TOKENS = 4000
MAX_PROMPT_LENGTH = 8000

class OllamaClient:
    def __init__(self):
        self.base_url = OLLAMA_API
        self.tags_url = OLLAMA_TAGS_API
        self.available_models = []
        self.last_model_check = None

    async def check_connection(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(self.tags_url)
                if response.status_code == 200:
                    data = response.json()
                    self.available_models = [m["name"] for m in data.get("models", [])]
                    self.last_model_check = datetime.now()
                    return True
                return False
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False

    async def get_best_model(self, preferred_model: str) -> str:
        if not self.available_models:
            await self.check_connection()
        
        if preferred_model in self.available_models:
            return preferred_model
        
        for fallback in FALLBACK_MODELS:
            if fallback in self.available_models:
                return fallback
        
        return self.available_models[0] if self.available_models else DEFAULT_MODEL

    async def generate_response(self, model: str, prompt: str, stream: bool = False) -> dict:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "num_ctx": 4096,
                "num_predict": MAX_TOKENS
            }
        }

        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                if stream:
                    return await self._handle_stream_response(client, payload)
                else:
                    response = await client.post(self.base_url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return {
                        "success": True,
                        "response": data.get("response", "").strip(),
                        "model": data.get("model", model),
                        "done": data.get("done", True)
                    }
        except httpx.TimeoutException:
            return {"success": False, "error": "Request timeout, server sedang sibuk"}
        except httpx.ConnectError:
            return {"success": False, "error": "Tidak dapat terhubung ke Ollama server"}
        except httpx.HTTPStatusError as e:
            return {"success": False, "error": f"HTTP error {e.response.status_code}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

    async def _handle_stream_response(self, client: httpx.AsyncClient, payload: dict):
        try:
            async with client.stream("POST", self.base_url, json=payload) as response:
                response.raise_for_status()
                full_text = ""
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                full_text += data["response"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                
                return {
                    "success": True,
                    "response": full_text.strip(),
                    "model": payload["model"],
                    "done": True
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

ollama_client = OllamaClient()

def detect_language(text: str) -> str:
    indonesian_words = ['adalah', 'yang', 'dan', 'dengan', 'untuk', 'dari', 'dalam', 'pada', 'ke', 'di', 'apa', 'bagaimana', 'mengapa', 'kapan', 'dimana', 'siapa']
    english_words = ['is', 'are', 'the', 'and', 'with', 'for', 'from', 'in', 'on', 'to', 'at', 'what', 'how', 'why', 'when', 'where', 'who']
    
    text_lower = text.lower()
    id_count = sum(1 for word in indonesian_words if word in text_lower)
    en_count = sum(1 for word in english_words if word in text_lower)
    
    return 'id' if id_count > en_count else 'en'

def try_eval_math(expr: str) -> Optional[str]:
    expr = re.sub(r'[^\d+\-*/.() ]', '', expr.strip())
    if re.fullmatch(r"[0-9+\-*/().\s]+", expr) and any(op in expr for op in ['+', '-', '*', '/']):
        try:
            result = eval(expr, {"__builtins__": {}})
            return f"Hasil: {result}"
        except:
            return None
    return None

def create_smart_prompt(prompt: str, language: str) -> str:
    if language == 'id':
        system_prompt = """Kamu adalah Mantap AI, asisten AI yang cerdas dan helpful.
Jawab dengan bahasa Indonesia yang natural dan mudah dipahami.
Berikan jawaban yang akurat, ringkas, dan informatif.
Jika tidak tahu jawaban pasti, katakan dengan jujur."""
    else:
        system_prompt = """You are Mantap AI, an intelligent and helpful AI assistant.
Answer in natural English that's easy to understand.
Provide accurate, concise, and informative responses.
If you don't know the answer for certain, be honest about it."""
    
    return f"{system_prompt}\n\nUser: {prompt}\nAssistant:"

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Mantap AI Backend...")
    connection_status = await ollama_client.check_connection()
    if connection_status:
        logger.info(f"✅ Connected to Ollama. Available models: {len(ollama_client.available_models)}")
    else:
        logger.warning("⚠️ Ollama server not available at startup")
    
    yield
    
    logger.info("🔄 Shutting down Mantap AI Backend...")

app = FastAPI(
    title="Mantap AI",
    description="Intelligent AI Assistant Backend",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
if os.path.exists("templates"):
    templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if os.path.exists("templates/index.html"):
        return templates.TemplateResponse("index.html", {"request": request})
    
    return HTMLResponse("""
    <!DOCTYPE html>
    <html><head><title>Mantap AI</title></head>
    <body style="font-family:Arial;padding:20px;text-align:center;">
        <h1>🤖 Mantap AI Backend</h1>
        <p>Backend server is running successfully!</p>
        <p><a href="/health">Health Check</a> | <a href="/models">Available Models</a></p>
    </body></html>
    """)

@app.get("/health")
async def health_check():
    connection_ok = await ollama_client.check_connection()
    return {
        "status": "healthy" if connection_ok else "degraded",
        "timestamp": datetime.now().isoformat(),
        "ollama_connected": connection_ok,
        "available_models": len(ollama_client.available_models),
        "service": "Mantap AI v2.0"
    }

@app.post("/chat")
async def chat(
    prompt: str = Form(...), 
    model: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None
):
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    prompt = prompt.strip()
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(status_code=400, detail=f"Prompt too long (max {MAX_PROMPT_LENGTH} chars)")

    session_id = str(uuid.uuid4())[:8]
    logger.info(f"[{session_id}] Processing request: {len(prompt)} chars")

    math_result = try_eval_math(prompt)
    if math_result:
        return {
            "reply": math_result,
            "model": "built-in-calculator",
            "status": "success",
            "session_id": session_id
        }

    if not await ollama_client.check_connection():
        raise HTTPException(
            status_code=503, 
            detail="AI service temporarily unavailable. Please try again."
        )

    language = detect_language(prompt)
    enhanced_prompt = create_smart_prompt(prompt, language)
    
    target_model = model or DEFAULT_MODEL
    best_model = await ollama_client.get_best_model(target_model)
    
    logger.info(f"[{session_id}] Using model: {best_model}, Language: {language}")

    result = await ollama_client.generate_response(best_model, enhanced_prompt)
    
    if not result["success"]:
        for fallback_model in FALLBACK_MODELS:
            if fallback_model != best_model:
                fallback = await ollama_client.get_best_model(fallback_model)
                logger.info(f"[{session_id}] Trying fallback: {fallback}")
                result = await ollama_client.generate_response(fallback, enhanced_prompt)
                if result["success"]:
                    break
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "AI service error"))

    response_text = result["response"]
    if not response_text:
        response_text = "Maaf, saya tidak dapat memberikan jawaban untuk pertanyaan ini." if language == 'id' else "Sorry, I cannot provide an answer to this question."

    return {
        "reply": response_text,
        "model": result.get("model", best_model),
        "status": "success",
        "session_id": session_id,
        "language": language
    }

@app.post("/chat-stream")
async def chat_stream(
    prompt: str = Form(...),
    model: Optional[str] = Form(None)
):
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    prompt = prompt.strip()
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(status_code=400, detail="Prompt too long")

    if not await ollama_client.check_connection():
        raise HTTPException(status_code=503, detail="AI service unavailable")

    language = detect_language(prompt)
    enhanced_prompt = create_smart_prompt(prompt, language)
    target_model = model or DEFAULT_MODEL
    best_model = await ollama_client.get_best_model(target_model)

    async def generate_stream():
        try:
            payload = {
                "model": best_model,
                "prompt": enhanced_prompt,
                "stream": True,
                "options": {
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "num_predict": MAX_TOKENS
                }
            }

            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                async with client.stream("POST", OLLAMA_API, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "response" in data and data["response"]:
                                    yield f"data: {json.dumps({'text': data['response']})}\n\n"
                                if data.get("done", False):
                                    yield f"data: {json.dumps({'done': True, 'model': best_model})}\n\n"
                                    break
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

@app.get("/models")
async def get_available_models():
    try:
        await ollama_client.check_connection()
        return {
            "models": ollama_client.available_models,
            "default": DEFAULT_MODEL,
            "fallbacks": FALLBACK_MODELS,
            "last_updated": ollama_client.last_model_check.isoformat() if ollama_client.last_model_check else None
        }
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return {
            "models": [DEFAULT_MODEL] + FALLBACK_MODELS,
            "default": DEFAULT_MODEL,
            "error": str(e)
        }

@app.get("/status")
async def get_system_status():
    connection_ok = await ollama_client.check_connection()
    return {
        "online": connection_ok,
        "models_available": len(ollama_client.available_models),
        "default_model": DEFAULT_MODEL,
        "max_tokens": MAX_TOKENS,
        "timeout": TIMEOUT,
        "supported_languages": ["Indonesian", "English", "Auto-detect"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )