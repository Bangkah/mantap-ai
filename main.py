from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import logging
from typing import Optional
import os
import re
from contextlib import asynccontextmanager

# ======================
# CONFIG & LOGGER
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MantapAI")

OLLAMA_API = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "phi3"  # lebih pintar defaultnya
FALLBACK_MODELS = ["mistral:7b", "llama2:latest"]
TIMEOUT = 90
MAX_TOKENS = 2500

# ======================
# OLLAMA CLIENT
# ======================
class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_API):
        self.base_url = base_url

    def check_connection(self) -> bool:
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def generate_response(self, model: str, prompt: str) -> dict:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": MAX_TOKENS
            }
        }
        try:
            logger.info(f"Sending request to Ollama with model: {model}")
            response = requests.post(self.base_url, json=payload, timeout=TIMEOUT)
            response.raise_for_status()
            data = response.json()
            return {
                "success": True,
                "response": data.get("response", ""),
                "model": data.get("model", model),
                "done": data.get("done", True)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

ollama_client = OllamaClient()

# ======================
# EVAL MATH
# ======================
def try_eval_math(expr: str) -> Optional[str]:
    expr = expr.strip()
    if re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        try:
            result = eval(expr, {"__builtins__": {}})
            return str(result)
        except Exception:
            return None
    return None

# ======================
# FASTAPI APP
# ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    if ollama_client.check_connection():
        logger.info("✅ Ollama server connected successfully")
    else:
        logger.warning("⚠️ Ollama server not available")
    yield

app = FastAPI(
    title="Mantap AI",
    description="AI Assistant Backend using Ollama",
    version="1.2.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static & Templates
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
if os.path.exists("templates"):
    templates = Jinja2Templates(directory="templates")

# ======================
# ROUTES
# ======================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not os.path.exists("templates"):
        return HTMLResponse("""
        <h1>Mantap AI Backend</h1>
        <p>Backend is running! 🚀</p>
        <p>Test API: <a href="/health">/health</a></p>
        """)
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    ollama_status = ollama_client.check_connection()
    return {
        "status": "healthy" if ollama_status else "degraded",
        "ollama_connected": ollama_status,
        "message": "Mantap AI Backend is running"
    }

@app.post("/chat")
async def chat(prompt: str = Form(...), model: Optional[str] = Form(DEFAULT_MODEL)):
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt tidak boleh kosong")
    if len(prompt) > 10000:
        raise HTTPException(status_code=400, detail="Prompt terlalu panjang (maks 10.000 karakter)")

    # Math eval langsung return
    math_result = try_eval_math(prompt)
    if math_result:
        return {"reply": math_result, "model": "python-math", "status": "success"}

    if not ollama_client.check_connection():
        raise HTTPException(status_code=503, detail="Ollama server tidak tersedia")

    # tambahin konteks agar lebih pintar & dukung bahasa Indonesia
    enhanced_prompt = f"""
    Kamu adalah asisten AI pintar bernama Mantap AI. 
    Gunakan bahasa Indonesia atau Inggris sesuai input user.
    Pertanyaan: {prompt}
    Jawaban:
    """

    # coba model utama dulu, kalau gagal pakai fallback
    models_to_try = [model] + FALLBACK_MODELS
    for m in models_to_try:
        result = ollama_client.generate_response(m, enhanced_prompt)
        if result["success"]:
            return {
                "reply": result["response"].strip(),
                "model": result.get("model", m),
                "status": "success"
            }

    raise HTTPException(status_code=500, detail="Semua model gagal merespons.")

@app.get("/models")
async def get_models():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        response.raise_for_status()
        data = response.json()
        models = [model["name"] for model in data.get("models", [])]
        return {"models": models, "default": DEFAULT_MODEL}
    except Exception as e:
        return {"models": [DEFAULT_MODEL], "default": DEFAULT_MODEL, "error": str(e)}

# ======================
# ENTRY POINT
# ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
