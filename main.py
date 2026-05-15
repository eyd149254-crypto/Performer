"""
AgriMind AI — main.py
─────────────────────────────────────────────────────────────────
Routes :
  GET  /                    → index.html
  GET  /status              → healthcheck
  POST /api/gemini          → proxy hybride :
                              has_image=false → Groq llama-3.3-70b  (chat/intrants)
                              has_image=true  → Gemini gemini-2.5-pro (diagnostic image)
  POST /api/hf              → proxy HuggingFace ResNet50 (évite CORS navigateur)
  POST /api/alert/submit    → Groq classifie → broadcast WS
  GET  /api/alerts          → liste alertes récentes
  WS   /ws/alerts           → push temps réel

Variables Render :
  GROQ_API_KEY              → gsk_...
  GEMINI_API_KEY            → AIza...
  HF_TOKEN                  → hf_... (optionnel — améliore priorité & rate-limit HF)
─────────────────────────────────────────────────────────────────
"""

import os, json, time, asyncio, uuid
from typing import Any, Optional
from collections import deque

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
HF_TOKEN       = os.environ.get("HF_TOKEN", "")          # optionnel

GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_BASE  = "https://generativelanguage.googleapis.com/v1beta/models"
HF_MODEL_URL = "https://api-inference.huggingface.co/models/mesabo/agri-plant-disease-resnet50"

MODEL_CHAT   = "llama-3.3-70b-versatile"   # Groq  — chat + alertes + intrants
MODEL_VISION = "gemini-2.5-pro"             # Gemini — diagnostic image

print("✅ Groq OK"      if GROQ_API_KEY   else "⚠️  GROQ_API_KEY manquante")
print("✅ Gemini OK"    if GEMINI_API_KEY else "⚠️  GEMINI_API_KEY manquante")
print("✅ HF Token OK"  if HF_TOKEN       else "ℹ️  HF_TOKEN non défini (modèle public OK)")

# ══════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════
app = FastAPI(title="AgriMind AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

active_ws:  list[WebSocket] = []
alerts_db:  deque           = deque(maxlen=200)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# ══════════════════════════════════════════════════════════════════
#  MODÈLES PYDANTIC
# ══════════════════════════════════════════════════════════════════
class ApiRequest(BaseModel):
    messages:   list[Any]
    system:     Optional[str] = None
    has_image:  bool          = False
    max_tokens: int           = 1000

class AlertSubmit(BaseModel):
    message:  str
    lat:      Optional[float] = None
    lon:      Optional[float] = None
    accuracy: Optional[float] = None

# ══════════════════════════════════════════════════════════════════
#  GROQ — chat + alertes (texte uniquement)
# ══════════════════════════════════════════════════════════════════
def to_groq_messages(messages: list[Any], system: Optional[str]) -> list[dict]:
    """Convertit format Anthropic → OpenAI/Groq (texte seulement)."""
    result = []
    if system:
        result.append({"role": "system", "content": system})
    for m in messages:
        role    = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, str):
            result.append({"role": role, "content": content})
        elif isinstance(content, list):
            text = " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
            if text:
                result.append({"role": role, "content": text})
    return result

async def call_groq(messages: list[dict], max_tokens: int = 1000, timeout: float = 60.0) -> tuple[int, dict]:
    payload = {
        "model":       MODEL_CHAT,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": 0.4,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json=payload,
        )
    raw = resp.json()
    if resp.status_code == 200:
        text = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        return 200, {"content": [{"type": "text", "text": text}]}
    return resp.status_code, raw

# ══════════════════════════════════════════════════════════════════
#  GEMINI — diagnostic image (vision)
# ══════════════════════════════════════════════════════════════════
def to_gemini_contents(messages: list[Any]) -> list[dict]:
    """Convertit format Anthropic → Gemini (texte + images base64)."""
    contents = []
    for m in messages:
        role    = "user" if m.get("role") == "user" else "model"
        content = m.get("content", "")
        parts   = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    parts.append({"text": block.get("text", "")})
                elif block.get("type") == "image":
                    src = block.get("source", {})
                    if src.get("type") == "base64":
                        parts.append({
                            "inline_data": {
                                "mime_type": src.get("media_type", "image/jpeg"),
                                "data":      src.get("data", ""),
                            }
                        })
        if parts:
            contents.append({"role": role, "parts": parts})
    return contents

async def call_gemini(contents: list[dict], system: Optional[str] = None,
                      max_tokens: int = 1000, timeout: float = 120.0) -> tuple[int, dict]:
    url     = f"{GEMINI_BASE}/{MODEL_VISION}:generateContent?key={GEMINI_API_KEY}"
    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.4},
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"})

    raw = resp.json()
    if resp.status_code == 200:
        try:
            text = raw["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            text = ""
        return 200, {"content": [{"type": "text", "text": text}]}
    return resp.status_code, raw

# ══════════════════════════════════════════════════════════════════
#  BROADCAST WS
# ══════════════════════════════════════════════════════════════════
async def broadcast(data: dict):
    dead = []
    for ws in active_ws:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        active_ws.remove(ws)

# ══════════════════════════════════════════════════════════════════
#  PROXY HYBRIDE — /api/gemini
#  has_image=false → Groq (rapide, gratuit, texte)
#  has_image=true  → Gemini Vision (diagnostic image)
# ══════════════════════════════════════════════════════════════════
@app.post("/api/gemini")
async def hybrid_proxy(req: ApiRequest):

    if req.has_image:
        # ── GEMINI VISION ──────────────────────────────────────────
        if not GEMINI_API_KEY:
            return JSONResponse(status_code=500,
                                content={"error": "GEMINI_API_KEY non configurée sur Render."})

        contents = to_gemini_contents(req.messages)
        print(f"🔬 /api/gemini → Gemini Vision  msgs={len(req.messages)}")

        try:
            status, body = await call_gemini(contents, req.system, req.max_tokens)
            if status == 200:
                preview = body["content"][0]["text"][:100].replace("\n", " ")
                print(f"✅ Gemini OK → «{preview}…»")
            else:
                print(f"❌ Gemini {status}: {body}")
            return JSONResponse(status_code=status, content=body)
        except httpx.TimeoutException:
            return JSONResponse(status_code=504, content={"error": "Timeout Gemini Vision — réessayez."})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    else:
        # ── GROQ CHAT ──────────────────────────────────────────────
        if not GROQ_API_KEY:
            return JSONResponse(status_code=500,
                                content={"error": "GROQ_API_KEY non configurée sur Render."})

        messages = to_groq_messages(req.messages, req.system)
        print(f"🤖 /api/gemini → Groq Chat  msgs={len(messages)}")

        try:
            status, body = await call_groq(messages, req.max_tokens)
            if status == 200:
                preview = body["content"][0]["text"][:100].replace("\n", " ")
                print(f"✅ Groq OK → «{preview}…»")
            else:
                print(f"❌ Groq {status}: {body}")
            return JSONResponse(status_code=status, content=body)
        except httpx.TimeoutException:
            return JSONResponse(status_code=504, content={"error": "Timeout Groq — réessayez."})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

# ══════════════════════════════════════════════════════════════════
#  PROXY HUGGINGFACE — /api/hf
#  Résout les erreurs CORS quand le navigateur appelle HF directement.
#  Le backend fait l'appel server-to-server, sans restriction CORS.
#  HF_TOKEN (optionnel) : améliore la priorité et évite le rate-limiting.
# ══════════════════════════════════════════════════════════════════
@app.post("/api/hf")
async def hf_proxy(request: Request):
    """Proxy HuggingFace Inference API ResNet50 → pas de CORS, token sécurisé."""
    body = await request.body()

    if not body:
        return JSONResponse(status_code=400, content={"error": "Corps de requête vide — image manquante."})

    headers = {"Content-Type": "image/jpeg"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    print(f"🤖 /api/hf → HuggingFace ResNet50  size={len(body)} bytes  token={'yes' if HF_TOKEN else 'no'}")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(HF_MODEL_URL, content=body, headers=headers)

        # Relayer le statut et le JSON tel quel (503 = modèle en chargement, géré côté client)
        try:
            payload = resp.json()
        except Exception:
            payload = {"error": f"Réponse non-JSON de HuggingFace (HTTP {resp.status_code})"}

        if resp.status_code == 200:
            print(f"✅ HF OK → {len(payload)} prédictions")
        else:
            print(f"⚠️  HF {resp.status_code}: {payload}")

        return JSONResponse(status_code=resp.status_code, content=payload)

    except httpx.TimeoutException:
        print("❌ HF Timeout")
        return JSONResponse(status_code=504, content={"error": "Timeout HuggingFace — réessayez dans quelques secondes."})
    except Exception as e:
        print(f"❌ HF Erreur: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ══════════════════════════════════════════════════════════════════
#  ALERT SUBMIT → Groq classifie → broadcast
# ══════════════════════════════════════════════════════════════════
CLASSIFY_SYSTEM = """Tu es un système de classification d'alertes agricoles pour le Burkina Faso.
Un agriculteur envoie un signalement en texte libre. Extrais et structure l'information.
Réponds UNIQUEMENT avec un objet JSON valide. Aucun texte avant ou après. Pas de markdown.

Format :
{
  "type": "maladie" | "ravageur" | "meteo" | "sol" | "autre",
  "culture": "nom de la culture ou null",
  "danger": "faible" | "modere" | "eleve" | "critique",
  "resume": "résumé clair max 120 caractères pour d'autres agriculteurs",
  "conseil": "action immédiate recommandée max 150 caractères",
  "confiance": 0 à 100
}

Règles danger :
- critique : propagation rapide imminente, perte totale probable
- eleve    : dégâts significatifs si non traité sous 48h
- modere   : surveillance nécessaire, traitement préventif
- faible   : information utile, pas d'urgence"""

@app.post("/api/alert/submit")
async def alert_submit(payload: AlertSubmit):
    if not GROQ_API_KEY:
        return JSONResponse(status_code=500, content={"error": "GROQ_API_KEY non configurée."})

    msg = payload.message.strip()
    if not msg:
        return JSONResponse(status_code=400, content={"error": "message vide"})

    gps_ctx = ""
    if payload.lat is not None and payload.lon is not None:
        gps_ctx = f"\nGPS : {payload.lat:.4f}, {payload.lon:.4f}"
        if payload.accuracy:
            gps_ctx += f" (±{payload.accuracy:.0f}m)"

    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {"role": "user",   "content": f"Signalement:{gps_ctx}\n\n{msg}"},
    ]

    print(f"📩 ALERT: «{msg[:80]}»  GPS=({payload.lat},{payload.lon})")

    try:
        status, body = await call_groq(messages, max_tokens=300, timeout=30.0)
        if status != 200:
            raise Exception(f"Groq error {status}: {body}")
        raw_text = body["content"][0]["text"].strip()
        raw_text = raw_text.lstrip("```json").lstrip("```").rstrip("```").strip()
        classification = json.loads(raw_text)
    except Exception as e:
        print(f"⚠️  Classification failed: {e} — fallback")
        classification = {
            "type": "autre", "culture": None, "danger": "modere",
            "resume": msg[:120], "conseil": "Consulter un technicien agricole.",
            "confiance": 30,
        }

    alert = {
        "id":        str(uuid.uuid4())[:8],
        "timestamp": time.time(),
        "message":   msg,
        "lat":       payload.lat,
        "lon":       payload.lon,
        "accuracy":  payload.accuracy,
        **classification,
    }

    alerts_db.appendleft(alert)
    print(f"✅ ALERT id={alert['id']}  danger={alert['danger']}  type={alert['type']}")
    await broadcast({"event": "new_alert", "alert": alert})

    return {"status": "ok", "alert": alert}

# ══════════════════════════════════════════════════════════════════
#  GET ALERTS
# ══════════════════════════════════════════════════════════════════
@app.get("/api/alerts")
async def get_alerts(limit: int = 50):
    return {"alerts": list(alerts_db)[:limit]}

# ══════════════════════════════════════════════════════════════════
#  WEBSOCKET
# ══════════════════════════════════════════════════════════════════
@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    await websocket.accept()
    active_ws.append(websocket)
    print(f"🟢 WS CONNECTED ({len(active_ws)} actif(s))")
    try:
        await websocket.send_json({"event": "init", "alerts": list(alerts_db)[:50]})
    except Exception:
        pass
    try:
        while True:
            await asyncio.wait_for(websocket.receive_text(), timeout=30)
    except (WebSocketDisconnect, asyncio.TimeoutError, Exception):
        pass
    finally:
        if websocket in active_ws:
            active_ws.remove(websocket)
        print(f"🔴 WS DISCONNECTED ({len(active_ws)} actif(s))")

# ══════════════════════════════════════════════════════════════════
#  STATUS
# ══════════════════════════════════════════════════════════════════
@app.get("/status")
async def status():
    return {
        "groq":         bool(GROQ_API_KEY),
        "gemini":       bool(GEMINI_API_KEY),
        "hf_token":     bool(HF_TOKEN),
        "model_chat":   MODEL_CHAT,
        "model_vision": MODEL_VISION,
        "hf_model":     HF_MODEL_URL.split("/")[-1],
        "ws_clients":   len(active_ws),
        "alerts_count": len(alerts_db),
    }

# ══════════════════════════════════════════════════════════════════
#  LANCEMENT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 AgriMind AI démarrage port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
