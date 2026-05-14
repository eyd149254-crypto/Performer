"""
AgriMind AI — main.py (Groq API)
─────────────────────────────────────────────────────────────────
Routes :
  GET  /                    → index.html
  GET  /status              → healthcheck
  POST /api/gemini          → proxy Groq (chatbot + diagnostic image + intrants)
                              has_image=true  → llama-3.2-90b-vision-preview
                              has_image=false → llama-3.3-70b-versatile
  POST /api/alert/submit    → signalement + GPS → Groq classifie → broadcast WS
  GET  /api/alerts          → liste alertes récentes
  WS   /ws/alerts           → push temps réel

Variables Render :
  GROQ_API_KEY              → gsk_...
─────────────────────────────────────────────────────────────────
"""

import os, json, time, asyncio, uuid
from typing import Any, Optional
from collections import deque

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

MODEL_CHAT   = "llama-3.3-70b-versatile"        # chat + alertes + intrants
MODEL_VISION = "llama-3.2-90b-vision-preview"   # diagnostic image

print("✅ Groq API key OK" if GROQ_API_KEY else "⚠️  GROQ_API_KEY manquante")

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

# ── état en mémoire ───────────────────────────────────────────────
active_ws:  list[WebSocket] = []
alerts_db:  deque           = deque(maxlen=200)

# ── statique ──────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# ══════════════════════════════════════════════════════════════════
#  MODÈLES PYDANTIC
# ══════════════════════════════════════════════════════════════════
class GroqRequest(BaseModel):
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
#  HELPERS
# ══════════════════════════════════════════════════════════════════
def anthropic_to_groq_messages(messages: list[Any],
                                system: Optional[str]) -> list[dict]:
    """
    Convertit le format Anthropic [{role, content}] vers OpenAI/Groq.
    Gère texte et images (base64).
    Injecte le system prompt en tête si fourni.
    """
    result = []

    if system:
        result.append({"role": "system", "content": system})

    for m in messages:
        role    = m.get("role", "user")
        content = m.get("content", "")

        if isinstance(content, str):
            result.append({"role": role, "content": content})

        elif isinstance(content, list):
            parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")

                if btype == "text":
                    parts.append({"type": "text", "text": block.get("text", "")})

                elif btype == "image":
                    src = block.get("source", {})
                    if src.get("type") == "base64":
                        mime = src.get("media_type", "image/jpeg")
                        data = src.get("data", "")
                        parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{data}"
                            }
                        })

            if parts:
                result.append({"role": role, "content": parts})

    return result


async def call_groq(model: str, messages: list[dict],
                    max_tokens: int = 1000,
                    timeout: float = 60.0) -> tuple[int, dict]:
    """
    Appelle Groq et retourne (status, body normalisé).
    body normalisé : { "content": [{"type":"text","text":"..."}] }
    """
    payload = {
        "model":      model,
        "messages":   messages,
        "max_tokens": max_tokens,
        "temperature": 0.4,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json=payload,
        )

    raw = resp.json()

    if resp.status_code == 200:
        try:
            text = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            text = ""
        return 200, {"content": [{"type": "text", "text": text}]}
    else:
        return resp.status_code, raw


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
#  PROXY GROQ — chatbot + diagnostic image + intrants
#  Route gardée /api/gemini pour ne pas changer le frontend
# ══════════════════════════════════════════════════════════════════
@app.post("/api/gemini")
async def groq_proxy(req: GroqRequest):
    if not GROQ_API_KEY:
        return JSONResponse(status_code=500,
                            content={"error": "GROQ_API_KEY non configurée sur Render."})

    model    = MODEL_VISION if req.has_image else MODEL_CHAT
    messages = anthropic_to_groq_messages(req.messages, req.system)

    print(f"🤖 /api/gemini→Groq  model={model}  msgs={len(messages)}  img={'oui' if req.has_image else 'non'}")

    try:
        timeout = 90.0 if req.has_image else 60.0
        status, body = await call_groq(model, messages, req.max_tokens, timeout)

        if status == 200:
            preview = body["content"][0]["text"][:100].replace("\n", " ")
            print(f"✅ Groq OK ({model}) → «{preview}…»")
        else:
            print(f"❌ Groq {status}: {body}")

        return JSONResponse(status_code=status, content=body)

    except httpx.TimeoutException:
        return JSONResponse(status_code=504,
                            content={"error": "Timeout Groq — réessayez."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ══════════════════════════════════════════════════════════════════
#  ALERT SUBMIT
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
        return JSONResponse(status_code=500,
                            content={"error": "GROQ_API_KEY non configurée."})

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
        {"role": "user",   "content": f"Signalement:{gps_ctx}\n\n{msg}"}
    ]

    print(f"📩 ALERT SUBMIT: «{msg[:80]}»  GPS=({payload.lat},{payload.lon})")

    try:
        status, body = await call_groq(MODEL_CHAT, messages, 300, 30.0)
        if status != 200:
            raise Exception(f"Groq error {status}: {body}")

        raw_text = body["content"][0]["text"]
        raw_text = raw_text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
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
        "model_chat":   MODEL_CHAT,
        "model_vision": MODEL_VISION,
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
