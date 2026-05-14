"""
AgriSmart — main.py (Gemini API)
─────────────────────────────────────────────────────────────────
Routes :
  GET  /                    → index.html
  GET  /status              → healthcheck
  POST /api/gemini          → proxy Gemini (chatbot + diagnostic image + intrants)
                              body: { model?, system?, messages[], has_image? }
                              → flash si pas d'image, pro si image détectée
  POST /api/alert/submit    → signalement texte + GPS → Gemini Flash classifie
                              → broadcast WebSocket
  GET  /api/alerts          → liste alertes récentes (max 200)
  WS   /ws/alerts           → push temps réel

Variables Railway :
  GEMINI_API_KEY            → clé AIza... (Google AI Studio)
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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE    = "https://generativelanguage.googleapis.com/v1beta/models"

MODEL_FLASH = "gemini-2.0-flash"      # chat + alertes + intrants
MODEL_PRO   = "gemini-2.5-pro"        # diagnostic image

print("✅ Gemini API key OK" if GEMINI_API_KEY else "⚠️  GEMINI_API_KEY manquante")

# ══════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════
app = FastAPI(title="AgriSmart API")

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
class GeminiRequest(BaseModel):
    # messages au format Anthropic {role, content} — on convertit côté serveur
    messages:   list[Any]
    system:     Optional[str] = None
    has_image:  bool          = False   # flag envoyé par le frontend
    max_tokens: int           = 1000

class AlertSubmit(BaseModel):
    message:  str
    lat:      Optional[float] = None
    lon:      Optional[float] = None
    accuracy: Optional[float] = None

# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def anthropic_to_gemini_contents(messages: list[Any]) -> list[dict]:
    """
    Convertit le format Anthropic [{role, content}] vers le format
    Gemini [{role, parts:[{text}|{inline_data}]}].
    content peut être : str | list[{type,text}|{type,image,source}]
    """
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
                btype = block.get("type", "")

                if btype == "text":
                    parts.append({"text": block.get("text", "")})

                elif btype == "image":
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


async def call_gemini(model: str, contents: list[dict],
                      system: Optional[str] = None,
                      max_tokens: int = 1000,
                      timeout: float = 90.0) -> tuple[int, dict]:
    """
    Appelle l'API Gemini generateContent et retourne (status, body_dict).
    body_dict suit le même format de réponse qu'on utilise côté client :
      { "content": [{"type":"text","text":"..."}] }
    """
    url = f"{GEMINI_BASE}/{model}:generateContent?key={GEMINI_API_KEY}"

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.4,
        },
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload,
                                 headers={"Content-Type": "application/json"})

    raw = resp.json()

    # Normaliser la réponse au format {content:[{type,text}]}
    # pour que le frontend n'ait pas à changer
    if resp.status_code == 200:
        try:
            text = raw["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            text = ""
        normalized = {"content": [{"type": "text", "text": text}]}
        return 200, normalized
    else:
        # Retourner l'erreur Gemini telle quelle
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
#  PROXY GEMINI — chatbot + diagnostic image + intrants
# ══════════════════════════════════════════════════════════════════
@app.post("/api/gemini")
async def gemini_proxy(req: GeminiRequest):
    if not GEMINI_API_KEY:
        return JSONResponse(status_code=500,
                            content={"error": "GEMINI_API_KEY non configurée sur Railway."})

    # Choix du modèle selon présence d'image
    model = MODEL_PRO if req.has_image else MODEL_FLASH

    contents = anthropic_to_gemini_contents(req.messages)

    print(f"🤖 /api/gemini  model={model}  msgs={len(req.messages)}  img={'oui' if req.has_image else 'non'}")

    try:
        status, body = await call_gemini(
            model      = model,
            contents   = contents,
            system     = req.system,
            max_tokens = req.max_tokens,
            timeout    = 120.0 if req.has_image else 60.0,
        )

        if status == 200:
            preview = (body.get("content", [{}])[0].get("text", "")[:100]
                       .replace("\n", " "))
            print(f"✅ Gemini OK ({model}) → «{preview}…»")
        else:
            print(f"❌ Gemini {status}: {body}")

        return JSONResponse(status_code=status, content=body)

    except httpx.TimeoutException:
        return JSONResponse(status_code=504,
                            content={"error": "Timeout Gemini — réessayez."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ══════════════════════════════════════════════════════════════════
#  ALERT SUBMIT — signalement → Gemini Flash classifie → broadcast
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
    if not GEMINI_API_KEY:
        return JSONResponse(status_code=500,
                            content={"error": "GEMINI_API_KEY non configurée."})

    msg = payload.message.strip()
    if not msg:
        return JSONResponse(status_code=400, content={"error": "message vide"})

    gps_ctx = ""
    if payload.lat is not None and payload.lon is not None:
        gps_ctx = f"\nGPS : {payload.lat:.4f}, {payload.lon:.4f}"
        if payload.accuracy:
            gps_ctx += f" (±{payload.accuracy:.0f}m)"

    contents = [{"role": "user", "parts": [
        {"text": f"Signalement:{gps_ctx}\n\n{msg}"}
    ]}]

    print(f"📩 ALERT SUBMIT: «{msg[:80]}»  GPS=({payload.lat},{payload.lon})")

    try:
        status, body = await call_gemini(
            model      = MODEL_FLASH,
            contents   = contents,
            system     = CLASSIFY_SYSTEM,
            max_tokens = 300,
            timeout    = 30.0,
        )
        if status != 200:
            raise Exception(f"Gemini error {status}: {body}")

        raw_text = body.get("content", [{}])[0].get("text", "{}")
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
        "gemini":       bool(GEMINI_API_KEY),
        "model_flash":  MODEL_FLASH,
        "model_pro":    MODEL_PRO,
        "ws_clients":   len(active_ws),
        "alerts_count": len(alerts_db),
    }

# ══════════════════════════════════════════════════════════════════
#  LANCEMENT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 AgriSmart démarrage port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
