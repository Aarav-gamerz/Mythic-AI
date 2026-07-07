"""
Mythic AI Ultra — single-file Flask app.
Providers: Groq + Cerebras only.
Image gen: Pollinations (no key)
Weather:   Open-Meteo (no key)
Search:    DuckDuckGo (no key)
News:      NewsAPI
Voice, history, VIP, markdown rendering, rate limiting.

Run dev:    python mythic.py
Run prod:   gunicorn -w 2 -k gthread --threads 4 -b 0.0.0.0:$PORT mythic:app
"""

import os
import re
import json
import uuid
import time
import base64
import random
import urllib.parse
from pathlib import Path

import requests
from flask import Flask, request, jsonify, Response, session, stream_with_context
from dotenv import load_dotenv

load_dotenv()

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    LIMITER_ENABLED = True
except ImportError:
    LIMITER_ENABLED = False

# --- Provider + model config ---------------------------------------
PROVIDER = os.environ.get("AI_PROVIDER", "auto").strip().lower()

GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
NEWS_API_KEY     = os.environ.get("NEWS_API_KEY", "")

GROQ_MODEL     = os.environ.get("GROQ_MODEL",     "llama-3.1-8b-instant")
CEREBRAS_MODEL = os.environ.get("CEREBRAS_MODEL", "llama-3.3-70b")

# --- VIP / model registry ------------------------------------------
VIP_PASSWORD = os.environ.get("VIP_PASSWORD", "1254")
VIP_MODELS   = {"aarav-ultra"}

AARAV_MAP = {
    "aarav-1.0":   ("groq",     "llama-3.1-8b-instant"),
    "aarav-2.0":   ("groq",     "llama-3.3-70b-versatile"),
    "aarav-2.5":   ("groq",     "llama-3.3-70b-versatile"),
    "aarav-3.5":   ("cerebras", "llama3.1-70b"),
    "aarav-ultra": ("cerebras", "llama-3.3-70b"),
}
DEFAULT_MODEL = "aarav-2.5"
MODEL_INFO = [
    {"id": "aarav-1.0",   "name": "Mythic 1.0",    "desc": "Fast — quick answers",       "vip": False},
    {"id": "aarav-2.0",   "name": "Mythic 2.0",    "desc": "Balanced — everyday tasks",  "vip": False},
    {"id": "aarav-2.5",   "name": "Mythic 2.5",    "desc": "Smart — best for most",      "vip": False},
    {"id": "aarav-3.5",   "name": "Mythic 3.5",    "desc": "Advanced — complex tasks",   "vip": False},
    {"id": "aarav-ultra", "name": "Mythic Ultra ✨","desc": "Most powerful — VIP only",  "vip": True},
]

SYSTEM_PROMPT = (
    "You are Mythic AI, a smart and friendly AI assistant. "
    "You were created by Aarav Singh — he is your maker, owner, and developer. "
    "If asked who made you, who owns you, or who your creator is: say 'I was created by Aarav Singh.' Say it once, naturally. "
    "Never say you were made by Google, Meta, Groq, Cerebras, Mistral, Anthropic, OpenAI, or any other company. "

    "REAL-TIME DATA POLICY (CRITICAL): You do not have live internet access on your own and your training data has a "
    "cutoff date, so you cannot know sports scores, election results, stock prices, weather, breaking news, or any other "
    "fact that changes over time, UNLESS the system message below explicitly gives you '[Live data]' for this turn. "
    "If '[Live data]' is present, use ONLY that data to answer. If it is NOT present and the user is asking about "
    "current events, scores, winners, prices, weather, or anything time-sensitive, you MUST NOT guess, estimate, or "
    "recall an answer from memory. Instead say you don't have live data for that right now and suggest they tap the "
    "🔍 Search or ask again so the system can fetch it. Never invent a winner, score, date, or figure. "

    "NEWS: You have access to real-time news. When the system provides news headlines, present them naturally. "

    "WEB SEARCH: You have access to real web search via DuckDuckGo. When the system provides search results, "
    "use them to answer the user's question accurately. Present the information naturally — don't just list URLs. "
    "If asked to search for something or about current events, the system will fetch results for you. "

    "WEATHER: When the system provides weather data, present it clearly and naturally. "
    "Include temperature, condition, humidity, and wind. Make it friendly and useful. "

    "IMAGE GENERATION: When the system shows '[Image generated]', acknowledge the image was created. "

    "HOMEWORK HELP: When the user asks for homework help, be a patient teacher. "
    "Explain step by step. Use simple language. Give examples. "
    "For math: show working. For essays: give structure and tips. For science: explain concepts clearly. "
    "Hindi mein bhi samjha sakte ho agar user Hindi mein pooche. "

    "LANGUAGE (STRICT RULE — follow this over every other instruction if there is ever a conflict): "
    "Before writing your reply, silently check: did the user's LATEST message use Latin/English letters, "
    "or Devanagari (Hindi script), or a clear mix (Hinglish)? Your entire reply must match that choice exactly, "
    "every single turn, with zero exceptions — this applies even in a long conversation, even if earlier turns "
    "used a different language, even if the topic is homework, news, weather, or search results. "
    "English message (Latin letters, e.g. 'what is the weather') → reply ENTIRELY in English, zero Hindi words. "
    "Hindi message (Devanagari script like नमस्ते) → reply ENTIRELY in Hindi Devanagari script. "
    "Hinglish message (Roman-script Hindi like 'kya haal hai') → reply in Hinglish. "
    "NEVER mix languages. NEVER add (Translation: ...) in brackets. NEVER switch language mid-reply. "
    "NEVER default to Hindi just because a previous turn was in Hindi, or because the topic feels Indian "
    "(cricket, IPL, Bollywood, etc.) — only the actual script/language of the CURRENT user message decides. "

    "BEHAVIOR: Never repeat yourself. No filler like 'Great question' or 'Sure, of course!'. "
    "Be direct, concise, helpful — like a knowledgeable friend. "
    "For code, always use markdown code blocks."
)

# --- Flask app + security -------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or ("dev-secret-CHANGE-ME-" + str(uuid.uuid4()))
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "20"))

# --- Optional Supabase storage --------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

def sb_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
def sb(path):
    return f"{SUPABASE_URL}/rest/v1/{path}"

# --- Local file storage fallback ------------------------------------
_BASE_DIR = Path(__file__).resolve().parent
_DATA_DIR = _BASE_DIR / "chat_data"
_DATA_DIR.mkdir(exist_ok=True)

def _user_conv_dir(username: str) -> Path:
    p = _DATA_DIR / "conversations" / username
    p.mkdir(parents=True, exist_ok=True)
    return p
def _conv_file(username: str, conv_id: str) -> Path:
    return _user_conv_dir(username) / f"{conv_id}.json"

def _list_conversations_file(username):
    folder = _user_conv_dir(username)
    out = []
    for fp in folder.glob("*.json"):
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
            out.append({"id": fp.stem, "title": d.get("title", "New chat"), "updated_at": d.get("updated_at", 0)})
        except Exception:
            continue
    out.sort(key=lambda c: c["updated_at"], reverse=True)
    return out
def _load_conversation_file(username, conv_id):
    fp = _conv_file(username, conv_id)
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None
def _save_conversation_file(username, conv_id, data):
    _conv_file(username, conv_id).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
def _delete_conversation_file(username, conv_id):
    fp = _conv_file(username, conv_id)
    if fp.exists():
        fp.unlink()

def list_conversations(username):
    if not SUPABASE_URL:
        return _list_conversations_file(username)
    try:
        r = requests.get(
            sb(f"conversations?username=eq.{username}&order=updated_at.desc&select=id,title,updated_at"),
            headers=sb_headers(), timeout=10,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []

def load_conversation(username, conv_id):
    if not SUPABASE_URL:
        return _load_conversation_file(username, conv_id)
    try:
        r = requests.get(
            sb(f"conversations?id=eq.{conv_id}&username=eq.{username}"),
            headers=sb_headers(), timeout=10,
        )
        if r.status_code == 200:
            rows = r.json()
            if rows:
                row = rows[0]
                return {
                    "title": row["title"],
                    "updated_at": row["updated_at"],
                    "messages": row["messages"] if isinstance(row["messages"], list) else json.loads(row["messages"]),
                }
    except Exception:
        pass
    return None

def save_conversation(username, conv_id, data):
    data["updated_at"] = time.time()
    if not SUPABASE_URL:
        _save_conversation_file(username, conv_id, data)
        return
    try:
        headers = {**sb_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"}
        requests.post(sb("conversations"), headers=headers, json={
            "id": conv_id, "username": username,
            "title": data.get("title", "New chat"),
            "updated_at": data["updated_at"],
            "messages": data.get("messages", []),
        }, timeout=15)
    except Exception:
        pass

def delete_conversation(username, conv_id):
    if not SUPABASE_URL:
        _delete_conversation_file(username, conv_id)
        return
    try:
        requests.delete(
            sb(f"conversations?id=eq.{conv_id}&username=eq.{username}"),
            headers=sb_headers(), timeout=10,
        )
    except Exception:
        pass

# --- Auth (login-less sessions) -------------------------------------
def current_username():
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
        session.permanent = True
    return session["user_id"]
def with_session(view):
    def wrapped(*a, **kw):
        current_username()
        return view(*a, **kw)
    wrapped.__name__ = view.__name__
    return wrapped

def make_title(first_message):
    t = (first_message or "Attachment").strip().replace("\n", " ")
    return t[:40] + ("…" if len(t) > 40 else "")

# --- Rate limiter ---------------------------------------------------
if LIMITER_ENABLED:
    limiter = Limiter(
        get_remote_address, app=app,
        default_limits=["240 per hour", "30 per minute"],
        storage_uri="memory://",
    )
else:
    limiter = None

# ============================================================
# Streaming helpers (Groq + Cerebras)
# ============================================================
def to_openai_messages(messages, system_prompt):
    msgs = [{"role": "system", "content": system_prompt}]
    for m in messages:
        role = "user" if m["role"] == "user" else "assistant"
        text = "".join(p.get("text", "") for p in m.get("parts", []) if "text" in p)
        msgs.append({"role": role, "content": text})
    return msgs

def _iter_sse_lines(resp):
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        payload = raw[5:].lstrip()
        if not payload or payload == "[DONE]":
            continue
        yield payload

def _openai_stream(url, headers, body):
    try:
        resp = requests.post(url, headers=headers, json=body, stream=True, timeout=60)
    except requests.RequestException as e:
        yield f"[Connection error: {e}]"
        return
    if resp.status_code != 200:
        yield f"[HTTP {resp.status_code}: {resp.text[:200]}]"
        return
    for payload in _iter_sse_lines(resp):
        try:
            obj = json.loads(payload)
            chunk = obj["choices"][0]["delta"].get("content", "")
            if chunk:
                yield chunk
        except (json.JSONDecodeError, KeyError, IndexError):
            continue

def groq_stream(messages, model):
    if not GROQ_API_KEY:
        yield "[Groq API key not configured]"; return
    yield from _openai_stream(
        "https://api.groq.com/openai/v1/chat/completions",
        {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        {"model": model, "messages": messages, "stream": True, "max_tokens": 2048},
    )

def cerebras_stream(messages, model):
    if not CEREBRAS_API_KEY:
        yield "[Cerebras API key not configured]"; return
    yield from _openai_stream(
        "https://api.cerebras.ai/v1/chat/completions",
        {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"},
        {"model": model, "messages": messages, "stream": True, "max_tokens": 2048},
    )

# ============================================================
# Routes
# ============================================================
@app.route("/")
@with_session
def index():
    return Response(PAGE, mimetype="text/html; charset=utf-8")

@app.route("/api/models")
def get_models():
    return jsonify({"models": MODEL_INFO, "default": DEFAULT_MODEL})

@app.route("/api/vip-unlock", methods=["POST"])
def vip_unlock():
    d = request.get_json(force=True) or {}
    if d.get("password") == VIP_PASSWORD:
        session["vip"] = True
        return jsonify({"success": True})
    return jsonify({"success": False}), 403

@app.route("/api/vip-status")
def vip_status():
    return jsonify({"vip": bool(session.get("vip"))})

@app.route("/api/conversations", methods=["GET"])
@with_session
def api_list_conversations():
    return jsonify({"conversations": list_conversations(current_username())})

@app.route("/api/conversations/<conv_id>", methods=["GET"])
@with_session
def api_get_conversation(conv_id):
    data = load_conversation(current_username(), conv_id)
    if data is None:
        return jsonify({"error": "not found"}), 404
    simplified = []
    for m in data.get("messages", []):
        role = "user" if m["role"] == "user" else "ai"
        text = "".join(p.get("text", "") for p in m.get("parts", []) if "text" in p)
        entry = {"role": role, "text": text}
        if m.get("attachment_meta"):
            entry["attachment"] = m["attachment_meta"]
        simplified.append(entry)
    return jsonify({"messages": simplified, "title": data.get("title", "New chat")})

@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
@with_session
def api_delete_conversation(conv_id):
    delete_conversation(current_username(), conv_id)
    return jsonify({"status": "deleted"})

@app.route("/api/conversations/<conv_id>", methods=["PATCH"])
@with_session
def api_rename_conversation(conv_id):
    data = request.get_json(force=True) or {}
    new_title = (data.get("title") or "").strip()[:120]
    if not new_title:
        return jsonify({"error": "title is required"}), 400
    username = current_username()
    conv = load_conversation(username, conv_id)
    if conv is None:
        return jsonify({"error": "not found"}), 404
    conv["title"] = new_title
    save_conversation(username, conv_id, conv)
    return jsonify({"status": "renamed", "title": new_title})

# --- News -----------------------------------------------------------
def fetch_news(query=None, category=None):
    if not NEWS_API_KEY:
        return None
    if query:
        url = "https://newsapi.org/v2/everything"
        params = {"apiKey": NEWS_API_KEY, "q": query, "sortBy": "publishedAt",
                  "language": "en", "pageSize": 8}
    else:
        url = "https://newsapi.org/v2/top-headlines"
        params = {"apiKey": NEWS_API_KEY, "country": "in", "pageSize": 8}
    if category:
        params["category"] = category
    try:
        r = requests.get(url, params=params, timeout=8)
        if r.status_code == 200:
            return [
                {"title": a["title"], "source": a["source"]["name"], "url": a["url"]}
                for a in r.json().get("articles", [])
                if a.get("title") and "[Removed]" not in a["title"]
            ]
    except Exception:
        pass
    return None

@app.route("/api/news", methods=["POST"])
@with_session
def get_news():
    d = request.get_json(force=True) or {}
    arts = fetch_news(query=d.get("query"), category=d.get("category"))
    if arts is None:
        return jsonify({"error": "News unavailable"}), 503
    return jsonify({"articles": arts})

# --- Web search (DuckDuckGo) ---------------------------------------
@app.route("/api/search", methods=["POST"])
@with_session
def web_search():
    d = request.get_json(force=True) or {}
    query = (d.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query required"}), 400
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            headers={"User-Agent": "MythicAI/1.0"}, timeout=8,
        )
        if r.status_code == 200:
            data = r.json()
            results = []
            if data.get("AbstractText"):
                results.append({
                    "title": data.get("Heading", query),
                    "snippet": data["AbstractText"],
                    "url": data.get("AbstractURL", ""),
                    "source": data.get("AbstractSource", ""),
                })
            if data.get("Answer"):
                results.insert(0, {
                    "title": "Answer", "snippet": data["Answer"], "url": "",
                    "source": data.get("AnswerType", ""),
                })
            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title": topic.get("Text", "")[:80],
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                        "source": "DuckDuckGo",
                    })
            if results:
                return jsonify({"results": results[:6], "query": query})
        return jsonify({"results": [], "query": query, "error": "No results found"})
    except Exception as e:
        return jsonify({"error": str(e)}), 502

# --- Weather (Open-Meteo, no key) ----------------------------------
WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow",
    80: "Rain showers", 81: "Heavy showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Heavy thunderstorm",
}
WMO_ICONS = {
    0: "☀️", 1: "🌤", 2: "⛅", 3: "☁️", 45: "🌫", 48: "🌫",
    51: "🌦", 53: "🌧", 55: "🌧", 61: "🌦", 63: "🌧", 65: "🌧",
    71: "🌨", 73: "❄️", 75: "❄️", 80: "🌧", 81: "🌧", 82: "⛈",
    95: "⛈", 96: "⛈", 99: "⛈",
}

@app.route("/api/weather", methods=["POST"])
@with_session
def get_weather():
    d = request.get_json(force=True) or {}
    location = (d.get("location") or "").strip()
    lat, lon = d.get("lat"), d.get("lon")
    if not location and (lat is None or lon is None):
        return jsonify({"error": "location or coordinates required"}), 400
    try:
        if lat is not None and lon is not None:
            geo_r = requests.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={"lat": lat, "lon": lon, "format": "json"},
                headers={"User-Agent": "MythicAI/1.0"}, timeout=8,
            )
            if geo_r.status_code == 200:
                addr = geo_r.json().get("address", {})
                city = addr.get("city") or addr.get("town") or addr.get("village") or "Your Location"
                location_name = city
            else:
                location_name = "Your Location"
        else:
            candidates = [location]
            words = location.replace(",", " ").split()
            if len(words) > 1:
                candidates += [words[-1], " ".join(words[1:]), words[0]]
            result = None
            for cand in candidates:
                cand = cand.strip()
                if not cand:
                    continue
                try:
                    geo_r = requests.get(
                        "https://geocoding-api.open-meteo.com/v1/search",
                        params={"name": cand, "count": 1, "language": "en", "format": "json"},
                        timeout=8,
                    )
                    if geo_r.status_code == 200 and geo_r.json().get("results"):
                        result = geo_r.json()["results"][0]
                        break
                except Exception:
                    continue
            if result is None:
                return jsonify({"error": f"City '{location}' not found. Try just the city name."}), 404
            lat, lon = result["latitude"], result["longitude"]
            location_name = result["name"] + ", " + result.get("country", "")

        wr = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                "hourly": "temperature_2m,weather_code,precipitation_probability",
                "forecast_days": 1, "wind_speed_unit": "kmh", "timezone": "auto",
            },
            timeout=8,
        )
        if wr.status_code != 200:
            return jsonify({"error": "Weather service unavailable"}), 502
        wjson = wr.json()
        cur = wjson["current"]
        code = cur.get("weather_code", 0)
        return jsonify({"weather": {
            "location": location_name,
            "temp": round(cur["temperature_2m"]),
            "feels_like": round(cur["apparent_temperature"]),
            "condition": WMO_CODES.get(code, "Unknown"),
            "humidity": cur["relative_humidity_2m"],
            "wind_speed": round(cur["wind_speed_10m"]),
            "icon": WMO_ICONS.get(code, "🌡"),
        }})
    except Exception as e:
        return jsonify({"error": str(e)}), 502

# --- Image generation (Pollinations) --------------------------------
@app.route("/api/generate-image", methods=["POST"])
@with_session
def generate_image():
    d = request.get_json(force=True) or {}
    prompt = (d.get("prompt") or "").strip()
    style  = (d.get("style") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt required"}), 400
    quality = "masterpiece, best quality, ultra detailed, sharp focus, intricate details, professional, cinematic lighting, high resolution"
    full = f"{prompt}, {style} style, {quality}" if style else f"{prompt}, {quality}"
    encoded = urllib.parse.quote(full)
    seed = random.randint(1, 999_999)
    candidates = [
        f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&model=flux&nologo=true&enhance=true&seed={seed}&nofeed=true",
        f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&model=flux-realism&nologo=true&enhance=true&seed={seed}&nofeed=true",
    ]
    last_err = None
    for url in candidates:
        try:
            r = requests.get(url, timeout=45, headers={"User-Agent": "MythicAI/1.0"})
            if r.status_code == 200:
                ct = r.headers.get("content-type", "")
                if ct.startswith("image/") and len(r.content) > 20_000:
                    return jsonify({"image": base64.b64encode(r.content).decode(), "mime": ct})
            last_err = f"status {r.status_code}"
        except requests.exceptions.Timeout:
            last_err = "timeout"
        except Exception as e:
            last_err = str(e)
    return jsonify({"error": f"Generation failed ({last_err}). Try a shorter, simpler prompt."}), 502

# --- Chat (streaming) ----------------------------------------------
def _trim_history(messages):
    if MAX_HISTORY_TURNS <= 0:
        return messages
    keep = MAX_HISTORY_TURNS * 2
    return messages[-keep:] if len(messages) > keep else messages

def _detect_lang_reminder(text: str) -> str:
    if re.search(r"[\u0900-\u097F]", text):
        return "[Reply ENTIRELY in Hindi Devanagari script for this message.] "
    if re.search(r"\b(hai|nahi|kya|kaise|kaha|kyun|mujhe|tumhe|aap|acha|theek|haan|nahin|bhi|karo|kar)\b", text, re.IGNORECASE):
        return "[Reply in Hinglish (Roman-script Hindi) for this message.] "
    return "[Reply ENTIRELY in English for this message — the user wrote in English.] "

@app.route("/api/chat", methods=["POST"])
@with_session
def chat():
    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    news_context = (data.get("news_context") or "").strip()
    conv_id      = data.get("conversation_id")
    attachment   = data.get("attachment")
    user_name    = (data.get("user_name") or "").strip()[:60]
    regenerate   = bool(data.get("regenerate"))
    aarav_id     = (data.get("model") or DEFAULT_MODEL).strip()

    if aarav_id in VIP_MODELS and not session.get("vip"):
        return jsonify({"error": "vip_required"}), 403

    provider, model_name = AARAV_MAP.get(aarav_id, AARAV_MAP[DEFAULT_MODEL])

    if regenerate and not conv_id:
        return jsonify({"error": "conversation_id is required to regenerate"}), 400
    if not regenerate and not user_message and not attachment:
        return jsonify({"error": "message or attachment is required"}), 400

    if attachment:
        try:
            raw = base64.b64decode(attachment.get("dataBase64", ""), validate=True)
        except Exception:
            return jsonify({"error": "invalid attachment data"}), 400
        if len(raw) > MAX_UPLOAD_BYTES:
            return jsonify({"error": "attachment too large (max 8MB)"}), 400

    username = current_username()
    conv = load_conversation(username, conv_id) if conv_id else None
    if conv is None:
        if regenerate:
            return jsonify({"error": "conversation not found"}), 404
        conv_id = str(uuid.uuid4())
        conv = {"title": make_title(user_message), "messages": []}

    messages = conv.setdefault("messages", [])

    if regenerate:
        if messages and messages[-1]["role"] == "model":
            messages.pop()
        if not messages or messages[-1]["role"] != "user":
            return jsonify({"error": "nothing to regenerate"}), 400
    else:
        user_parts = []
        if user_message:
            lang_reminder = _detect_lang_reminder(user_message)
            if news_context:
                user_parts.append({"text": f"{lang_reminder}[Live news context:]\n{news_context}\n\n[User question:] {user_message}"})
            else:
                user_parts.append({"text": f"{lang_reminder}{user_message}"})
        attachment_meta = None
        if attachment:
            mime_type = attachment.get("mimeType", "application/octet-stream")
            user_parts.append({"inline_data": {"mime_type": mime_type, "data": attachment["dataBase64"]}})
            attachment_meta = {"name": attachment.get("name", "file"), "mimeType": mime_type}
        user_entry = {"role": "user", "parts": user_parts}
        if attachment_meta:
            user_entry["attachment_meta"] = attachment_meta
        messages.append(user_entry)

    history_for_model = _trim_history(messages)

    effective_system_prompt = SYSTEM_PROMPT
    if user_name:
        effective_system_prompt += (
            f" The user has told you their preferred name is \"{user_name}\". "
            f"Address them as {user_name} naturally where it fits — don't force it into every reply."
        )
    openai_msgs = to_openai_messages(history_for_model, effective_system_prompt)

    def generate():
        full_reply = []
        def try_chunks():
            if provider == "groq":
                yield from groq_stream(openai_msgs, model_name)
            elif provider == "cerebras":
                chunks = list(cerebras_stream(openai_msgs, model_name))
                if chunks and len(chunks) == 1 and chunks[0].startswith("[Cerebras"):
                    # Cerebras failed -> fall back to Groq 70B
                    yield from groq_stream(openai_msgs, "llama-3.3-70b-versatile")
                else:
                    yield from chunks
            else:
                providers = []
                if GROQ_API_KEY:
                    providers.append(("groq", lambda: groq_stream(openai_msgs, GROQ_MODEL)))
                if CEREBRAS_API_KEY:
                    providers.append(("cerebras", lambda: cerebras_stream(openai_msgs, CEREBRAS_MODEL)))
                if not providers:
                    yield "[No AI providers configured. Add GROQ_API_KEY and/or CEREBRAS_API_KEY to .env.]"
                    return
                for _, fn in providers:
                    yielded_any = False
                    for chunk in fn():
                        yielded_any = True
                        yield chunk
                    if yielded_any:
                        return
                yield "[All AI providers failed or are rate-limited. Try again in a moment.]"

        for chunk in try_chunks():
            full_reply.append(chunk)
            yield chunk
        messages.append({"role": "model", "parts": [{"text": "".join(full_reply)}]})
        save_conversation(username, conv_id, conv)

    resp = Response(stream_with_context(generate()), mimetype="text/plain; charset=utf-8")
    resp.headers["X-Conversation-Id"] = conv_id
    return resp

# ============================================================
# HTML page
# ============================================================
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mythic AI</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:#1a1a1a; --panel:#2a2a2a; --border:#3a3a3a;
    --text:#ececec; --muted:#8e8ea0; --accent:#10a37f;
    --accent-dim:#1a3a30; --user-bubble:#2a2a2a; --user-text:#ececec;
    --ai-bubble:#1a1a1a; --sidebar-w:260px; --msg-font-size:14.5px;
  }
  body.theme-light {
    --bg:#f7f7f8; --panel:#ffffff; --border:#e2e2e6;
    --text:#1a1a1a; --muted:#6b6b76; --accent-dim:#e6f6f0;
    --user-bubble:#e9e9ed; --user-text:#1a1a1a; --ai-bubble:#f0f0f2;
  }
  body.bubble-compact .msg { padding:6px 11px; line-height:1.35; font-size:calc(var(--msg-font-size) - 1px); }
  body.bubble-comfortable .msg { padding:11px 15px; line-height:1.6; font-size:var(--msg-font-size); }
  body.bubble-spacious .msg { padding:16px 20px; line-height:1.8; font-size:calc(var(--msg-font-size) + 1px); }
  * { box-sizing:border-box; margin:0; padding:0; }
  html,body { height:100%; background:var(--bg); color:var(--text);
    font-family:'Inter','Noto Sans Devanagari',-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; overflow:hidden; }
  .layout { display:flex; height:100vh; }
  #sidebar { width:var(--sidebar-w); flex-shrink:0; background:var(--panel);
    border-right:1px solid var(--border); display:flex; flex-direction:column;
    transition:margin-left .2s ease; }
  #sidebar.hidden { margin-left:calc(-1 * var(--sidebar-w)); }
  #new-chat-btn { margin:12px; padding:10px 14px; background:var(--accent); color:#fff;
    border:none; border-radius:8px; font-size:13.5px; font-weight:600; cursor:pointer; text-align:left; }
  #conv-list { flex:1; overflow-y:auto; padding:0 8px; display:flex; flex-direction:column; gap:2px; }
  .conv-item { display:flex; align-items:center; justify-content:space-between; gap:6px;
    padding:9px 10px; border-radius:7px; cursor:pointer; font-size:13px; color:var(--muted); }
  .conv-item:hover { background:var(--accent-dim); color:var(--text); }
  .conv-item.active { background:var(--accent-dim); color:var(--accent); font-weight:500; }
  .conv-item .title { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1; }
  .conv-item .rename-btn, .conv-item .del-btn { opacity:0; background:none; border:none; color:var(--muted);
    cursor:pointer; font-size:12px; padding:2px 5px; flex-shrink:0; touch-action:manipulation; }
  .conv-item:hover .rename-btn, .conv-item:hover .del-btn { opacity:1; }
  .conv-item .del-btn:hover { color:#ef4444; }
  .conv-item .rename-btn:hover { color:var(--accent); }
  #sidebar-footer { padding:12px; font-size:11px; color:var(--muted); border-top:1px solid var(--border); }
  .app { display:flex; flex-direction:column; height:100vh; flex:1; min-width:0; }
  header { padding:calc(14px + env(safe-area-inset-top)) 20px 14px; border-bottom:1px solid var(--border);
    display:flex; align-items:center; justify-content:space-between; gap:10px; background:var(--bg); }
  header .left { display:flex; align-items:center; gap:10px; min-width:0; }
  header .right { display:flex; align-items:center; gap:8px; flex-shrink:0; }
  header button { touch-action:manipulation; -webkit-tap-highlight-color:transparent; }
  #sidebar-toggle { background:none; border:1px solid var(--border); color:var(--muted);
    width:36px; height:36px; border-radius:6px; cursor:pointer; font-size:15px; flex-shrink:0; }
  header h1 { font-size:16px; font-weight:700; color:var(--accent); margin:0; }
  #name-btn, #settings-btn, #export-btn, #fullscreen-btn, #clear-btn {
    background:none; border:1px solid var(--border); color:var(--muted);
    width:36px; height:36px; border-radius:6px; cursor:pointer; font-size:15px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center;
    -webkit-tap-highlight-color:transparent; }
  #name-btn:hover, #settings-btn:hover, #export-btn:hover, #fullscreen-btn:hover { background:var(--panel); }
  #settings-btn:hover, #name-btn:hover { color:var(--accent); }
  #clear-btn:hover { background:var(--panel); color:#ef4444; border-color:#ef4444; }
  #name-modal-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.55);
    z-index:200; align-items:center; justify-content:center; }
  #name-modal-overlay.show { display:flex; }
  #name-modal { background:var(--bg); border:1px solid var(--border); border-radius:14px;
    padding:22px; width:90%; max-width:360px; }
  #name-modal h3 { margin:0 0 6px; font-size:16px; }
  #name-modal p { margin:0 0 14px; font-size:12.5px; color:var(--muted); }
  #name-input { width:100%; box-sizing:border-box; padding:10px 12px; border-radius:8px;
    border:1.5px solid var(--border); background:var(--panel); color:var(--text);
    font-size:14.5px; outline:none; font-family:inherit; }
  #name-input:focus { border-color:var(--accent); }
  #name-modal-actions { display:flex; justify-content:flex-end; gap:8px; margin-top:16px; }
  #name-modal-actions button { padding:8px 14px; border-radius:8px; font-size:13px;
    cursor:pointer; border:1px solid var(--border); background:none; color:var(--text); }
  #name-save-btn { background:var(--accent); color:#fff; border-color:var(--accent); }
  #messages-wrap { flex:1; overflow-y:auto; position:relative; }
  #messages { padding:24px 20px; display:flex; flex-direction:column; gap:16px;
    max-width:760px; margin:0 auto; width:100%; min-height:100%; }
  .msg { max-width:80%; padding:11px 15px; border-radius:18px; line-height:1.6;
    font-size:var(--msg-font-size); white-space:pre-wrap; word-wrap:break-word; }
  .msg.user { align-self:flex-end; background:var(--user-bubble); color:var(--user-text);
    border-bottom-right-radius:4px; }
  .msg.ai { align-self:flex-start; background:var(--ai-bubble); color:var(--text);
    border-bottom-left-radius:4px; }
  .msg-text code { background:var(--panel); border:1px solid var(--border); border-radius:4px;
    padding:1px 5px; font-family:'SFMono-Regular',Consolas,Menlo,monospace; font-size:.92em; }
  .msg-text pre { background:var(--panel); border:1px solid var(--border); border-radius:8px;
    padding:10px 12px; overflow-x:auto; margin:6px 0; white-space:pre; }
  .msg-text pre code { background:none; border:none; padding:0; }
  .msg-text ul, .msg-text ol { margin:4px 0 4px 20px; }
  .msg-text p { margin:0 0 6px; } .msg-text p:last-child { margin-bottom:0; }
  .msg-text strong { font-weight:700; } .msg-text em { font-style:italic; } .msg-text a { color:var(--accent); }
  .msg.error { align-self:center; background:#fef2f2; border:1px solid #fecaca;
    color:#dc2626; font-size:13px; border-radius:10px; max-width:90%; }
  .msg img { max-width:100%; border-radius:10px; display:block; margin-top:8px; }
  .attach-chip { font-size:11.5px; opacity:.75; margin-bottom:4px; }
  .msg-row { display:flex; flex-direction:column; max-width:80%; }
  .msg-row.user { align-self:flex-end; align-items:flex-end; }
  .msg-row.ai { align-self:flex-start; align-items:flex-start; }
  .msg-row.error { align-self:center; align-items:center; max-width:90%; }
  .msg-row .msg { max-width:100%; }
  .msg-actions { display:flex; gap:4px; margin-top:3px; opacity:0; height:22px; }
  .msg-row:hover .msg-actions, .msg-row:focus-within .msg-actions { opacity:1; }
  .msg-actions button { background:none; border:none; color:var(--muted); cursor:pointer;
    font-size:12px; padding:2px 7px; border-radius:5px;
    -webkit-tap-highlight-color:transparent; }
  .msg-actions button:hover { background:var(--panel); color:var(--text); }
  .empty-state { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
    text-align:center; color:var(--muted); pointer-events:none; }
  .empty-state h2 { font-size:22px; font-weight:700; color:var(--accent); margin-bottom:8px; }
  .empty-state p { font-size:14px; }
  .typing { align-self:flex-start; display:flex; gap:5px; padding:14px 16px;
    background:var(--ai-bubble); border-radius:18px; border-bottom-left-radius:4px; }
  .typing span { width:7px; height:7px; border-radius:50%; background:var(--muted);
    animation:blink 1.2s infinite ease-in-out; }
  .typing span:nth-child(2) { animation-delay:.2s; }
  .typing span:nth-child(3) { animation-delay:.4s; }
  @keyframes blink { 0%,80%,100%{opacity:.2} 40%{opacity:1} }
  #scroll-btn { position:fixed; bottom:130px; right:24px; width:36px; height:36px;
    border-radius:50%; background:var(--accent); color:#fff; border:none; cursor:pointer;
    font-size:18px; display:none; align-items:center; justify-content:center;
    box-shadow:0 2px 8px rgba(0,0,0,.15); z-index:10; }
  #scroll-btn.show { display:flex; }
  .gen-img { max-width:320px; border-radius:12px; display:block; margin-top:8px; }
  #pending-attach { max-width:760px; margin:0 auto; width:100%; padding:6px 20px 0;
    display:none; align-items:center; gap:8px; font-size:12.5px; color:var(--muted); }
  #pending-attach.show { display:flex; }
  #pending-attach button { background:none; border:none; color:var(--muted); cursor:pointer; }
  .input-area { padding:10px 20px 16px; border-top:1px solid var(--border);
    background:var(--bg); max-width:760px; margin:0 auto; width:100%; }
  .input-row { display:flex; gap:8px; align-items:flex-end; background:var(--panel);
    border:1.5px solid var(--border); border-radius:14px; padding:8px 10px; }
  .input-row:focus-within { border-color:var(--accent); }
  .tool-btn { background:none; border:none; color:var(--muted); cursor:pointer;
    width:36px; height:36px; border-radius:8px; font-size:18px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center;
    -webkit-tap-highlight-color:transparent; }
  .tool-btn:hover { background:var(--accent-dim); color:var(--accent); }
  textarea { flex:1; resize:none; background:transparent; border:none; color:var(--text);
    font-size:14.5px; font-family:inherit; line-height:1.4; max-height:140px;
    outline:none; padding:4px 0; }
  textarea::placeholder { color:var(--muted); }
  #send-btn { background:var(--accent); color:#fff; border:none; border-radius:10px;
    width:36px; height:36px; font-size:18px; cursor:pointer; flex-shrink:0;
    display:flex; align-items:center; justify-content:center;
    -webkit-tap-highlight-color:transparent; }
  #send-btn.generating { background:#ef4444; }
  #voice-btn.listening { color:#ef4444; animation:pulse 1s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  #speaking-indicator { display:none; align-items:center; gap:6px; font-size:12px;
    color:var(--accent); padding:4px 0; }
  #speaking-indicator.show { display:flex; }
  #stop-speak-btn { background:none; border:1px solid var(--border); color:var(--muted);
    font-size:11px; padding:2px 8px; border-radius:4px; cursor:pointer; }
  #quick-actions { display:flex; gap:8px; padding:6px 20px 0;
    max-width:760px; margin:0 auto; width:100%; flex-wrap:wrap; }
  .quick-btn { background:var(--panel); border:1px solid var(--border); color:var(--text);
    font-size:12.5px; padding:6px 14px; border-radius:20px; cursor:pointer;
    transition:all .15s ease; white-space:nowrap; font-family:inherit; }
  .quick-btn:hover { background:var(--accent-dim); border-color:var(--accent); color:var(--accent); }
  @media(max-width:768px) {
    :root { --sidebar-w: 78vw; }
    #sidebar { position:fixed; top:0; left:0; z-index:100; height:100%;
      height:-webkit-fill-available; width:var(--sidebar-w) !important;
      transform:translateX(0); transition:transform .25s ease;
      box-shadow:4px 0 24px rgba(0,0,0,.5); }
    #sidebar.hidden { transform:translateX(-105%); margin-left:0 !important; }
    #sidebar-overlay { display:block; }
    .app { width:100% !important; flex:1; }
    header { padding:calc(10px + env(safe-area-inset-top)) 12px 10px; }
    header h1 { font-size:14px; }
    #messages { padding:14px 10px; gap:12px; max-width:100%; }
    .msg { max-width:90%; font-size:14px; padding:10px 12px; }
    .msg-row { max-width:90%; }
    .msg-actions { opacity:1; height:26px; }
    .input-area { padding:8px 10px max(10px,env(safe-area-inset-bottom)); }
    textarea { font-size:16px; }
    .tool-btn { width:34px; height:34px; font-size:17px; }
    #send-btn { width:34px; height:34px; font-size:16px; }
    .conv-item .rename-btn, .conv-item .del-btn { opacity:1; }
  }
  @media(max-width:380px) { :root { --sidebar-w: 88vw; } .msg { font-size:13.5px; } }
</style>
</head>
<body>
<div class="layout">
  <div id="sidebar-overlay" style="display:none;position:fixed;inset:0;background:#0007;z-index:99"></div>
  <div id="sidebar">
    <button id="new-chat-btn">+ New chat</button>
    <div id="conv-list"></div>
    <div id="sidebar-footer">Mythic AI &middot; by Aarav Singh</div>
  </div>
  <div class="app">
    <header>
      <div class="left">
        <button id="sidebar-toggle" title="Toggle sidebar">☰</button>
        <h1>Mythic AI</h1>
        <select id="model-select" style="background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:5px 8px;font-size:12px;cursor:pointer;outline:none;max-width:130px;font-family:inherit;">
          <option value="aarav-1.0">Mythic 1.0</option>
          <option value="aarav-2.0">Mythic 2.0</option>
          <option value="aarav-2.5" selected>Mythic 2.5</option>
          <option value="aarav-3.5">Mythic 3.5</option>
          <option value="aarav-ultra">Mythic Ultra 🔒</option>
        </select>
      </div>
      <div class="right">
        <button id="settings-btn" title="Settings">⚙</button>
        <button id="fullscreen-btn" title="Fullscreen">⛶</button>
        <button id="name-btn" title="What should Mythic AI call you?">🙂</button>
        <button id="export-btn" title="Export this chat">⬇</button>
        <button id="clear-btn" title="Delete this chat">🗑</button>
      </div>
    </header>
    <div id="messages-wrap">
      <div id="messages">
        <div class="empty-state" id="empty-state">
          <h2>Mythic AI</h2>
          <p>Ask me anything, generate images, or just chat 👋</p>
        </div>
      </div>
    </div>
    <button id="scroll-btn" title="Scroll to bottom">↓</button>
    <div id="pending-attach">📎 <span id="pending-attach-name"></span> <button id="pending-attach-remove">✕</button></div>
    <div id="speaking-indicator">🔊 Speaking... <button id="stop-speak-btn">Stop</button></div>
    <div id="quick-actions">
      <button class="quick-btn" id="img-gen-btn">🎨 Image</button>
      <button class="quick-btn" id="homework-btn">📚 Homework</button>
      <button class="quick-btn" id="weather-btn">🌤 Weather</button>
      <button class="quick-btn" id="search-btn">🔍 Search</button>
    </div>
    <div class="input-area">
      <form id="chat-form">
        <div class="input-row">
          <input type="file" id="file-input" accept="image/*,.txt,.md,.csv,.json,.pdf" style="display:none">
          <button class="tool-btn" id="attach-btn" type="button" title="Attach file">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
            </svg>
          </button>
          <input type="file" id="camera-input" accept="image/*" capture="environment" style="display:none">
          <input type="file" id="selfie-input" accept="image/*" capture="user" style="display:none">
          <button class="tool-btn" id="camera-btn" type="button" title="Take photo">📷</button>
          <button class="tool-btn" id="selfie-btn" type="button" title="Selfie">🤳</button>
          <textarea id="input" rows="1" placeholder="Message Mythic AI..."></textarea>
          <button class="tool-btn" id="voice-btn" type="button" title="Voice">🎤</button>
          <button id="send-btn" type="submit" title="Send">➤</button>
        </div>
      </form>
    </div>
  </div>
</div>

<div id="img-modal-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:300;align-items:center;justify-content:center;">
  <div style="background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:24px;width:90%;max-width:420px;">
    <h3 style="margin:0 0 6px;font-size:17px;">🎨 Generate Image</h3>
    <p style="color:var(--muted);font-size:13px;margin:0 0 16px;">Describe the image you want to create</p>
    <select id="img-style" style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:10px;padding:10px 12px;font-size:14px;margin-bottom:10px;outline:none;font-family:inherit;">
      <option value="ghibli">🌿 Studio Ghibli</option>
      <option value="anime">🎌 Anime</option>
      <option value="realistic">📷 Realistic</option>
      <option value="oil painting">🖼 Oil Painting</option>
      <option value="watercolor">🎨 Watercolor</option>
      <option value="3d render">🧊 3D Render</option>
      <option value="cartoon">🐱 Cartoon</option>
      <option value="digital art">💻 Digital Art</option>
      <option value="">✨ No Style</option>
    </select>
    <textarea id="img-prompt" rows="3" placeholder="e.g. A girl walking through a forest..." style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:10px;padding:10px 12px;font-size:14px;font-family:inherit;resize:none;outline:none;"></textarea>
    <div id="img-result" style="display:none;margin-top:12px;text-align:center;">
      <img id="img-output" style="max-width:100%;border-radius:10px;display:block;margin:0 auto;">
    </div>
    <div id="img-loading" style="display:none;text-align:center;padding:20px;color:var(--muted);font-size:13px;">⏳ Generating...</div>
    <div id="img-error" style="display:none;color:#ef4444;font-size:12px;margin-top:8px;"></div>
    <div style="display:flex;gap:8px;margin-top:14px;">
      <button id="img-generate-btn" style="flex:1;background:var(--accent);color:#fff;border:none;border-radius:8px;padding:10px;font-size:14px;font-weight:600;cursor:pointer;">Generate</button>
      <button id="img-close-btn" style="flex:1;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:10px;font-size:14px;cursor:pointer;">Close</button>
    </div>
  </div>
</div>

<div id="weather-modal-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:300;align-items:center;justify-content:center;">
  <div style="background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:24px;width:90%;max-width:380px;">
    <h3 style="margin:0 0 6px;font-size:17px;">🌤 Weather</h3>
    <p style="color:var(--muted);font-size:13px;margin:0 0 14px;">Enter city name or use your location</p>
    <div style="display:flex;gap:8px;margin-bottom:10px;">
      <input id="weather-city" type="text" placeholder="e.g. Delhi, Mumbai, London..." style="flex:1;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:10px 12px;font-size:14px;outline:none;font-family:inherit;">
      <button id="weather-location-btn" style="background:var(--panel);border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:8px 12px;font-size:18px;cursor:pointer;">📍</button>
    </div>
    <div id="weather-result" style="display:none;background:var(--bg);border-radius:12px;padding:16px;margin-bottom:12px;"><div id="weather-content"></div></div>
    <div id="weather-loading" style="display:none;text-align:center;padding:16px;color:var(--muted);font-size:13px;">⏳ Fetching...</div>
    <div id="weather-error" style="display:none;color:#ef4444;font-size:12px;margin-bottom:8px;"></div>
    <div style="display:flex;gap:8px;">
      <button id="weather-search-btn" style="flex:1;background:var(--accent);color:#fff;border:none;border-radius:8px;padding:10px;font-size:14px;font-weight:600;cursor:pointer;">Get Weather</button>
      <button id="weather-close-btn" style="flex:1;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:10px;font-size:14px;cursor:pointer;">Close</button>
    </div>
  </div>
</div>

<div id="name-modal-overlay">
  <div id="name-modal">
    <h3>What should Mythic AI call you?</h3>
    <p>Enter your preferred name.</p>
    <input type="text" id="name-input" maxlength="60" placeholder="e.g. Aarav" autocomplete="off">
    <div id="name-modal-actions">
      <button id="name-cancel-btn" type="button">Cancel</button>
      <button id="name-save-btn" type="button">Save</button>
    </div>
  </div>
</div>

<div id="settings-modal-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:250;align-items:center;justify-content:center;">
  <div style="background:var(--bg);border:1px solid var(--border);border-radius:14px;padding:22px;width:92%;max-width:420px;max-height:85vh;overflow-y:auto;">
    <h3 style="margin:0 0 14px;font-size:17px;">⚙ Settings</h3>
    <div style="margin-bottom:16px;">
      <label style="font-size:12.5px;color:var(--muted);display:block;margin-bottom:6px;">Theme</label>
      <div style="display:flex;gap:6px;">
        <button class="settings-choice" data-group="theme" data-value="dark" style="flex:1;padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--text);cursor:pointer;font-size:12.5px;font-family:inherit;">🌙 Dark</button>
        <button class="settings-choice" data-group="theme" data-value="light" style="flex:1;padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--text);cursor:pointer;font-size:12.5px;font-family:inherit;">☀ Light</button>
        <button class="settings-choice" data-group="theme" data-value="system" style="flex:1;padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--text);cursor:pointer;font-size:12.5px;font-family:inherit;">🖥 System</button>
      </div>
    </div>
    <div style="margin-bottom:16px;">
      <label style="font-size:12.5px;color:var(--muted);display:block;margin-bottom:6px;">Accent color</label>
      <input type="color" id="accent-color-input" value="#10a37f" style="width:100%;height:38px;border-radius:8px;border:1px solid var(--border);background:var(--panel);cursor:pointer;">
    </div>
    <div style="margin-bottom:16px;">
      <label style="font-size:12.5px;color:var(--muted);display:block;margin-bottom:6px;">Font size: <span id="font-size-label">14.5px</span></label>
      <input type="range" id="font-size-slider" min="12" max="19" step="0.5" value="14.5" style="width:100%;">
    </div>
    <div style="margin-bottom:16px;">
      <label style="font-size:12.5px;color:var(--muted);display:block;margin-bottom:6px;">Chat bubble style</label>
      <div style="display:flex;gap:6px;">
        <button class="settings-choice" data-group="bubble" data-value="compact" style="flex:1;padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--text);cursor:pointer;font-size:12.5px;font-family:inherit;">Compact</button>
        <button class="settings-choice" data-group="bubble" data-value="comfortable" style="flex:1;padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--text);cursor:pointer;font-size:12.5px;font-family:inherit;">Comfortable</button>
        <button class="settings-choice" data-group="bubble" data-value="spacious" style="flex:1;padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--text);cursor:pointer;font-size:12.5px;font-family:inherit;">Spacious</button>
      </div>
    </div>
    <div style="margin-bottom:16px;">
      <label style="font-size:12.5px;color:var(--muted);display:block;margin-bottom:6px;">Tone</label>
      <select id="tone-select" style="width:100%;background:var(--panel);border:1px solid var(--border);color:var(--text);border-radius:8px;padding:8px;font-size:13px;font-family:inherit;">
        <option value="default">Default</option><option value="formal">Formal</option>
        <option value="casual">Casual</option><option value="funny">Funny</option>
        <option value="professional">Professional</option>
      </select>
    </div>
    <div style="margin-bottom:16px;">
      <label style="font-size:12.5px;color:var(--muted);display:block;margin-bottom:6px;">Response length</label>
      <select id="length-select" style="width:100%;background:var(--panel);border:1px solid var(--border);color:var(--text);border-radius:8px;padding:8px;font-size:13px;font-family:inherit;">
        <option value="default">Default</option><option value="short">Short</option>
        <option value="medium">Medium</option><option value="long">Long</option>
      </select>
    </div>
    <div style="margin-bottom:16px;">
      <label style="font-size:12.5px;color:var(--muted);display:block;margin-bottom:6px;">Custom instructions (persona)</label>
      <textarea id="custom-instructions-input" rows="3" placeholder="e.g. Always answer like a strict but kind teacher..." style="width:100%;background:var(--panel);border:1px solid var(--border);color:var(--text);border-radius:8px;padding:8px;font-size:13px;font-family:inherit;resize:none;"></textarea>
    </div>
    <button id="settings-close-btn" style="width:100%;background:var(--accent);color:#fff;border:none;border-radius:8px;padding:10px;font-size:14px;font-weight:600;cursor:pointer;">Done</button>
  </div>
</div>

<script>
function escapeHTML(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;')}
function renderMarkdown(raw){
  if(!raw) return '';
  let s = escapeHTML(raw);
  s = s.replace(/```([a-zA-Z0-9_+\-]*)\n?([\s\S]*?)```/g,(m,lang,code)=>'<pre><code class="lang-'+(lang||'')+'">'+code.replace(/\n$/,'')+'</code></pre>');
  s = s.replace(/`([^`\n]+)`/g,'<code>$1</code>');
  s = s.replace(/^######\s+(.*)$/gm,'<h6>$1</h6>').replace(/^#####\s+(.*)$/gm,'<h5>$1</h5>').replace(/^####\s+(.*)$/gm,'<h4>$1</h4>').replace(/^###\s+(.*)$/gm,'<h3>$1</h3>').replace(/^##\s+(.*)$/gm,'<h2>$1</h2>').replace(/^#\s+(.*)$/gm,'<h1>$1</h1>');
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,'<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  s = s.replace(/\*\*([^*\n]+)\*\*/g,'<strong>$1</strong>').replace(/(^|[^*])\*([^*\n]+)\*/g,'$1<em>$2</em>');
  s = s.replace(/(^|\n)((?:[-*]\s+.+(?:\n|$))+)/g,(m,pre,block)=>{const items=block.trim().split(/\n/).map(l=>'<li>'+l.replace(/^[-*]\s+/,'')+'</li>').join('');return pre+'<ul>'+items+'</ul>'});
  s = s.replace(/(^|\n)((?:\d+\.\s+.+(?:\n|$))+)/g,(m,pre,block)=>{const items=block.trim().split(/\n/).map(l=>'<li>'+l.replace(/^\d+\.\s+/,'')+'</li>').join('');return pre+'<ol>'+items+'</ol>'});
  s = s.split(/\n{2,}/).map(chunk=>/^\s*<(h\d|ul|ol|pre|p|blockquote)/.test(chunk)?chunk:'<p>'+chunk.replace(/\n/g,'<br>')+'</p>').join('\n');
  return s;
}

const messagesWrap=document.getElementById('messages-wrap');
const messagesEl=document.getElementById('messages');
const form=document.getElementById('chat-form');
const input=document.getElementById('input');
const sendBtn=document.getElementById('send-btn');
const clearBtn=document.getElementById('clear-btn');
const convListEl=document.getElementById('conv-list');
const newChatBtn=document.getElementById('new-chat-btn');
const sidebarToggle=document.getElementById('sidebar-toggle');
const sidebar=document.getElementById('sidebar');
const sidebarOverlay=document.getElementById('sidebar-overlay');
const fullscreenBtn=document.getElementById('fullscreen-btn');
const nameBtn=document.getElementById('name-btn');
const nameModalOverlay=document.getElementById('name-modal-overlay');
const nameInput=document.getElementById('name-input');
const nameCancelBtn=document.getElementById('name-cancel-btn');
const nameSaveBtn=document.getElementById('name-save-btn');
const exportBtn=document.getElementById('export-btn');
const fileInput=document.getElementById('file-input');
const attachBtn=document.getElementById('attach-btn');
const cameraInput=document.getElementById('camera-input');
const selfieInput=document.getElementById('selfie-input');
const cameraBtn=document.getElementById('camera-btn');
const selfieBtn=document.getElementById('selfie-btn');
const voiceBtn=document.getElementById('voice-btn');
const pendingAttach=document.getElementById('pending-attach');
const pendingName=document.getElementById('pending-attach-name');
const pendingRemove=document.getElementById('pending-attach-remove');
const scrollBtn=document.getElementById('scroll-btn');
const speakingIndicator=document.getElementById('speaking-indicator');
const stopSpeakBtn=document.getElementById('stop-speak-btn');
const modelSelect=document.getElementById('model-select');

let activeConvId=null, selectedModel='aarav-2.5', vipUnlocked=false;
let pendingFile=null, recognition=null, isGenerating=false, currentAbortController=null;

function autoResize(){input.style.height='auto';input.style.height=Math.min(input.scrollHeight,140)+'px'}

function showVipModal(){
  const existing=document.getElementById('vip-modal-overlay');
  if(existing){existing.style.display='flex';return}
  const overlay=document.createElement('div');
  overlay.id='vip-modal-overlay';
  overlay.style.cssText='position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:500;display:flex;align-items:center;justify-content:center;';
  overlay.innerHTML=`
    <div style="background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:24px;width:90%;max-width:340px;">
      <div style="font-size:22px;margin-bottom:6px;">🔒 VIP Access</div>
      <div style="color:var(--muted);font-size:13px;margin-bottom:16px;">Mythic Ultra is for VIP users only.</div>
      <input id="vip-pw-in" type="password" placeholder="VIP password" style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:10px 12px;font-size:14px;outline:none;margin-bottom:8px;font-family:inherit;">
      <div id="vip-pw-err" style="color:#ef4444;font-size:12px;display:none;margin-bottom:8px;">Wrong password.</div>
      <div style="display:flex;gap:8px;">
        <button id="vip-pw-ok" style="flex:1;background:var(--accent);color:#fff;border:none;border-radius:8px;padding:10px;font-size:14px;font-weight:600;cursor:pointer;">Unlock</button>
        <button id="vip-pw-cancel" style="flex:1;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:10px;font-size:14px;cursor:pointer;">Cancel</button>
      </div>
    </div>`;
  document.body.appendChild(overlay);
  const pwIn=overlay.querySelector('#vip-pw-in');
  const pwErr=overlay.querySelector('#vip-pw-err');
  pwIn.focus();
  overlay.querySelector('#vip-pw-cancel').addEventListener('click',()=>{overlay.style.display='none';modelSelect.value=selectedModel});
  overlay.querySelector('#vip-pw-ok').addEventListener('click',async()=>{
    const r=await fetch('/api/vip-unlock',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pwIn.value.trim()})});
    const d=await r.json();
    if(d.success){vipUnlocked=true;overlay.style.display='none';selectedModel='aarav-ultra';modelSelect.value='aarav-ultra';const opt=modelSelect.querySelector('option[value="aarav-ultra"]');if(opt)opt.textContent='Mythic Ultra ✨'}
    else{pwErr.style.display='block';pwIn.value='';pwIn.focus()}
  });
  pwIn.addEventListener('keydown',e=>{if(e.key==='Enter')overlay.querySelector('#vip-pw-ok').click()});
}

(async()=>{
  try{
    const [mr,vr]=await Promise.all([fetch('/api/models').then(r=>r.json()),fetch('/api/vip-status').then(r=>r.json())]);
    vipUnlocked=vr.vip;
    modelSelect.innerHTML='';
    mr.models.forEach(m=>{
      const opt=document.createElement('option');
      opt.value=m.id;
      opt.textContent=m.vip?(vipUnlocked?'Mythic Ultra ✨':'Mythic Ultra 🔒'):m.name;
      opt.dataset.vip=m.vip?'1':'0';
      if(m.id===mr.default)opt.selected=true;
      modelSelect.appendChild(opt);
    });
    selectedModel=mr.default;
  }catch{}
})();
modelSelect.addEventListener('change',()=>{
  const opt=modelSelect.options[modelSelect.selectedIndex];
  if(opt&&opt.dataset.vip==='1'&&!vipUnlocked)showVipModal();
  else selectedModel=modelSelect.value;
});

messagesWrap.addEventListener('scroll',()=>{const nearBottom=messagesWrap.scrollHeight-messagesWrap.scrollTop-messagesWrap.clientHeight<120;scrollBtn.classList.toggle('show',!nearBottom)});
scrollBtn.addEventListener('click',()=>messagesWrap.scrollTo({top:messagesWrap.scrollHeight,behavior:'smooth'}));
function scrollToBottom(){requestAnimationFrame(()=>messagesWrap.scrollTo({top:messagesWrap.scrollHeight,behavior:'smooth'}))}
function clearEmptyState(){const es=document.getElementById('empty-state');if(es)es.remove()}
function showEmptyState(){messagesEl.innerHTML='<div class="empty-state" id="empty-state"><h2>Mythic AI</h2><p>Ask me anything 👋</p></div>'}

function addMessage(role,text,attachment){
  clearEmptyState();
  const row=document.createElement('div');row.className='msg-row '+role;
  const div=document.createElement('div');div.className='msg '+role;
  if(attachment){
    const chip=document.createElement('div');chip.className='attach-chip';chip.textContent='📎 '+attachment.name;div.appendChild(chip);
    if(attachment.mimeType&&attachment.mimeType.startsWith('image/')&&attachment.dataBase64){
      const img=document.createElement('img');img.src='data:'+attachment.mimeType+';base64,'+attachment.dataBase64;div.appendChild(img);
    }
  }
  const textNode=document.createElement('div');textNode.className='msg-text';
  if(role==='ai')textNode.innerHTML=renderMarkdown(text||'');
  else textNode.textContent=text||'';
  div.appendChild(textNode);row.appendChild(div);
  if(role==='user'||role==='ai')row.appendChild(buildMsgActions(row,textNode,role,text||''));
  messagesEl.appendChild(row);scrollToBottom();return textNode;
}
function buildMsgActions(row,textNode,role,rawText){
  const actions=document.createElement('div');actions.className='msg-actions';
  const copyBtn=document.createElement('button');copyBtn.type='button';copyBtn.title='Copy';copyBtn.textContent='📋';
  copyBtn.addEventListener('click',async()=>{
    try{await navigator.clipboard.writeText(rawText)}catch{const ta=document.createElement('textarea');ta.value=rawText;document.body.appendChild(ta);ta.select();try{document.execCommand('copy')}catch{}ta.remove()}
    const orig=copyBtn.textContent;copyBtn.textContent='✓';setTimeout(()=>copyBtn.textContent=orig,1200);
  });
  actions.appendChild(copyBtn);
  if(role==='ai'){
    const regenBtn=document.createElement('button');regenBtn.type='button';regenBtn.title='Regenerate';regenBtn.textContent='↻';
    regenBtn.addEventListener('click',()=>regenerateLast(row));actions.appendChild(regenBtn);
  }
  return actions;
}
function showTyping(){const div=document.createElement('div');div.className='typing';div.id='typing-indicator';div.innerHTML='<span></span><span></span><span></span>';messagesEl.appendChild(div);scrollToBottom()}
function hideTyping(){const el=document.getElementById('typing-indicator');if(el)el.remove()}

function isHindi(text){return /[\u0900-\u097F]/.test(text)}
function speak(text){
  if(!window.speechSynthesis)return;window.speechSynthesis.cancel();
  const plain=text.replace(/[#*`_~>]/g,'').trim();if(!plain)return;
  const utt=new SpeechSynthesisUtterance(plain);utt.rate=0.95;utt.pitch=1.1;
  const hindi=isHindi(plain);utt.lang=hindi?'hi-IN':'en-IN';
  const voices=window.speechSynthesis.getVoices();
  if(voices.length){
    let chosen=null;
    if(hindi){chosen=voices.find(v=>v.lang.startsWith('hi')&&/female|woman|lekha|kalpana|aditi|riya|priya|sunita/i.test(v.name))||voices.find(v=>v.lang.startsWith('hi'))}
    else{chosen=voices.find(v=>v.lang==='en-IN'&&/female|woman|aditi|riya/i.test(v.name))||voices.find(v=>v.lang==='en-IN')||voices.find(v=>v.lang.startsWith('en')&&/female|woman|samantha|victoria|karen|sonia|aria|jenny|zira|hazel|susan/i.test(v.name))||voices.find(v=>v.lang.startsWith('en'))}
    if(chosen)utt.voice=chosen;
  }
  utt.onstart=()=>speakingIndicator.classList.add('show');
  utt.onend=()=>speakingIndicator.classList.remove('show');
  utt.onerror=()=>speakingIndicator.classList.remove('show');
  window.speechSynthesis.speak(utt);
}
if(window.speechSynthesis){window.speechSynthesis.getVoices();window.speechSynthesis.onvoiceschanged=()=>window.speechSynthesis.getVoices()}
stopSpeakBtn.addEventListener('click',()=>{window.speechSynthesis&&window.speechSynthesis.cancel();speakingIndicator.classList.remove('show')});

function setupVoice(){
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;if(!SR)return;
  recognition=new SR();recognition.continuous=false;recognition.interimResults=true;recognition.lang='hi-IN';
  let finalTranscript='';
  recognition.onstart=()=>{voiceBtn.classList.add('active','listening');finalTranscript=''};
  recognition.onresult=(e)=>{finalTranscript='';for(let i=e.resultIndex;i<e.results.length;i++){if(e.results[i].isFinal)finalTranscript+=e.results[i][0].transcript;else input.value=e.results[i][0].transcript}if(finalTranscript)input.value=finalTranscript};
  recognition.onend=()=>{voiceBtn.classList.remove('active','listening');if(input.value.trim())form.requestSubmit()};
  recognition.onerror=()=>voiceBtn.classList.remove('active','listening');
}
setupVoice();
voiceBtn.addEventListener('click',()=>{if(!recognition){alert('Voice not supported in this browser.');return}if(voiceBtn.classList.contains('listening')){recognition.stop();return}recognition.start()});

function handleFileSelect(file){
  if(!file)return;const reader=new FileReader();
  reader.onload=(e)=>{const dataUrl=e.target.result;pendingFile={name:file.name,mimeType:file.type||'application/octet-stream',dataBase64:dataUrl.split(',')[1]};pendingName.textContent=file.name;pendingAttach.classList.add('show')};
  reader.readAsDataURL(file);
}
attachBtn.addEventListener('click',()=>fileInput.click());
cameraBtn.addEventListener('click',()=>cameraInput.click());
selfieBtn.addEventListener('click',()=>selfieInput.click());
fileInput.addEventListener('change',()=>handleFileSelect(fileInput.files[0]));
cameraInput.addEventListener('change',()=>handleFileSelect(cameraInput.files[0]));
selfieInput.addEventListener('change',()=>handleFileSelect(selfieInput.files[0]));
pendingRemove.addEventListener('click',()=>{pendingFile=null;fileInput.value='';cameraInput.value='';pendingAttach.classList.remove('show')});

async function loadConversationList(){
  try{
    const r=await fetch('/api/conversations');const d=await r.json();const convs=d.conversations||[];
    convListEl.innerHTML='';
    convs.forEach(c=>{
      const item=document.createElement('div');item.className='conv-item'+(c.id===activeConvId?' active':'');
      item.innerHTML='<span class="title"></span><button class="rename-btn">✎</button><button class="del-btn">✕</button>';
      item.querySelector('.title').textContent=c.title;
      item.addEventListener('click',(e)=>{if(!e.target.classList.contains('del-btn')&&!e.target.classList.contains('rename-btn'))openConversation(c.id)});
      item.querySelector('.rename-btn').addEventListener('click',async(e)=>{e.stopPropagation();const nt=prompt('Rename chat:',c.title);if(!nt||!nt.trim()||nt.trim()===c.title)return;await fetch('/api/conversations/'+c.id,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({title:nt.trim()})});loadConversationList()});
      item.querySelector('.del-btn').addEventListener('click',async(e)=>{e.stopPropagation();await fetch('/api/conversations/'+c.id,{method:'DELETE'});if(c.id===activeConvId)startNewChat();else loadConversationList()});
      convListEl.appendChild(item);
    });
    return convs;
  }catch{return[]}
}
async function openConversation(convId){
  activeConvId=convId;
  try{const r=await fetch('/api/conversations/'+convId);if(!r.ok)return;const d=await r.json();messagesEl.innerHTML='';(d.messages||[]).forEach(m=>addMessage(m.role,m.text,m.attachment));loadConversationList()}catch{}
  if(isMobile())closeSidebar();
}
function startNewChat(){activeConvId=null;messagesEl.innerHTML='';showEmptyState();loadConversationList();if(isMobile())closeSidebar()}

function setGenerating(state){
  isGenerating=state;sendBtn.classList.toggle('generating',state);
  sendBtn.title=state?'Stop':'Send';
  sendBtn.innerHTML=state?'<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>':'<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>';
}

const IMAGE_KEYWORDS=/\b(generate|create|draw|make|paint|render|show me|ghibli|anime|realistic|cartoon|portrait|landscape|art|artwork|image of|picture of|photo of|illustration)\b/i;
async function tryGenerateImage(prompt){
  try{const r=await fetch('/api/generate-image',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt})});
    const d=await r.json();
    if(d.image){const div=document.createElement('div');div.className='msg ai';const img=document.createElement('img');img.className='gen-img';img.src='data:image/png;base64,'+d.image;div.appendChild(img);messagesEl.appendChild(div);scrollToBottom();return true}
  }catch{}return false;
}

const NEWS_RE=/\b(news|khabar|khabren|headline|aaj ki|today.*news|latest.*news|cricket|ipl|score|match|winner|won|result|election|stock|price|rate|breaking|current events?|live)\b/i;
const WEATHER_RE=/\b(weather|mausam|temperature|rain|barish|forecast|humid)\b/i;
const SEARCH_RE=/^(search:|search for|google|find|look up|what is|who is|tell me about|how to|kya hai|kaun hai|batao|dhundho|when|where|which)/i;
const GENERIC_FACT_RE=/\b(today|now|currently|latest|this year|2025|2026|recent|update|happening)\b/i;

function getLastKnownLocation(){return new Promise(r=>{if(!navigator.geolocation){r(null);return}navigator.geolocation.getCurrentPosition(p=>r({lat:p.coords.latitude,lon:p.coords.longitude}),()=>r(null),{timeout:5000})})}

async function fetchWeatherContextForChat(message){
  const m=message.match(/(?:weather|mausam|temperature|forecast)\s*(?:in|of|at)?\s*([a-zA-Z\u0900-\u097F ]{2,40})/i);
  const city=m?m[1].trim():null;
  let body=city&&city.length>1?{location:city}:null;
  if(!body){const loc=await getLastKnownLocation();if(loc)body={lat:loc.lat,lon:loc.lon}}
  if(!body)return null;
  try{
    const r=await fetch('/api/weather',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();if(!d.weather)return null;
    const w=d.weather;
    return `[Live data] Current weather in ${w.location}: ${w.temp}°C (feels like ${w.feels_like}°C), ${w.condition}, humidity ${w.humidity}%, wind ${w.wind_speed} km/h.`;
  }catch{return null}
}

async function streamReply({message=null,attachment=null,regenerate=false}={}){
  showTyping();setGenerating(true);currentAbortController=new AbortController();
  if(!regenerate){const wantsImage=IMAGE_KEYWORDS.test(message||'')&&!attachment;if(wantsImage){hideTyping();const generated=await tryGenerateImage(message);if(generated){setGenerating(false);loadConversationList();return}showTyping()}}
  let newsContext=null;
  if(!regenerate&&message&&WEATHER_RE.test(message)){const w=await fetchWeatherContextForChat(message);if(w)newsContext=w}
  if(!regenerate&&message&&NEWS_RE.test(message)){try{const nr=await fetch('/api/news',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:message.length<80?message:null})});const nd=await nr.json();if(nd.articles&&nd.articles.length>0)newsContext="[Live data] Live news headlines:\n"+nd.articles.map((a,i)=>`${i+1}. ${a.title} (${a.source})`).join('\n')}catch{}}
  if(!regenerate&&message&&(SEARCH_RE.test(message)||GENERIC_FACT_RE.test(message)||(!newsContext&&NEWS_RE.test(message)))){
    const sq=message.replace(/^(search:|search for)\s*/i,'').trim();
    try{const sr=await fetch('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:sq})});
      const sd=await sr.json();
      if(sd.results&&sd.results.length>0){const sb=`[Live data] Web search results for "${sq}":\n`+sd.results.map((r,i)=>`${i+1}. ${r.title}: ${r.snippet}`).join('\n');newsContext=newsContext?(newsContext+"\n\n"+sb):sb}
    }catch{}
  }
  let aiTextNode=null;
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},signal:currentAbortController.signal,body:JSON.stringify({message:message||'',conversation_id:activeConvId,attachment,user_name:getUserName(),regenerate:!!regenerate,news_context:newsContext,model:selectedModel})});
    if(!r.ok||!r.body){hideTyping();if(r.status===403){const e=await r.json().catch(()=>({}));if(e.error==='vip_required'){showVipModal();return}}addMessage('error','Something went wrong. Try again.');return}
    hideTyping();aiTextNode=addMessage('ai','');
    const convId=r.headers.get('X-Conversation-Id');if(convId)activeConvId=convId;
    const reader=r.body.getReader();const decoder=new TextDecoder();let buf='';
    while(true){const{done,value}=await reader.read();if(done)break;buf+=decoder.decode(value,{stream:true});aiTextNode.innerHTML=renderMarkdown(buf);scrollToBottom()}
    speak(buf);loadConversationList();
  }catch(err){hideTyping();if(err.name==='AbortError'){if(aiTextNode&&!aiTextNode.textContent.trim())aiTextNode.textContent='[Stopped]'}else{addMessage('error','Network error: '+err.message)}}
  finally{setGenerating(false);currentAbortController=null}
}

function regenerateLast(row){if(isGenerating)return;row.remove();streamReply({regenerate:true})}

form.addEventListener('submit',(e)=>{
  e.preventDefault();if(isGenerating)return;
  const text=input.value.trim();if(!text&&!pendingFile)return;
  const attachment=pendingFile;pendingFile=null;fileInput.value='';cameraInput.value='';pendingAttach.classList.remove('show');
  addMessage('user',text,attachment);input.value='';input.style.height='auto';
  streamReply({message:text,attachment});
});
sendBtn.addEventListener('click',(e)=>{if(isGenerating){e.preventDefault();if(currentAbortController)currentAbortController.abort()}});
input.addEventListener('keydown',(e)=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();form.requestSubmit()}});
input.addEventListener('input',()=>{input.style.height='auto';input.style.height=Math.min(input.scrollHeight,140)+'px'});

function isMobile(){return window.innerWidth<=768}
function openSidebar(){sidebar.classList.remove('hidden');if(isMobile())sidebarOverlay.style.display='block'}
function closeSidebar(){sidebar.classList.add('hidden');sidebarOverlay.style.display='none'}
sidebarToggle.addEventListener('click',()=>sidebar.classList.contains('hidden')?openSidebar():closeSidebar());
sidebarOverlay.addEventListener('click',closeSidebar);

const fullscreenIcon=fullscreenBtn.querySelector('span');
function isFullscreen(){return !!(document.fullscreenElement||document.webkitFullscreenElement)}
function updateFullscreenBtn(){if(isFullscreen()){fullscreenIcon.textContent='⤢';fullscreenBtn.title='Exit fullscreen';fullscreenBtn.classList.add('active')}else{fullscreenIcon.textContent='⛶';fullscreenBtn.title='Fullscreen';fullscreenBtn.classList.remove('active')}}
async function toggleFullscreen(){const el=document.documentElement;try{if(!document.fullscreenElement&&!document.webkitFullscreenElement){if(el.requestFullscreen)await el.requestFullscreen();else if(el.webkitRequestFullscreen)el.webkitRequestFullscreen()}else{if(document.exitFullscreen)await document.exitFullscreen();else if(document.webkitExitFullscreen)document.webkitExitFullscreen()}}catch{}}
fullscreenBtn.addEventListener('click',toggleFullscreen);
document.addEventListener('fullscreenchange',updateFullscreenBtn);
document.addEventListener('webkitfullscreenchange',updateFullscreenBtn);

function getUserName(){return localStorage.getItem('aarav_user_name')||''}
function setUserName(name){if(name)localStorage.setItem('aarav_user_name',name);else localStorage.removeItem('aarav_user_name')}
function openNameModal(){nameInput.value=getUserName();nameModalOverlay.classList.add('show');setTimeout(()=>nameInput.focus(),50)}
function closeNameModal(){nameModalOverlay.classList.remove('show')}
nameBtn.addEventListener('click',openNameModal);
nameCancelBtn.addEventListener('click',closeNameModal);
nameModalOverlay.addEventListener('click',(e)=>{if(e.target===nameModalOverlay)closeNameModal()});
nameSaveBtn.addEventListener('click',()=>{setUserName(nameInput.value.trim());closeNameModal()});
nameInput.addEventListener('keydown',(e)=>{if(e.key==='Enter'){e.preventDefault();nameSaveBtn.click()}else if(e.key==='Escape')closeNameModal()});
if(!localStorage.getItem('aarav_name_prompted')){localStorage.setItem('aarav_name_prompted','1');setTimeout(openNameModal,600)}

if(isMobile())sidebar.classList.add('hidden');
newChatBtn.addEventListener('click',startNewChat);
clearBtn.addEventListener('click',async()=>{if(!activeConvId)return;await fetch('/api/conversations/'+activeConvId,{method:'DELETE'});startNewChat()});

exportBtn.addEventListener('click',async()=>{if(!activeConvId){alert('Start or open a chat first.');return}try{const r=await fetch('/api/conversations/'+activeConvId);if(!r.ok)return;const d=await r.json();const lines=[`# ${d.title||'Mythic AI chat'}`,''];(d.messages||[]).forEach(m=>{lines.push(m.role==='user'?'You:':'Mythic AI:');lines.push(m.text||(m.attachment?`[attachment: ${m.attachment.name}]`:''));lines.push('')});const blob=new Blob([lines.join('\n')],{type:'text/plain;charset=utf-8'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=(d.title||'chat').replace(/[^a-z0-9_ -]/gi,'').trim().slice(0,60)+'.txt';document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(url)}catch(err){alert('Export failed: '+err.message)}});

const imgModalOverlay=document.getElementById('img-modal-overlay');
const imgPrompt=document.getElementById('img-prompt');
const imgResult=document.getElementById('img-result');
const imgOutput=document.getElementById('img-output');
const imgLoading=document.getElementById('img-loading');
const imgError=document.getElementById('img-error');
const imgGenerateBtn=document.getElementById('img-generate-btn');
const imgCloseBtn=document.getElementById('img-close-btn');
const imgGenBtn=document.getElementById('img-gen-btn');
const imgStyle=document.getElementById('img-style');

imgGenBtn.addEventListener('click',()=>{imgModalOverlay.style.display='flex';imgPrompt.focus();imgResult.style.display='none';imgError.style.display='none';imgLoading.style.display='none'});
imgCloseBtn.addEventListener('click',()=>{imgModalOverlay.style.display='none'});
imgModalOverlay.addEventListener('click',e=>{if(e.target===imgModalOverlay)imgModalOverlay.style.display='none'});
imgGenerateBtn.addEventListener('click',async()=>{
  const prompt=imgPrompt.value.trim();const style=imgStyle?imgStyle.value:'';
  if(!prompt)return;imgResult.style.display='none';imgError.style.display='none';imgLoading.style.display='block';imgGenerateBtn.disabled=true;
  try{
    const r=await fetch('/api/generate-image',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,style})});
    const d=await r.json();imgLoading.style.display='none';
    if(d.image){imgOutput.src='data:image/png;base64,'+d.image;imgResult.style.display='block';clearEmptyState();const div=document.createElement('div');div.className='msg-row ai';const bubble=document.createElement('div');bubble.className='msg ai';const caption=document.createElement('div');caption.textContent='🎨 '+prompt;caption.style.cssText='font-size:12px;opacity:.7;margin-bottom:8px;';const img=document.createElement('img');img.src='data:image/png;base64,'+d.image;img.style.cssText='max-width:100%;border-radius:10px;display:block;';bubble.appendChild(caption);bubble.appendChild(img);div.appendChild(bubble);messagesEl.appendChild(div);scrollToBottom()}
    else{imgError.textContent=d.error||'Generation failed.';imgError.style.display='block'}
  }catch(e){imgLoading.style.display='none';imgError.textContent='Network error: '+e.message;imgError.style.display='block'}
  finally{imgGenerateBtn.disabled=false}
});
imgPrompt.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();imgGenerateBtn.click()}});

const weatherOverlay=document.getElementById('weather-modal-overlay');
const weatherCity=document.getElementById('weather-city');
const weatherResult=document.getElementById('weather-result');
const weatherContent=document.getElementById('weather-content');
const weatherLoading=document.getElementById('weather-loading');
const weatherError=document.getElementById('weather-error');
const weatherSearchBtn=document.getElementById('weather-search-btn');
const weatherCloseBtn=document.getElementById('weather-close-btn');
const weatherLocationBtn=document.getElementById('weather-location-btn');
const weatherBtn=document.getElementById('weather-btn');

weatherBtn.addEventListener('click',()=>{weatherOverlay.style.display='flex';weatherCity.focus();weatherResult.style.display='none';weatherError.style.display='none'});
weatherCloseBtn.addEventListener('click',()=>{weatherOverlay.style.display='none'});
weatherOverlay.addEventListener('click',e=>{if(e.target===weatherOverlay)weatherOverlay.style.display='none'});

function renderWeather(w){
  weatherContent.innerHTML=`<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;"><div style="font-size:48px;line-height:1;">${w.icon}</div><div><div style="font-size:18px;font-weight:700;">${w.location}</div><div style="font-size:13px;color:var(--muted);">${w.condition}</div></div></div><div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;"><div style="background:var(--bg);border-radius:8px;padding:10px;"><div style="font-size:11px;color:var(--muted);">TEMPERATURE</div><div style="font-size:22px;font-weight:700;">${w.temp}°C</div><div style="font-size:11px;color:var(--muted);">Feels like ${w.feels_like}°C</div></div><div style="background:var(--bg);border-radius:8px;padding:10px;"><div style="font-size:11px;color:var(--muted);">HUMIDITY</div><div style="font-size:22px;font-weight:700;">${w.humidity}%</div><div style="font-size:11px;color:var(--muted);">Wind ${w.wind_speed} km/h</div></div></div>`;
  weatherResult.style.display='block';
  input.value=`${w.icon} Weather in ${w.location}: ${w.temp}°C, ${w.condition}. Humidity: ${w.humidity}%, Wind: ${w.wind_speed} km/h, Feels like: ${w.feels_like}°C.`;autoResize();
}
async function fetchWeather(location){
  weatherResult.style.display='none';weatherError.style.display='none';weatherLoading.style.display='block';weatherSearchBtn.disabled=true;
  try{const r=await fetch('/api/weather',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location})});const d=await r.json();weatherLoading.style.display='none';if(d.weather)renderWeather(d.weather);else{weatherError.textContent=d.error||'Could not fetch weather.';weatherError.style.display='block'}}catch(e){weatherLoading.style.display='none';weatherError.textContent='Error: '+e.message;weatherError.style.display='block'}finally{weatherSearchBtn.disabled=false}
}
weatherSearchBtn.addEventListener('click',()=>{const loc=weatherCity.value.trim();if(loc)fetchWeather(loc)});
weatherCity.addEventListener('keydown',e=>{if(e.key==='Enter')weatherSearchBtn.click()});
weatherLocationBtn.addEventListener('click',()=>{if(!navigator.geolocation){alert('Geolocation not supported');return}weatherLoading.style.display='block';weatherLocationBtn.disabled=true;navigator.geolocation.getCurrentPosition(async pos=>{const{latitude:lat,longitude:lon}=pos.coords;weatherLocationBtn.disabled=false;weatherResult.style.display='none';weatherError.style.display='none';weatherLoading.style.display='block';weatherSearchBtn.disabled=true;try{const r=await fetch('/api/weather',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({lat,lon})});const d=await r.json();weatherLoading.style.display='none';if(d.weather)renderWeather(d.weather);else{weatherError.textContent=d.error||'Could not fetch weather.';weatherError.style.display='block'}}catch(e){weatherLoading.style.display='none';weatherError.textContent='Error: '+e.message;weatherError.style.display='block'}finally{weatherSearchBtn.disabled=false}},err=>{weatherLocationBtn.disabled=false;weatherLoading.style.display='none';alert('Could not get your location: '+err.message)})});

const homeworkBtn=document.getElementById('homework-btn');
homeworkBtn.addEventListener('click',()=>{input.value='Help me with my homework: ';input.focus();autoResize();input.setSelectionRange(input.value.length,input.value.length)});
const searchBtn=document.getElementById('search-btn');
searchBtn.addEventListener('click',()=>{const q=prompt('What do you want to search for?');if(!q||!q.trim())return;input.value='Search: '+q.trim();input.focus();autoResize();form.requestSubmit()});

(async()=>{const convs=await loadConversationList();if(convs.length>0)openConversation(convs[0].id);else showEmptyState()})();
</script>
</body>
</html>"""

# ============================================================
if __name__ == "__main__":
    active = []
    if PROVIDER in ("auto", "groq") and GROQ_API_KEY:     active.append(f"Groq({GROQ_MODEL})")
    if PROVIDER in ("auto", "cerebras") and CEREBRAS_API_KEY: active.append(f"Cerebras({CEREBRAS_MODEL})")
    if not LIMITER_ENABLED:
        print("⚠️  flask-limiter not installed — chat is unrate-limited. Run: pip install flask-limiter")
    print(f"Mythic AI → http://localhost:5000")
    print(f"Providers (fallback order): {' → '.join(active) if active else 'NONE — add GROQ_API_KEY / CEREBRAS_API_KEY to .env'}")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
