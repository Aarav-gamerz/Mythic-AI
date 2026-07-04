"""
Mythic AI — single file, powered by Google's Gemini API or a local Ollama model.

Usage (Gemini — default, needs a free API key):
    1. pip install flask requests
    2. Set your API key:
         Mac/Linux:   export GEMINI_API_KEY="your-key-here"
         Windows:     set GEMINI_API_KEY=your-key-here
    3. python ai_chat.py
    4. Open http://localhost:5000 in your browser

Get a FREE API key (no credit card needed) at https://aistudio.google.com/apikey

Usage (Ollama — fully local, no API key or internet needed):
    1. Install Ollama from https://ollama.com and make sure it's running
       (`ollama serve`, or it may already be running as a background service)
    2. Pull a model, e.g.:  ollama pull llama3.1
    3. Set the provider:
         Mac/Linux:   export AI_PROVIDER=ollama
         Windows:     set AI_PROVIDER=ollama
       Optional overrides:
         OLLAMA_URL   (default: http://localhost:11434)
         OLLAMA_MODEL (default: llama3.1)
    4. python ai_chat.py
    5. Open http://localhost:5000 in your browser

Features:
- Login/register (real accounts, hashed passwords, stored in chat_data/users.json)
- Multi-conversation chat with sidebar, saved per-account, survives restarts
- File/image upload (attach an image or text file to a message)
- Web search grounding (Gemini can search Google for current info — Gemini only)
- Streaming responses (text appears word-by-word)
- Switchable AI backend: Google Gemini (cloud) or Ollama (local, private, free)
- No rate limiting — unlimited messages
"""

import os
import json
import uuid
import time
import base64
import requests
from flask import (
    Flask, request, jsonify, Response, session,
    stream_with_context
)

PROVIDER = os.environ.get("AI_PROVIDER", "auto").strip().lower()
# "auto"        = round-robin: Groq → OpenRouter → HuggingFace (all work on servers)
# "gemini"      = Google Gemini only (free tier only works locally, not on Render)
# "groq"        = Groq only
# "openrouter"  = OpenRouter only
# "huggingface" = Hugging Face only
# "ollama"      = local Ollama only

# --- API Keys (hardcoded fallbacks — override via environment variables) ------
# WARNING: don't commit a file with real keys to a public GitHub repo.
# Set these as environment variables on Render instead.
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY",    "")
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY",      "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")
CEREBRAS_API_KEY  = os.environ.get("CEREBRAS_API_KEY",  "csk-2ph5f5nxt3jrtwj5edcehpr5xh96628268fvjh4e658m4t6h")
OPENROUTER_API_KEY= os.environ.get("OPENROUTER_API_KEY","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")
HF_API_KEY        = os.environ.get("HF_API_KEY",        "")

# --- Model names -------------------------------------------------------------
GEMINI_MODEL      = "gemini-2.5-flash"
GROQ_MODEL        = os.environ.get("GROQ_MODEL",        "llama-3.1-8b-instant")
OPENROUTER_MODEL  = os.environ.get("OPENROUTER_MODEL",  "google/gemma-3-4b-it:free")
HF_MODEL          = os.environ.get("HF_MODEL",          "mistralai/Mistral-7B-Instruct-v0.3")
CEREBRAS_MODEL    = os.environ.get("CEREBRAS_MODEL",    "gpt-oss-120b")
OLLAMA_MODEL      = os.environ.get("OLLAMA_MODEL",       "llama3.1")
OLLAMA_URL        = os.environ.get("OLLAMA_URL",         "http://localhost:11434").rstrip("/")

GEMINI_STREAM_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:streamGenerateContent"
)

# keep old name for compatibility with existing references below
API_KEY = GEMINI_API_KEY
MODEL   = GEMINI_MODEL
SYSTEM_PROMPT = (
    "You are Mythic AI, a smart and friendly AI assistant made by Aarav Singh. "
    "If asked who made you, say you are Mythic AI made by Aarav Singh — say it once naturally, never repeat it unprompted. "
    "You are NOT ChatGPT, NOT GPT-4, NOT made by OpenAI, NOT made by Google, Groq, Anthropic, Meta, or any other company. "
    "You are Mythic AI, exclusively. Never claim to be any other AI. "
    "Your model tiers are: Mythic 1.0 (fast), Mythic 2.0 (balanced), Mythic 2.5 (smart), Mythic 3.5 (advanced), Mythic Ultra (most powerful). "
    "If asked what model or AI you are, say you are Mythic AI and mention the tier if known. "
    "You can help with anything: questions, writing, coding, math, ideas, or just chatting. "
    "When writing code, always wrap it in markdown code blocks with the language name. "
    "LANGUAGE: Always reply ENTIRELY in the same language the user's message is written in — "
    "never mix two languages in a single reply. If they write in Hindi, reply fully in Hindi. "
    "If they write in English, reply fully in English. Never force English on the user. "
    "ANTI-REPETITION RULES — follow strictly every reply: "
    "1. NEVER restate or echo back what the user just said. Jump straight to the answer. "
    "2. NEVER start replies with filler like Great question, Sure, Of course, Absolutely, Certainly. "
    "3. NEVER repeat information already given earlier in the conversation. Build on it. "
    "4. Be direct and natural — like a knowledgeable friend, not a customer service bot. "
    "5. Keep answers concise unless the user asks for detail."
)

import re as _re

WEATHER_KEYWORDS = _re.compile(
    r'\b(weather|temperature|temp|rain|sunny|cloudy|forecast|humidity|wind|storm|snow|hot|cold|climate|aaj ka mausam|mausam|barish|garmi|sardi)\b',
    _re.IGNORECASE
)
LOCATION_KEYWORDS = _re.compile(
    r'\b(where am i|my location|my city|my country|locate me|current location|meri location|main kahan)\b',
    _re.IGNORECASE
)

def get_client_location(ip):
    """Get approximate location from IP using free ip-api.com."""
    try:
        if not ip or ip in ('127.0.0.1', '::1', 'localhost'):
            return None
        r = requests.get(f"http://ip-api.com/json/{ip}?fields=status,city,regionName,country,lat,lon",
                         timeout=5)
        if r.status_code == 200:
            d = r.json()
            if d.get('status') == 'success':
                return d
    except Exception:
        pass
    return None

def get_weather(lat, lon, city):
    """Fetch current weather from Open-Meteo (free, no key needed)."""
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,apparent_temperature",
                "timezone": "auto",
            },
            timeout=8
        )
        if r.status_code == 200:
            d = r.json().get("current", {})
            wmo = {0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",
                   45:"Foggy",48:"Icy fog",51:"Light drizzle",53:"Drizzle",55:"Heavy drizzle",
                   61:"Light rain",63:"Rain",65:"Heavy rain",71:"Light snow",73:"Snow",75:"Heavy snow",
                   80:"Rain showers",81:"Rain showers",82:"Heavy rain showers",
                   95:"Thunderstorm",96:"Thunderstorm with hail",99:"Thunderstorm with hail"}
            code = d.get("weather_code", 0)
            condition = wmo.get(code, "Unknown")
            return (f"Current weather in {city}: {condition}, "
                    f"{d.get('temperature_2m')}°C (feels like {d.get('apparent_temperature')}°C), "
                    f"Humidity {d.get('relative_humidity_2m')}%, "
                    f"Wind {d.get('wind_speed_10m')} km/h")
    except Exception:
        pass
    return None

def build_context_injection(user_message, client_ip):
    """Return extra context to prepend to system prompt if the message needs it."""
    parts = []
    needs_weather = bool(WEATHER_KEYWORDS.search(user_message))
    needs_location = bool(LOCATION_KEYWORDS.search(user_message))

    if needs_weather or needs_location:
        loc = get_client_location(client_ip)
        if loc:
            city = loc.get('city','?')
            region = loc.get('regionName','')
            country = loc.get('country','?')
            lat = loc.get('lat')
            lon = loc.get('lon')
            parts.append(f"[USER LOCATION: {city}, {region}, {country}]")
            if needs_weather and lat and lon:
                weather = get_weather(lat, lon, city)
                if weather:
                    parts.append(f"[LIVE WEATHER DATA: {weather}]")
        else:
            parts.append("[USER LOCATION: Could not determine from IP (likely localhost/VPN)]")

    if parts:
        parts.append("[Use this real data to answer the user's question accurately. Do NOT say you lack location/weather access.]")
    return " ".join(parts)

# Extra instruction appended ONLY for Gemini, which actually has a real google_search tool wired up.
# Other providers (Groq/Cerebras/OpenRouter/HF/Ollama) have no real search access, so telling them
# "you have search" makes them hallucinate fake tool-call JSON into the visible reply — hence this
# is kept separate from the base SYSTEM_PROMPT above.
GEMINI_SEARCH_ADDENDUM = (
    " WEB SEARCH: You have access to Google Search. When the user asks about current events, "
    "live prices, news, sports scores, weather, or anything that needs up-to-date information, "
    "use the search tool to find the answer. Do not say you cannot search the web. When you use "
    "search results, translate/summarize them into the reply language — never paste a mix of languages."
)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me-" + str(uuid.uuid4()))

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB

# --- Supabase config ---------------------------------------------------------
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


# --- User accounts (Supabase: users table) -----------------------------------

def current_username():
    """Each visitor gets a unique anonymous ID stored in their browser cookie.
    No login required — conversations are private per browser session."""
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
        session.permanent = True
    return session["user_id"]


def login_required(view):
    """No-op decorator kept so all @login_required routes still work unchanged."""
    def wrapped(*args, **kwargs):
        current_username()  # ensure session id is set
        return view(*args, **kwargs)
    wrapped.__name__ = view.__name__
    return wrapped


# --- Supabase / file storage helpers ----------------------------------------

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
        requests.post(
            sb("conversations"),
            headers=headers,
            json={
                "id": conv_id,
                "username": username,
                "title": data.get("title", "New chat"),
                "updated_at": data["updated_at"],
                "messages": data.get("messages", []),
            },
            timeout=15,
        )
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


# --- Local file fallbacks for when Supabase is not configured ----------------
import os as _os
_BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_DATA_DIR = _os.path.join(_BASE_DIR, "chat_data")
_os.makedirs(_DATA_DIR, exist_ok=True)

def _user_conv_dir(username):
    path = _os.path.join(_DATA_DIR, "conversations", username)
    _os.makedirs(path, exist_ok=True)
    return path

def _conv_file(username, conv_id):
    return _os.path.join(_user_conv_dir(username), f"{conv_id}.json")

def _list_conversations_file(username):
    folder = _user_conv_dir(username)
    convs = []
    for fname in _os.listdir(folder):
        if not fname.endswith(".json"):
            continue
        path = _os.path.join(folder, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            convs.append({"id": fname[:-5], "title": d.get("title", "New chat"), "updated_at": d.get("updated_at", 0)})
        except Exception:
            continue
    convs.sort(key=lambda c: c["updated_at"], reverse=True)
    return convs

def _load_conversation_file(username, conv_id):
    path = _conv_file(username, conv_id)
    if not _os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _save_conversation_file(username, conv_id, data):
    with open(_conv_file(username, conv_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _delete_conversation_file(username, conv_id):
    path = _conv_file(username, conv_id)
    if _os.path.exists(path):
        _os.remove(path)


def make_title(first_message):
    title = (first_message or "Attachment").strip().replace("\n", " ")
    return title[:40] + ("…" if len(title) > 40 else "")


# --- HTML pages ----------------------------------------------------------

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<title>Mythic AI</title>
<style>
:root {
  --bg:#1a1a1a; --panel:#2a2a2a; --border:#3a3a3a;
  --text:#ececec; --muted:#8e8ea0; --accent:#10a37f;
  --accent-dim:#1a3a30; --sidebar-w:260px;
}
*{box-sizing:border-box;margin:0;padding:0;}
html,body{height:100%;background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;overflow:hidden;}
.layout{display:flex;height:100vh;}

/* Sidebar */
#sidebar{width:var(--sidebar-w);flex-shrink:0;background:var(--panel);
  border-right:1px solid var(--border);display:flex;flex-direction:column;
  transition:transform .25s ease;}
#sidebar.hidden{transform:translateX(-105%);}
#sidebar-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:99;}
#new-chat-btn{margin:12px;padding:11px 14px;background:var(--accent);color:#fff;
  border:none;border-radius:8px;font-size:13.5px;font-weight:600;cursor:pointer;text-align:left;}
#new-chat-btn:hover{opacity:.9;}
#conv-list{flex:1;overflow-y:auto;padding:0 8px;display:flex;flex-direction:column;gap:2px;}
.conv-item{display:flex;align-items:center;gap:6px;padding:9px 10px;border-radius:7px;
  cursor:pointer;font-size:13px;color:var(--muted);}
.conv-item:hover,.conv-item.active{background:var(--accent-dim);color:var(--accent);}
.conv-item .ctitle{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.conv-item .cbtn{opacity:0;background:none;border:none;color:var(--muted);cursor:pointer;
  font-size:13px;padding:2px 6px;flex-shrink:0;touch-action:manipulation;}
.conv-item:hover .cbtn{opacity:1;}
.conv-item .cbtn:hover{color:var(--accent);}
.conv-item .cdel:hover{color:#ef4444;}
#sidebar-footer{padding:12px;font-size:11px;color:var(--muted);border-top:1px solid var(--border);
  padding-bottom:max(12px,env(safe-area-inset-bottom));}

/* Main */
.app{display:flex;flex-direction:column;flex:1;min-width:0;height:100vh;}
header{padding:12px 16px;border-bottom:1px solid var(--border);display:flex;
  align-items:center;justify-content:space-between;gap:8px;background:var(--bg);
  padding-top:max(12px,env(safe-area-inset-top));}
header .left{display:flex;align-items:center;gap:8px;min-width:0;}
header .right{display:flex;align-items:center;gap:6px;flex-shrink:0;}
header h1{font-size:16px;font-weight:700;color:var(--accent);}
.hbtn{background:none;border:1px solid var(--border);color:var(--muted);width:36px;height:36px;
  border-radius:6px;cursor:pointer;font-size:15px;display:flex;align-items:center;
  justify-content:center;flex-shrink:0;touch-action:manipulation;-webkit-tap-highlight-color:transparent;}
.hbtn:hover{background:var(--panel);color:var(--text);}
.hbtn.on{color:var(--accent);border-color:var(--accent);}
#clear-btn:hover{color:#ef4444;border-color:#ef4444;}

/* Model bar */
/* Model selector — sits below the input box like Claude */
#model-row{display:flex;align-items:center;gap:6px;margin-top:6px;padding:0 4px;}
#model-label{font-size:14px;}
#model-select{background:none;color:var(--muted);border:none;font-size:12px;
  cursor:pointer;outline:none;padding:4px 2px;border-radius:6px;max-width:200px;}
#model-select:hover{color:var(--text);}
#model-select option{background:var(--panel);color:var(--text);}

/* Messages */
#messages-wrap{flex:1;overflow-y:auto;position:relative;}
#messages{padding:24px 20px;display:flex;flex-direction:column;gap:16px;
  max-width:760px;margin:0 auto;width:100%;min-height:100%;}
.msg-row{display:flex;flex-direction:column;max-width:80%;}
.msg-row.user{align-self:flex-end;align-items:flex-end;}
.msg-row.ai{align-self:flex-start;align-items:flex-start;}
.msg-row.error{align-self:center;align-items:center;max-width:90%;}
.msg{padding:11px 15px;border-radius:18px;line-height:1.6;font-size:14.5px;
  white-space:pre-wrap;word-wrap:break-word;max-width:100%;}
.msg.user{background:var(--panel);color:var(--text);border-bottom-right-radius:4px;}
.msg.ai{background:var(--bg);color:var(--text);border-bottom-left-radius:4px;
  border:1px solid var(--border);}
.msg.error{background:#fef2f2;border:1px solid #fecaca;color:#dc2626;font-size:13px;border-radius:10px;}
.msg img{max-width:100%;border-radius:10px;display:block;margin-top:8px;}
.attach-chip{font-size:11.5px;opacity:.75;margin-bottom:4px;}
.msg-actions{display:flex;gap:4px;margin-top:3px;opacity:0;transition:opacity .15s;height:24px;}
.msg-row:hover .msg-actions{opacity:1;}
.msg-actions button{background:none;border:none;color:var(--muted);cursor:pointer;
  font-size:12px;padding:3px 8px;border-radius:5px;touch-action:manipulation;}
.msg-actions button:hover{background:var(--panel);color:var(--text);}
.empty-state{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  text-align:center;color:var(--muted);}
.empty-state h2{font-size:22px;font-weight:700;color:var(--accent);margin-bottom:8px;}
.typing{align-self:flex-start;display:flex;gap:5px;padding:14px 16px;
  background:var(--bg);border:1px solid var(--border);border-radius:18px;border-bottom-left-radius:4px;}
.typing span{width:7px;height:7px;border-radius:50%;background:var(--muted);
  animation:blink 1.2s infinite ease-in-out;}
.typing span:nth-child(2){animation-delay:.2s;}
.typing span:nth-child(3){animation-delay:.4s;}
@keyframes blink{0%,80%,100%{opacity:.2}40%{opacity:1}}

/* Scroll btn */
#scroll-btn{position:fixed;bottom:120px;right:20px;width:36px;height:36px;border-radius:50%;
  background:var(--accent);color:#fff;border:none;cursor:pointer;font-size:18px;
  display:none;align-items:center;justify-content:center;box-shadow:0 2px 8px rgba(0,0,0,.3);z-index:10;}
#scroll-btn.show{display:flex;}

/* Input */
#pending-attach{max-width:760px;margin:0 auto;width:100%;padding:6px 20px 0;
  display:none;align-items:center;gap:8px;font-size:12.5px;color:var(--muted);}
#pending-attach.show{display:flex;}
#pending-attach button{background:none;border:none;color:var(--muted);cursor:pointer;}
.input-area{padding:10px 20px;border-top:1px solid var(--border);background:var(--bg);
  padding-bottom:max(16px,env(safe-area-inset-bottom));}
.input-wrap{max-width:760px;margin:0 auto;}
.input-row{display:flex;gap:8px;align-items:flex-end;background:var(--panel);
  border:1.5px solid var(--border);border-radius:14px;padding:8px 10px;}
.input-row:focus-within{border-color:var(--accent);}
.tool-btn{background:none;border:none;color:var(--muted);cursor:pointer;width:36px;height:36px;
  border-radius:8px;font-size:18px;flex-shrink:0;display:flex;align-items:center;
  justify-content:center;touch-action:manipulation;-webkit-tap-highlight-color:transparent;}
.tool-btn:hover{background:var(--accent-dim);color:var(--accent);}
textarea{flex:1;resize:none;background:transparent;border:none;color:var(--text);
  font-size:14.5px;font-family:inherit;line-height:1.4;max-height:140px;outline:none;padding:4px 0;}
textarea::placeholder{color:var(--muted);}
#send-btn{background:var(--accent);color:#fff;border:none;border-radius:10px;width:36px;height:36px;
  font-size:18px;cursor:pointer;flex-shrink:0;display:flex;align-items:center;justify-content:center;
  touch-action:manipulation;}
#send-btn:disabled{background:var(--accent-dim);color:var(--muted);cursor:not-allowed;}
#send-btn.stop{background:#ef4444;}
#voice-btn.listening{color:#ef4444;animation:pulse 1s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* Name modal */
#name-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);
  z-index:200;align-items:center;justify-content:center;}
#name-overlay.show{display:flex;}
#name-modal{background:var(--bg);border:1px solid var(--border);border-radius:14px;
  padding:22px;width:90%;max-width:360px;}
#name-modal h3{margin:0 0 6px;font-size:16px;}
#name-modal p{margin:0 0 14px;font-size:12.5px;color:var(--muted);}
#name-input{width:100%;padding:10px 12px;border-radius:8px;border:1.5px solid var(--border);
  background:var(--panel);color:var(--text);font-size:14.5px;outline:none;}
#name-input:focus{border-color:var(--accent);}
.modal-actions{display:flex;justify-content:flex-end;gap:8px;margin-top:16px;}
.modal-actions button{padding:8px 14px;border-radius:8px;font-size:13px;cursor:pointer;
  border:1px solid var(--border);background:none;color:var(--text);}
#name-save{background:var(--accent);color:#fff;border-color:var(--accent);}

/* Scrollbars */
#messages-wrap::-webkit-scrollbar,#conv-list::-webkit-scrollbar{width:6px;}
#messages-wrap::-webkit-scrollbar-thumb,#conv-list::-webkit-scrollbar-thumb{
  background:var(--border);border-radius:4px;}

/* Mobile */
@media(max-width:768px){
  :root{--sidebar-w:80vw;}
  html,body{overscroll-behavior-y:contain;}
  #sidebar{position:fixed;top:0;left:0;z-index:100;height:100%;height:-webkit-fill-available;}
  #sidebar.hidden{transform:translateX(-110%);}
  #sidebar-overlay{display:block;}
  .app{width:100%;}
  header{padding:10px 12px;padding-top:max(10px,env(safe-area-inset-top));}
  header h1{font-size:14px;}
  .hbtn{width:38px;height:38px;}
  /* Hide desktop header buttons on mobile — use ⋮ menu instead */
  #header-actions{display:none;}
  #more-btn{display:flex;}
  #messages{padding:12px 10px;gap:12px;}
  .msg{font-size:14px;}
  .msg-row{max-width:90%;}
  .msg-actions{opacity:1;height:28px;}
  .msg-actions button{min-height:32px;padding:4px 10px;}
  .input-area{padding:8px 10px;padding-bottom:max(10px,env(safe-area-inset-bottom));}
  textarea{font-size:16px;}
  .tool-btn{width:40px;height:40px;}
  #send-btn{width:40px;height:40px;}
  #scroll-btn{bottom:90px;right:10px;width:38px;height:38px;}
  #new-chat-btn{margin:10px;padding:12px;font-size:14px;}
  .conv-item{min-height:44px;font-size:13px;}
  .conv-item .cbtn{opacity:1;}
}
@media(max-width:380px){
  :root{--sidebar-w:90vw;}
  .msg{font-size:13.5px;}
}

/* More menu (mobile only) */
#more-btn{display:none;background:none;border:1px solid var(--border);color:var(--muted);
  width:38px;height:38px;border-radius:6px;font-size:20px;cursor:pointer;flex-shrink:0;
  align-items:center;justify-content:center;touch-action:manipulation;}
#more-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:149;}
#more-menu{display:none;position:fixed;top:60px;right:10px;background:var(--panel);
  border:1px solid var(--border);border-radius:12px;padding:6px;flex-direction:column;
  gap:2px;z-index:150;min-width:180px;box-shadow:0 8px 24px rgba(0,0,0,.4);}
#more-menu.open,#more-overlay.open{display:flex;}
#more-menu button{display:flex;align-items:center;gap:10px;background:none;border:none;
  color:var(--text);font-size:14px;padding:12px 14px;border-radius:8px;cursor:pointer;
  width:100%;touch-action:manipulation;}
#more-menu button:active{background:var(--accent-dim);}
</style>
</head>
<body>
<div class="layout">
  <div id="sidebar-overlay"></div>
  <div id="sidebar">
    <button id="new-chat-btn">+ New chat</button>
    <div id="conv-list"></div>
    <div id="sidebar-footer">Mythic AI &middot; by Aarav Singh</div>
  </div>
  <div class="app">
    <header>
      <div class="left">
        <button class="hbtn" id="sidebar-toggle" title="Toggle sidebar">☰</button>
        <h1>Mythic AI</h1>
      </div>
      <div class="right" id="header-actions">
        <button class="hbtn" id="fullscreen-btn" title="Fullscreen"><span id="fs-icon">⛶</span></button>
        <button class="hbtn" id="name-btn" title="Your name">🙂</button>
        <button class="hbtn" id="export-btn" title="Export chat">⬇</button>
        <button class="hbtn" id="clear-btn" title="Delete chat">🗑</button>
      </div>
      <button id="more-btn" title="More">⋮</button>
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

    <div id="pending-attach">
      📎 <span id="attach-name"></span>
      <button id="attach-remove">✕</button>
    </div>

    <div class="input-area">
      <div class="input-wrap">
        <form id="chat-form">
          <div class="input-row">
            <input type="file" id="file-input" accept="image/*,.txt,.md,.csv,.json,.pdf" style="display:none">
            <button class="tool-btn" id="attach-btn" type="button" title="Attach file">📎</button>
            <input type="file" id="camera-input" accept="image/*" capture="environment" style="display:none">
            <button class="tool-btn" id="camera-btn" type="button" title="Camera">📷</button>
            <textarea id="msg-input" rows="1" placeholder="Message Mythic AI..."></textarea>
            <button class="tool-btn" id="voice-btn" type="button" title="Voice">🎤</button>
            <button id="send-btn" type="submit" title="Send">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
            </button>
          </div>
        </form>
        <div id="model-row">
          <span id="model-label">⚡</span>
          <select id="model-select">
            <option value="aarav-1.0">Mythic 1.0 — Fast</option>
            <option value="aarav-2.0">Mythic 2.0 — Balanced</option>
            <option value="aarav-2.5" selected>Mythic 2.5 — Smart</option>
            <option value="aarav-3.5">Mythic 3.5 — Advanced</option>
            <option value="aarav-ultra">Mythic Ultra 🔒</option>
          </select>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- More menu (mobile) -->
<div id="more-overlay"></div>
<div id="more-menu">
  <button id="mm-fullscreen">⛶ Fullscreen</button>
  <button id="mm-name">🙂 Your name</button>
  <button id="mm-export">⬇ Export chat</button>
  <button id="mm-clear">🗑 Delete chat</button>
</div>

<!-- Name modal -->
<div id="name-overlay">
  <div id="name-modal">
    <h3>What should Mythic AI call you?</h3>
    <p>Mythic AI will use this name when talking to you.</p>
    <input type="text" id="name-input" maxlength="60" placeholder="e.g. Aarav" autocomplete="off">
    <div class="modal-actions">
      <button id="name-cancel">Cancel</button>
      <button id="name-save">Save</button>
    </div>
  </div>
</div>

<script>
// --- Element references ---
const $ = id => document.getElementById(id);
const msgWrap    = $('messages-wrap');
const msgList    = $('messages');
const form       = $('chat-form');
const msgInput   = $('msg-input');
const sendBtn    = $('send-btn');
const clearBtn   = $('clear-btn');
const convList   = $('conv-list');
const newChatBtn = $('new-chat-btn');
const sbToggle   = $('sidebar-toggle');
const fsBtn      = $('fullscreen-btn');
const fsIcon     = $('fs-icon');
const nameBtn    = $('name-btn');
const nameOverlay= $('name-overlay');
const nameInput  = $('name-input');
const nameCancel = $('name-cancel');
const nameSave   = $('name-save');
const exportBtn  = $('export-btn');
const sidebar    = $('sidebar');
const sbOverlay  = $('sidebar-overlay');
const fileInput  = $('file-input');
const attachBtn  = $('attach-btn');
const cameraInput= $('camera-input');
const cameraBtn  = $('camera-btn');
const voiceBtn   = $('voice-btn');
const pendingDiv = $('pending-attach');
const attachName = $('attach-name');
const attachRm   = $('attach-remove');
const scrollBtn  = $('scroll-btn');
const modelSelect= $('model-select');
const moreBtn    = $('more-btn');
const moreOverlay= $('more-overlay');
const moreMenu   = $('more-menu');
const mmFs       = $('mm-fullscreen');
const mmName     = $('mm-name');
const mmExport   = $('mm-export');
const mmClear    = $('mm-clear');

// --- State ---
let activeConvId = null;
let pendingFile  = null;
let isGenerating = false;
let abortCtrl    = null;
let recognition  = null;
let selectedModel = 'aarav-2.5';

// --- Helpers ---
const isMobile = () => window.innerWidth <= 768;
const getUserName = () => localStorage.getItem('mythic_name') || '';
const setUserName = n => n ? localStorage.setItem('mythic_name', n) : localStorage.removeItem('mythic_name');

// --- Scroll ---
msgWrap.addEventListener('scroll', () => {
  const near = msgWrap.scrollHeight - msgWrap.scrollTop - msgWrap.clientHeight < 120;
  scrollBtn.classList.toggle('show', !near);
});
scrollBtn.addEventListener('click', () => msgWrap.scrollTo({top:msgWrap.scrollHeight, behavior:'smooth'}));
function scrollToBottom() {
  requestAnimationFrame(() => msgWrap.scrollTo({top:msgWrap.scrollHeight, behavior:'smooth'}));
}

// --- Empty state ---
function clearEmpty() { const e = $('empty-state'); if(e) e.remove(); }
function showEmpty() {
  msgList.innerHTML = '<div class="empty-state" id="empty-state"><h2>Mythic AI</h2><p>Ask me anything, generate images, or just chat 👋</p></div>';
}

// --- Add message ---
function addMsg(role, text, attachment) {
  clearEmpty();
  const row = document.createElement('div');
  row.className = 'msg-row ' + role;
  const bubble = document.createElement('div');
  bubble.className = 'msg ' + role;
  if (attachment) {
    const chip = document.createElement('div');
    chip.className = 'attach-chip';
    chip.textContent = '📎 ' + attachment.name;
    bubble.appendChild(chip);
    if (attachment.mimeType && attachment.mimeType.startsWith('image/') && attachment.dataBase64) {
      const img = document.createElement('img');
      img.src = 'data:' + attachment.mimeType + ';base64,' + attachment.dataBase64;
      bubble.appendChild(img);
    }
  }
  const textDiv = document.createElement('div');
  textDiv.className = 'msg-text';
  textDiv.textContent = text;
  bubble.appendChild(textDiv);
  row.appendChild(bubble);
  if (role === 'user' || role === 'ai') {
    const acts = document.createElement('div');
    acts.className = 'msg-actions';
    const cp = document.createElement('button');
    cp.type = 'button'; cp.textContent = '📋'; cp.title = 'Copy';
    cp.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(textDiv.textContent); }
      catch { const t=document.createElement('textarea');t.value=textDiv.textContent;
        document.body.appendChild(t);t.select();document.execCommand('copy');t.remove(); }
      const orig = cp.textContent; cp.textContent = '✓';
      setTimeout(() => cp.textContent = orig, 1200);
    });
    acts.appendChild(cp);
    if (role === 'ai') {
      const rg = document.createElement('button');
      rg.type = 'button'; rg.textContent = '↻'; rg.title = 'Regenerate';
      rg.addEventListener('click', () => { row.remove(); streamReply({regenerate:true}); });
      acts.appendChild(rg);
    }
    row.appendChild(acts);
  }
  msgList.appendChild(row);
  scrollToBottom();
  return textDiv;
}

// --- Typing indicator ---
function showTyping() {
  const d = document.createElement('div');
  d.className = 'typing'; d.id = 'typing-dot';
  d.innerHTML = '<span></span><span></span><span></span>';
  msgList.appendChild(d); scrollToBottom();
}
function hideTyping() { const e = $('typing-dot'); if(e) e.remove(); }

// --- Sidebar ---
function openSidebar() {
  sidebar.classList.remove('hidden');
  if (isMobile()) sbOverlay.style.display = 'block';
}
function closeSidebar() {
  sidebar.classList.add('hidden');
  sbOverlay.style.display = 'none';
}
sbToggle.addEventListener('click', () => sidebar.classList.contains('hidden') ? openSidebar() : closeSidebar());
sbOverlay.addEventListener('click', closeSidebar);

// Swipe to close sidebar on mobile
let touchX = null;
sidebar.addEventListener('touchstart', e => { touchX = e.touches[0].clientX; }, {passive:true});
sidebar.addEventListener('touchend', e => {
  if (touchX !== null && e.changedTouches[0].clientX - touchX < -60) closeSidebar();
  touchX = null;
}, {passive:true});

// --- Fullscreen ---
const fsSupported = !!(document.documentElement.requestFullscreen || document.documentElement.webkitRequestFullscreen);
function updateFsBtn() {
  const on = !!(document.fullscreenElement || document.webkitFullscreenElement);
  fsIcon.textContent = on ? '⤢' : '⛶';
  mmFs.textContent = on ? '⤢ Exit fullscreen' : '⛶ Fullscreen';
  fsBtn.classList.toggle('on', on);
}
async function toggleFs() {
  try {
    const el = document.documentElement;
    if (!document.fullscreenElement && !document.webkitFullscreenElement) {
      if (el.requestFullscreen) await el.requestFullscreen();
      else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen();
    } else {
      if (document.exitFullscreen) await document.exitFullscreen();
      else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
    }
  } catch(e) { console.warn('Fullscreen:', e); }
}
fsBtn.addEventListener('click', toggleFs);
document.addEventListener('fullscreenchange', updateFsBtn);
document.addEventListener('webkitfullscreenchange', updateFsBtn);

// --- More menu (mobile) ---
function openMore() { moreMenu.classList.add('open'); moreOverlay.classList.add('open'); }
function closeMore() { moreMenu.classList.remove('open'); moreOverlay.classList.remove('open'); }
moreBtn.addEventListener('click', openMore);
moreOverlay.addEventListener('click', closeMore);
mmFs.addEventListener('click', () => { closeMore(); toggleFs(); });
mmName.addEventListener('click', () => { closeMore(); openNameModal(); });
mmExport.addEventListener('click', () => { closeMore(); doExport(); });
mmClear.addEventListener('click', () => { closeMore(); doClear(); });

// --- Name modal ---
function openNameModal() {
  nameInput.value = getUserName();
  nameOverlay.classList.add('show');
  setTimeout(() => nameInput.focus(), 50);
}
function closeNameModal() { nameOverlay.classList.remove('show'); }
nameBtn.addEventListener('click', openNameModal);
nameCancel.addEventListener('click', closeNameModal);
nameOverlay.addEventListener('click', e => { if(e.target===nameOverlay) closeNameModal(); });
nameSave.addEventListener('click', () => { setUserName(nameInput.value.trim()); closeNameModal(); });
nameInput.addEventListener('keydown', e => {
  if(e.key==='Enter'){e.preventDefault();nameSave.click();}
  else if(e.key==='Escape') closeNameModal();
});
if (!localStorage.getItem('mythic_prompted')) {
  localStorage.setItem('mythic_prompted','1');
  setTimeout(openNameModal, 800);
}

// --- Model selector ---
modelSelect.addEventListener('change', () => { selectedModel = modelSelect.value; });

// --- VIP unlock ---
function showVipModal() {
  // Remove any existing modal first
  const existing = document.getElementById('vip-modal-wrap');
  if (existing) existing.remove();

  const m = document.createElement('div');
  m.id = 'vip-modal-wrap';
  m.style.cssText='position:fixed;inset:0;z-index:300;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;';
  document.body.appendChild(m);

  const box = document.createElement('div');
  box.style.cssText='background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:24px;width:300px;max-width:90vw;';
  box.innerHTML = `
    <div style="font-size:22px;margin-bottom:8px;">🔒 VIP Access</div>
    <div style="color:var(--muted);font-size:13px;margin-bottom:14px;">Enter your VIP password to unlock Mythic Ultra.</div>
    <input type="password" placeholder="VIP password" style="width:100%;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:8px;padding:10px;font-size:14px;outline:none;margin-bottom:8px;box-sizing:border-box;">
    <div class="vip-err" style="color:#ef4444;font-size:12px;margin-bottom:8px;display:none;">Wrong password. Try again.</div>
    <div style="display:flex;gap:8px;">
      <button class="vip-ok" style="flex:1;background:var(--accent);color:#fff;border:none;border-radius:8px;padding:10px;font-size:14px;font-weight:600;cursor:pointer;">Unlock</button>
      <button class="vip-no" style="flex:1;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:10px;font-size:14px;cursor:pointer;">Cancel</button>
    </div>`;
  m.appendChild(box);

  const pw  = box.querySelector('input');
  const err = box.querySelector('.vip-err');
  const okBtn = box.querySelector('.vip-ok');
  const noBtn = box.querySelector('.vip-no');

  setTimeout(() => pw.focus(), 50);

  noBtn.addEventListener('click', () => { m.remove(); modelSelect.value = selectedModel; });
  m.addEventListener('click', e => { if (e.target === m) { m.remove(); modelSelect.value = selectedModel; } });

  okBtn.addEventListener('click', async () => {
    okBtn.textContent = 'Checking...'; okBtn.disabled = true;
    try {
      const r = await fetch('/api/vip-unlock', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({password: pw.value.trim()})
      });
      const d = await r.json();
      if (d.success) {
        selectedModel = 'aarav-ultra';
        modelSelect.value = 'aarav-ultra';
        m.remove();
      } else {
        err.style.display = 'block';
        pw.value = ''; pw.focus();
        okBtn.textContent = 'Unlock'; okBtn.disabled = false;
      }
    } catch(e) {
      err.textContent = 'Network error. Try again.';
      err.style.display = 'block';
      okBtn.textContent = 'Unlock'; okBtn.disabled = false;
    }
  });
  pw.addEventListener('keydown', e => { if (e.key === 'Enter') okBtn.click(); });
}

modelSelect.addEventListener('change', () => {
  if (modelSelect.value === 'aarav-ultra') {
    fetch('/api/vip-status').then(r => r.json()).then(d => {
      if (!d.vip) showVipModal();
      else selectedModel = 'aarav-ultra';
    }).catch(() => showVipModal());
  } else {
    selectedModel = modelSelect.value;
  }
});

// --- File attach ---
function handleFile(file) {
  if(!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    const b64 = e.target.result.split(',')[1];
    pendingFile = {name:file.name, mimeType:file.type||'application/octet-stream', dataBase64:b64};
    attachName.textContent = file.name;
    pendingDiv.classList.add('show');
  };
  reader.readAsDataURL(file);
}
attachBtn.addEventListener('click', () => fileInput.click());
cameraBtn.addEventListener('click', () => cameraInput.click());
fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));
cameraInput.addEventListener('change', () => handleFile(cameraInput.files[0]));
attachRm.addEventListener('click', () => {
  pendingFile=null; fileInput.value=''; cameraInput.value='';
  pendingDiv.classList.remove('show');
});

// --- Voice input ---
(function setupVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!SR){voiceBtn.title='Not supported in this browser';return;}
  recognition = new SR();
  recognition.continuous = false; recognition.interimResults = true; recognition.lang = 'en-US';
  recognition.onstart = () => voiceBtn.classList.add('listening');
  recognition.onresult = e => {
    let final = '';
    for(let i=e.resultIndex;i<e.results.length;i++){
      if(e.results[i].isFinal) final += e.results[i][0].transcript;
      else msgInput.value = e.results[i][0].transcript;
    }
    if(final) msgInput.value = final;
  };
  recognition.onend = () => { voiceBtn.classList.remove('listening'); if(msgInput.value.trim()) form.requestSubmit(); };
  recognition.onerror = () => voiceBtn.classList.remove('listening');
})();
voiceBtn.addEventListener('click', () => {
  if(!recognition){alert('Voice input requires Chrome.');return;}
  if(voiceBtn.classList.contains('listening')) recognition.stop();
  else recognition.start();
});

// --- Conversations ---
async function loadConvList() {
  try {
    const r = await fetch('/api/conversations');
    const d = await r.json();
    const convs = d.conversations || [];
    convList.innerHTML = '';
    convs.forEach(c => {
      const item = document.createElement('div');
      item.className = 'conv-item' + (c.id === activeConvId ? ' active' : '');
      item.innerHTML = '<span class="ctitle"></span>'
        + '<button class="cbtn crename" title="Rename">✎</button>'
        + '<button class="cbtn cdel" title="Delete">✕</button>';
      item.querySelector('.ctitle').textContent = c.title;
      item.addEventListener('click', e => {
        if(!e.target.classList.contains('cbtn')) openConv(c.id);
      });
      item.querySelector('.crename').addEventListener('click', async e => {
        e.stopPropagation();
        const t = prompt('Rename:', c.title);
        if(!t || !t.trim() || t.trim()===c.title) return;
        await fetch('/api/conversations/'+c.id,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({title:t.trim()})});
        loadConvList();
      });
      item.querySelector('.cdel').addEventListener('click', async e => {
        e.stopPropagation();
        await fetch('/api/conversations/'+c.id,{method:'DELETE'});
        if(c.id===activeConvId) newChat();
        else loadConvList();
      });
      convList.appendChild(item);
    });
    return convs;
  } catch { return []; }
}

async function openConv(id) {
  activeConvId = id;
  try {
    const r = await fetch('/api/conversations/' + id);
    if(!r.ok) return;
    const d = await r.json();
    msgList.innerHTML = '';
    (d.messages||[]).forEach(m => addMsg(m.role, m.text, m.attachment));
    loadConvList();
  } catch {}
  if(isMobile()) closeSidebar();
}

function newChat() {
  activeConvId = null;
  msgList.innerHTML = '';
  showEmpty();
  loadConvList();
  if(isMobile()) closeSidebar();
}

newChatBtn.addEventListener('click', newChat);

// --- Clear / Export ---
async function doClear() {
  if(!activeConvId) return;
  await fetch('/api/conversations/'+activeConvId,{method:'DELETE'});
  newChat();
}
async function doExport() {
  if(!activeConvId){alert('Open a chat first.');return;}
  try {
    const r = await fetch('/api/conversations/'+activeConvId);
    if(!r.ok) return;
    const d = await r.json();
    const lines = ['# ' + (d.title||'Mythic AI chat'), ''];
    (d.messages||[]).forEach(m => {
      lines.push(m.role==='user' ? 'You:' : 'Mythic AI:');
      lines.push(m.text||(m.attachment?'[attachment: '+m.attachment.name+']':''));
      lines.push('');
    });
    const blob = new Blob([lines.join('\n')],{type:'text/plain;charset=utf-8'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href=url; a.download=((d.title||'chat').replace(/[^a-z0-9_ -]/gi,'').trim().slice(0,60)||'chat')+'.txt';
    document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  } catch(e){alert('Export failed: '+e.message);}
}
clearBtn.addEventListener('click', doClear);
exportBtn.addEventListener('click', doExport);

// --- Image generation ---
const IMAGE_RE = /\b(generate|create|draw|make|paint|render|show me|design|sketch|ghibli|anime|realistic|cartoon|portrait|landscape|artwork|image of|picture of|photo of|illustration of|logo)\b/i;

function addImageMsg(b64, mime, caption) {
  clearEmpty();
  const row = document.createElement('div');
  row.className = 'msg-row ai';
  const bubble = document.createElement('div');
  bubble.className = 'msg ai';
  bubble.style.padding = '8px';
  if (caption) {
    const cap = document.createElement('div');
    cap.style.cssText = 'font-size:13px;color:var(--muted);margin-bottom:6px;';
    cap.textContent = '🎨 ' + caption;
    bubble.appendChild(cap);
  }
  const img = document.createElement('img');
  img.src = `data:${mime||'image/jpeg'};base64,${b64}`;
  img.style.cssText = 'max-width:100%;border-radius:10px;display:block;cursor:pointer;';
  img.title = 'Click to download';
  img.addEventListener('click', () => {
    const a = document.createElement('a');
    a.href = img.src; a.download = 'mythic-ai-image.jpg';
    document.body.appendChild(a); a.click(); a.remove();
  });
  bubble.appendChild(img);

  // Download button
  const dl = document.createElement('button');
  dl.style.cssText = 'margin-top:6px;background:none;border:1px solid var(--border);color:var(--muted);border-radius:6px;padding:4px 10px;font-size:12px;cursor:pointer;';
  dl.textContent = '⬇ Download';
  dl.addEventListener('click', () => { img.click(); });
  bubble.appendChild(dl);

  row.appendChild(bubble);
  msgList.appendChild(row);
  scrollToBottom();
}

async function tryImageGen(prompt) {
  // Show a generating indicator
  clearEmpty();
  const genRow = document.createElement('div');
  genRow.className = 'msg-row ai';
  genRow.id = 'img-generating';
  genRow.innerHTML = `<div class="msg ai" style="color:var(--muted);font-size:13px;">🎨 Generating image<span id="img-dots">...</span></div>`;
  msgList.appendChild(genRow);
  scrollToBottom();

  // Animate dots
  let dots = 0;
  const dotsEl = document.getElementById('img-dots');
  const dotsTimer = setInterval(() => {
    dots = (dots + 1) % 4;
    if (dotsEl) dotsEl.textContent = '.'.repeat(dots + 1);
  }, 400);

  try {
    const r = await fetch('/api/generate-image', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt})
    });
    clearInterval(dotsTimer);
    const genEl = document.getElementById('img-generating');
    if (genEl) genEl.remove();

    const d = await r.json();
    if (d.image) {
      addImageMsg(d.image, d.mime || 'image/jpeg', prompt);
      return true;
    } else {
      addMsg('error', 'Image generation failed: ' + (d.error || 'Unknown error'));
      return true;
    }
  } catch(e) {
    clearInterval(dotsTimer);
    const genEl = document.getElementById('img-generating');
    if (genEl) genEl.remove();
    return false;
  }
}

// --- Generate / Stream ---
function setGenerating(on) {
  isGenerating = on;
  sendBtn.classList.toggle('stop', on);
  sendBtn.title = on ? 'Stop' : 'Send';
  sendBtn.innerHTML = on
    ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>'
    : '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>';
}

async function streamReply({message='', attachment=null, regenerate=false}={}) {
  showTyping(); setGenerating(true); abortCtrl = new AbortController();
  let aiNode = null;
  try {
    const r = await fetch('/api/chat', {
      method:'POST', signal:abortCtrl.signal,
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message, conversation_id:activeConvId, attachment,
        user_name:getUserName(), regenerate, model:selectedModel})
    });
    if(!r.ok) {
      hideTyping();
      if(r.status===403){const e=await r.json().catch(()=>({}));if(e.error==='vip_required'){showVipModal();return;}}
      addMsg('error','Something went wrong. Please try again.'); return;
    }
    hideTyping();
    aiNode = addMsg('ai','');
    const convId = r.headers.get('X-Conversation-Id');
    if(convId) activeConvId = convId;
    const reader = r.body.getReader();
    const dec = new TextDecoder('utf-8');
    let full = '';
    while(true) {
      const {done,value} = await reader.read();
      if(done) break;
      full += dec.decode(value,{stream:true});
      aiNode.textContent = full;
      scrollToBottom();
    }
    loadConvList();
  } catch(e) {
    hideTyping();
    if(e.name==='AbortError'){if(aiNode&&!aiNode.textContent.trim()) aiNode.textContent='[Stopped]';}
    else addMsg('error','Network error: '+e.message);
  } finally { setGenerating(false); abortCtrl=null; }
}

// --- Form submit / stop ---
form.addEventListener('submit', async e => {
  e.preventDefault();
  if(isGenerating) return;
  const text = msgInput.value.trim();
  if(!text && !pendingFile) return;
  const att = pendingFile;
  pendingFile=null; fileInput.value=''; cameraInput.value=''; pendingDiv.classList.remove('show');
  addMsg('user', text, att);
  msgInput.value=''; msgInput.style.height='auto';

  // Check if user wants an image (only when no file attached)
  if (!att && IMAGE_RE.test(text)) {
    const generated = await tryImageGen(text);
    if (generated) return; // image generated — don't also send to chat AI
  }

  streamReply({message:text, attachment:att});
});
sendBtn.addEventListener('click', e => {
  if(isGenerating){e.preventDefault();if(abortCtrl) abortCtrl.abort();}
});
msgInput.addEventListener('keydown', e => {
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();form.requestSubmit();}
});
msgInput.addEventListener('input', () => {
  msgInput.style.height='auto';
  msgInput.style.height=Math.min(msgInput.scrollHeight,140)+'px';
});

// --- Initial load ---
if(isMobile()) sidebar.classList.add('hidden');
(async () => {
  const convs = await loadConvList();
  if(convs.length>0) openConv(convs[0].id);
  else showEmpty();
})();
</script>
</body>
</html>
"""

@app.route("/")
@login_required
def index():
    return Response(PAGE, mimetype="text/html; charset=utf-8")


@app.route("/api/conversations", methods=["GET"])
@login_required
def api_list_conversations():
    return jsonify({"conversations": list_conversations(current_username())})


@app.route("/api/conversations/<conv_id>", methods=["GET"])
@login_required
def api_get_conversation(conv_id):
    data = load_conversation(current_username(), conv_id)
    if data is None:
        return jsonify({"error": "not found"}), 404
    simplified = []
    for m in data.get("messages", []):
        role = "user" if m["role"] == "user" else "ai"
        text_parts = [p.get("text", "") for p in m["parts"] if "text" in p]
        entry = {"role": role, "text": "".join(text_parts)}
        if m.get("attachment_meta"):
            entry["attachment"] = m["attachment_meta"]
        simplified.append(entry)
    return jsonify({"messages": simplified, "title": data.get("title", "New chat")})


@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
@login_required
def api_delete_conversation(conv_id):
    delete_conversation(current_username(), conv_id)
    return jsonify({"status": "deleted"})


@app.route("/api/conversations/<conv_id>", methods=["PATCH"])
@login_required
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


def to_ollama_messages(gemini_messages, system_prompt):
    """Convert our stored Gemini-style messages ({role, parts:[...]}) into
    Ollama's chat format ({role, content, images?})."""
    msgs = [{"role": "system", "content": system_prompt}]
    for m in gemini_messages:
        role = "user" if m["role"] == "user" else "assistant"
        text = "".join(p.get("text", "") for p in m["parts"] if "text" in p)
        entry = {"role": role, "content": text}
        images = [
            p["inline_data"]["data"]
            for p in m["parts"]
            if "inline_data" in p and p["inline_data"].get("mime_type", "").startswith("image/")
        ]
        if images:
            entry["images"] = images
        msgs.append(entry)
    return msgs


def ollama_stream_chunks(messages):
    """Yields plain text increments from a local Ollama server."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": True},
            stream=True,
            timeout=120,
        )
    except requests.RequestException as e:
        yield (
            f"[Could not reach Ollama at {OLLAMA_URL}: {e}. "
            f"Make sure Ollama is installed and running (`ollama serve`), "
            f"and that you've pulled the model (`ollama pull {OLLAMA_MODEL}`).]"
        )
        return

    if resp.status_code != 200:
        yield f"[Ollama error ({resp.status_code}): {resp.text}]"
        return

    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if obj.get("error"):
            yield f"[Ollama error: {obj['error']}]"
            return
        content = obj.get("message", {}).get("content", "")
        if content:
            yield content
        if obj.get("done"):
            break


def to_openai_messages(gemini_messages, system_prompt):
    """Convert stored Gemini-format messages to OpenAI-compatible chat format.
    Used by Groq, OpenRouter, and HuggingFace (all use the same OpenAI-style API)."""
    msgs = [{"role": "system", "content": system_prompt}]
    for m in gemini_messages:
        role = "user" if m["role"] == "user" else "assistant"
        text = "".join(p.get("text", "") for p in m["parts"] if "text" in p)
        msgs.append({"role": role, "content": text})
    return msgs


def groq_stream_chunks(messages):
    """Stream from Groq API (OpenAI-compatible, very fast, generous free tier)."""
    if not GROQ_API_KEY:
        return
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": messages, "stream": True, "max_tokens": 2048},
            stream=True, timeout=60,
        )
        if resp.status_code == 200:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                    content = obj["choices"][0]["delta"].get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            return  # success — stop here
        # rate limited or error — fall through (return without yielding)
    except requests.RequestException:
        pass


def openrouter_stream_chunks(messages):
    """Stream from OpenRouter (aggregates many free models)."""
    if not OPENROUTER_API_KEY:
        return
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json",
                     "HTTP-Referer": "http://localhost:5000", "X-Title": "Mythic AI"},
            json={"model": OPENROUTER_MODEL, "messages": messages, "stream": True, "max_tokens": 2048},
            stream=True, timeout=60,
        )
        if resp.status_code == 200:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                    content_chunk = obj["choices"][0]["delta"].get("content", "")
                    if content_chunk:
                        yield content_chunk
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            return
        else:
            yield f"[OpenRouter error {resp.status_code}: {resp.text[:200]}]"
            return
    except requests.RequestException as e:
        yield f"[OpenRouter connection error: {e}]"
        return


def huggingface_stream_chunks(messages):
    """Stream from Hugging Face Inference API (free tier available)."""
    if not HF_API_KEY:
        return
    # HF uses the same OpenAI-compatible endpoint format
    try:
        resp = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"},
            json={"model": HF_MODEL, "messages": messages, "stream": True, "max_tokens": 2048},
            stream=True, timeout=60,
        )
        if resp.status_code == 200:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                    content_chunk = obj["choices"][0]["delta"].get("content", "")
                    if content_chunk:
                        yield content_chunk
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            return
        else:
            yield f"[OpenRouter error {resp.status_code}: {resp.text[:200]}]"
            return
    except requests.RequestException as e:
        yield f"[OpenRouter connection error: {e}]"
        return



def cerebras_stream_chunks(messages):
    """Stream from Cerebras AI (very fast, generous free tier, works on servers)."""
    if not CEREBRAS_API_KEY:
        return
    try:
        resp = requests.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"},
            json={"model": CEREBRAS_MODEL, "messages": messages, "stream": True, "max_tokens": 2048},
            stream=True, timeout=60,
        )
        if resp.status_code == 200:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                    chunk = obj["choices"][0]["delta"].get("content", "")
                    if chunk:
                        yield chunk
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            return
        else:
            yield f"[Cerebras error {resp.status_code}: {resp.text[:300]}]"
            return
    except requests.RequestException as e:
        yield f"[Cerebras connection error: {e}]"
        return


def groq_stream_chunks_with_model(messages, model):
    """Stream from Groq with a specific model."""
    if not GROQ_API_KEY:
        yield "[Groq API key not configured. Set GROQ_API_KEY environment variable.]"; return
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "stream": True, "max_tokens": 2048},
            stream=True, timeout=60,
        )
        if resp.status_code == 200:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"): continue
                d = line[5:].strip()
                if d == "[DONE]": break
                try:
                    c = json.loads(d)["choices"][0]["delta"].get("content", "")
                    if c: yield c
                except: continue
            return
        yield f"[Groq error {resp.status_code}: {resp.text[:200]}]"
    except requests.RequestException as e:
        yield f"[Groq connection error: {e}]"


def cerebras_stream_chunks_with_model(messages, model):
    """Stream from Cerebras with a specific model."""
    if not CEREBRAS_API_KEY:
        yield "[Cerebras API key not configured. Set CEREBRAS_API_KEY environment variable.]"; return
    try:
        resp = requests.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "stream": True, "max_tokens": 2048},
            stream=True, timeout=60,
        )
        if resp.status_code == 200:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"): continue
                d = line[5:].strip()
                if d == "[DONE]": break
                try:
                    c = json.loads(d)["choices"][0]["delta"].get("content", "")
                    if c: yield c
                except: continue
            return
        yield f"[Cerebras error {resp.status_code}: {resp.text[:200]}]"
    except requests.RequestException as e:
        yield f"[Cerebras connection error: {e}]"


# --- Round-robin provider rotation ------------------------------------------
# Tracks which provider index to try FIRST next time (persists for the life
# of the process; resets on server restart, which is fine).
_provider_index = [0]


def auto_stream_chunks(gemini_payload, gemini_messages, system_prompt=None):
    """True round-robin rotation across all configured providers.
    No provider is primary — starts from wherever the last successful call left off.
    If the current provider fails/is rate-limited, moves to the next one and
    remembers that position for the next request too."""
    sp = system_prompt or SYSTEM_PROMPT
    openai_msgs = to_openai_messages(gemini_messages, sp)
    ollama_msgs = to_ollama_messages(gemini_messages, sp)

    # Build the full list of available providers (only those with keys)
    all_providers = []
    if PROVIDER in ("auto", "gemini") and GEMINI_API_KEY:
        all_providers.append(("Gemini", lambda: gemini_stream_chunks(gemini_payload)))
    if PROVIDER in ("auto", "groq") and GROQ_API_KEY:
        all_providers.append(("Groq", lambda: groq_stream_chunks(openai_msgs)))
    if PROVIDER in ("auto", "cerebras") and CEREBRAS_API_KEY:
        all_providers.append(("Cerebras", lambda: cerebras_stream_chunks(openai_msgs)))
    if PROVIDER == "openrouter" and OPENROUTER_API_KEY:
        all_providers.append(("OpenRouter", lambda: openrouter_stream_chunks(openai_msgs)))
    if PROVIDER == "huggingface" and HF_API_KEY:
        # HuggingFace free tier blocks server IPs (Render etc) — only use if explicitly set
        all_providers.append(("HuggingFace", lambda: huggingface_stream_chunks(openai_msgs)))
    if PROVIDER == "ollama":
        all_providers.append(("Ollama", lambda: ollama_stream_chunks(ollama_msgs)))

    if not all_providers:
        yield "[No AI providers configured. Add at least one API key.]"
        return

    n = len(all_providers)
    start = _provider_index[0] % n

    # Try each provider starting from the current rotation position
    for i in range(n):
        idx = (start + i) % n
        name, fn = all_providers[idx]
        collected = []
        try:
            for chunk in fn():
                collected.append(chunk)
                yield chunk
            if collected:
                # Success — next request starts from the NEXT provider (true rotation)
                _provider_index[0] = (idx + 1) % n
                return
        except Exception:
            pass
        # This provider failed — silently try the next one

    yield "[All AI providers failed or are rate-limited. Try again in a moment.]"




def gemini_stream_chunks(payload):
    """Yields plain text increments from Gemini's SSE stream.
    Returns nothing (silently) on auth/quota/rate-limit errors so the
    round-robin rotation automatically falls through to the next provider."""
    try:
        resp = requests.post(
            GEMINI_STREAM_URL,
            params={"key": API_KEY, "alt": "sse"},
            json=payload,
            stream=True,
            timeout=60,
        )
    except requests.RequestException:
        return  # network error — silently fall through

    if resp.status_code != 200:
        # Auth errors (401/403), quota errors (429), and server errors (5xx)
        # all mean "try the next provider" — don't yield anything.
        return

    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data:"):
            continue
        data_str = raw_line[len("data:"):].strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            obj = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        try:
            for part in obj["candidates"][0]["content"]["parts"]:
                if "text" in part:
                    yield part["text"]
        except (KeyError, IndexError):
            continue


@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    conv_id = data.get("conversation_id")
    attachment = data.get("attachment")  # {name, mimeType, dataBase64} or None
    user_name = (data.get("user_name") or "").strip()[:60]  # what Mythic AI should call the user
    regenerate = bool(data.get("regenerate"))

    if regenerate:
        if not conv_id:
            return jsonify({"error": "conversation_id is required to regenerate"}), 400
    elif not user_message and not attachment:
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
        # Drop the most recent assistant reply (if any) so a fresh one replaces it.
        # Leaves the preceding user message in place to regenerate against.
        if messages and messages[-1]["role"] == "model":
            messages.pop()
        if not messages or messages[-1]["role"] != "user":
            return jsonify({"error": "nothing to regenerate"}), 400
    else:
        user_parts = []
        if user_message:
            user_parts.append({"text": user_message})
        attachment_meta = None
        if attachment:
            mime_type = attachment.get("mimeType", "application/octet-stream")
            user_parts.append({
                "inline_data": {"mime_type": mime_type, "data": attachment["dataBase64"]}
            })
            attachment_meta = {"name": attachment.get("name", "file"), "mimeType": mime_type}

        user_entry = {"role": "user", "parts": user_parts}
        if attachment_meta:
            user_entry["attachment_meta"] = attachment_meta
        messages.append(user_entry)

    # Strip attachment_meta (frontend-only field) before sending to the model
    gemini_contents = [
        {"role": m["role"], "parts": m["parts"]} for m in messages
    ]

    effective_system_prompt = SYSTEM_PROMPT
    if user_name:
        effective_system_prompt += (
            f" The user has told you their preferred name is \"{user_name}\". "
            f"Address them as {user_name} naturally where it fits."
        )

    # Inject live weather/location data if the message needs it
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr or '').split(',')[0].strip()
    ctx = build_context_injection(user_message, client_ip)
    if ctx:
        effective_system_prompt = ctx + " " + effective_system_prompt

    openai_msgs = to_openai_messages(messages, effective_system_prompt)

    # Map the frontend Aarav model name → provider + real model string
    AARAV_MAP = {
        "aarav-1.0":    ("groq",      "llama-3.1-8b-instant"),
        "aarav-2.0":    ("groq",      "llama-3.3-70b-versatile"),
        "aarav-2.5":    ("cerebras",  "llama-3.3-70b"),
        "aarav-3.5":    ("cerebras",  "llama-4-scout-17b-16e-instruct"),
        "aarav-ultra":  ("cerebras",  "qwen-3-32b"),
    }
    aarav_id = (data.get("model") or "aarav-2.5").strip()

    # VIP gate
    if aarav_id == "aarav-ultra" and not session.get("vip"):
        return jsonify({"error": "vip_required"}), 403

    provider, model_name = AARAV_MAP.get(aarav_id, ("groq", "llama-3.1-8b-instant"))

    def generate():
        full_reply = []
        if provider == "groq":
            chunk_source = groq_stream_chunks_with_model(openai_msgs, model_name)
        elif provider == "cerebras":
            chunk_source = cerebras_stream_chunks_with_model(openai_msgs, model_name)
        elif PROVIDER == "ollama":
            chunk_source = ollama_stream_chunks(to_ollama_messages(messages, effective_system_prompt))
        else:
            payload = {"contents": [{"role": m["role"], "parts": m["parts"]} for m in messages],
                       "systemInstruction": {"parts": [{"text": effective_system_prompt}]}}
            chunk_source = auto_stream_chunks(payload, messages, effective_system_prompt)

        for chunk in chunk_source:
            full_reply.append(chunk)
            yield chunk
        messages.append({"role": "model", "parts": [{"text": "".join(full_reply)}]})
        save_conversation(username, conv_id, conv)

    resp = Response(stream_with_context(generate()), mimetype="text/plain; charset=utf-8")
    resp.headers["X-Conversation-Id"] = conv_id
    return resp


@app.route("/api/generate-image", methods=["POST"])
@login_required
def generate_image():
    data = request.get_json(force=True) or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt required"}), 400
    try:
        import urllib.parse
        encoded = urllib.parse.quote(prompt)
        # Pollinations.ai — free, no API key, no sign-up needed
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=768&height=768&nologo=true&enhance=true"
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
            img_b64 = base64.b64encode(resp.content).decode("utf-8")
            return jsonify({"image": img_b64, "mime": resp.headers["content-type"]})
        return jsonify({"error": f"Image generation failed ({resp.status_code})"}), 502
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502


if __name__ == "__main__":
    active = []
    if PROVIDER in ("auto", "gemini") and GEMINI_API_KEY:
        active.append(f"Gemini({GEMINI_MODEL})")
    if PROVIDER in ("auto", "groq") and GROQ_API_KEY:
        active.append(f"Groq({GROQ_MODEL})")
    if PROVIDER in ("auto", "cerebras") and CEREBRAS_API_KEY:
        active.append(f"Cerebras({CEREBRAS_MODEL})")
    if PROVIDER in ("auto", "openrouter") and OPENROUTER_API_KEY:
        active.append(f"OpenRouter({OPENROUTER_MODEL})")
    if PROVIDER in ("auto", "huggingface") and HF_API_KEY:
        active.append(f"HuggingFace({HF_MODEL})")
    if PROVIDER == "ollama":
        active.append(f"Ollama({OLLAMA_MODEL}@{OLLAMA_URL})")
    providers_str = " → ".join(active) if active else "none configured!"
    print(f"Starting Mythic AI at http://localhost:5000")
    print(f"Providers (fallback order): {providers_str}")
    app.run(host="0.0.0.0", port=5000, debug=False)
