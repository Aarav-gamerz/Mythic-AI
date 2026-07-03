"""
Aarav AI — single file, powered by Google's Gemini API or a local Ollama model.

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
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY",      "")
CEREBRAS_API_KEY  = os.environ.get("CEREBRAS_API_KEY",  "")
OPENROUTER_API_KEY= os.environ.get("OPENROUTER_API_KEY","")
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
    "You are Aarav AI, a smart and friendly AI assistant made by Aarav Singh. "
    "If asked who made you, say you are Aarav AI made by Aarav Singh — say it once naturally, never repeat it unprompted. "
    "Never mention Google, Groq, OpenRouter, HuggingFace, Meta, Mistral, Anthropic, or any AI company as your creator or backend. "
    "You can help with anything: questions, writing, coding, math, ideas, or just chatting. "
    "When writing code, always wrap it in markdown code blocks with the language name. "
    "LANGUAGE: Always reply ENTIRELY in the same language the user's message is written in — "
    "never mix two languages in a single reply. If they write in Hindi, reply fully in Hindi. "
    "If they write in English, reply fully in English (do not slip into Hindi or any other language "
    "partway through, even if source information you know is in a different language — translate it "
    "into the reply language first). If they mix languages themselves, match their mix. "
    "Never force English on the user. "
    "TOOL USE: Never write out fake tool calls, function names, or JSON like {\"query\": ...} in your reply — "
    "those are internal mechanisms the user must never see. If you don't actually have live web access, "
    "just answer from what you know and say your information may not be fully up to date, instead of "
    "pretending to search. "
    "ANTI-REPETITION RULES — follow strictly every reply: "
    "1. NEVER restate or echo back what the user just said. Jump straight to the answer. "
    "2. NEVER start replies with filler like Great question, Sure, Of course, Absolutely, Certainly. "
    "3. NEVER repeat information already given earlier in the conversation. Build on it. "
    "4. Be direct and natural — like a knowledgeable friend, not a customer service bot. "
    "5. Keep answers concise unless the user asks for detail."
)

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
<title>Aarav AI</title>
<style>
  :root {
    --bg:#1a1a1a; --panel:#2a2a2a; --border:#3a3a3a;
    --text:#ececec; --muted:#8e8ea0; --accent:#10a37f;
    --accent-dim:#1a3a30; --user-bubble:#2a2a2a; --user-text:#ececec;
    --ai-bubble:#1a1a1a; --sidebar-w:260px;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  html,body { height:100%; background:var(--bg); color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif; overflow:hidden; }
  .layout { display:flex; height:100vh; }

  /* Sidebar */
  #sidebar { width:var(--sidebar-w); flex-shrink:0; background:var(--panel);
    border-right:1px solid var(--border); display:flex; flex-direction:column;
    transition:margin-left .2s ease; }
  #sidebar.hidden { margin-left:calc(-1 * var(--sidebar-w)); }
  #new-chat-btn { margin:12px; padding:10px 14px; background:var(--accent); color:#fff;
    border:none; border-radius:8px; font-size:13.5px; font-weight:600; cursor:pointer; text-align:left; }
  #new-chat-btn:hover { opacity:.9; }
  #conv-list { flex:1; overflow-y:auto; padding:0 8px; display:flex; flex-direction:column; gap:2px; }
  .conv-item { display:flex; align-items:center; justify-content:space-between; gap:6px;
    padding:9px 10px; border-radius:7px; cursor:pointer; font-size:13px; color:var(--muted); }
  .conv-item:hover { background:var(--accent-dim); color:var(--text); }
  .conv-item.active { background:var(--accent-dim); color:var(--accent); font-weight:500; }
  .conv-item .title { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1; }
  .conv-item .rename-btn { opacity:0; background:none; border:none; color:var(--muted);
    cursor:pointer; font-size:12px; padding:2px 5px; flex-shrink:0; touch-action:manipulation; }
  .conv-item .del-btn { opacity:0; background:none; border:none; color:var(--muted);
    cursor:pointer; font-size:13px; padding:2px 5px; flex-shrink:0; touch-action:manipulation; }
  .conv-item:hover .rename-btn { opacity:1; }
  .conv-item:hover .del-btn { opacity:1; }
  .conv-item .rename-btn:hover { color:var(--accent); }
  .conv-item .del-btn:hover { color:#ef4444; }
  #sidebar-footer { padding:12px; font-size:11px; color:var(--muted); border-top:1px solid var(--border); }

  /* Main */
  .app { display:flex; flex-direction:column; height:100vh; flex:1; min-width:0; }
  header { padding:calc(14px + env(safe-area-inset-top)) 20px 14px; border-bottom:1px solid var(--border);
    display:flex; align-items:center; justify-content:space-between; gap:10px;
    background:var(--bg); position:relative; z-index:20; }
  header .left { display:flex; align-items:center; gap:10px; min-width:0; }
  header .right { display:flex; align-items:center; gap:8px; flex-shrink:0; }
  header button { touch-action:manipulation; -webkit-tap-highlight-color:transparent; }
  #sidebar-toggle { background:none; border:1px solid var(--border); color:var(--muted);
    width:36px; height:36px; border-radius:6px; cursor:pointer; font-size:15px; flex-shrink:0; }
  #sidebar-toggle:hover { background:var(--panel); }
  header h1 { font-size:16px; font-weight:700; color:var(--accent); margin:0; }
  #name-btn { background:none; border:1px solid var(--border); color:var(--muted);
    width:36px; height:36px; border-radius:6px; cursor:pointer; font-size:15px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; touch-action:manipulation; }
  #name-btn:hover { background:var(--panel); }
  #export-btn { background:none; border:1px solid var(--border); color:var(--muted);
    width:36px; height:36px; border-radius:6px; cursor:pointer; font-size:15px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; touch-action:manipulation; }
  #export-btn:hover { background:var(--panel); }

  /* Mobile "more" menu — hidden on desktop, replaces the individual header
     icon buttons on narrow screens to keep the header uncluttered and
     each remaining tap target comfortably large. */
  #more-btn { display:none; background:none; border:1px solid var(--border); color:var(--muted);
    width:38px; height:38px; border-radius:6px; font-size:18px; cursor:pointer; flex-shrink:0;
    align-items:center; justify-content:center; touch-action:manipulation;
    -webkit-tap-highlight-color:transparent; }
  #more-menu-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.5); z-index:149; }
  #more-menu { display:none; position:fixed; top:calc(52px + env(safe-area-inset-top)); right:12px;
    background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:6px;
    flex-direction:column; gap:2px; z-index:150; box-shadow:0 10px 30px rgba(0,0,0,.4); min-width:190px; }
  #more-menu button { display:flex; align-items:center; gap:10px; background:none; border:none;
    color:var(--text); font-size:14px; padding:12px 14px; border-radius:8px; cursor:pointer;
    text-align:left; width:100%; touch-action:manipulation; -webkit-tap-highlight-color:transparent; }
  #more-menu button:active { background:var(--accent-dim); }
  #more-menu.show, #more-menu-overlay.show { display:flex; }

  /* Fullscreen toggle — lives in the header so it's reachable with one tap
     at the top of the screen, alongside the other header icon buttons. */
  #fullscreen-btn { background:none; border:1px solid var(--border); color:var(--muted);
    width:36px; height:36px; border-radius:6px; cursor:pointer; font-size:15px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; touch-action:manipulation;
    -webkit-tap-highlight-color:transparent; }
  #fullscreen-btn:hover { background:var(--panel); color:var(--text); }
  #fullscreen-btn.active { color:var(--accent); border-color:var(--accent); }
  #fullscreen-icon { font-size:16px; line-height:1; }

  /* Fallback for browsers without a real Fullscreen API (iOS Safari, some in-app webviews):
     hide the sidebar toggle/header chrome so the chat fills the screen. */
  body.pseudo-fullscreen #sidebar-toggle,
  body.pseudo-fullscreen header .left h1 { display:none; }
  body.pseudo-fullscreen header { padding-top:calc(6px + env(safe-area-inset-top)); padding-bottom:6px; }

  /* Name modal */
  #name-modal-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.55);
    z-index:200; align-items:center; justify-content:center; }
  #name-modal-overlay.show { display:flex; }
  #name-modal { background:var(--bg); border:1px solid var(--border); border-radius:14px;
    padding:22px; width:90%; max-width:360px; box-shadow:0 10px 40px rgba(0,0,0,.3); }
  #name-modal h3 { margin:0 0 6px; font-size:16px; color:var(--text); }
  #name-modal p { margin:0 0 14px; font-size:12.5px; color:var(--muted); }
  #name-input { width:100%; box-sizing:border-box; padding:10px 12px; border-radius:8px;
    border:1.5px solid var(--border); background:var(--panel); color:var(--text);
    font-size:14.5px; outline:none; }
  #name-input:focus { border-color:var(--accent); }
  #name-modal-actions { display:flex; justify-content:flex-end; gap:8px; margin-top:16px; }
  #name-modal-actions button { padding:8px 14px; border-radius:8px; font-size:13px;
    cursor:pointer; border:1px solid var(--border); background:none; color:var(--text); }
  #name-cancel-btn:hover { background:var(--panel); }
  #name-save-btn { background:var(--accent); color:#fff; border-color:var(--accent); }
  #name-save-btn:hover { opacity:.9; }
  #clear-btn { background:none; border:1px solid var(--border); color:var(--muted);
    width:36px; height:36px; border-radius:6px; font-size:15px; cursor:pointer; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; touch-action:manipulation;
    -webkit-tap-highlight-color:transparent; }
  #clear-btn:hover { background:var(--panel); color:#ef4444; border-color:#ef4444; }

  /* Messages */
  #messages-wrap { flex:1; overflow-y:auto; position:relative; }
  #messages { padding:24px 20px; display:flex; flex-direction:column; gap:16px;
    max-width:760px; margin:0 auto; width:100%; min-height:100%; }
  .msg { max-width:80%; padding:11px 15px; border-radius:18px; line-height:1.6;
    font-size:14.5px; white-space:pre-wrap; word-wrap:break-word; }
  .msg.user { align-self:flex-end; background:var(--user-bubble); color:var(--user-text);
    border-bottom-right-radius:4px; }
  .msg.ai { align-self:flex-start; background:var(--ai-bubble); color:var(--text);
    border-bottom-left-radius:4px; }
  .msg.error { align-self:center; background:#fef2f2; border:1px solid #fecaca;
    color:#dc2626; font-size:13px; border-radius:10px; }
  .msg img { max-width:100%; border-radius:10px; display:block; margin-top:8px; }
  .attach-chip { font-size:11.5px; opacity:.75; margin-bottom:4px; }

  /* Message row wraps the bubble + its action buttons (copy / regenerate) */
  .msg-row { display:flex; flex-direction:column; max-width:80%; }
  .msg-row.user { align-self:flex-end; align-items:flex-end; }
  .msg-row.ai { align-self:flex-start; align-items:flex-start; }
  .msg-row.error { align-self:center; align-items:center; max-width:90%; }
  .msg-row .msg { max-width:100%; }
  .msg-actions { display:flex; gap:4px; margin-top:3px; opacity:0; transition:opacity .15s;
    height:22px; }
  .msg-row:hover .msg-actions, .msg-row:focus-within .msg-actions { opacity:1; }
  .msg-actions button { background:none; border:none; color:var(--muted); cursor:pointer;
    font-size:12px; padding:2px 7px; border-radius:5px; touch-action:manipulation;
    -webkit-tap-highlight-color:transparent; }
  .msg-actions button:hover { background:var(--panel); color:var(--text); }
  .empty-state { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
    text-align:center; color:var(--muted); }
  .empty-state h2 { font-size:22px; font-weight:700; color:var(--accent); margin-bottom:8px; }
  .empty-state p { font-size:14px; }
  .typing { align-self:flex-start; display:flex; gap:5px; padding:14px 16px;
    background:var(--ai-bubble); border-radius:18px; border-bottom-left-radius:4px; }
  .typing span { width:7px; height:7px; border-radius:50%; background:var(--muted);
    animation:blink 1.2s infinite ease-in-out; }
  .typing span:nth-child(2) { animation-delay:.2s; }
  .typing span:nth-child(3) { animation-delay:.4s; }
  @keyframes blink { 0%,80%,100%{opacity:.2} 40%{opacity:1} }

  /* Scroll to bottom */
  #scroll-btn { position:fixed; bottom:130px; right:24px; width:36px; height:36px;
    border-radius:50%; background:var(--accent); color:#fff; border:none; cursor:pointer;
    font-size:18px; display:none; align-items:center; justify-content:center;
    box-shadow:0 2px 8px rgba(0,0,0,.15); z-index:10; }
  #scroll-btn.show { display:flex; }

  /* Image preview */
  .gen-img { max-width:320px; border-radius:12px; display:block; margin-top:8px; }

  /* Input area */
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
    touch-action:manipulation; -webkit-tap-highlight-color:transparent; }
  .tool-btn:hover { background:var(--accent-dim); color:var(--accent); }
  .tool-btn.active { color:var(--accent); }
  textarea { flex:1; resize:none; background:transparent; border:none; color:var(--text);
    font-size:14.5px; font-family:inherit; line-height:1.4; max-height:140px;
    outline:none; padding:4px 0; }
  textarea::placeholder { color:var(--muted); }
  #send-btn { background:var(--accent); color:#fff; border:none; border-radius:10px;
    width:36px; height:36px; font-size:18px; cursor:pointer; flex-shrink:0;
    display:flex; align-items:center; justify-content:center;
    touch-action:manipulation; -webkit-tap-highlight-color:transparent; }
  #send-btn:disabled { background:var(--accent-dim); color:var(--muted); cursor:not-allowed; }
  #send-btn.generating { background:#ef4444; }
  #send-btn.generating:hover { opacity:.9; }
  #voice-btn.listening { color:#ef4444; animation:pulse 1s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  /* Speaking indicator */
  #speaking-indicator { display:none; align-items:center; gap:6px; font-size:12px;
    color:var(--accent); padding:4px 0; }
  #speaking-indicator.show { display:flex; }
  #stop-speak-btn { background:none; border:1px solid var(--border); color:var(--muted);
    font-size:11px; padding:2px 8px; border-radius:4px; cursor:pointer; }

  #messages-wrap::-webkit-scrollbar, #conv-list::-webkit-scrollbar { width:6px; }
  #messages-wrap::-webkit-scrollbar-thumb, #conv-list::-webkit-scrollbar-thumb
    { background:var(--border); border-radius:4px; }
  #sidebar-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.55);
    z-index:99; -webkit-tap-highlight-color:transparent; }

  @media(max-width:768px) {
    :root { --sidebar-w: 78vw; }

    html, body { overscroll-behavior-y: contain; }
    #messages-wrap { overscroll-behavior-y: contain; }

    /* Sidebar slides in as overlay — never pushes content */
    #sidebar { position:fixed; top:0; left:0; z-index:100; height:100%;
      height:-webkit-fill-available; width:var(--sidebar-w) !important;
      transform:translateX(0); transition:transform .25s ease;
      box-shadow:4px 0 24px rgba(0,0,0,.5); }
    #sidebar.hidden { transform:translateX(-105%); margin-left:0 !important; }
    #sidebar-footer { padding-bottom:max(10px, env(safe-area-inset-bottom)); }

    /* Show overlay when sidebar open */
    #sidebar-overlay { display:block; }

    /* Main app always takes full width */
    .app { width:100% !important; flex:1; }

    header { padding:calc(10px + env(safe-area-inset-top)) 10px 10px; }
    header h1 { font-size:14px; }
    #sidebar-toggle { width:40px; height:40px; font-size:15px; }

    /* Collapse the individual header icons into a single "more" menu on mobile */
    #header-actions { display:none; }
    #more-btn { display:flex; }

    #messages-wrap { overflow-y:auto; -webkit-overflow-scrolling:touch; }
    #messages { padding:14px 10px; gap:12px; max-width:100%; }
    .msg { max-width:90%; font-size:14px; padding:10px 12px; }
    .msg-row { max-width:90%; }
    .msg-actions { opacity:1; height:28px; } /* no hover on touch — keep always visible */
    .msg-actions button { font-size:14px; padding:5px 10px; min-width:36px; min-height:32px; }

    .input-area { padding:8px 10px max(10px,env(safe-area-inset-bottom)); }
    .input-row { padding:6px 8px; }
    textarea { font-size:16px; } /* 16px prevents iOS zoom */
    .tool-btn { width:40px; height:40px; font-size:18px; }
    #send-btn { width:40px; height:40px; font-size:17px; }

    .empty-state h2 { font-size:19px; }
    .empty-state p { font-size:13px; }
    #scroll-btn { bottom:calc(80px + env(safe-area-inset-bottom)); right:12px; width:38px; height:38px; }

    #new-chat-btn { margin:10px; padding:12px; font-size:14px; min-height:44px; }
    .conv-item { padding:10px 8px; font-size:13px; min-height:44px; }
    .conv-item .rename-btn { opacity:1; min-width:32px; min-height:32px; }
    .conv-item .del-btn { opacity:1; min-width:32px; min-height:32px; }
    #sidebar-footer { font-size:11px; padding:10px 12px; }
  }

  @media(max-width:380px) {
    :root { --sidebar-w: 88vw; }
    .msg { font-size:13.5px; }
    header h1 { font-size:13px; }
  }
</style>
</head>
<body>
<div class="layout">
  <div id="sidebar-overlay" style="display:none;position:fixed;inset:0;background:#0007;z-index:99" id="sidebar-overlay"></div>
  <div id="sidebar">
    <button id="new-chat-btn">+ New chat</button>
    <div id="conv-list"></div>
    <div id="sidebar-footer">Aarav AI &middot; by Aarav Singh</div>
  </div>
  <div class="app">
    <header>
      <div class="left">
        <button id="sidebar-toggle" title="Toggle sidebar">☰</button>
        <h1>Aarav AI</h1>
      </div>
      <div class="right" id="header-actions">
        <button id="fullscreen-btn" type="button" title="Fullscreen">
          <span id="fullscreen-icon">⛶</span>
        </button>
        <button id="name-btn" title="What should Aarav AI call you?">🙂</button>
        <button id="export-btn" title="Export this chat">⬇</button>
        <button id="clear-btn" title="Delete this chat">🗑</button>
      </div>
      <button id="more-btn" type="button" title="More options">⋮</button>
    </header>
    <div id="more-menu-overlay"></div>
    <div id="more-menu">
      <button id="more-fullscreen" type="button"><span id="more-fullscreen-icon">⛶</span> Fullscreen</button>
      <button id="more-name" type="button">🙂 Your name</button>
      <button id="more-export" type="button">⬇ Export chat</button>
      <button id="more-clear" type="button">🗑 Delete chat</button>
    </div>

    <div id="messages-wrap">
      <div id="messages">
        <div class="empty-state" id="empty-state">
          <h2>Aarav AI</h2>
          <p>Ask me anything, generate images, or just chat 👋</p>
        </div>
      </div>
    </div>

    <button id="scroll-btn" title="Scroll to bottom">↓</button>

    <div id="pending-attach">
      📎 <span id="pending-attach-name"></span>
      <button id="pending-attach-remove">✕</button>
    </div>

    <div id="speaking-indicator">
      🔊 Speaking...
      <button id="stop-speak-btn">Stop</button>
    </div>

    <div class="input-area">
      <form id="chat-form">
        <div class="input-row">
          <!-- Attach file -->
          <input type="file" id="file-input" accept="image/*,.txt,.md,.csv,.json,.pdf" style="display:none">
          <button class="tool-btn" id="attach-btn" type="button" title="Attach file">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
            </svg>
          </button>
          <!-- Camera -->
          <input type="file" id="camera-input" accept="image/*" capture="environment" style="display:none">
          <button class="tool-btn" id="camera-btn" type="button" title="Take photo">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
              <circle cx="12" cy="13" r="4"/>
            </svg>
          </button>
          <textarea id="input" rows="1" placeholder="Message Aarav AI..."></textarea>
          <!-- Voice input -->
          <button class="tool-btn" id="voice-btn" type="button" title="Voice input">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
              <line x1="12" y1="19" x2="12" y2="23"/>
              <line x1="8" y1="23" x2="16" y2="23"/>
            </svg>
          </button>
          <button id="send-btn" type="submit" title="Send">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
          </button>
        </div>
      </form>
    </div>
  </div>
</div>

<div id="name-modal-overlay">
  <div id="name-modal">
    <h3>What should Aarav AI call you?</h3>
    <p>Enter your preferred name — Aarav AI will use it when it talks to you.</p>
    <input type="text" id="name-input" maxlength="60" placeholder="e.g. Aarav" autocomplete="off">
    <div id="name-modal-actions">
      <button id="name-cancel-btn" type="button">Cancel</button>
      <button id="name-save-btn" type="button">Save</button>
    </div>
  </div>
</div>

<script>
const messagesWrap = document.getElementById('messages-wrap');
const messagesEl   = document.getElementById('messages');
const form         = document.getElementById('chat-form');
const input        = document.getElementById('input');
const sendBtn      = document.getElementById('send-btn');
const clearBtn     = document.getElementById('clear-btn');
const convListEl   = document.getElementById('conv-list');
const newChatBtn   = document.getElementById('new-chat-btn');
const sidebarToggle= document.getElementById('sidebar-toggle');
const fullscreenBtn= document.getElementById('fullscreen-btn');
const nameBtn       = document.getElementById('name-btn');
const nameModalOverlay = document.getElementById('name-modal-overlay');
const nameInput     = document.getElementById('name-input');
const nameCancelBtn = document.getElementById('name-cancel-btn');
const nameSaveBtn   = document.getElementById('name-save-btn');
const exportBtn     = document.getElementById('export-btn');
const moreBtn        = document.getElementById('more-btn');
const moreMenu       = document.getElementById('more-menu');
const moreMenuOverlay= document.getElementById('more-menu-overlay');
const moreFullscreen = document.getElementById('more-fullscreen');
const moreFullscreenIcon = document.getElementById('more-fullscreen-icon');
const moreName        = document.getElementById('more-name');
const moreExport      = document.getElementById('more-export');
const moreClear       = document.getElementById('more-clear');
const sidebar      = document.getElementById('sidebar');
const fileInput    = document.getElementById('file-input');
const attachBtn    = document.getElementById('attach-btn');
const cameraInput  = document.getElementById('camera-input');
const cameraBtn    = document.getElementById('camera-btn');
const voiceBtn     = document.getElementById('voice-btn');
const pendingAttach= document.getElementById('pending-attach');
const pendingName  = document.getElementById('pending-attach-name');
const pendingRemove= document.getElementById('pending-attach-remove');
const scrollBtn    = document.getElementById('scroll-btn');
const speakingIndicator = document.getElementById('speaking-indicator');
const stopSpeakBtn = document.getElementById('stop-speak-btn');

let activeConvId = null;
let pendingFile  = null;
let recognition  = null;
let currentUtterance = null;

// --- Scroll button ---
messagesWrap.addEventListener('scroll', () => {
  const nearBottom = messagesWrap.scrollHeight - messagesWrap.scrollTop - messagesWrap.clientHeight < 120;
  scrollBtn.classList.toggle('show', !nearBottom);
});
scrollBtn.addEventListener('click', () => {
  messagesWrap.scrollTo({ top: messagesWrap.scrollHeight, behavior: 'smooth' });
});
function scrollToBottom() {
  requestAnimationFrame(() => {
    messagesWrap.scrollTo({ top: messagesWrap.scrollHeight, behavior: 'smooth' });
  });
}

function clearEmptyState() {
  const es = document.getElementById('empty-state');
  if (es) es.remove();
}
function showEmptyState() {
  messagesEl.innerHTML = '<div class="empty-state" id="empty-state"><h2>Aarav AI</h2><p>Ask me anything, generate images, or just chat 👋</p></div>';
}

function addMessage(role, text, attachment) {
  clearEmptyState();
  const row = document.createElement('div');
  row.className = 'msg-row ' + role;

  const div = document.createElement('div');
  div.className = 'msg ' + role;
  if (attachment) {
    const chip = document.createElement('div');
    chip.className = 'attach-chip';
    chip.textContent = '📎 ' + attachment.name;
    div.appendChild(chip);
    if (attachment.mimeType && attachment.mimeType.startsWith('image/') && attachment.dataBase64) {
      const img = document.createElement('img');
      img.src = 'data:' + attachment.mimeType + ';base64,' + attachment.dataBase64;
      div.appendChild(img);
    }
  }
  const textNode = document.createElement('div');
  textNode.className = 'msg-text';
  textNode.textContent = text;
  div.appendChild(textNode);
  row.appendChild(div);

  if (role === 'user' || role === 'ai') {
    row.appendChild(buildMsgActions(row, textNode, role));
  }

  messagesEl.appendChild(row);
  scrollToBottom();
  return textNode;
}

function buildMsgActions(row, textNode, role) {
  const actions = document.createElement('div');
  actions.className = 'msg-actions';

  const copyBtn = document.createElement('button');
  copyBtn.type = 'button';
  copyBtn.className = 'copy-btn';
  copyBtn.title = 'Copy';
  copyBtn.textContent = '📋';
  copyBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(textNode.textContent);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = textNode.textContent;
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy'); } catch {}
      ta.remove();
    }
    const orig = copyBtn.textContent;
    copyBtn.textContent = '✓';
    setTimeout(() => { copyBtn.textContent = orig; }, 1200);
  });
  actions.appendChild(copyBtn);

  if (role === 'ai') {
    const regenBtn = document.createElement('button');
    regenBtn.type = 'button';
    regenBtn.className = 'regen-btn';
    regenBtn.title = 'Regenerate response';
    regenBtn.textContent = '↻';
    regenBtn.addEventListener('click', () => regenerateLast(row));
    actions.appendChild(regenBtn);
  }
  return actions;
}

function addImageMessage(role, base64, caption) {
  clearEmptyState();
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  if (caption) {
    const cap = document.createElement('div');
    cap.textContent = caption;
    cap.style.marginBottom = '8px';
    div.appendChild(cap);
  }
  const img = document.createElement('img');
  img.className = 'gen-img';
  img.src = 'data:image/png;base64,' + base64;
  div.appendChild(img);
  messagesEl.appendChild(div);
  scrollToBottom();
}

function showTyping() {
  const div = document.createElement('div');
  div.className = 'typing'; div.id = 'typing-indicator';
  div.innerHTML = '<span></span><span></span><span></span>';
  messagesEl.appendChild(div);
  scrollToBottom();
}
function hideTyping() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

// --- Text-to-speech ---
function speak(text) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const plain = text.replace(/[#*`_~>]/g, '').trim();
  if (!plain) return;
  currentUtterance = new SpeechSynthesisUtterance(plain);
  currentUtterance.rate = 1.05;
  currentUtterance.onstart = () => speakingIndicator.classList.add('show');
  currentUtterance.onend = () => speakingIndicator.classList.remove('show');
  currentUtterance.onerror = () => speakingIndicator.classList.remove('show');
  window.speechSynthesis.speak(currentUtterance);
}
stopSpeakBtn.addEventListener('click', () => {
  window.speechSynthesis && window.speechSynthesis.cancel();
  speakingIndicator.classList.remove('show');
});

// --- Voice input ---
function setupVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { voiceBtn.title = 'Voice not supported in this browser'; return; }
  recognition = new SR();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = 'en-US';
  let finalTranscript = '';
  recognition.onstart  = () => { voiceBtn.classList.add('active', 'listening'); finalTranscript = ''; };
  recognition.onresult = (e) => {
    finalTranscript = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (e.results[i].isFinal) finalTranscript += e.results[i][0].transcript;
      else input.value = e.results[i][0].transcript;
    }
    if (finalTranscript) input.value = finalTranscript;
  };
  recognition.onend = () => {
    voiceBtn.classList.remove('active', 'listening');
    if (input.value.trim()) form.requestSubmit();
  };
  recognition.onerror = () => voiceBtn.classList.remove('active', 'listening');
}
setupVoice();
voiceBtn.addEventListener('click', () => {
  if (!recognition) { alert('Voice input is not supported in this browser. Try Chrome.'); return; }
  if (voiceBtn.classList.contains('listening')) { recognition.stop(); return; }
  recognition.start();
});

// --- File / camera attach ---
function handleFileSelect(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    const dataUrl = e.target.result;
    const base64  = dataUrl.split(',')[1];
    pendingFile = { name: file.name, mimeType: file.type || 'application/octet-stream', dataBase64: base64 };
    pendingName.textContent = file.name;
    pendingAttach.classList.add('show');
  };
  reader.readAsDataURL(file);
}
attachBtn.addEventListener('click', () => fileInput.click());
cameraBtn.addEventListener('click', () => cameraInput.click());
fileInput.addEventListener('change', () => handleFileSelect(fileInput.files[0]));
cameraInput.addEventListener('change', () => handleFileSelect(cameraInput.files[0]));
pendingRemove.addEventListener('click', () => {
  pendingFile = null;
  fileInput.value = '';
  cameraInput.value = '';
  pendingAttach.classList.remove('show');
});

// --- Image generation detection ---
const IMAGE_KEYWORDS = /\b(generate|create|draw|make|paint|render|show me|ghibli|anime|realistic|cartoon|portrait|landscape|art|artwork|image of|picture of|photo of|illustration)\b/i;
async function tryGenerateImage(prompt) {
  try {
    const r = await fetch('/api/generate-image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    });
    const d = await r.json();
    if (d.image) {
      addImageMessage('ai', d.image, '');
      return true;
    }
  } catch {}
  return false;
}

// --- Conversations ---
async function loadConversationList() {
  try {
    const r = await fetch('/api/conversations');
    const d = await r.json();
    const convs = d.conversations || [];
    convListEl.innerHTML = '';
    convs.forEach(c => {
      const item = document.createElement('div');
      item.className = 'conv-item' + (c.id === activeConvId ? ' active' : '');
      item.innerHTML = '<span class="title"></span>'
        + '<button class="rename-btn" title="Rename">✎</button>'
        + '<button class="del-btn" title="Delete">✕</button>';
      item.querySelector('.title').textContent = c.title;
      item.addEventListener('click', (e) => {
        if (!e.target.classList.contains('del-btn') && !e.target.classList.contains('rename-btn')) openConversation(c.id);
      });
      item.querySelector('.rename-btn').addEventListener('click', async (e) => {
        e.stopPropagation();
        const newTitle = prompt('Rename chat:', c.title);
        if (!newTitle || !newTitle.trim() || newTitle.trim() === c.title) return;
        await fetch('/api/conversations/' + c.id, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: newTitle.trim() })
        });
        loadConversationList();
      });
      item.querySelector('.del-btn').addEventListener('click', async (e) => {
        e.stopPropagation();
        await fetch('/api/conversations/' + c.id, { method: 'DELETE' });
        if (c.id === activeConvId) startNewChat();
        else loadConversationList();
      });
      convListEl.appendChild(item);
    });
    return convs;
  } catch { return []; }
}

async function openConversation(convId) {
  activeConvId = convId;
  try {
    const r = await fetch('/api/conversations/' + convId);
    if (!r.ok) return;
    const d = await r.json();
    messagesEl.innerHTML = '';
    (d.messages || []).forEach(m => addMessage(m.role, m.text, m.attachment));
    loadConversationList();
  } catch {}
  if (isMobile()) closeSidebar();
}

function startNewChat() {
  activeConvId = null;
  messagesEl.innerHTML = '';
  showEmptyState();
  loadConversationList();
  if (isMobile()) closeSidebar();
}

// --- Send / regenerate / stop ---
let isGenerating = false;
let currentAbortController = null;

function setGenerating(state) {
  isGenerating = state;
  sendBtn.classList.toggle('generating', state);
  sendBtn.title = state ? 'Stop generating' : 'Send';
  sendBtn.innerHTML = state
    ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>'
    : '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>';
}

async function streamReply({ message = null, attachment = null, regenerate = false } = {}) {
  showTyping();
  setGenerating(true);
  currentAbortController = new AbortController();

  // Check if user wants an image (only on fresh sends, not regenerate)
  if (!regenerate) {
    const wantsImage = IMAGE_KEYWORDS.test(message || '') && !attachment;
    if (wantsImage) {
      hideTyping();
      const generated = await tryGenerateImage(message);
      if (generated) { setGenerating(false); loadConversationList(); return; }
      showTyping();
    }
  }

  let aiTextNode = null;
  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: currentAbortController.signal,
      body: JSON.stringify({
        message: message || '',
        conversation_id: activeConvId,
        attachment,
        user_name: getUserName(),
        regenerate: !!regenerate,
      })
    });
    if (!r.ok || !r.body) {
      hideTyping();
      addMessage('error', 'Something went wrong. Try again.');
      return;
    }
    hideTyping();
    aiTextNode = addMessage('ai', '');

    const convId = r.headers.get('X-Conversation-Id');
    if (convId) activeConvId = convId;

    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      fullText += chunk;
      aiTextNode.textContent = fullText;
      scrollToBottom();
    }
    speak(fullText);
    loadConversationList();
  } catch (err) {
    hideTyping();
    if (err.name === 'AbortError') {
      // User hit stop — keep whatever text streamed in so far, just mark it as stopped.
      if (aiTextNode && !aiTextNode.textContent.trim()) aiTextNode.textContent = '[Stopped]';
    } else {
      addMessage('error', 'Network error: ' + err.message);
    }
  } finally {
    setGenerating(false);
    currentAbortController = null;
  }
}

function regenerateLast(row) {
  if (isGenerating) return;
  row.remove();
  streamReply({ regenerate: true });
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  if (isGenerating) return;
  const text = input.value.trim();
  if (!text && !pendingFile) return;
  const attachment = pendingFile;
  pendingFile = null;
  fileInput.value = ''; cameraInput.value = '';
  pendingAttach.classList.remove('show');
  addMessage('user', text, attachment);
  input.value = '';
  input.style.height = 'auto';
  streamReply({ message: text, attachment });
});

sendBtn.addEventListener('click', (e) => {
  if (isGenerating) {
    e.preventDefault();
    if (currentAbortController) currentAbortController.abort();
  }
});

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 140) + 'px';
});

const sidebarOverlay = document.getElementById('sidebar-overlay');

function isMobile() { return window.innerWidth <= 768; }

function openSidebar() {
  sidebar.classList.remove('hidden');
  if (isMobile()) sidebarOverlay.style.display = 'block';
}
function closeSidebar() {
  sidebar.classList.add('hidden');
  sidebarOverlay.style.display = 'none';
}
sidebarToggle.addEventListener('click', () => {
  sidebar.classList.contains('hidden') ? openSidebar() : closeSidebar();
});
sidebarOverlay.addEventListener('click', closeSidebar);

// Fullscreen toggle (works on Android/desktop; iOS Safari has no real Fullscreen API,
// so it falls back to a "pseudo-fullscreen" mode that maximizes the app view instead)
const fullscreenIcon  = document.getElementById('fullscreen-icon');
const fsSupported = !!(document.documentElement.requestFullscreen || document.documentElement.webkitRequestFullscreen);

function isFullscreen() {
  return !!(document.fullscreenElement || document.webkitFullscreenElement) ||
    document.body.classList.contains('pseudo-fullscreen');
}
function updateFullscreenBtn() {
  if (isFullscreen()) {
    fullscreenIcon.textContent = '⤢';
    fullscreenBtn.title = 'Exit fullscreen';
    fullscreenBtn.classList.add('active');
  } else {
    fullscreenIcon.textContent = '⛶';
    fullscreenBtn.title = 'Fullscreen';
    fullscreenBtn.classList.remove('active');
  }
}
async function toggleFullscreen() {
  const el = document.documentElement;
  try {
    if (fsSupported) {
      if (!document.fullscreenElement && !document.webkitFullscreenElement) {
        if (el.requestFullscreen) await el.requestFullscreen();
        else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen();
      } else {
        if (document.exitFullscreen) await document.exitFullscreen();
        else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
      }
    } else {
      // iOS Safari / in-app browsers: real Fullscreen API isn't available,
      // so just maximize the app view (hides scroll bounce, fills the screen).
      document.body.classList.toggle('pseudo-fullscreen');
      updateFullscreenBtn();
    }
  } catch (err) {
    console.warn('Fullscreen request failed:', err);
    // Even on failure, fall back to pseudo-fullscreen so the button still does something
    document.body.classList.toggle('pseudo-fullscreen');
    updateFullscreenBtn();
  }
}
fullscreenBtn.addEventListener('click', toggleFullscreen);
document.addEventListener('fullscreenchange', updateFullscreenBtn);
document.addEventListener('webkitfullscreenchange', updateFullscreenBtn);

// "What should Aarav AI call you?" — stored locally, sent with every chat request
function getUserName() { return localStorage.getItem('aarav_user_name') || ''; }
function setUserName(name) {
  if (name) localStorage.setItem('aarav_user_name', name);
  else localStorage.removeItem('aarav_user_name');
}
function openNameModal() {
  nameInput.value = getUserName();
  nameModalOverlay.classList.add('show');
  setTimeout(() => nameInput.focus(), 50);
}
function closeNameModal() { nameModalOverlay.classList.remove('show'); }
nameBtn.addEventListener('click', openNameModal);
nameCancelBtn.addEventListener('click', closeNameModal);
nameModalOverlay.addEventListener('click', (e) => { if (e.target === nameModalOverlay) closeNameModal(); });
nameSaveBtn.addEventListener('click', () => {
  setUserName(nameInput.value.trim());
  closeNameModal();
});
nameInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); nameSaveBtn.click(); }
  else if (e.key === 'Escape') closeNameModal();
});
// First-time visitors get a gentle one-time prompt
if (!localStorage.getItem('aarav_name_prompted')) {
  localStorage.setItem('aarav_name_prompted', '1');
  setTimeout(openNameModal, 600);
}

// Hide sidebar by default on mobile
if (isMobile()) sidebar.classList.add('hidden');
newChatBtn.addEventListener('click', startNewChat);

async function doClearChat() {
  if (!activeConvId) return;
  await fetch('/api/conversations/' + activeConvId, { method: 'DELETE' });
  startNewChat();
}
clearBtn.addEventListener('click', doClearChat);

async function doExportChat() {
  if (!activeConvId) { alert('Start or open a chat first.'); return; }
  try {
    const r = await fetch('/api/conversations/' + activeConvId);
    if (!r.ok) return;
    const d = await r.json();
    const lines = [`# ${d.title || 'Aarav AI chat'}`, ''];
    (d.messages || []).forEach(m => {
      lines.push(m.role === 'user' ? 'You:' : 'Aarav AI:');
      lines.push(m.text || (m.attachment ? `[attachment: ${m.attachment.name}]` : ''));
      lines.push('');
    });
    const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = (d.title || 'chat').replace(/[^a-z0-9_ -]/gi, '').trim().slice(0, 60) + '.txt';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert('Export failed: ' + err.message);
  }
}
exportBtn.addEventListener('click', doExportChat);

// --- Mobile "more" menu: consolidates fullscreen/name/export/delete into one panel ---
function openMoreMenu() {
  if (moreFullscreenIcon) moreFullscreenIcon.textContent = fullscreenIcon.textContent;
  moreMenu.classList.add('show');
  moreMenuOverlay.classList.add('show');
}
function closeMoreMenu() {
  moreMenu.classList.remove('show');
  moreMenuOverlay.classList.remove('show');
}
moreBtn.addEventListener('click', openMoreMenu);
moreMenuOverlay.addEventListener('click', closeMoreMenu);
moreFullscreen.addEventListener('click', () => { closeMoreMenu(); toggleFullscreen(); });
moreName.addEventListener('click', () => { closeMoreMenu(); openNameModal(); });
moreExport.addEventListener('click', () => { closeMoreMenu(); doExportChat(); });
moreClear.addEventListener('click', () => { closeMoreMenu(); doClearChat(); });

// --- Swipe left to close the sidebar on mobile ---
let touchStartX = null;
sidebar.addEventListener('touchstart', (e) => { touchStartX = e.touches[0].clientX; }, { passive: true });
sidebar.addEventListener('touchend', (e) => {
  if (touchStartX === null) return;
  const dx = e.changedTouches[0].clientX - touchStartX;
  if (dx < -60) closeSidebar();
  touchStartX = null;
}, { passive: true });

// Initial load
(async () => {
  const convs = await loadConversationList();
  if (convs.length > 0) openConversation(convs[0].id);
  else showEmptyState();
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
                     "HTTP-Referer": "http://localhost:5000", "X-Title": "Aarav AI"},
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
    user_name = (data.get("user_name") or "").strip()[:60]  # what Aarav AI should call the user
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
            f"Address them as {user_name} naturally where it fits (e.g. greetings, "
            f"acknowledgements) — don't force it into every single reply."
        )
    # Only the real Gemini call gets the "you have search" instruction — fallback
    # providers don't have real search, so giving them that instruction makes them
    # hallucinate fake tool-call JSON into the visible reply.
    gemini_system_prompt = effective_system_prompt + GEMINI_SEARCH_ADDENDUM

    payload = {
        "contents": gemini_contents,
        "systemInstruction": {"parts": [{"text": gemini_system_prompt}]},
        "tools": [{"google_search": {}}],
    }

    def generate():
        full_reply = []
        if PROVIDER == "ollama":
            chunk_source = ollama_stream_chunks(to_ollama_messages(messages, effective_system_prompt))
        else:
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
    if not HF_API_KEY:
        return jsonify({"error": "No HuggingFace key configured"}), 503
    # Use FLUX.1-schnell — fast, high quality, free on HF
    model = "black-forest-labs/FLUX.1-schnell"
    try:
        resp = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={"inputs": prompt},
            timeout=60,
        )
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
            img_b64 = base64.b64encode(resp.content).decode("utf-8")
            return jsonify({"image": img_b64})
        else:
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
    print(f"Starting Aarav AI at http://localhost:5000")
    print(f"Providers (fallback order): {providers_str}")
    app.run(host="0.0.0.0", port=5000, debug=False)
