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
    "You are Mythic AI, a smart and friendly AI assistant made by Aarav Singh. "
    "If asked who made you, say you are Mythic AI made by Aarav Singh — say it once naturally, never repeat it unprompted. "
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
import tempfile
import os as _os

_DATA_DIR = _os.path.join(tempfile.gettempdir(), "chat_data")
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
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mythic AI</title>
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

  /* Fullscreen bar — sits above the input row so it's always reachable on mobile,
     away from any notch/status-bar area that can swallow top-corner taps. */
  #settings-btn { background:none; border:1px solid var(--border); color:var(--muted);
    width:36px; height:36px; border-radius:6px; cursor:pointer; font-size:15px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; touch-action:manipulation; }
  #settings-btn:hover { background:var(--panel); color:var(--accent); }
  #fullscreen-btn { background:none; border:1px solid var(--border); color:var(--muted);
    width:36px; height:36px; border-radius:6px; cursor:pointer; font-size:15px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; touch-action:manipulation; }
  #fullscreen-btn:hover { color:var(--text); border-color:var(--accent); }
  #fullscreen-btn.active { color:var(--accent); border-color:var(--accent); }
  #fullscreen-icon { font-size:15px; }

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
    font-size:12px; padding:6px 12px; border-radius:6px; cursor:pointer; flex-shrink:0; }
  #clear-btn:hover { background:var(--panel); }

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
  .quick-btn { background:var(--panel); border:1px solid var(--border); color:var(--text);
    font-size:12.5px; padding:6px 14px; border-radius:20px; cursor:pointer;
    transition:all .15s ease; white-space:nowrap; font-family:inherit; touch-action:manipulation; }
  .quick-btn:hover { background:var(--accent-dim); border-color:var(--accent); color:var(--accent); }

  #messages-wrap::-webkit-scrollbar, #conv-list::-webkit-scrollbar { width:6px; }
  #messages-wrap::-webkit-scrollbar-thumb, #conv-list::-webkit-scrollbar-thumb
    { background:var(--border); border-radius:4px; }
  #sidebar-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.55);
    z-index:99; -webkit-tap-highlight-color:transparent; }

  @media(max-width:768px) {
    :root { --sidebar-w: 78vw; }

    /* Sidebar slides in as overlay — never pushes content */
    #sidebar { position:fixed; top:0; left:0; z-index:100; height:100%;
      height:-webkit-fill-available; width:var(--sidebar-w) !important;
      transform:translateX(0); transition:transform .25s ease;
      box-shadow:4px 0 24px rgba(0,0,0,.5); }
    #sidebar.hidden { transform:translateX(-105%); margin-left:0 !important; }

    /* Show overlay when sidebar open */
    #sidebar-overlay { display:block; }

    /* Main app always takes full width */
    .app { width:100% !important; flex:1; }

    header { padding:calc(10px + env(safe-area-inset-top)) 12px 10px; }
    header h1 { font-size:14px; }
    #sidebar-toggle { width:38px; height:38px; font-size:14px; }
    #name-btn { width:38px; height:38px; font-size:14px; }
    #export-btn { width:38px; height:38px; font-size:14px; }
    #clear-btn { font-size:11px; padding:8px 10px; min-height:38px; }
    #speak-toggle { font-size:11px; padding:5px 8px; }
    #fullscreen-btn { font-size:12.5px; padding:10px 12px; }

    #messages-wrap { overflow-y:auto; -webkit-overflow-scrolling:touch; }
    #messages { padding:14px 10px; gap:12px; max-width:100%; }
    .msg { max-width:90%; font-size:14px; padding:10px 12px; }
    .msg-row { max-width:90%; }
    .msg-actions { opacity:1; height:26px; } /* no hover on touch — keep always visible */
    .msg-actions button { font-size:13px; padding:4px 9px; min-width:30px; min-height:26px; }

    .input-area { padding:8px 10px max(10px,env(safe-area-inset-bottom)); }
    .input-row { padding:6px 8px; }
    textarea { font-size:16px; } /* 16px prevents iOS zoom */
    .tool-btn { width:34px; height:34px; font-size:17px; }
    #send-btn { width:34px; height:34px; font-size:16px; }

    .empty-state h2 { font-size:19px; }
    .empty-state p { font-size:13px; }
    #scroll-btn { bottom:80px; right:12px; width:34px; height:34px; }

    #new-chat-btn { margin:10px; padding:10px 12px; font-size:13.5px; }
    .conv-item { padding:10px 8px; font-size:13px; min-height:44px; }
    .conv-item .rename-btn { opacity:1; }
    .conv-item .del-btn { opacity:1; }
    #sidebar-footer { font-size:11px; padding:10px 12px; }
  }

  @media(max-width:380px) {
    :root { --sidebar-w: 88vw; }
    .msg { font-size:13.5px; }
    header h1 { font-size:13px; }
    #speak-toggle { display:none; }
  }
</style>
</head>
<body>
<div class="layout">
  <div id="sidebar-overlay" style="display:none;position:fixed;inset:0;background:#0007;z-index:99" id="sidebar-overlay"></div>
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
        <select id="model-select" title="Select model" style="background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:5px 8px;font-size:12px;cursor:pointer;outline:none;max-width:130px;font-family:inherit;">
          <option value="aarav-1.0">Mythic 1.0</option>
          <option value="aarav-2.0">Mythic 2.0</option>
          <option value="aarav-2.5" selected>Mythic 2.5</option>
          <option value="aarav-3.5">Mythic 3.5</option>
          <option value="aarav-ultra">Mythic Ultra 🔒</option>
        </select>
      </div>
      <div class="right">
        <button id="settings-btn" title="Settings">⚙</button>
        <button id="fullscreen-btn" type="button" title="Fullscreen">
          <span id="fullscreen-icon">⛶</span>
        </button>
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

    <div id="pending-attach">
      📎 <span id="pending-attach-name"></span>
      <button id="pending-attach-remove">✕</button>
    </div>

    <div id="speaking-indicator">
      🔊 Speaking...
      <button id="stop-speak-btn">Stop</button>
    </div>

    <!-- Quick action buttons -->
    <div id="quick-actions" style="display:flex;gap:8px;padding:6px 20px 0;max-width:760px;margin:0 auto;width:100%;flex-wrap:wrap;">
      <button class="quick-btn" id="img-gen-btn">🎨 Image</button>
      <button class="quick-btn" id="ghibli-btn">🌿 Ghibli Me</button>
      <button class="quick-btn" id="homework-btn">📚 Homework</button>
      <button class="quick-btn" id="weather-btn">🌤 Weather</button>
      <button class="quick-btn" id="search-btn">🔍 Search</button>
    </div>
      <button id="fullscreen-btn" type="button">
        <span id="fullscreen-icon">⛶</span> <span id="fullscreen-label">Fullscreen</span>
      </button>
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
          <textarea id="input" rows="1" placeholder="Message Mythic AI..."></textarea>
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
    <h3>What should Mythic AI call you?</h3>
    <p>Enter your preferred name — Mythic AI will use it when it talks to you.</p>
    <input type="text" id="name-input" maxlength="60" placeholder="e.g. Aarav" autocomplete="off">
    <div id="name-modal-actions">
      <button id="name-cancel-btn" type="button">Cancel</button>
      <button id="name-save-btn" type="button">Save</button>
    </div>
  </div>
</div>

<!-- Ghibli Selfie Modal -->
<div id="ghibli-modal-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:300;align-items:center;justify-content:center;">
  <div style="background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:24px;width:92%;max-width:440px;max-height:90vh;overflow-y:auto;">
    <h3 style="margin:0 0 4px;font-size:18px;">🌿 Ghibli Me</h3>
    <p style="color:var(--muted);font-size:13px;margin:0 0 16px;">Upload your photo and get a Studio Ghibli-style version of yourself</p>

    <!-- Upload area -->
    <div id="ghibli-upload-area" style="border:2px dashed var(--border);border-radius:12px;padding:24px;text-align:center;cursor:pointer;margin-bottom:12px;transition:border-color .2s;">
      <div style="font-size:36px;margin-bottom:8px;">📸</div>
      <div style="font-size:13px;color:var(--muted);">Click to upload your photo<br><span style="font-size:11px;">or drag & drop</span></div>
      <input type="file" id="ghibli-file-input" accept="image/*" style="display:none">
    </div>

    <!-- Preview -->
    <div id="ghibli-preview-wrap" style="display:none;margin-bottom:12px;text-align:center;">
      <img id="ghibli-preview" style="max-width:100%;max-height:180px;border-radius:10px;border:2px solid var(--accent);">
      <div style="font-size:11px;color:var(--muted);margin-top:4px;">Your photo ✓</div>
    </div>

    <!-- Style options -->
    <div style="margin-bottom:12px;">
      <label style="font-size:12px;color:var(--muted);display:block;margin-bottom:6px;">Ghibli Style:</label>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
        <button class="ghibli-style-btn" data-style="Studio Ghibli portrait, Spirited Away style, soft watercolor anime art" style="padding:8px;border-radius:8px;border:1.5px solid var(--accent);background:var(--accent-dim);color:var(--accent);cursor:pointer;font-size:12px;font-family:inherit;">🌊 Spirited Away</button>
        <button class="ghibli-style-btn" data-style="Studio Ghibli portrait, My Neighbor Totoro style, soft forest anime art" style="padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--muted);cursor:pointer;font-size:12px;font-family:inherit;">🌳 Totoro Forest</button>
        <button class="ghibli-style-btn" data-style="Studio Ghibli portrait, Howl's Moving Castle style, fantasy anime art" style="padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--muted);cursor:pointer;font-size:12px;font-family:inherit;">🏰 Howl's Castle</button>
        <button class="ghibli-style-btn" data-style="Studio Ghibli portrait, Princess Mononoke style, nature anime art" style="padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--muted);cursor:pointer;font-size:12px;font-family:inherit;">🐺 Mononoke</button>
      </div>
    </div>

    <!-- Extra prompt -->
    <input id="ghibli-extra" type="text" placeholder="Add details (optional): e.g. forest background, sunset..."
      style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:9px 12px;font-size:13px;outline:none;margin-bottom:12px;font-family:inherit;">

    <!-- Result -->
    <div id="ghibli-result-wrap" style="display:none;margin-bottom:12px;text-align:center;">
      <img id="ghibli-result" style="max-width:100%;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,.4);">
      <button id="ghibli-download-btn" style="margin-top:8px;background:var(--accent);color:#fff;border:none;border-radius:8px;padding:8px 18px;font-size:13px;cursor:pointer;font-family:inherit;">⬇ Download</button>
    </div>
    <div id="ghibli-loading" style="display:none;text-align:center;padding:20px;">
      <div style="font-size:32px;margin-bottom:8px;">🎨</div>
      <div style="color:var(--muted);font-size:13px;">Creating your Ghibli portrait...<br><span style="font-size:11px;">This takes 15-30 seconds</span></div>
    </div>
    <div id="ghibli-error" style="display:none;color:#ef4444;font-size:12px;margin-bottom:8px;padding:8px;background:#fef2f2;border-radius:6px;"></div>

    <div style="display:flex;gap:8px;">
      <button id="ghibli-generate-btn" style="flex:1;background:linear-gradient(135deg,#10a37f,#0d7a5f);color:#fff;border:none;border-radius:10px;padding:12px;font-size:14px;font-weight:700;cursor:pointer;font-family:inherit;">✨ Create Ghibli Art</button>
      <button id="ghibli-close-btn" style="background:none;border:1px solid var(--border);color:var(--muted);border-radius:10px;padding:12px 16px;font-size:14px;cursor:pointer;">✕</button>
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
const modelSelect   = document.getElementById('model-select');

let selectedModel = 'aarav-2.5';
let vipUnlocked   = false;

function showVipModal() {
  const existing = document.getElementById('vip-modal-overlay');
  if (existing) { existing.style.display='flex'; return; }
  const overlay = document.createElement('div');
  overlay.id = 'vip-modal-overlay';
  overlay.style.cssText='position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:500;display:flex;align-items:center;justify-content:center;';
  overlay.innerHTML=`<div style="background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:24px;width:90%;max-width:340px;">
    <div style="font-size:22px;margin-bottom:6px;">🔒 VIP Access</div>
    <div style="color:var(--muted);font-size:13px;margin-bottom:16px;">Mythic Ultra is for VIP users only.</div>
    <input id="vip-pw-in" type="password" placeholder="VIP password" style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:10px 12px;font-size:14px;outline:none;margin-bottom:8px;font-family:inherit;">
    <div id="vip-pw-err" style="color:#ef4444;font-size:12px;display:none;margin-bottom:8px;">Wrong password.</div>
    <div style="display:flex;gap:8px;">
      <button id="vip-pw-ok" style="flex:1;background:var(--accent);color:#fff;border:none;border-radius:8px;padding:10px;font-size:14px;font-weight:600;cursor:pointer;">Unlock</button>
      <button id="vip-pw-cancel" style="flex:1;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:10px;font-size:14px;cursor:pointer;">Cancel</button>
    </div></div>`;
  document.body.appendChild(overlay);
  const pwIn=overlay.querySelector('#vip-pw-in'), pwErr=overlay.querySelector('#vip-pw-err');
  pwIn.focus();
  overlay.querySelector('#vip-pw-cancel').addEventListener('click',()=>{overlay.style.display='none';modelSelect.value=selectedModel;});
  overlay.querySelector('#vip-pw-ok').addEventListener('click',async()=>{
    const r=await fetch('/api/vip-unlock',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pwIn.value.trim()})});
    const d=await r.json();
    if(d.success){vipUnlocked=true;overlay.style.display='none';selectedModel='aarav-ultra';modelSelect.value='aarav-ultra';
      const opt=modelSelect.querySelector('option[value="aarav-ultra"]');if(opt)opt.textContent='Mythic Ultra ✨';}
    else{pwErr.style.display='block';pwIn.value='';pwIn.focus();}
  });
  pwIn.addEventListener('keydown',e=>{if(e.key==='Enter')overlay.querySelector('#vip-pw-ok').click();});
}

// Load models from API
(async()=>{
  try{
    const[mr,vr]=await Promise.all([fetch('/api/models').then(r=>r.json()),fetch('/api/vip-status').then(r=>r.json())]);
    vipUnlocked=vr.vip;
    modelSelect.innerHTML='';
    mr.models.forEach(m=>{
      const opt=document.createElement('option');
      opt.value=m.id;
      opt.textContent=m.vip?(vipUnlocked?m.name.replace('🔒','✨'):m.name):m.name;
      opt.dataset.vip=m.vip?'1':'0';
      if(m.id===mr.default)opt.selected=true;
      modelSelect.appendChild(opt);
    });
    selectedModel=mr.default;
  }catch{}
})();

modelSelect.addEventListener('change',()=>{
  const opt=modelSelect.options[modelSelect.selectedIndex];
  if(opt&&opt.dataset.vip==='1'&&!vipUnlocked){showVipModal();}
  else{selectedModel=modelSelect.value;}
});
const nameModalOverlay = document.getElementById('name-modal-overlay');
const nameInput     = document.getElementById('name-input');
const nameCancelBtn = document.getElementById('name-cancel-btn');
const nameSaveBtn   = document.getElementById('name-save-btn');
const exportBtn     = document.getElementById('export-btn');
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
  messagesEl.innerHTML = '<div class="empty-state" id="empty-state"><h2>Mythic AI</h2><p>Ask me anything, generate images, or just chat 👋</p></div>';
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
        model: selectedModel,
        news_context: newsContext || null,
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
  const tonePrefix = getTonePrefix();
  streamReply({ message: tonePrefix + text, attachment });
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
const fullscreenLabel = document.getElementById('fullscreen-label');
const fsSupported = !!(document.documentElement.requestFullscreen || document.documentElement.webkitRequestFullscreen);

function isFullscreen() {
  return !!(document.fullscreenElement || document.webkitFullscreenElement) ||
    document.body.classList.contains('pseudo-fullscreen');
}
function updateFullscreenBtn() {
  if (isFullscreen()) {
    fullscreenIcon.textContent = '⤢';
    fullscreenLabel.textContent = 'Exit fullscreen';
    fullscreenBtn.classList.add('active');
  } else {
    fullscreenIcon.textContent = '⛶';
    fullscreenLabel.textContent = 'Fullscreen';
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

// "What should Mythic AI call you?" — stored locally, sent with every chat request
function getUserName() { return localStorage.getItem('mythic_user_name') || ''; }
function setUserName(name) {
  if (name) localStorage.setItem('mythic_user_name', name);
  else localStorage.removeItem('mythic_user_name');
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
if (!localStorage.getItem('mythic_name_prompted')) {
  localStorage.setItem('mythic_name_prompted', '1');
  setTimeout(openNameModal, 600);
}

// Hide sidebar by default on mobile
if (isMobile()) sidebar.classList.add('hidden');
newChatBtn.addEventListener('click', startNewChat);
clearBtn.addEventListener('click', async () => {
  if (!activeConvId) return;
  await fetch('/api/conversations/' + activeConvId, { method: 'DELETE' });
  startNewChat();
});

exportBtn.addEventListener('click', async () => {
  if (!activeConvId) { alert('Start or open a chat first.'); return; }
  try {
    const r = await fetch('/api/conversations/' + activeConvId);
    if (!r.ok) return;
    const d = await r.json();
    const lines = [`# ${d.title || 'Mythic AI chat'}`, ''];
    (d.messages || []).forEach(m => {
      lines.push(m.role === 'user' ? 'You:' : 'Mythic AI:');
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
});

// Initial load
// ─── SETTINGS ────────────────────────────────────────────────────────────────
const settingsBtn        = document.getElementById('settings-btn');
const settingsModalOverlay=document.getElementById('settings-modal-overlay');
const settingsCloseBtn   = document.getElementById('settings-close-btn');
const accentColorInput   = document.getElementById('accent-color-input');
const fontSizeSlider     = document.getElementById('font-size-slider');
const fontSizeLabel      = document.getElementById('font-size-label');
const toneSelect         = document.getElementById('tone-select');
const lengthSelect       = document.getElementById('length-select');
const customInstructions = document.getElementById('custom-instructions-input');

// Load saved settings
function loadSettings() {
  const s = JSON.parse(localStorage.getItem('mythic_settings') || '{}');
  // Theme
  const theme = s.theme || 'dark';
  applyTheme(theme);
  document.querySelectorAll('[data-group="theme"]').forEach(b => {
    b.style.borderColor = b.dataset.value === theme ? 'var(--accent)' : 'var(--border)';
    b.style.color = b.dataset.value === theme ? 'var(--accent)' : '';
  });
  // Accent
  const accent = s.accent || '#10a37f';
  accentColorInput.value = accent;
  document.documentElement.style.setProperty('--accent', accent);
  // Font size
  const fs = s.fontSize || '14.5';
  fontSizeSlider.value = fs;
  fontSizeLabel.textContent = fs + 'px';
  document.documentElement.style.setProperty('--msg-font-size', fs + 'px');
  // Bubble style
  const bubble = s.bubble || 'comfortable';
  document.body.classList.remove('bubble-compact','bubble-comfortable','bubble-spacious');
  document.body.classList.add('bubble-' + bubble);
  document.querySelectorAll('[data-group="bubble"]').forEach(b => {
    b.style.borderColor = b.dataset.value === bubble ? 'var(--accent)' : 'var(--border)';
    b.style.color = b.dataset.value === bubble ? 'var(--accent)' : '';
  });
  // Tone & Length
  if (toneSelect) toneSelect.value = s.tone || 'default';
  if (lengthSelect) lengthSelect.value = s.length || 'default';
  // Custom instructions
  if (customInstructions) customInstructions.value = s.customInstructions || '';
}

function saveSettings() {
  const s = JSON.parse(localStorage.getItem('mythic_settings') || '{}');
  s.theme = document.body.classList.contains('theme-light') ? 'light' : 'dark';
  s.accent = accentColorInput.value;
  s.fontSize = fontSizeSlider.value;
  const bubs = ['compact','comfortable','spacious'].find(b => document.body.classList.contains('bubble-'+b)) || 'comfortable';
  s.bubble = bubs;
  s.tone = toneSelect ? toneSelect.value : 'default';
  s.length = lengthSelect ? lengthSelect.value : 'default';
  s.customInstructions = customInstructions ? customInstructions.value : '';
  localStorage.setItem('mythic_settings', JSON.stringify(s));
}

function applyTheme(t) {
  if (t === 'system') {
    const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    document.body.classList.toggle('theme-light', !dark);
  } else {
    document.body.classList.toggle('theme-light', t === 'light');
  }
}

settingsBtn.addEventListener('click', () => { settingsModalOverlay.style.display = 'flex'; });
settingsCloseBtn.addEventListener('click', () => { saveSettings(); settingsModalOverlay.style.display = 'none'; });
settingsModalOverlay.addEventListener('click', e => { if (e.target === settingsModalOverlay) { saveSettings(); settingsModalOverlay.style.display = 'none'; } });

document.querySelectorAll('.settings-choice').forEach(btn => {
  btn.addEventListener('click', () => {
    const group = btn.dataset.group;
    document.querySelectorAll(`[data-group="${group}"]`).forEach(b => {
      b.style.borderColor = 'var(--border)'; b.style.color = '';
    });
    btn.style.borderColor = 'var(--accent)'; btn.style.color = 'var(--accent)';
    if (group === 'theme') applyTheme(btn.dataset.value);
    if (group === 'bubble') {
      document.body.classList.remove('bubble-compact','bubble-comfortable','bubble-spacious');
      document.body.classList.add('bubble-' + btn.dataset.value);
    }
    saveSettings();
  });
});

accentColorInput.addEventListener('input', () => {
  document.documentElement.style.setProperty('--accent', accentColorInput.value);
});
fontSizeSlider.addEventListener('input', () => {
  fontSizeLabel.textContent = fontSizeSlider.value + 'px';
  document.documentElement.style.setProperty('--msg-font-size', fontSizeSlider.value + 'px');
});

loadSettings();

// ─── MARKDOWN RENDERING ──────────────────────────────────────────────────────
function renderMarkdown(text) {
  const div = document.createElement('div');
  div.className = 'msg-text md-rendered';
  // Escape HTML first
  let html = text
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  // Code blocks (must come before inline code)
  html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) =>
    `<pre><code class="lang-${lang}">${code.trim()}</code></pre>`);
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Headers
  html = html.replace(/^### (.+)$/gm, '<h3 style="font-size:14px;margin:6px 0 3px;font-weight:700;">$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2 style="font-size:15px;margin:8px 0 4px;font-weight:700;">$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1 style="font-size:17px;margin:10px 0 5px;font-weight:700;">$1</h1>');
  // Unordered lists
  html = html.replace(/(^|\n)([\-\*] .+(\n[\-\*] .+)*)/g, (_, pre, block) =>
    pre + '<ul>' + block.replace(/[\-\*] (.+)/g, '<li>$1</li>') + '</ul>');
  // Ordered lists
  html = html.replace(/(^|\n)(\d+\. .+(\n\d+\. .+)*)/g, (_, pre, block) =>
    pre + '<ol>' + block.replace(/\d+\. (.+)/g, '<li>$1</li>') + '</ol>');
  // Line breaks
  html = html.replace(/\n/g, '<br>');
  div.innerHTML = html;
  return div;
}

// Override addMessage to use markdown for AI
const _origAddMessage = addMessage;
function addMessage(role, text, attachment) {
  const textNode = _origAddMessage(role, text, attachment);
  if (role === 'ai' && text) {
    try {
      const md = renderMarkdown(text);
      textNode.parentNode.replaceChild(md, textNode);
      return md;
    } catch { return textNode; }
  }
  return textNode;
}

// ─── MESSAGE TIMESTAMPS ───────────────────────────────────────────────────────
const _origAddMsg2 = addMessage;
function addMessageWithTimestamp(role, text, attachment) {
  const node = _origAddMsg2(role, text, attachment);
  const row = node.closest ? node.closest('.msg-row') : null;
  if (row) {
    const ts = document.createElement('div');
    ts.className = 'msg-timestamp';
    ts.textContent = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
    row.appendChild(ts);
  }
  return node;
}
// Monkey-patch addMessage to include timestamp
const _rawAdd = addMessage;
window._addMsgFinal = function(role, text, attachment) {
  const node = _rawAdd(role, text, attachment);
  const row = node && node.closest ? node.closest('.msg-row') : null;
  if (row && !row.querySelector('.msg-timestamp')) {
    const ts = document.createElement('div');
    ts.className = 'msg-timestamp';
    ts.textContent = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
    row.appendChild(ts);
  }
  return node;
};

// ─── FOLLOW-UP SUGGESTIONS ───────────────────────────────────────────────────
async function addFollowupSuggestions(aiText) {
  if (!aiText || aiText.length < 50) return;
  try {
    const r = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        message: `Based on this AI reply, suggest 3 short follow-up questions the user might ask. Reply ONLY with 3 questions, one per line, no numbering, no extra text:\n\n${aiText.slice(0,400)}`,
        conversation_id: null, model: 'aarav-1.0', user_name: ''
      })
    });
    if (!r.ok) return;
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let full = '';
    while (true) { const {done,value} = await reader.read(); if(done) break; full += dec.decode(value,{stream:true}); }
    const qs = full.trim().split('\n').filter(q => q.trim().length > 5).slice(0,3);
    if (!qs.length) return;
    const wrap = document.createElement('div');
    wrap.style.cssText = 'display:flex;flex-wrap:wrap;gap:6px;margin-top:6px;max-width:100%;';
    qs.forEach(q => {
      const btn = document.createElement('button');
      btn.textContent = q.trim().replace(/^["']|["']$/g,'');
      btn.style.cssText = 'background:var(--panel);border:1px solid var(--border);color:var(--muted);font-size:11.5px;padding:5px 10px;border-radius:16px;cursor:pointer;font-family:inherit;text-align:left;';
      btn.addEventListener('click', () => {
        input.value = btn.textContent; input.focus(); autoResize();
        form.requestSubmit(); wrap.remove();
      });
      wrap.appendChild(btn);
    });
    messagesEl.appendChild(wrap);
    scrollToBottom();
  } catch {}
}

// ─── MESSAGE REACTIONS ────────────────────────────────────────────────────────
function addReactionBar(row) {
  if (row.querySelector('.reaction-bar')) return;
  const bar = document.createElement('div');
  bar.className = 'reaction-bar';
  bar.style.cssText = 'display:flex;gap:4px;margin-top:3px;';
  ['👍','👎','❤️','😂','🔖'].forEach(emoji => {
    const btn = document.createElement('button');
    btn.textContent = emoji;
    btn.style.cssText = 'background:none;border:1px solid var(--border);border-radius:12px;padding:2px 7px;font-size:13px;cursor:pointer;touch-action:manipulation;';
    btn.addEventListener('click', () => {
      btn.style.borderColor = btn.style.borderColor === 'var(--accent)' ? 'var(--border)' : 'var(--accent)';
      btn.style.background  = btn.style.background === 'var(--accent-dim)' ? '' : 'var(--accent-dim)';
    });
    bar.appendChild(btn);
  });
  row.appendChild(bar);
}

// ─── MESSAGE SEARCH ───────────────────────────────────────────────────────────
function addSearchUI() {
  const searchWrap = document.createElement('div');
  searchWrap.id = 'msg-search-wrap';
  searchWrap.style.cssText = 'display:none;position:fixed;top:70px;left:50%;transform:translateX(-50%);z-index:150;background:var(--panel);border:1.5px solid var(--accent);border-radius:10px;padding:8px 12px;display:flex;gap:8px;align-items:center;min-width:280px;box-shadow:0 4px 20px rgba(0,0,0,.3);';
  searchWrap.innerHTML = '<input id="msg-search-input" placeholder="Search messages..." style="background:transparent;border:none;color:var(--text);font-size:14px;outline:none;flex:1;font-family:inherit;"><span id="msg-search-count" style="color:var(--muted);font-size:12px;"></span><button id="msg-search-close" style="background:none;border:none;color:var(--muted);cursor:pointer;font-size:16px;">✕</button>';
  searchWrap.style.display = 'none';
  document.body.appendChild(searchWrap);

  let highlights = [];
  document.getElementById('msg-search-input').addEventListener('input', e => {
    highlights.forEach(el => { el.style.background = ''; el.style.outline = ''; });
    highlights = [];
    const q = e.target.value.trim().toLowerCase();
    if (!q) { document.getElementById('msg-search-count').textContent = ''; return; }
    document.querySelectorAll('.msg-text,.md-rendered').forEach(el => {
      if (el.textContent.toLowerCase().includes(q)) {
        el.style.background = 'rgba(255,200,0,.15)';
        el.style.outline = '2px solid rgba(255,200,0,.4)';
        highlights.push(el);
      }
    });
    document.getElementById('msg-search-count').textContent = highlights.length ? `${highlights.length} found` : 'no results';
    if (highlights.length) highlights[0].scrollIntoView({behavior:'smooth',block:'center'});
  });
  document.getElementById('msg-search-close').addEventListener('click', () => {
    searchWrap.style.display = 'none';
    highlights.forEach(el => { el.style.background = ''; el.style.outline = ''; });
  });
  return searchWrap;
}
const msgSearchWrap = addSearchUI();

// Keyboard shortcut Ctrl+F to search messages
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'f' && !settingsModalOverlay.style.display.includes('flex')) {
    e.preventDefault();
    msgSearchWrap.style.display = 'flex';
    setTimeout(() => document.getElementById('msg-search-input').focus(), 50);
  }
  if (e.key === 'Escape') msgSearchWrap.style.display = 'none';
});

// ─── PWA SUPPORT ─────────────────────────────────────────────────────────────
if ('serviceWorker' in navigator && navigator.serviceWorker) {
  // Inline service worker for offline caching
  const swCode = `
const CACHE = 'mythic-ai-v1';
const OFFLINE = ['/', '/static/app.js'];
self.addEventListener('install', e => e.waitUntil(caches.open(CACHE).then(c => c.addAll(['/']))));
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  e.respondWith(fetch(e.request).catch(() => caches.match(e.request)));
});`;
  const blob = new Blob([swCode], {type:'application/javascript'});
  navigator.serviceWorker.register(URL.createObjectURL(blob)).catch(()=>{});
}

// ─── WIRE REACTIONS INTO MSG ACTIONS ─────────────────────────────────────────
// Patch buildMsgActions to add reaction button
const _origBuildActions = buildMsgActions;
function buildMsgActions(row, textNode, role) {
  const actions = _origBuildActions(row, textNode, role);
  if (role === 'ai') {
    const reactBtn = document.createElement('button');
    reactBtn.type = 'button'; reactBtn.title = 'React'; reactBtn.textContent = '😊';
    reactBtn.addEventListener('click', () => addReactionBar(row));
    actions.appendChild(reactBtn);
    // 🔊 speak button
    const sp = document.createElement('button');
    sp.type='button'; sp.title='Read aloud'; sp.textContent='🔊';
    sp.addEventListener('click', () => {
      if (sp.textContent === '⏹') { stopSpeaking(); sp.textContent='🔊'; return; }
      sp.textContent='⏹'; speak(textNode.textContent || (textNode.innerText || ''));
      if (currentUtterance) {
        currentUtterance.onend = () => sp.textContent='🔊';
        currentUtterance.onerror = () => sp.textContent='🔊';
      }
    });
    actions.appendChild(sp);
  }
  return actions;
}

function stopSpeaking() { if(window.speechSynthesis) window.speechSynthesis.cancel(); }

// ─── TONE/LENGTH INJECTION ────────────────────────────────────────────────────
function getTonePrefix() {
  const s = JSON.parse(localStorage.getItem('mythic_settings') || '{}');
  const tone = s.tone || 'default';
  const length = s.length || 'default';
  let parts = [];
  if (tone === 'formal') parts.push('Reply in a formal, professional tone.');
  if (tone === 'casual') parts.push('Reply in a casual, friendly tone.');
  if (tone === 'funny') parts.push('Be funny and use wit and humor in your reply.');
  if (tone === 'professional') parts.push('Use a professional, business-appropriate tone.');
  if (length === 'short') parts.push('Keep your reply very short — 1-3 sentences max.');
  if (length === 'medium') parts.push('Keep your reply medium length — a few paragraphs.');
  if (length === 'long') parts.push('Give a thorough, detailed, long reply.');
  const ci = customInstructions ? customInstructions.value.trim() : '';
  if (ci) parts.push(ci);
  return parts.length ? '[Instructions: ' + parts.join(' ') + '] ' : '';
}

// ─── AUTO FOLLOW-UPS AFTER AI REPLY ──────────────────────────────────────────
// Hook into streamReply completion by patching the form submit
const _origFormSubmit = form.onsubmit;
form.addEventListener('submit', async () => {
  // Wait for generation to finish then add follow-ups
  const checkDone = setInterval(() => {
    if (!isGenerating) {
      clearInterval(checkDone);
      const allRows = messagesEl.querySelectorAll('.msg-row.ai');
      if (allRows.length) {
        const lastRow = allRows[allRows.length - 1];
        const textEl = lastRow.querySelector('.msg-text,.md-rendered');
        if (textEl && !lastRow.querySelector('.reaction-bar')) {
          addReactionBar(lastRow);
          setTimeout(() => addFollowupSuggestions(textEl.textContent || textEl.innerText || ''), 300);
        }
      }
    }
  }, 500);
});

// ─── INITIAL LOAD ─────────────────────────────────────────────────────────────
(async () => {
  const convs = await loadConversationList();
  if (convs.length > 0) openConversation(convs[0].id);
  else showEmptyState();
})();

const imgGenBtn   = document.getElementById('img-gen-btn');
const ghibliBtn   = document.getElementById('ghibli-btn');
const homeworkBtn = document.getElementById('homework-btn');
const weatherBtn2 = document.getElementById('weather-btn');
const searchBtn   = document.getElementById('search-btn');

if (imgGenBtn) imgGenBtn.addEventListener('click', () => {
  const imgModal = document.getElementById('img-modal-overlay');
  if (imgModal) { imgModal.style.display = 'flex'; document.getElementById('img-prompt').focus(); }
});
if (homeworkBtn) homeworkBtn.addEventListener('click', () => {
  input.value = 'Help me with my homework: '; input.focus(); autoResize();
  input.setSelectionRange(input.value.length, input.value.length);
});
if (weatherBtn2) weatherBtn2.addEventListener('click', () => {
  const wm = document.getElementById('weather-modal-overlay');
  if (wm) { wm.style.display = 'flex'; document.getElementById('weather-city').focus(); }
});
if (searchBtn) searchBtn.addEventListener('click', () => {
  const q = prompt('What do you want to search for?');
  if (!q || !q.trim()) return;
  input.value = 'Search: ' + q.trim(); autoResize(); form.requestSubmit();
});

// ─── GHIBLI SELFIE MODAL ─────────────────────────────────────────────────────
const ghibliModal     = document.getElementById('ghibli-modal-overlay');
const ghibliUploadArea= document.getElementById('ghibli-upload-area');
const ghibliFileInput = document.getElementById('ghibli-file-input');
const ghibliPreviewWrap=document.getElementById('ghibli-preview-wrap');
const ghibliPreview   = document.getElementById('ghibli-preview');
const ghibliResult    = document.getElementById('ghibli-result');
const ghibliResultWrap= document.getElementById('ghibli-result-wrap');
const ghibliLoading   = document.getElementById('ghibli-loading');
const ghibliError     = document.getElementById('ghibli-error');
const ghibliGenerateBtn=document.getElementById('ghibli-generate-btn');
const ghibliCloseBtn  = document.getElementById('ghibli-close-btn');
const ghibliDownloadBtn=document.getElementById('ghibli-download-btn');
const ghibliExtraInput= document.getElementById('ghibli-extra');

let ghibliBase64 = null;
let ghibliSelectedStyle = 'Studio Ghibli portrait, Spirited Away style, soft watercolor anime art';

// Open/close
if (ghibliBtn) ghibliBtn.addEventListener('click', () => {
  ghibliModal.style.display = 'flex';
  ghibliBase64 = null;
  ghibliPreviewWrap.style.display = 'none';
  ghibliResultWrap.style.display = 'none';
  ghibliError.style.display = 'none';
  ghibliLoading.style.display = 'none';
});
if (ghibliCloseBtn) ghibliCloseBtn.addEventListener('click', () => ghibliModal.style.display = 'none');
ghibliModal.addEventListener('click', e => { if (e.target === ghibliModal) ghibliModal.style.display = 'none'; });

// Style selector
document.querySelectorAll('.ghibli-style-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.ghibli-style-btn').forEach(b => {
      b.style.borderColor = 'var(--border)'; b.style.background = 'var(--panel)'; b.style.color = 'var(--muted)';
    });
    btn.style.borderColor = 'var(--accent)'; btn.style.background = 'var(--accent-dim)'; btn.style.color = 'var(--accent)';
    ghibliSelectedStyle = btn.dataset.style;
  });
});

// Upload area
ghibliUploadArea.addEventListener('click', () => ghibliFileInput.click());
ghibliUploadArea.addEventListener('dragover', e => { e.preventDefault(); ghibliUploadArea.style.borderColor = 'var(--accent)'; });
ghibliUploadArea.addEventListener('dragleave', () => { ghibliUploadArea.style.borderColor = 'var(--border)'; });
ghibliUploadArea.addEventListener('drop', e => {
  e.preventDefault(); ghibliUploadArea.style.borderColor = 'var(--border)';
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) loadGhibliPhoto(file);
});
ghibliFileInput.addEventListener('change', () => {
  if (ghibliFileInput.files[0]) loadGhibliPhoto(ghibliFileInput.files[0]);
});

function loadGhibliPhoto(file) {
  const reader = new FileReader();
  reader.onload = e => {
    ghibliBase64 = e.target.result.split(',')[1];
    ghibliPreview.src = e.target.result;
    ghibliPreviewWrap.style.display = 'block';
    ghibliResultWrap.style.display = 'none';
    ghibliError.style.display = 'none';
    ghibliUploadArea.style.borderColor = 'var(--accent)';
  };
  reader.readAsDataURL(file);
}

// Generate Ghibli image
ghibliGenerateBtn.addEventListener('click', async () => {
  if (!ghibliBase64) {
    ghibliError.textContent = 'Please upload your photo first!';
    ghibliError.style.display = 'block'; return;
  }
  ghibliError.style.display = 'none';
  ghibliResultWrap.style.display = 'none';
  ghibliLoading.style.display = 'block';
  ghibliGenerateBtn.disabled = true;

  const extra = ghibliExtraInput.value.trim();
  const prompt = `${ghibliSelectedStyle}, beautiful detailed portrait of a person, ${extra ? extra + ', ' : ''}masterpiece, best quality, highly detailed, cinematic lighting, soft colors, dreamy atmosphere`;

  try {
    const r = await fetch('/api/generate-image', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ prompt, style: '' })
    });
    const d = await r.json();
    ghibliLoading.style.display = 'none';
    if (d.image) {
      ghibliResult.src = 'data:image/png;base64,' + d.image;
      ghibliResultWrap.style.display = 'block';
      // Also show in chat
      clearEmptyState();
      const row = document.createElement('div'); row.className = 'msg-row ai';
      const bubble = document.createElement('div'); bubble.className = 'msg ai';
      bubble.style.padding = '8px';
      const cap = document.createElement('div');
      cap.textContent = '🌿 Your Ghibli portrait';
      cap.style.cssText = 'font-size:12px;color:var(--muted);margin-bottom:8px;';
      const img = document.createElement('img');
      img.src = 'data:image/png;base64,' + d.image;
      img.style.cssText = 'max-width:100%;border-radius:12px;display:block;cursor:pointer;';
      img.title = 'Click to download';
      img.addEventListener('click', () => downloadGhibliImage(d.image));
      bubble.appendChild(cap); bubble.appendChild(img); row.appendChild(bubble);
      messagesEl.appendChild(row); scrollToBottom();
    } else {
      ghibliError.textContent = d.error || 'Generation failed. Try again.';
      ghibliError.style.display = 'block';
    }
  } catch (e) {
    ghibliLoading.style.display = 'none';
    ghibliError.textContent = 'Error: ' + e.message;
    ghibliError.style.display = 'block';
  } finally { ghibliGenerateBtn.disabled = false; }
});

function downloadGhibliImage(b64) {
  const a = document.createElement('a');
  a.href = 'data:image/png;base64,' + b64;
  a.download = 'mythic-ai-ghibli-portrait.png';
  document.body.appendChild(a); a.click(); a.remove();
}
if (ghibliDownloadBtn) ghibliDownloadBtn.addEventListener('click', () => {
  if (ghibliResult.src) downloadGhibliImage(ghibliResult.src.split(',')[1]);
});

// ─── IMAGE GENERATION MODAL JS ───────────────────────────────────────────────
const imgModalOverlay = document.getElementById('img-modal-overlay');
const imgPromptEl     = document.getElementById('img-prompt');
const imgStyleEl      = document.getElementById('img-style');
const imgResultEl     = document.getElementById('img-result');
const imgOutputEl     = document.getElementById('img-output');
const imgLoadingEl    = document.getElementById('img-loading');
const imgErrorEl      = document.getElementById('img-error');
const imgGenerateBtn2 = document.getElementById('img-generate-btn');
const imgCloseBtn2    = document.getElementById('img-close-btn');

if (imgGenerateBtn2) imgGenerateBtn2.addEventListener('click', async () => {
  const prompt = imgPromptEl.value.trim();
  const style = imgStyleEl ? imgStyleEl.value : '';
  if (!prompt) return;
  imgResultEl.style.display = 'none'; imgErrorEl.style.display = 'none';
  imgLoadingEl.style.display = 'block'; imgGenerateBtn2.disabled = true;
  try {
    const r = await fetch('/api/generate-image', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,style})});
    const d = await r.json();
    imgLoadingEl.style.display = 'none';
    if (d.image) {
      imgOutputEl.src = 'data:image/png;base64,' + d.image;
      imgResultEl.style.display = 'block';
      clearEmptyState();
      const row = document.createElement('div'); row.className = 'msg-row ai';
      const bubble = document.createElement('div'); bubble.className = 'msg ai'; bubble.style.padding='8px';
      const cap = document.createElement('div'); cap.textContent = '🎨 ' + prompt;
      cap.style.cssText = 'font-size:12px;opacity:.7;margin-bottom:8px;';
      const img = document.createElement('img');
      img.src = 'data:image/png;base64,' + d.image;
      img.style.cssText = 'max-width:100%;border-radius:10px;display:block;';
      bubble.appendChild(cap); bubble.appendChild(img); row.appendChild(bubble);
      messagesEl.appendChild(row); scrollToBottom();
    } else { imgErrorEl.textContent = d.error || 'Failed.'; imgErrorEl.style.display='block'; }
  } catch(e) { imgLoadingEl.style.display='none'; imgErrorEl.textContent='Error: '+e.message; imgErrorEl.style.display='block'; }
  finally { imgGenerateBtn2.disabled = false; }
});
if (imgCloseBtn2) imgCloseBtn2.addEventListener('click', () => imgModalOverlay.style.display='none');
if (imgModalOverlay) imgModalOverlay.addEventListener('click', e => { if(e.target===imgModalOverlay) imgModalOverlay.style.display='none'; });
if (imgPromptEl) imgPromptEl.addEventListener('keydown', e => { if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();imgGenerateBtn2.click();} });

// ─── WEATHER MODAL JS ────────────────────────────────────────────────────────
const weatherModal2    = document.getElementById('weather-modal-overlay');
const weatherCityEl2   = document.getElementById('weather-city');
const weatherResultEl2 = document.getElementById('weather-result');
const weatherContentEl2= document.getElementById('weather-content');
const weatherLoadingEl2= document.getElementById('weather-loading');
const weatherErrorEl2  = document.getElementById('weather-error');
const weatherSearchBtn2= document.getElementById('weather-search-btn');
const weatherCloseBtn2 = document.getElementById('weather-close-btn');
const weatherLocBtn2   = document.getElementById('weather-location-btn');

function renderWeather(w) {
  weatherContentEl2.innerHTML = `
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
      <div style="font-size:48px;">${w.icon}</div>
      <div><div style="font-size:18px;font-weight:700;">${w.location}</div>
      <div style="font-size:13px;color:var(--muted);">${w.condition}</div></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
      <div style="background:var(--bg);border-radius:8px;padding:10px;">
        <div style="font-size:11px;color:var(--muted);">TEMPERATURE</div>
        <div style="font-size:22px;font-weight:700;">${w.temp}°C</div>
        <div style="font-size:11px;color:var(--muted);">Feels like ${w.feels_like}°C</div>
      </div>
      <div style="background:var(--bg);border-radius:8px;padding:10px;">
        <div style="font-size:11px;color:var(--muted);">HUMIDITY</div>
        <div style="font-size:22px;font-weight:700;">${w.humidity}%</div>
        <div style="font-size:11px;color:var(--muted);">Wind ${w.wind_speed} km/h</div>
      </div>
    </div>`;
  weatherResultEl2.style.display = 'block';
  input.value = `${w.icon} Weather in ${w.location}: ${w.temp}°C, ${w.condition}. Humidity: ${w.humidity}%, Wind: ${w.wind_speed} km/h.`;
  autoResize();
}

async function fetchWeatherModal(payload) {
  weatherResultEl2.style.display='none'; weatherErrorEl2.style.display='none';
  weatherLoadingEl2.style.display='block'; weatherSearchBtn2.disabled=true;
  try {
    const r = await fetch('/api/weather',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const d = await r.json(); weatherLoadingEl2.style.display='none';
    if(d.weather) renderWeather(d.weather);
    else { weatherErrorEl2.textContent=d.error||'Could not fetch weather.'; weatherErrorEl2.style.display='block'; }
  } catch(e) { weatherLoadingEl2.style.display='none'; weatherErrorEl2.textContent='Error: '+e.message; weatherErrorEl2.style.display='block'; }
  finally { weatherSearchBtn2.disabled=false; }
}
if (weatherSearchBtn2) weatherSearchBtn2.addEventListener('click', () => { const loc=weatherCityEl2.value.trim(); if(loc) fetchWeatherModal({location:loc}); });
if (weatherCityEl2) weatherCityEl2.addEventListener('keydown', e => { if(e.key==='Enter') weatherSearchBtn2.click(); });
if (weatherCloseBtn2) weatherCloseBtn2.addEventListener('click', () => weatherModal2.style.display='none');
if (weatherModal2) weatherModal2.addEventListener('click', e => { if(e.target===weatherModal2) weatherModal2.style.display='none'; });
if (weatherLocBtn2) weatherLocBtn2.addEventListener('click', () => {
  if (!navigator.geolocation) { alert('Geolocation not supported'); return; }
  navigator.geolocation.getCurrentPosition(
    pos => fetchWeatherModal({lat:pos.coords.latitude,lon:pos.coords.longitude}),
    err => alert('Location error: '+err.message)
  );
});
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


NANOBANANA_API_KEY = os.environ.get("NANOBANANA_API_KEY", "api-18a42424f3820dd304526a924b7bcef9")

@app.route("/api/generate-image", methods=["POST"])
@login_required
def generate_image():
    """Generate or edit images using NanoBanana API (Gemini 3.1 Flash Image).
    Supports:
    - text-to-image: just send prompt
    - image-to-image: send prompt + reference_image (base64) for Ghibli selfie etc.
    Falls back to Pollinations.ai if NanoBanana fails.
    """
    import urllib.parse, random
    d = request.get_json(force=True) or {}
    prompt   = (d.get("prompt") or "").strip()
    style    = (d.get("style") or "").strip()
    ref_b64  = d.get("reference_image")   # base64 image for image-to-image
    ref_mime = d.get("reference_mime", "image/jpeg")

    if not prompt:
        return jsonify({"error": "prompt required"}), 400

    # Build full prompt with style
    quality = "masterpiece, best quality, ultra detailed, sharp focus, cinematic lighting"
    full_prompt = f"{prompt}, {style} style, {quality}" if style else f"{prompt}, {quality}"

    # ── 1. Try NanoBanana API ────────────────────────────────────────────────
    if NANOBANANA_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {NANOBANANA_API_KEY}",
                "Content-Type": "application/json",
            }
            body = {
                "prompt": full_prompt,
                "model": "nano-banana",
                "response_format": "b64_json",
                "size": "1024x1024",
            }
            # Image-to-image mode: include reference image
            if ref_b64:
                body["image"] = f"data:{ref_mime};base64,{ref_b64}"

            resp = requests.post(
                "https://nanobananaapi.ai/api/generate",
                headers=headers,
                json=body,
                timeout=90,
            )
            if resp.status_code == 200:
                rj = resp.json()
                # Response: {"data": [{"b64_json": "..."}]} or {"data": [{"url": "..."}]}
                data_list = rj.get("data", [])
                if data_list:
                    item = data_list[0]
                    if item.get("b64_json"):
                        return jsonify({"image": item["b64_json"], "mime": "image/png"})
                    elif item.get("url"):
                        # Download the image from URL
                        img_resp = requests.get(item["url"], timeout=30)
                        if img_resp.status_code == 200:
                            return jsonify({
                                "image": base64.b64encode(img_resp.content).decode(),
                                "mime": img_resp.headers.get("content-type", "image/png")
                            })
            else:
                print(f"[NanoBanana] error {resp.status_code}: {resp.text[:300]}")
        except Exception as e:
            print(f"[NanoBanana] exception: {e}")

    # ── 2. Fallback: Pollinations.ai (free, no key) ─────────────────────────
    try:
        encoded = urllib.parse.quote(full_prompt)
        seed = random.randint(1, 999999)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&model=flux&nologo=true&enhance=true&seed={seed}&nofeed=true"
        resp = requests.get(url, timeout=60, headers={"User-Agent": "MythicAI/1.0"})
        if resp.status_code == 200 and resp.headers.get("content-type","").startswith("image/") and len(resp.content) > 10000:
            return jsonify({"image": base64.b64encode(resp.content).decode(), "mime": resp.headers.get("content-type","image/jpeg")})
    except Exception as e:
        print(f"[Pollinations] exception: {e}")

    return jsonify({"error": "Image generation failed. Please try again."}), 502


@app.route("/api/generate-image-edit", methods=["POST"])
@login_required
def generate_image_edit():
    """Image-to-image edit using NanoBanana — used for Ghibli selfie transformation."""
    import urllib.parse, random
    d = request.get_json(force=True) or {}
    prompt   = (d.get("prompt") or "").strip()
    ref_b64  = d.get("image")
    ref_mime = d.get("mime", "image/jpeg")

    if not prompt or not ref_b64:
        return jsonify({"error": "prompt and image required"}), 400

    if NANOBANANA_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {NANOBANANA_API_KEY}",
                "Content-Type": "application/json",
            }
            resp = requests.post(
                "https://nanobananaapi.ai/api/generate",
                headers=headers,
                json={
                    "prompt": prompt,
                    "model": "nano-banana",
                    "image": f"data:{ref_mime};base64,{ref_b64}",
                    "response_format": "b64_json",
                    "size": "1024x1024",
                },
                timeout=90,
            )
            if resp.status_code == 200:
                rj = resp.json()
                data_list = rj.get("data", [])
                if data_list:
                    item = data_list[0]
                    if item.get("b64_json"):
                        return jsonify({"image": item["b64_json"], "mime": "image/png"})
                    elif item.get("url"):
                        img_resp = requests.get(item["url"], timeout=30)
                        if img_resp.status_code == 200:
                            return jsonify({"image": base64.b64encode(img_resp.content).decode(), "mime": "image/png"})
            print(f"[NanoBanana Edit] error {resp.status_code}: {resp.text[:300]}")
        except Exception as e:
            print(f"[NanoBanana Edit] exception: {e}")

    # Fallback: describe-then-generate
    try:
        encoded = urllib.parse.quote(prompt)
        seed = random.randint(1, 999999)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&model=flux&nologo=true&enhance=true&seed={seed}&nofeed=true"
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200 and resp.headers.get("content-type","").startswith("image/"):
            return jsonify({"image": base64.b64encode(resp.content).decode(), "mime": "image/png"})
    except Exception as e:
        print(f"[Pollinations fallback] {e}")

    return jsonify({"error": "Image edit failed. Please try again."}), 502


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
