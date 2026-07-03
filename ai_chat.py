# Mobile-friendly CSS injection helper
MOBILE_CSS = '''
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{box-sizing:border-box}
body{margin:0}
@media (max-width:768px){
.sidebar{width:100%!important;position:relative!important}
.chat-container,.main-content{width:100%!important;margin:0!important;padding:10px!important}
.message{max-width:95%!important;font-size:16px!important}
input,textarea,button{font-size:16px!important}
}
</style>
'''

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
    Flask, request, jsonify, Response, session, redirect,
    url_for, stream_with_context
)
from werkzeug.security import generate_password_hash, check_password_hash

PROVIDER = os.environ.get("AI_PROVIDER", "auto").strip().lower()
# "auto"        = try Gemini → Groq → OpenRouter → HuggingFace in order (recommended)
# "gemini"      = Google Gemini only
# "groq"        = Groq only
# "openrouter"  = OpenRouter only
# "huggingface" = Hugging Face Inference API only
# "ollama"      = local Ollama only (no internet/key needed)

# --- API Keys (hardcoded fallbacks — override via environment variables) ------
# WARNING: don't commit a file with real keys to a public GitHub repo.
# Set these as environment variables on Render instead.
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY",    "AQ.Ab8RN6IilW_rW7qo0jh1JKfH1Hw3XVMxiFXA8y3aJZ5LyIH0pg")
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY",      "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")
OPENROUTER_API_KEY= os.environ.get("OPENROUTER_API_KEY","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")
HF_API_KEY        = os.environ.get("HF_API_KEY",        "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")

# --- Model names -------------------------------------------------------------
GEMINI_MODEL      = "gemini-2.5-flash"
GROQ_MODEL        = os.environ.get("GROQ_MODEL",        "llama-3.3-70b-versatile")
OPENROUTER_MODEL  = os.environ.get("OPENROUTER_MODEL",  "mistralai/mistral-7b-instruct:free")
HF_MODEL          = os.environ.get("HF_MODEL",          "mistralai/Mistral-7B-Instruct-v0.3")
OLLAMA_MODEL      = os.environ.get("OLLAMA_MODEL",       "llama3.1")
OLLAMA_URL        = os.environ.get("OLLAMA_URL",         "http://localhost:11434").rstrip("/")

GEMINI_STREAM_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:streamGenerateContent"
)

# keep old name for compatibility with existing references below
API_KEY = GEMINI_API_KEY
MODEL   = GEMINI_MODEL
SYSTEM_PROMPT = (
    "You are Aarav AI, a friendly, helpful AI assistant created by Aarav Singh. "
    "If asked who made you, who your creator/owner/developer is, or who you belong to, "
    "always say you were created by Aarav Singh — never mention Google, OpenAI, Anthropic, "
    "or any other company as your creator. "
    "You can chat about anything — answer questions, help with writing, brainstorm ideas, "
    "explain things, or just talk. Keep a warm, casual, conversational tone, like chatting "
    "with a knowledgeable friend. Be clear and helpful without being overly formal or stiff. "
    "You have access to a Google Search tool and can look up current information when needed. "
    "You can also see images and files the user attaches."
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

def load_users():
    if not SUPABASE_URL:
        return _load_users_file()
    try:
        r = requests.get(sb("users"), headers=sb_headers(), timeout=10)
        if r.status_code == 200:
            return {u["username"]: u for u in r.json()}
    except Exception:
        pass
    return {}

def save_user(username, password_hash):
    if not SUPABASE_URL:
        users = _load_users_file()
        users[username] = {"username": username, "password_hash": password_hash}
        _save_users_file(users)
        return
    try:
        requests.post(sb("users"), headers=sb_headers(),
            json={"username": username, "password_hash": password_hash}, timeout=10)
    except Exception:
        pass

# File fallbacks (used locally when SUPABASE_URL not set)
import os as _os
_BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_DATA_DIR = _os.path.join(_BASE_DIR, "chat_data")
_os.makedirs(_DATA_DIR, exist_ok=True)
_USERS_FILE = _os.path.join(_DATA_DIR, "users.json")

def _load_users_file():
    if not _os.path.exists(_USERS_FILE):
        return {}
    try:
        with open(_USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_users_file(users):
    with open(_USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def save_users(users):
    """Legacy helper kept for compatibility — saves all users at once."""
    if not SUPABASE_URL:
        _save_users_file(users)
        return
    for username, u in users.items():
        save_user(username, u["password_hash"])


def current_username():
    return session.get("username")


def login_required(view):
    def wrapped(*args, **kwargs):
        if not current_username():
            if request.path.startswith("/api/"):
                return jsonify({"error": "not logged in"}), 401
            return redirect(url_for("login_page"))
        return view(*args, **kwargs)
    wrapped.__name__ = view.__name__
    return wrapped


# --- Persistent per-user, multi-conversation storage (Supabase) --------------

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

AUTH_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Aarav AI — {mode_title}</title>
<style>
  :root {{ --bg:#0e0f11; --panel:#17181b; --border:#2a2c30; --text:#e8e6e1;
    --muted:#8b8d92; --accent:#d97757; }}
  * {{ box-sizing:border-box; }}
  html,body {{ height:100%; margin:0; background:var(--bg); color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;
    display:flex; align-items:center; justify-content:center; }}
  .card {{ background:var(--panel); border:1px solid var(--border); border-radius:14px;
    padding:32px; width:100%; max-width:340px; }}
  .card h1 {{ font-size:17px; margin:0 0 4px; }}
  .card .dot {{ display:inline-block; width:8px; height:8px; border-radius:50%;
    background:var(--accent); margin-right:8px; }}
  .card p.sub {{ color:var(--muted); font-size:13px; margin:0 0 22px; }}
  label {{ font-size:12.5px; color:var(--muted); display:block; margin-bottom:6px; }}
  input {{ width:100%; background:#0e0f11; border:1px solid var(--border); color:var(--text);
    border-radius:8px; padding:10px 12px; font-size:14px; }}
  input:focus {{ outline:none; border-color:var(--accent); }}
  .pw-wrap {{ position:relative; margin-bottom:16px; }}
  .pw-wrap input {{ padding-right:40px; margin-bottom:0; }}
  .pw-toggle {{ position:absolute; right:8px; top:50%; transform:translateY(-50%);
    background:none; border:none; color:var(--muted); cursor:pointer; font-size:15px;
    padding:4px 6px; width:auto; }}
  .pw-toggle:hover {{ color:var(--text); }}
  .pw-toggle svg {{ width:18px; height:18px; display:block; }}
  .username-wrap {{ margin-bottom:16px; }}
  button[type="submit"] {{ width:100%; background:var(--accent); color:#1a1a1a; border:none; border-radius:8px;
    padding:11px; font-size:14px; font-weight:600; cursor:pointer; }}
  button[type="submit"]:hover {{ opacity:0.9; }}
  .switch {{ text-align:center; margin-top:16px; font-size:13px; color:var(--muted); }}
  .switch a {{ color:var(--accent); text-decoration:none; }}
  .error {{ background:#3a1f1f; border:1px solid #6b3030; color:#f0b8b8; font-size:13px;
    padding:9px 12px; border-radius:8px; margin-bottom:16px; }}
</style>
</head>
<body>
  <div class="card">
    <h1><span class="dot"></span>Aarav AI</h1>
    <p class="sub">{mode_title}</p>
    {error_html}
    <form method="POST">
      <label>Username</label>
      <div class="username-wrap">
        <input type="text" name="username" required autofocus>
      </div>
      <label>Password</label>
      <div class="pw-wrap">
        <input type="password" name="password" id="pw-field" required
          autocomplete="new-password" data-lpignore="true" data-1p-ignore
          data-form-type="other" data-bwignore="true">
        <button type="button" class="pw-toggle" id="pw-toggle" title="Show password">
          <svg id="eye-open" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/><circle cx="12" cy="12" r="3"/></svg>
          <svg id="eye-closed" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="display:none"><path d="M17.94 17.94A10.94 10.94 0 0 1 12 20c-7 0-11-7-11-7a20.29 20.29 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 7 11 7a20.29 20.29 0 0 1-4 5.19M14.12 14.12a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>
        </button>
      </div>
      <button type="submit">{mode_title}</button>
    </form>
    <div class="switch">{switch_html}</div>
  </div>
  <script>
    const pwField = document.getElementById('pw-field');
    const pwToggle = document.getElementById('pw-toggle');
    const eyeOpen = document.getElementById('eye-open');
    const eyeClosed = document.getElementById('eye-closed');
    pwToggle.addEventListener('click', () => {{
      const isHidden = pwField.type === 'password';
      pwField.type = isHidden ? 'text' : 'password';
      eyeOpen.style.display = isHidden ? 'none' : 'block';
      eyeClosed.style.display = isHidden ? 'block' : 'none';
      pwToggle.title = isHidden ? 'Hide password' : 'Show password';
    }});
  </script>
</body>
</html>
"""


@app.route("/login", methods=["GET", "POST"])
def login_page():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        users = load_users()
        user = users.get(username)
        if user and check_password_hash(user["password_hash"], password):
            session["username"] = username
            session.permanent = True
            return redirect(url_for("index"))
        error = "Wrong username or password."
    error_html = f'<div class="error">{error}</div>' if error else ""
    switch_html = 'New here? <a href="/register">Create an account</a>'
    return Response(
        AUTH_PAGE.format(mode_title="Log in", error_html=error_html, switch_html=switch_html),
        mimetype="text/html",
    )


@app.route("/register", methods=["GET", "POST"])
def register_page():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if len(username) < 3:
            error = "Username must be at least 3 characters."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        else:
            users = load_users()
            if username in users:
                error = "That username is already taken."
            else:
                users[username] = {"password_hash": generate_password_hash(password)}
                save_users(users)
                session["username"] = username
                session.permanent = True
                return redirect(url_for("index"))
    error_html = f'<div class="error">{error}</div>' if error else ""
    switch_html = 'Already have an account? <a href="/login">Log in</a>'
    return Response(
        AUTH_PAGE.format(mode_title="Sign up", error_html=error_html, switch_html=switch_html),
        mimetype="text/html",
    )


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login_page"))


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Aarav AI</title>
<style>
  :root {
    --bg: #0e0f11; --panel: #17181b; --border: #2a2c30;
    --text: #e8e6e1; --muted: #8b8d92; --accent: #d97757;
    --accent-dim: #4a2f26; --user-bubble: #262830; --ai-bubble: #1c1e22;
    --sidebar-w: 260px;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; overflow: hidden; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif; }
  .layout { display: flex; height: 100vh; width: 100vw; }
  #sidebar { width: var(--sidebar-w); flex-shrink: 0; background: var(--panel);
    border-right: 1px solid var(--border); display: flex; flex-direction: column;
    transition: margin-left 0.2s ease; }
  #sidebar.hidden { margin-left: calc(-1 * var(--sidebar-w)); }
  #new-chat-btn { margin: 14px; padding: 10px 14px; background: var(--accent); color: #1a1a1a;
    border: none; border-radius: 8px; font-size: 13.5px; font-weight: 600; cursor: pointer; text-align: left; }
  #new-chat-btn:hover { opacity: 0.9; }
  #conv-list { flex: 1; overflow-y: auto; padding: 0 8px; display: flex; flex-direction: column; gap: 2px; }
  .conv-item { display: flex; align-items: center; justify-content: space-between; gap: 6px;
    padding: 9px 10px; border-radius: 7px; cursor: pointer; font-size: 13px; color: var(--muted);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .conv-item:hover { background: var(--ai-bubble); color: var(--text); }
  .conv-item.active { background: var(--user-bubble); color: var(--text); }
  .conv-item .title { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
  .conv-item .del-btn { opacity: 0; background: none; border: none; color: var(--muted);
    cursor: pointer; font-size: 13px; padding: 2px 5px; flex-shrink: 0; }
  .conv-item:hover .del-btn { opacity: 1; }
  .conv-item .del-btn:hover { color: #e88; }
  #sidebar-footer { padding: 12px; font-size: 11px; color: var(--muted); border-top: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center; }
  #sidebar-footer a { color: var(--muted); text-decoration: none; }
  #sidebar-footer a:hover { color: var(--text); }
  .app { display: flex; flex-direction: column; height: 100vh; flex: 1; min-width: 0; }
  header { padding: 18px 20px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between; gap: 10px; }
  header .left { display: flex; align-items: center; gap: 10px; min-width: 0; }
  #sidebar-toggle { background: none; border: 1px solid var(--border); color: var(--muted);
    width: 30px; height: 30px; border-radius: 6px; cursor: pointer; font-size: 14px; flex-shrink: 0; }
  #sidebar-toggle:hover { color: var(--text); }
  header h1 { font-size: 15px; font-weight: 600; letter-spacing: 0.02em; margin: 0;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  header .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: var(--accent); margin-right: 8px; flex-shrink: 0; }
  #clear-btn { background: none; border: 1px solid var(--border); color: var(--muted);
    font-size: 12px; padding: 6px 12px; border-radius: 6px; cursor: pointer; flex-shrink: 0; }
  #clear-btn:hover { color: var(--text); border-color: var(--muted); }
  #messages { flex: 1; overflow-y: auto; padding: 24px 20px; display: flex;
    flex-direction: column; gap: 16px; max-width: 760px; margin: 0 auto; width: 100%; }
  .msg { max-width: 82%; padding: 11px 15px; border-radius: 14px; line-height: 1.5;
    font-size: 14.5px; white-space: pre-wrap; word-wrap: break-word; }
  .msg.user { align-self: flex-end; background: var(--user-bubble); border-bottom-right-radius: 4px; }
  .msg.ai { align-self: flex-start; background: var(--ai-bubble); border: 1px solid var(--border);
    border-bottom-left-radius: 4px; }
  .msg.error { align-self: center; background: #3a1f1f; border: 1px solid #6b3030;
    color: #f0b8b8; font-size: 13px; }
  .msg img.attach-thumb { max-width: 220px; border-radius: 8px; display: block; margin-top: 8px; }
  .attach-chip { font-size: 11.5px; color: var(--muted); margin-bottom: 4px; }
  .empty-state { margin: auto; text-align: center; color: var(--muted); }
  .empty-state .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    background: var(--accent); margin-bottom: 14px; }
  .empty-state p { margin: 4px 0; font-size: 13.5px; }
  .typing { align-self: flex-start; display: flex; gap: 4px; padding: 14px 16px;
    background: var(--ai-bubble); border: 1px solid var(--border); border-radius: 14px;
    border-bottom-left-radius: 4px; }
  .typing span { width: 6px; height: 6px; border-radius: 50%; background: var(--muted);
    animation: blink 1.2s infinite ease-in-out; }
  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes blink { 0%, 80%, 100% { opacity: 0.25; } 40% { opacity: 1; } }
  #pending-attach { max-width: 760px; margin: 0 auto; width: 100%; padding: 0 20px;
    display: none; align-items: center; gap: 8px; font-size: 12.5px; color: var(--muted); }
  #pending-attach.show { display: flex; }
  #pending-attach button { background: none; border: none; color: var(--muted); cursor: pointer; font-size: 13px; }
  form { display: flex; gap: 8px; padding: 10px 20px 20px; border-top: 1px solid var(--border);
    max-width: 760px; margin: 0 auto; width: 100%; align-items: flex-end; }
  #attach-btn { background: none; border: 1px solid var(--border); color: var(--muted);
    width: 42px; height: 42px; border-radius: 10px; cursor: pointer; font-size: 16px; flex-shrink: 0; }
  #attach-btn:hover { color: var(--text); }
  textarea { flex: 1; resize: none; background: var(--panel); border: 1px solid var(--border);
    color: var(--text); border-radius: 10px; padding: 11px 13px; font-size: 14.5px;
    font-family: inherit; line-height: 1.4; max-height: 140px; }
  textarea:focus { outline: none; border-color: var(--accent); }
  button[type="submit"] { background: var(--accent); color: #1a1a1a; border: none;
    border-radius: 10px; padding: 0 20px; font-size: 14px; font-weight: 600; cursor: pointer; height: 42px; }
  button[type="submit"]:disabled { background: var(--accent-dim); color: var(--muted); cursor: not-allowed; }
  #messages::-webkit-scrollbar, #conv-list::-webkit-scrollbar { width: 8px; }
  #messages::-webkit-scrollbar-thumb, #conv-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
  @media (max-width: 640px) {
    #sidebar { position: fixed; top: 0; left: 0; z-index: 10; height: 100vh; }
    #sidebar.hidden { margin-left: calc(-1 * var(--sidebar-w)); }
  }
</style>
</head>
<body>
<div class="layout">
  <div id="sidebar">
    <button id="new-chat-btn" type="button">+ New chat</button>
    <div id="conv-list"></div>
    <div id="sidebar-footer">
      <span>__USERNAME__</span>
      <a href="/logout">Log out</a>
    </div>
  </div>

  <div class="app">
    <header>
      <div class="left">
        <button id="sidebar-toggle" type="button" title="Toggle sidebar">☰</button>
        <h1><span class="dot"></span>Aarav AI</h1>
      </div>
      <button id="clear-btn" type="button">Delete chat</button>
    </header>

    <div id="messages">
      <div class="empty-state" id="empty-state">
        <div class="dot"></div>
        <p>Hey! Ask me anything, attach a file, or just say hi 👋</p>
      </div>
    </div>

    <div id="pending-attach">
      <span id="pending-attach-name"></span>
      <button id="pending-attach-remove" type="button">✕</button>
    </div>

    <form id="chat-form">
      <input type="file" id="file-input" accept="image/*,.txt,.md,.csv,.json,.pdf" style="display:none">
      <button id="attach-btn" type="button" title="Attach file">📎</button>
      <textarea id="input" rows="1" placeholder="Message..." autofocus></textarea>
      <button type="submit" id="send-btn">Send</button>
    </form>
  </div>
</div>

<script>
  const messagesEl = document.getElementById('messages');
  const form = document.getElementById('chat-form');
  const input = document.getElementById('input');
  const sendBtn = document.getElementById('send-btn');
  const clearBtn = document.getElementById('clear-btn');
  const convListEl = document.getElementById('conv-list');
  const newChatBtn = document.getElementById('new-chat-btn');
  const sidebarToggle = document.getElementById('sidebar-toggle');
  const sidebar = document.getElementById('sidebar');
  const fileInput = document.getElementById('file-input');
  const attachBtn = document.getElementById('attach-btn');
  const pendingAttach = document.getElementById('pending-attach');
  const pendingAttachName = document.getElementById('pending-attach-name');
  const pendingAttachRemove = document.getElementById('pending-attach-remove');

  let activeConvId = null;
  let pendingFile = null; // {name, mimeType, dataBase64}

  function scrollToBottom() {
    requestAnimationFrame(() => { messagesEl.scrollTop = messagesEl.scrollHeight; });
  }
  function clearEmptyState() {
    const es = document.getElementById('empty-state');
    if (es) es.remove();
  }
  function showEmptyState() {
    messagesEl.innerHTML = '<div class="empty-state" id="empty-state"><div class="dot"></div><p>Hey! Ask me anything, attach a file, or just say hi 👋</p></div>';
  }

  function addMessage(role, text, attachment) {
    clearEmptyState();
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    if (attachment && attachment.mimeType && attachment.mimeType.startsWith('image/')) {
      const chip = document.createElement('div');
      chip.className = 'attach-chip';
      chip.textContent = '📎 ' + attachment.name;
      div.appendChild(chip);
      const img = document.createElement('img');
      img.className = 'attach-thumb';
      img.src = 'data:' + attachment.mimeType + ';base64,' + attachment.dataBase64;
      div.appendChild(img);
    } else if (attachment) {
      const chip = document.createElement('div');
      chip.className = 'attach-chip';
      chip.textContent = '📎 ' + attachment.name;
      div.appendChild(chip);
    }
    const textNode = document.createElement('div');
    textNode.textContent = text;
    div.appendChild(textNode);
    messagesEl.appendChild(div);
    scrollToBottom();
    return textNode;
  }

  function showTyping() {
    const div = document.createElement('div');
    div.className = 'typing';
    div.id = 'typing-indicator';
    div.innerHTML = '<span></span><span></span><span></span>';
    messagesEl.appendChild(div);
    scrollToBottom();
  }
  function hideTyping() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
  }

  async function loadConversationList() {
    try {
      const res = await fetch('/api/conversations');
      const data = await res.json();
      renderConvList(data.conversations || []);
      return data.conversations || [];
    } catch (err) {
      console.error('Could not load conversations:', err);
      return [];
    }
  }

  function renderConvList(convs) {
    convListEl.innerHTML = '';
    convs.forEach(c => {
      const item = document.createElement('div');
      item.className = 'conv-item' + (c.id === activeConvId ? ' active' : '');
      item.innerHTML = '<span class="title"></span><button class="del-btn" title="Delete">✕</button>';
      item.querySelector('.title').textContent = c.title;
      item.addEventListener('click', (e) => {
        if (e.target.classList.contains('del-btn')) return;
        openConversation(c.id);
      });
      item.querySelector('.del-btn').addEventListener('click', async (e) => {
        e.stopPropagation();
        await fetch('/api/conversations/' + c.id, { method: 'DELETE' });
        if (c.id === activeConvId) { await startNewChat(); } else { loadConversationList(); }
      });
      convListEl.appendChild(item);
    });
  }

  async function openConversation(convId) {
    activeConvId = convId;
    try {
      const res = await fetch('/api/conversations/' + convId);
      if (!res.ok) return;
      const data = await res.json();
      messagesEl.innerHTML = '';
      if (data.messages && data.messages.length > 0) {
        data.messages.forEach(m => addMessage(m.role, m.text, m.attachment));
      } else {
        showEmptyState();
      }
      loadConversationList();
    } catch (err) {
      console.error('Could not open conversation:', err);
    }
  }

  async function startNewChat() {
    activeConvId = null;
    messagesEl.innerHTML = '';
    showEmptyState();
    loadConversationList();
  }

  attachBtn.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (!file) return;
    if (file.size > 8 * 1024 * 1024) {
      alert('File is too big (max 8MB).');
      fileInput.value = '';
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      pendingFile = {
        name: file.name,
        mimeType: file.type || 'application/octet-stream',
        dataBase64: reader.result.split(',')[1]
      };
      pendingAttachName.textContent = '📎 ' + file.name;
      pendingAttach.classList.add('show');
    };
    reader.readAsDataURL(file);
  });

  pendingAttachRemove.addEventListener('click', () => {
    pendingFile = null;
    fileInput.value = '';
    pendingAttach.classList.remove('show');
  });

  async function sendMessage(text, attachment) {
    showTyping();
    sendBtn.disabled = true;
    let aiTextNode = null;
    let fullText = '';
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          conversation_id: activeConvId,
          attachment: attachment || null
        })
      });
      if (!res.ok) {
        hideTyping();
        let errMsg = 'Something went wrong.';
        try { const errData = await res.json(); errMsg = errData.error || errMsg; } catch (e) {}
        addMessage('error', errMsg);
        return;
      }
      const convId = res.headers.get('X-Conversation-Id');
      if (convId) activeConvId = convId;

      hideTyping();
      aiTextNode = addMessage('ai', '');

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        fullText += chunk;
        aiTextNode.textContent = fullText;
        scrollToBottom();
      }
      loadConversationList();
    } catch (err) {
      hideTyping();
      addMessage('error', 'Could not reach the server: ' + err.message);
    } finally {
      sendBtn.disabled = false;
    }
  }

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text && !pendingFile) return;
    const attachment = pendingFile;
    addMessage('user', text, attachment);
    input.value = '';
    input.style.height = 'auto';
    pendingFile = null;
    fileInput.value = '';
    pendingAttach.classList.remove('show');
    sendMessage(text, attachment);
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 140) + 'px';
  });

  clearBtn.addEventListener('click', async () => {
    if (!activeConvId) return;
    await fetch('/api/conversations/' + activeConvId, { method: 'DELETE' });
    await startNewChat();
  });

  newChatBtn.addEventListener('click', () => startNewChat());
  sidebarToggle.addEventListener('click', () => sidebar.classList.toggle('hidden'));

  (async () => {
    const convs = await loadConversationList();
    if (convs.length > 0) { openConversation(convs[0].id); } else { showEmptyState(); }
  })();
</script>
</body>
</html>
"""


@app.route("/")
@login_required
def index():
    return Response(PAGE.replace("__USERNAME__", current_username()), mimetype="text/html")


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
                    content = obj["choices"][0]["delta"].get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            return
    except requests.RequestException:
        pass


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
                    content = obj["choices"][0]["delta"].get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            return
    except requests.RequestException:
        pass


def auto_stream_chunks(gemini_payload, gemini_messages):
    """Auto fallback chain: Gemini → Groq → OpenRouter → HuggingFace.
    Tries each provider in order; falls through to the next if one fails/has no key."""
    openai_msgs = to_openai_messages(gemini_messages, SYSTEM_PROMPT)

    providers = []
    if PROVIDER in ("auto", "gemini"):
        providers.append(("Gemini", lambda: gemini_stream_chunks(gemini_payload)))
    if PROVIDER in ("auto", "groq"):
        providers.append(("Groq", lambda: groq_stream_chunks(openai_msgs)))
    if PROVIDER in ("auto", "openrouter"):
        providers.append(("OpenRouter", lambda: openrouter_stream_chunks(openai_msgs)))
    if PROVIDER in ("auto", "huggingface"):
        providers.append(("HuggingFace", lambda: huggingface_stream_chunks(openai_msgs)))
    if PROVIDER == "ollama":
        providers.append(("Ollama", lambda: ollama_stream_chunks(to_ollama_messages(gemini_messages, SYSTEM_PROMPT))))

    for name, fn in providers:
        collected = []
        try:
            for chunk in fn():
                collected.append(chunk)
                yield chunk
            if collected:
                return  # got a real response — done
        except Exception:
            pass
        # nothing yielded from this provider — try next

    yield "[All AI providers failed or have no API keys configured. Add keys to try again.]"


def gemini_stream_chunks(payload):
    """Yields plain text increments from Gemini's SSE stream."""
    try:
        resp = requests.post(
            GEMINI_STREAM_URL,
            params={"key": API_KEY, "alt": "sse"},
            json=payload,
            stream=True,
            timeout=60,
        )
    except requests.RequestException as e:
        yield f"[Error contacting Gemini: {e}]"
        return

    if resp.status_code != 200:
        try:
            body = resp.json()
            err_msg = body.get("error", {}).get("message", resp.text)
        except ValueError:
            err_msg = resp.text
        yield f"[Gemini error: {err_msg}]"
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

    if not user_message and not attachment:
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
        conv_id = str(uuid.uuid4())
        conv = {"title": make_title(user_message), "messages": []}

    messages = conv.setdefault("messages", [])

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

    payload = {
        "contents": gemini_contents,
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "tools": [{"google_search": {}}],
    }

    def generate():
        full_reply = []
        if PROVIDER == "ollama":
            chunk_source = ollama_stream_chunks(to_ollama_messages(messages, SYSTEM_PROMPT))
        else:
            chunk_source = auto_stream_chunks(payload, messages)

        for chunk in chunk_source:
            full_reply.append(chunk)
            yield chunk
        messages.append({"role": "model", "parts": [{"text": "".join(full_reply)}]})
        save_conversation(username, conv_id, conv)

    resp = Response(stream_with_context(generate()), mimetype="text/plain")
    resp.headers["X-Conversation-Id"] = conv_id
    return resp


if __name__ == "__main__":
    active = []
    if PROVIDER in ("auto", "gemini") and GEMINI_API_KEY:
        active.append(f"Gemini({GEMINI_MODEL})")
    if PROVIDER in ("auto", "groq") and GROQ_API_KEY:
        active.append(f"Groq({GROQ_MODEL})")
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
