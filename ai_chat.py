"""
Mythic AI — single file, powered by Google's Gemini API or a local Ollama model.
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

GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY",    "")
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY",      "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")
CEREBRAS_API_KEY  = os.environ.get("CEREBRAS_API_KEY",  "csk-2ph5f5nxt3jrtwj5edcehpr5xh96628268fvjh4e658m4t6h")
OPENROUTER_API_KEY= os.environ.get("OPENROUTER_API_KEY","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")
HF_API_KEY        = os.environ.get("HF_API_KEY",        "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")
NEWS_API_KEY      = os.environ.get("NEWS_API_KEY",      "344a953f2d08489a865239c2f9f030e4")
NANO_BANANA_API_KEY = os.environ.get("NANO_BANANA_API_KEY", "52b727c25b537301bd162fe540c5267e")

VIP_PASSWORD = os.environ.get("VIP_PASSWORD", "1254")
VIP_MODELS   = {"mythic-vip"}

MYTHIC_MODELS = {
    "mythic-1":   ("groq",      "llama-3.1-8b-instant",      "Fast",     False),
    "mythic-2":   ("groq",      "llama-3.3-70b-versatile",   "Smart",    False),
    "mythic-pro": ("cerebras",  "llama-3.3-70b",             "Advanced", False),
    "mythic-vip": ("openrouter","meta-llama/llama-3.3-70b-instruct:free","Most Powerful 🔒",True),
}
DEFAULT_MODEL_ID = "mythic-2"

GEMINI_MODEL    = "gemini-2.5-flash"
GROQ_MODEL      = os.environ.get("GROQ_MODEL",     "llama-3.3-70b-versatile")
CEREBRAS_MODEL  = os.environ.get("CEREBRAS_MODEL", "llama-3.3-70b")
OPENROUTER_MODEL= os.environ.get("OPENROUTER_MODEL","meta-llama/llama-3.3-70b-instruct:free")
HF_MODEL        = os.environ.get("HF_MODEL",        "mistralai/Mistral-7B-Instruct-v0.3")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL",     "llama3.1")
OLLAMA_URL      = os.environ.get("OLLAMA_URL",       "http://localhost:11434").rstrip("/")

NANO_BANANA_MODEL = "gemini-2.0-flash-preview-image-generation"
NANO_BANANA_URL   = "https://api.nanobanana.ai/v1beta/models/" + NANO_BANANA_MODEL + ":generateContent"

GEMINI_STREAM_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:streamGenerateContent"
API_KEY = GEMINI_API_KEY
MODEL   = GEMINI_MODEL

SYSTEM_PROMPT = (
    "You are Mythic AI, a smart and friendly AI assistant made by Aarav Singh. "
    "If asked who made you, say you are Mythic AI made by Aarav Singh — say it once naturally, never repeat it unprompted. "
    "Never mention Google, Groq, Cerebras, OpenRouter, HuggingFace, Meta, Mistral, Anthropic, or any AI company as your creator or backend. "
    "You can help with anything: questions, writing, coding, math, ideas, or just chatting. "
    "When writing code, always wrap it in markdown code blocks with the language name. "
    "LANGUAGE: Always reply ENTIRELY in the same language the user writes in. "
    "English → English only. Hindi (Devanagari) → Hindi only. Hinglish → Hinglish only. NEVER mix. "
    "ANTI-REPETITION: Never restate what the user said. No filler like 'Great question' or 'Sure'. "
    "Be direct and natural — like a knowledgeable friend. Keep answers concise unless asked for detail."
)

GEMINI_SEARCH_ADDENDUM = (
    " WEB SEARCH: You have access to Google Search for current events, news, prices, sports scores. "
    "Use it when needed. Summarize results in the reply language."
)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-" + str(uuid.uuid4()))
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

def sb_headers():
    return {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json", "Prefer": "return=representation"}

def sb(path): return f"{SUPABASE_URL}/rest/v1/{path}"

def current_username():
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
        session.permanent = True
    return session["user_id"]

def login_required(view):
    def wrapped(*args, **kwargs):
        current_username()
        return view(*args, **kwargs)
    wrapped.__name__ = view.__name__
    return wrapped

def list_conversations(username):
    if not SUPABASE_URL:
        return _list_conversations_file(username)
    try:
        r = requests.get(sb(f"conversations?username=eq.{username}&order=updated_at.desc&select=id,title,updated_at"), headers=sb_headers(), timeout=10)
        if r.status_code == 200: return r.json()
    except: pass
    return []

def load_conversation(username, conv_id):
    if not SUPABASE_URL: return _load_conversation_file(username, conv_id)
    try:
        r = requests.get(sb(f"conversations?id=eq.{conv_id}&username=eq.{username}"), headers=sb_headers(), timeout=10)
        if r.status_code == 200 and r.json():
            row = r.json()[0]
            return {"title": row["title"], "updated_at": row["updated_at"],
                    "messages": row["messages"] if isinstance(row["messages"], list) else json.loads(row["messages"])}
    except: pass
    return None

def save_conversation(username, conv_id, data):
    data["updated_at"] = time.time()
    if not SUPABASE_URL: _save_conversation_file(username, conv_id, data); return
    try:
        h = {**sb_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"}
        requests.post(sb("conversations"), headers=h, json={"id": conv_id, "username": username,
            "title": data.get("title","New chat"), "updated_at": data["updated_at"], "messages": data.get("messages",[])}, timeout=15)
    except: pass

def delete_conversation(username, conv_id):
    if not SUPABASE_URL: _delete_conversation_file(username, conv_id); return
    try: requests.delete(sb(f"conversations?id=eq.{conv_id}&username=eq.{username}"), headers=sb_headers(), timeout=10)
    except: pass

import os as _os
_BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_DATA_DIR = _os.path.join(_BASE_DIR, "chat_data")
_os.makedirs(_DATA_DIR, exist_ok=True)

def _user_conv_dir(u):
    p = _os.path.join(_DATA_DIR, "conversations", u); _os.makedirs(p, exist_ok=True); return p
def _conv_file(u, c): return _os.path.join(_user_conv_dir(u), f"{c}.json")
def _list_conversations_file(u):
    folder = _user_conv_dir(u); convs = []
    for f in _os.listdir(folder):
        if not f.endswith(".json"): continue
        try:
            d = json.load(open(_os.path.join(folder,f)))
            convs.append({"id":f[:-5],"title":d.get("title","New chat"),"updated_at":d.get("updated_at",0)})
        except: pass
    return sorted(convs, key=lambda c: c["updated_at"], reverse=True)
def _load_conversation_file(u, c):
    p = _conv_file(u,c)
    if not _os.path.exists(p): return None
    try: return json.load(open(p))
    except: return None
def _save_conversation_file(u, c, data):
    with open(_conv_file(u,c),"w",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False,indent=2)
def _delete_conversation_file(u, c):
    p = _conv_file(u,c)
    if _os.path.exists(p): _os.remove(p)

def make_title(msg):
    if not msg or not msg.strip():
        return "New chat"
    t = msg.strip().replace("\n", " ")
    # Just use the first 40 chars of the user's actual message
    return t[:40] + ("…" if len(t) > 40 else "")

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<title>Mythic AI</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg:#1a1a1a; --panel:#2a2a2a; --border:#3a3a3a;
  --text:#ececec; --muted:#8e8ea0; --accent:#10a37f;
  --accent-dim:#1a3a30; --sidebar-w:260px;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--text);
  font-family:'Inter','Noto Sans Devanagari',-apple-system,sans-serif;overflow:hidden}
.layout{display:flex;height:100vh}
#sidebar{width:var(--sidebar-w);flex-shrink:0;background:var(--panel);border-right:1px solid var(--border);
  display:flex;flex-direction:column;transition:transform .25s ease}
#sidebar.hidden{transform:translateX(-105%)}
#new-chat-btn{margin:12px;padding:10px 14px;background:var(--accent);color:#fff;border:none;
  border-radius:8px;font-size:13.5px;font-weight:600;cursor:pointer;text-align:left}
#new-chat-btn:hover{opacity:.9}
#conv-list{flex:1;overflow-y:auto;padding:0 8px;display:flex;flex-direction:column;gap:2px}
.conv-item{display:flex;align-items:center;justify-content:space-between;gap:6px;
  padding:9px 10px;border-radius:7px;cursor:pointer;font-size:13px;color:var(--muted)}
.conv-item:hover,.conv-item.active{background:var(--accent-dim);color:var(--accent)}
.conv-item .title{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}
.conv-item .del-btn{opacity:0;background:none;border:none;color:var(--muted);cursor:pointer;
  font-size:13px;padding:2px 5px;flex-shrink:0}
.conv-item:hover .del-btn,.conv-item .del-btn{opacity:1}
.conv-item .del-btn:hover{color:#ef4444}
#sidebar-footer{padding:12px;font-size:11px;color:var(--muted);border-top:1px solid var(--border)}
.app{display:flex;flex-direction:column;height:100vh;flex:1;min-width:0}
header{padding:12px 16px;border-bottom:1px solid var(--border);display:flex;
  align-items:center;justify-content:space-between;gap:8px;background:var(--bg);
  padding-top:max(12px,env(safe-area-inset-top))}
header .left{display:flex;align-items:center;gap:8px;min-width:0}
header .right{display:flex;align-items:center;gap:6px;flex-shrink:0}
header h1{font-size:16px;font-weight:700;color:var(--accent);margin:0}
.hbtn{background:none;border:1px solid var(--border);color:var(--muted);width:36px;height:36px;
  border-radius:6px;cursor:pointer;font-size:15px;display:flex;align-items:center;
  justify-content:center;flex-shrink:0;touch-action:manipulation}
.hbtn:hover{background:var(--panel);color:var(--text)}
.hbtn.on{color:var(--accent);border-color:var(--accent)}
#clear-btn:hover{color:#ef4444;border-color:#ef4444}
#messages-wrap{flex:1;overflow-y:auto;position:relative;-webkit-overflow-scrolling:touch}
#messages{padding:20px 16px;display:flex;flex-direction:column;gap:14px;
  max-width:760px;margin:0 auto;width:100%;min-height:100%}
.msg-row{display:flex;flex-direction:column;max-width:82%}
.msg-row.user{align-self:flex-end;align-items:flex-end}
.msg-row.ai{align-self:flex-start;align-items:flex-start}
.msg-row.error{align-self:center;align-items:center;max-width:90%}
.msg{padding:10px 14px;border-radius:18px;line-height:1.6;font-size:14.5px;
  white-space:pre-wrap;word-wrap:break-word;max-width:100%}
.msg.user{background:var(--panel);color:var(--text);border-bottom-right-radius:4px}
.msg.ai{background:var(--bg);color:var(--text);border:1px solid var(--border);border-bottom-left-radius:4px}
.msg.error{background:#fef2f2;border:1px solid #fecaca;color:#dc2626;font-size:13px;border-radius:10px}
.msg img{max-width:100%;border-radius:10px;display:block;margin-top:8px}
.attach-chip{font-size:11.5px;opacity:.75;margin-bottom:4px}
.msg-actions{display:flex;gap:4px;margin-top:3px;opacity:0;transition:opacity .15s;height:24px}
.msg-row:hover .msg-actions,.msg-row:focus-within .msg-actions{opacity:1}
.msg-actions button{background:none;border:none;color:var(--muted);cursor:pointer;
  font-size:12px;padding:3px 8px;border-radius:5px}
.msg-actions button:hover{background:var(--panel);color:var(--text)}
.empty-state{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  text-align:center;color:var(--muted);width:90%}
.empty-state h2{font-size:22px;font-weight:700;color:var(--accent);margin-bottom:8px}
.typing{align-self:flex-start;display:flex;gap:5px;padding:14px 16px;
  background:var(--bg);border:1px solid var(--border);border-radius:18px;border-bottom-left-radius:4px}
.typing span{width:7px;height:7px;border-radius:50%;background:var(--muted);animation:blink 1.2s infinite ease-in-out}
.typing span:nth-child(2){animation-delay:.2s}
.typing span:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,80%,100%{opacity:.2}40%{opacity:1}}
#scroll-btn{position:fixed;bottom:130px;right:20px;width:36px;height:36px;border-radius:50%;
  background:var(--accent);color:#fff;border:none;cursor:pointer;font-size:18px;
  display:none;align-items:center;justify-content:center;box-shadow:0 2px 8px rgba(0,0,0,.3);z-index:10}
#scroll-btn.show{display:flex}
#pending-attach{max-width:760px;margin:0 auto;width:100%;padding:6px 16px 0;
  display:none;align-items:center;gap:8px;font-size:12.5px;color:var(--muted)}
#pending-attach.show{display:flex}
#pending-attach button{background:none;border:none;color:var(--muted);cursor:pointer}
#speaking-bar{display:none;align-items:center;gap:8px;font-size:12px;color:var(--accent);
  padding:4px 16px;max-width:760px;margin:0 auto;width:100%}
#speaking-bar.show{display:flex}
#stop-speak-btn{background:none;border:1px solid var(--border);color:var(--muted);
  font-size:11px;padding:2px 8px;border-radius:4px;cursor:pointer}
#quick-actions{display:flex;gap:6px;padding:6px 16px 0;max-width:760px;
  margin:0 auto;width:100%;flex-wrap:nowrap;overflow-x:auto;-webkit-overflow-scrolling:touch}
#quick-actions::-webkit-scrollbar{display:none}
.quick-btn{background:var(--panel);border:1px solid var(--border);color:var(--text);
  font-size:12px;padding:6px 12px;border-radius:20px;cursor:pointer;white-space:nowrap;
  font-family:inherit;flex-shrink:0}
.quick-btn:hover{background:var(--accent-dim);border-color:var(--accent);color:var(--accent)}
.input-wrap{max-width:760px;margin:0 auto;width:100%;padding:8px 16px max(12px,env(safe-area-inset-bottom))}
.input-row{display:flex;gap:8px;align-items:flex-end;background:var(--panel);
  border:1.5px solid var(--border);border-radius:14px;padding:8px 10px}
.input-row:focus-within{border-color:var(--accent)}
.tool-btn{background:none;border:none;color:var(--muted);cursor:pointer;width:36px;height:36px;
  border-radius:8px;font-size:18px;flex-shrink:0;display:flex;align-items:center;
  justify-content:center;touch-action:manipulation}
.tool-btn:hover{background:var(--accent-dim);color:var(--accent)}
.tool-btn.listening{color:#ef4444;animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
textarea{flex:1;resize:none;background:transparent;border:none;color:var(--text);
  font-size:14.5px;font-family:inherit;line-height:1.4;max-height:140px;outline:none;padding:4px 0}
textarea::placeholder{color:var(--muted)}
#send-btn{background:var(--accent);color:#fff;border:none;border-radius:10px;width:36px;height:36px;
  cursor:pointer;flex-shrink:0;display:flex;align-items:center;justify-content:center;touch-action:manipulation}
#send-btn:disabled{opacity:.4;cursor:not-allowed}
#send-btn.stop-mode{background:#ef4444}
#messages-wrap::-webkit-scrollbar,#conv-list::-webkit-scrollbar{width:6px}
#messages-wrap::-webkit-scrollbar-thumb,#conv-list::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
#sidebar-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:99}
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:300;
  align-items:center;justify-content:center;padding:16px}
.modal-overlay.show{display:flex}
.modal-box{background:var(--panel);border:1px solid var(--border);border-radius:16px;
  padding:22px;width:100%;max-width:420px;max-height:90vh;overflow-y:auto}
.modal-box h3{margin:0 0 6px;font-size:17px}
.modal-box p{color:var(--muted);font-size:13px;margin:0 0 14px}
.modal-input{width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);
  border-radius:10px;padding:10px 12px;font-size:14px;font-family:inherit;outline:none}
.modal-input:focus{border-color:var(--accent)}
.modal-btn-row{display:flex;gap:8px;margin-top:12px}
.btn-primary{flex:1;background:var(--accent);color:#fff;border:none;border-radius:8px;
  padding:10px;font-size:14px;font-weight:600;cursor:pointer;font-family:inherit}
.btn-secondary{flex:1;background:none;border:1px solid var(--border);color:var(--muted);
  border-radius:8px;padding:10px;font-size:14px;cursor:pointer;font-family:inherit}
.name-modal{background:var(--bg);border:1px solid var(--border);border-radius:14px;
  padding:22px;width:90%;max-width:360px}
@media(max-width:768px){
  :root{--sidebar-w:80vw}
  #sidebar{position:fixed;top:0;left:0;z-index:100;height:100%;height:-webkit-fill-available;
    box-shadow:4px 0 24px rgba(0,0,0,.5)}
  #sidebar-overlay{display:block}
  .app{width:100%}
  header{padding:10px 12px;padding-top:max(10px,env(safe-area-inset-top))}
  header h1{font-size:14px}
  .hbtn{width:34px;height:34px;font-size:14px}
  #messages{padding:12px 10px;gap:12px}
  .msg{font-size:14px;padding:9px 12px}
  .msg-row{max-width:92%}
  .msg-actions{opacity:1;height:28px}
  .msg-actions button{min-height:28px;padding:4px 10px}
  textarea{font-size:16px}
  .tool-btn{width:34px;height:34px;font-size:17px}
  #send-btn{width:34px;height:34px}
  #scroll-btn{bottom:100px;right:12px}
  .input-wrap{padding:6px 10px max(10px,env(safe-area-inset-bottom))}
}
@media(max-width:380px){
  :root{--sidebar-w:90vw}
  .msg{font-size:13.5px}
}
</style>
</head>
<body>
<div class="layout">
  <div id="sidebar-overlay"></div>
  <div id="sidebar" class="hidden">
    <button id="new-chat-btn">+ New chat</button>
    <div id="conv-list"></div>
    <div id="sidebar-footer">Mythic AI &middot; by Aarav Singh</div>
  </div>
  <div class="app">
    <header>
      <div class="left">
        <button class="hbtn" id="sidebar-toggle">☰</button>
        <h1>Mythic AI</h1>
        <select id="model-sel" title="Select model" style="background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:4px 8px;font-size:12px;cursor:pointer;outline:none;max-width:110px;font-family:inherit">
          <option value="mythic-1">Mythic 1</option>
          <option value="mythic-2" selected>Mythic 2</option>
          <option value="mythic-pro">Mythic Pro</option>
          <option value="mythic-vip">Mythic VIP 🔒</option>
        </select>
      </div>
      <div class="right">
        <button class="hbtn" id="speak-toggle" title="Toggle voice">🔇</button>
        <button class="hbtn" id="settings-btn" title="Settings">⚙️</button>
        <button class="hbtn" id="fs-btn" title="Fullscreen">⛶</button>
        <button class="hbtn" id="name-btn" title="Your name">🙂</button>
        <button class="hbtn" id="export-btn" title="Export">⬇</button>
        <button class="hbtn" id="clear-btn" title="Delete chat">🗑</button>
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
    <button id="scroll-btn">↓</button>
    <div id="pending-attach">📎 <span id="pa-name"></span><button id="pa-remove">✕</button></div>
    <div id="speaking-bar">🔊 Speaking... <button id="stop-speak-btn">Stop</button></div>
    <div id="quick-actions">
      <button class="quick-btn" id="img-btn">🎨 Image</button>
      <button class="quick-btn" id="ghibli-btn">🌿 Ghibli Me</button>
      <button class="quick-btn" id="hw-btn">📚 Homework</button>
      <button class="quick-btn" id="weather-btn">🌤 Weather</button>
      <button class="quick-btn" id="search-btn">🔍 Search</button>
      <button class="quick-btn" id="news-btn">📰 News</button>
    </div>
    <div class="input-wrap">
      <form id="chat-form">
        <div class="input-row">
          <input type="file" id="file-input" accept="image/*,.txt,.md,.csv,.json,.pdf" style="display:none">
          <button class="tool-btn" id="attach-btn" type="button" title="Attach">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
          </button>
          <input type="file" id="camera-input" accept="image/*" capture="environment" style="display:none">
          <input type="file" id="selfie-input" accept="image/*" capture="user" style="display:none">
          <button class="tool-btn" id="camera-btn" type="button" title="Camera">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
          </button>
          <button class="tool-btn" id="selfie-btn" type="button" title="Selfie">🤳</button>
          <textarea id="input" rows="1" placeholder="Message Mythic AI..."></textarea>
          <button class="tool-btn" id="voice-btn" type="button" title="Voice">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
          </button>
          <button id="send-btn" type="submit">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
          </button>
        </div>
      </form>
    </div>
  </div>
</div>

<!-- Name modal -->
<div class="modal-overlay" id="name-modal">
  <div class="name-modal">
    <h3 style="margin:0 0 6px;font-size:16px;">What should Mythic AI call you?</h3>
    <p style="color:var(--muted);font-size:13px;margin:0 0 14px;">Your name will be used in conversation.</p>
    <input class="modal-input" type="text" id="name-input" maxlength="60" placeholder="e.g. Aarav" autocomplete="off">
    <div class="modal-btn-row">
      <button class="btn-secondary" id="name-cancel">Cancel</button>
      <button class="btn-primary" id="name-save">Save</button>
    </div>
  </div>
</div>

<!-- Image modal -->
<div class="modal-overlay" id="img-modal">
  <div class="modal-box">
    <h3>🎨 Generate Image</h3>
    <p>Describe what you want to create</p>
    <select class="modal-input" id="img-style" style="margin-bottom:10px;cursor:pointer">
      <option value="">✨ Auto</option>
      <option value="ghibli">🌿 Studio Ghibli</option>
      <option value="anime">🎌 Anime</option>
      <option value="realistic">📷 Realistic</option>
      <option value="oil painting">🖼 Oil Painting</option>
      <option value="watercolor">🎨 Watercolor</option>
      <option value="3d render">🧊 3D Render</option>
      <option value="cartoon">🐱 Cartoon</option>
    </select>
    <textarea class="modal-input" id="img-prompt" rows="3" placeholder="e.g. A sunset over mountains..." style="margin-bottom:10px;resize:none"></textarea>
    <div id="img-result" style="display:none;text-align:center;margin-bottom:10px">
      <img id="img-out" style="max-width:100%;border-radius:10px">
    </div>
    <div id="img-loading" style="display:none;text-align:center;padding:16px;color:var(--muted)">⏳ Generating... (15-30 seconds)</div>
    <div id="img-err" style="display:none;color:#ef4444;font-size:12px;margin-bottom:8px"></div>
    <div class="modal-btn-row">
      <button class="btn-primary" id="img-go">Generate</button>
      <button class="btn-secondary" id="img-close">Close</button>
    </div>
  </div>
</div>

<!-- Ghibli Me modal -->
<div class="modal-overlay" id="ghibli-modal">
  <div class="modal-box">
    <h3>🌿 Ghibli Me</h3>
    <p>Upload your photo and get a Studio Ghibli version of yourself</p>
    <div id="ghibli-upload" style="border:2px dashed var(--border);border-radius:12px;padding:20px;text-align:center;cursor:pointer;margin-bottom:10px">
      <div style="font-size:32px;margin-bottom:6px">📸</div>
      <div style="font-size:13px;color:var(--muted)">Click to upload your photo</div>
      <input type="file" id="ghibli-file" accept="image/*" style="display:none">
    </div>
    <div id="ghibli-preview-wrap" style="display:none;text-align:center;margin-bottom:10px">
      <img id="ghibli-preview" style="max-height:160px;border-radius:10px;border:2px solid var(--accent)">
    </div>
    <input class="modal-input" type="text" id="ghibli-extra" placeholder="Extra details (optional)..." style="margin-bottom:10px">
    <div id="ghibli-result-wrap" style="display:none;text-align:center;margin-bottom:10px">
      <img id="ghibli-result" style="max-width:100%;border-radius:12px">
      <button id="ghibli-dl" style="margin-top:8px;background:var(--accent);color:#fff;border:none;border-radius:8px;padding:8px 18px;font-size:13px;cursor:pointer">⬇ Download</button>
    </div>
    <div id="ghibli-loading" style="display:none;text-align:center;padding:16px;color:var(--muted)">🎨 Creating your Ghibli portrait...</div>
    <div id="ghibli-err" style="display:none;color:#ef4444;font-size:12px;margin-bottom:8px;padding:8px;background:#fef2f2;border-radius:6px"></div>
    <div class="modal-btn-row">
      <button class="btn-primary" id="ghibli-go">✨ Create</button>
      <button class="btn-secondary" id="ghibli-close">Close</button>
    </div>
  </div>
</div>

<!-- Weather modal -->
<div class="modal-overlay" id="weather-modal">
  <div class="modal-box">
    <h3>🌤 Weather Forecast</h3>
    <p>Enter city name or use your location</p>
    <div style="display:flex;gap:8px;margin-bottom:10px">
      <input class="modal-input" type="text" id="weather-city" placeholder="e.g. Delhi, Mumbai..." style="flex:1">
      <button id="weather-loc-btn" style="background:var(--panel);border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:8px 12px;font-size:18px;cursor:pointer">📍</button>
    </div>
    <div id="weather-result" style="display:none;background:var(--bg);border-radius:12px;padding:14px;margin-bottom:10px"></div>
    <div id="weather-loading" style="display:none;text-align:center;padding:14px;color:var(--muted)">⏳ Fetching weather...</div>
    <div id="weather-err" style="display:none;color:#ef4444;font-size:12px;margin-bottom:8px"></div>
    <div class="modal-btn-row">
      <button class="btn-primary" id="weather-go">Get Weather</button>
      <button class="btn-secondary" id="weather-close">Close</button>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const wrap    = $('messages-wrap');
const msgs    = $('messages');
const form    = $('chat-form');
const inp     = $('input');
const sendBtn = $('send-btn');
const sidebar = $('sidebar');
const overlay = $('sidebar-overlay');

let activeId = null, pendingFile = null, isGen = false, abortCtrl = null;
let voiceOn = false, recognition = null;

function resize() { inp.style.height='auto'; inp.style.height=Math.min(inp.scrollHeight,140)+'px'; }
function getUserName() { return localStorage.getItem('mythic_name')||''; }
function setUserName(n) { n ? localStorage.setItem('mythic_name',n) : localStorage.removeItem('mythic_name'); }
function isMobile() { return window.innerWidth<=768; }

// Sidebar
function openSb() { sidebar.classList.remove('hidden'); if(isMobile()) overlay.style.display='block'; }
function closeSb() { sidebar.classList.add('hidden'); overlay.style.display='none'; }
$('sidebar-toggle').addEventListener('click', ()=> sidebar.classList.contains('hidden') ? openSb() : closeSb());
overlay.addEventListener('click', closeSb);

// Scroll
wrap.addEventListener('scroll', ()=>{
  const near = wrap.scrollHeight-wrap.scrollTop-wrap.clientHeight < 120;
  $('scroll-btn').classList.toggle('show', !near);
});
$('scroll-btn').addEventListener('click', ()=> wrap.scrollTo({top:wrap.scrollHeight,behavior:'smooth'}));
function scrollBottom() { requestAnimationFrame(()=> wrap.scrollTo({top:wrap.scrollHeight,behavior:'smooth'})); }

function clearEmpty() { const e=$('empty-state'); if(e) e.remove(); }
function showEmpty() { msgs.innerHTML='<div class="empty-state" id="empty-state"><h2>Mythic AI</h2><p>Ask me anything, generate images, or just chat 👋</p></div>'; }

function addMsg(role, text, attach) {
  clearEmpty();
  const row = document.createElement('div'); row.className='msg-row '+role;
  const bubble = document.createElement('div'); bubble.className='msg '+role;
  if(attach) {
    const chip=document.createElement('div'); chip.className='attach-chip'; chip.textContent='📎 '+attach.name; bubble.appendChild(chip);
    if(attach.mimeType&&attach.mimeType.startsWith('image/')&&attach.dataBase64) {
      const img=document.createElement('img'); img.src='data:'+attach.mimeType+';base64,'+attach.dataBase64; bubble.appendChild(img);
    }
  }
  const tn=document.createElement('div'); tn.className='msg-text'; tn.textContent=text; bubble.appendChild(tn); row.appendChild(bubble);
  if(role==='user'||role==='ai') {
    const acts=document.createElement('div'); acts.className='msg-actions';
    const cpBtn=document.createElement('button'); cpBtn.textContent='📋'; cpBtn.title='Copy';
    cpBtn.addEventListener('click', async()=>{
      try { await navigator.clipboard.writeText(tn.textContent); } catch { const t=document.createElement('textarea'); t.value=tn.textContent; document.body.appendChild(t); t.select(); document.execCommand('copy'); t.remove(); }
      const o=cpBtn.textContent; cpBtn.textContent='✓'; setTimeout(()=>cpBtn.textContent=o,1200);
    }); acts.appendChild(cpBtn);
    if(role==='ai') { const rg=document.createElement('button'); rg.textContent='↻'; rg.title='Regenerate'; rg.addEventListener('click',()=>{ if(!isGen){row.remove();streamReply({regenerate:true});} }); acts.appendChild(rg); }
    row.appendChild(acts);
  }
  msgs.appendChild(row); scrollBottom(); return tn;
}

function showTyping() { const d=document.createElement('div'); d.className='typing'; d.id='typ'; d.innerHTML='<span></span><span></span><span></span>'; msgs.appendChild(d); scrollBottom(); }
function hideTyping() { const e=$('typ'); if(e) e.remove(); }

// Voice output
function isHindi(t) { return /[\u0900-\u097F]/.test(t); }
function speak(text) {
  if(!voiceOn||!window.speechSynthesis) return;
  speechSynthesis.cancel();
  const plain=text.replace(/[#*`_~>]/g,'').trim(); if(!plain) return;
  const utt=new SpeechSynthesisUtterance(plain);
  utt.rate=0.95; utt.pitch=1.1;
  const hindi=isHindi(plain); utt.lang=hindi?'hi-IN':'en-IN';
  const voices=speechSynthesis.getVoices();
  let v=null;
  if(hindi) {
    v=voices.find(x=>x.lang.startsWith('hi')&&/female|woman|lekha|kalpana|aditi|riya/i.test(x.name))||voices.find(x=>x.lang.startsWith('hi'));
  } else {
    v=voices.find(x=>x.lang==='en-IN')||voices.find(x=>x.lang.startsWith('en')&&/female|woman|samantha|victoria|karen|sonia|aria|jenny|zira/i.test(x.name))||voices.find(x=>x.lang.startsWith('en'));
  }
  if(v) utt.voice=v;
  utt.onstart=()=>$('speaking-bar').classList.add('show');
  utt.onend=utt.onerror=()=>$('speaking-bar').classList.remove('show');
  speechSynthesis.speak(utt);
}
if(window.speechSynthesis){ speechSynthesis.getVoices(); speechSynthesis.onvoiceschanged=()=>speechSynthesis.getVoices(); }
$('stop-speak-btn').addEventListener('click',()=>{ speechSynthesis&&speechSynthesis.cancel(); $('speaking-bar').classList.remove('show'); });
const spkToggle=$('speak-toggle');
spkToggle.addEventListener('click',()=>{
  voiceOn=!voiceOn; spkToggle.textContent=voiceOn?'🔊':'🔇'; spkToggle.classList.toggle('on',voiceOn);
  if(!voiceOn){ speechSynthesis&&speechSynthesis.cancel(); $('speaking-bar').classList.remove('show'); }
});

// Voice input
(function(){
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition; if(!SR) return;
  recognition=new SR(); recognition.continuous=false; recognition.interimResults=true; recognition.lang='hi-IN';
  recognition.onstart=()=>$('voice-btn').classList.add('listening');
  recognition.onresult=e=>{ let fin=''; for(let i=e.resultIndex;i<e.results.length;i++){ if(e.results[i].isFinal) fin+=e.results[i][0].transcript; else inp.value=e.results[i][0].transcript; } if(fin) inp.value=fin; resize(); };
  recognition.onend=()=>{ $('voice-btn').classList.remove('listening'); if(inp.value.trim()) form.requestSubmit(); };
  recognition.onerror=()=>$('voice-btn').classList.remove('listening');
})();
$('voice-btn').addEventListener('click',()=>{ if(!recognition){alert('Voice not supported. Try Chrome.');return;} recognition.classList&&recognition.stop ? recognition.stop() : (document.getElementById('voice-btn').classList.contains('listening')?recognition.stop():recognition.start()); });
$('voice-btn').addEventListener('click',()=>{ if(!recognition){alert('Voice not supported. Try Chrome.');return;} $('voice-btn').classList.contains('listening')?recognition.stop():recognition.start(); });

// File attach
function handleFile(file){ if(!file) return; const r=new FileReader(); r.onload=e=>{ const b64=e.target.result.split(',')[1]; pendingFile={name:file.name,mimeType:file.type||'application/octet-stream',dataBase64:b64}; $('pa-name').textContent=file.name; $('pending-attach').classList.add('show'); }; r.readAsDataURL(file); }
$('attach-btn').addEventListener('click',()=>$('file-input').click());
$('camera-btn').addEventListener('click',()=>$('camera-input').click());
$('selfie-btn').addEventListener('click',()=>$('selfie-input').click());
$('file-input').addEventListener('change',()=>handleFile($('file-input').files[0]));
$('camera-input').addEventListener('change',()=>handleFile($('camera-input').files[0]));
$('selfie-input').addEventListener('change',()=>handleFile($('selfie-input').files[0]));
$('pa-remove').addEventListener('click',()=>{ pendingFile=null; $('file-input').value=''; $('camera-input').value=''; $('selfie-input').value=''; $('pending-attach').classList.remove('show'); });

// Fullscreen
const fsBtn=$('fs-btn');
async function toggleFs(){
  const el=document.documentElement;
  try{ if(!document.fullscreenElement&&!document.webkitFullscreenElement){ el.requestFullscreen?await el.requestFullscreen():el.webkitRequestFullscreen&&el.webkitRequestFullscreen(); } else { document.exitFullscreen?await document.exitFullscreen():document.webkitExitFullscreen&&document.webkitExitFullscreen(); } }catch(e){ document.body.classList.toggle('pseudo-fs'); }
  fsBtn.textContent=document.fullscreenElement||document.webkitFullscreenElement?'⤢':'⛶';
}
fsBtn.addEventListener('click',toggleFs);
document.addEventListener('fullscreenchange',()=>fsBtn.textContent=document.fullscreenElement?'⤢':'⛶');

// Name modal
function openNameModal(){ $('name-input').value=getUserName(); $('name-modal').classList.add('show'); setTimeout(()=>$('name-input').focus(),50); }
function closeNameModal(){ $('name-modal').classList.remove('show'); }
$('name-btn').addEventListener('click',openNameModal);
$('name-cancel').addEventListener('click',closeNameModal);
$('name-modal').addEventListener('click',e=>{ if(e.target===$('name-modal')) closeNameModal(); });
$('name-save').addEventListener('click',()=>{ setUserName($('name-input').value.trim()); closeNameModal(); });
$('name-input').addEventListener('keydown',e=>{ if(e.key==='Enter') $('name-save').click(); else if(e.key==='Escape') closeNameModal(); });
if(!localStorage.getItem('mythic_name_asked')){ localStorage.setItem('mythic_name_asked','1'); setTimeout(openNameModal,700); }

// Export
$('export-btn').addEventListener('click',async()=>{
  if(!activeId){alert('Open a chat first.');return;}
  try{ const r=await fetch('/api/conversations/'+activeId); if(!r.ok) return; const d=await r.json();
  const lines=[`# ${d.title||'Mythic AI chat'}`,'']; (d.messages||[]).forEach(m=>{ lines.push(m.role==='user'?'You:':'Mythic AI:'); lines.push(m.text||(m.attachment?'[attachment: '+m.attachment.name+']':'')); lines.push(''); });
  const blob=new Blob([lines.join('\n')],{type:'text/plain'}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download='mythic-chat.txt'; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url); }catch(e){alert('Export failed: '+e.message);}
});

// Conversations
async function loadList(){
  try{ const r=await fetch('/api/conversations'); const d=await r.json(); const convs=d.conversations||[];
  $('conv-list').innerHTML='';
  convs.forEach(c=>{ const item=document.createElement('div'); item.className='conv-item'+(c.id===activeId?' active':'');
    item.innerHTML='<span class="title"></span><button class="del-btn" title="Delete">✕</button>';
    item.querySelector('.title').textContent=c.title;
    item.addEventListener('click',e=>{ if(!e.target.classList.contains('del-btn')) openConv(c.id); });
    item.querySelector('.del-btn').addEventListener('click',async e=>{ e.stopPropagation(); await fetch('/api/conversations/'+c.id,{method:'DELETE'}); if(c.id===activeId) newChat(); else loadList(); });
    $('conv-list').appendChild(item);
  }); return convs; }catch{return[];}
}
async function openConv(id){ activeId=id; try{ const r=await fetch('/api/conversations/'+id); if(!r.ok) return; const d=await r.json(); msgs.innerHTML=''; (d.messages||[]).forEach(m=>addMsg(m.role,m.text,m.attachment)); loadList(); }catch{} if(isMobile()) closeSb(); }
function newChat(){ activeId=null; msgs.innerHTML=''; showEmpty(); loadList(); if(isMobile()) closeSb(); }
$('new-chat-btn').addEventListener('click',newChat);
$('clear-btn').addEventListener('click',async()=>{ if(!activeId) return; await fetch('/api/conversations/'+activeId,{method:'DELETE'}); newChat(); });

// Send / stream
function setGen(on){ isGen=on; sendBtn.classList.toggle('stop-mode',on); sendBtn.title=on?'Stop':'Send'; sendBtn.innerHTML=on?'<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>':'<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>'; }

const IMG_RE=/\b(generate|create|draw|make|paint|render|ghibli|anime|cartoon|portrait|image of|picture of|illustration)\b/i;
const NEWS_RE=/\b(news|khabar|headline|aaj ki|cricket|score|match|breaking)\b/i;
const SRCH_RE=/^search:\s*/i;

async function streamReply({message='',attachment=null,regenerate=false}={}){
  showTyping(); setGen(true); abortCtrl=new AbortController();
  let newsCtx=null;
  if(!regenerate&&message){
    if(SRCH_RE.test(message)){
      const q=message.replace(SRCH_RE,'').trim();
      try{ const r=await fetch('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q})}); const d=await r.json(); if(d.results&&d.results.length) newsCtx='Search results for "'+q+'":\n'+d.results.map((x,i)=>`${i+1}. ${x.title}: ${x.snippet}`).join('\n'); }catch{}
    } else if(NEWS_RE.test(message)){
      try{ const r=await fetch('/api/news',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:message.length<80?message:null})}); const d=await r.json(); if(d.articles&&d.articles.length) newsCtx=d.articles.map((a,i)=>`${i+1}. ${a.title} (${a.source})`).join('\n'); }catch{}
    }
  }
  let aiNode=null;
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},signal:abortCtrl.signal,
      body:JSON.stringify({message,conversation_id:activeId,attachment,user_name:getUserName(),regenerate:!!regenerate,news_context:newsCtx})});
    if(!r.ok||!r.body){hideTyping();addMsg('error','Something went wrong. Try again.');return;}
    hideTyping(); aiNode=addMsg('ai','');
    const cid=r.headers.get('X-Conversation-Id'); if(cid) activeId=cid;
    const reader=r.body.getReader(); const dec=new TextDecoder(); let full='';
    while(true){ const{done,value}=await reader.read(); if(done) break; full+=dec.decode(value,{stream:true}); aiNode.textContent=full; scrollBottom(); }
    speak(full); loadList();
  }catch(err){
    hideTyping();
    if(err.name==='AbortError'){ if(aiNode&&!aiNode.textContent.trim()) aiNode.textContent='[Stopped]'; }
    else addMsg('error','Network error: '+err.message);
  }finally{ setGen(false); abortCtrl=null; }
}

form.addEventListener('submit',e=>{ e.preventDefault(); if(isGen) return; const t=inp.value.trim(); if(!t&&!pendingFile) return; const att=pendingFile; pendingFile=null; $('file-input').value=''; $('camera-input').value=''; $('selfie-input').value=''; $('pending-attach').classList.remove('show'); addMsg('user',t,att); inp.value=''; inp.style.height='auto'; streamReply({message:t,attachment:att}); });
sendBtn.addEventListener('click',e=>{ if(isGen){ e.preventDefault(); abortCtrl&&abortCtrl.abort(); } });
inp.addEventListener('keydown',e=>{ if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();form.requestSubmit();} });
inp.addEventListener('input',resize);

// Quick buttons
$('hw-btn').addEventListener('click',()=>{ inp.value='Help me with my homework: '; inp.focus(); resize(); inp.setSelectionRange(inp.value.length,inp.value.length); });
$('search-btn').addEventListener('click',()=>{ const q=prompt('What to search for?'); if(!q||!q.trim()) return; addMsg('user','Search: '+q.trim()); streamReply({message:'Search: '+q.trim()}); });
$('news-btn').addEventListener('click',()=>{ addMsg('user','Today\'s top news'); streamReply({message:'Today\'s top news'}); });
$('img-btn').addEventListener('click',()=>{ $('img-modal').classList.add('show'); $('img-prompt').focus(); });
$('ghibli-btn').addEventListener('click',()=>{ $('ghibli-modal').classList.add('show'); });
$('weather-btn').addEventListener('click',()=>{ $('weather-modal').classList.add('show'); $('weather-city').focus(); });

// Image modal
const imgModal=$('img-modal');
$('img-close').addEventListener('click',()=>imgModal.classList.remove('show'));
imgModal.addEventListener('click',e=>{ if(e.target===imgModal) imgModal.classList.remove('show'); });
$('img-prompt').addEventListener('keydown',e=>{ if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();$('img-go').click();} });
$('img-go').addEventListener('click',async()=>{
  const prompt=$('img-prompt').value.trim(); const style=$('img-style').value;
  if(!prompt) return;
  $('img-result').style.display='none'; $('img-err').style.display='none'; $('img-loading').style.display='block'; $('img-go').disabled=true;
  try{
    const r=await fetch('/api/generate-image',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,style})});
    const d=await r.json(); $('img-loading').style.display='none';
    if(d.image){ $('img-out').src='data:image/png;base64,'+d.image; $('img-result').style.display='block'; clearEmpty(); const row=document.createElement('div'); row.className='msg-row ai'; const b=document.createElement('div'); b.className='msg ai'; b.style.padding='8px'; const cap=document.createElement('div'); cap.textContent='🎨 '+prompt; cap.style.cssText='font-size:12px;opacity:.7;margin-bottom:8px'; const img=document.createElement('img'); img.src='data:image/png;base64,'+d.image; img.style.cssText='max-width:100%;border-radius:10px'; b.appendChild(cap); b.appendChild(img); row.appendChild(b); msgs.appendChild(row); scrollBottom(); }
    else{ $('img-err').textContent=d.error||'Failed. Try again.'; $('img-err').style.display='block'; }
  }catch(e){ $('img-loading').style.display='none'; $('img-err').textContent='Error: '+e.message; $('img-err').style.display='block'; }
  finally{ $('img-go').disabled=false; }
});

// Ghibli modal
let ghibliB64=null;
const ghibliModal=$('ghibli-modal');
$('ghibli-close').addEventListener('click',()=>ghibliModal.classList.remove('show'));
ghibliModal.addEventListener('click',e=>{ if(e.target===ghibliModal) ghibliModal.classList.remove('show'); });
$('ghibli-upload').addEventListener('click',()=>$('ghibli-file').click());
$('ghibli-file').addEventListener('change',()=>{
  const file=$('ghibli-file').files[0]; if(!file) return;
  const r=new FileReader(); r.onload=e=>{ ghibliB64=e.target.result.split(',')[1]; $('ghibli-preview').src=e.target.result; $('ghibli-preview-wrap').style.display='block'; $('ghibli-result-wrap').style.display='none'; }; r.readAsDataURL(file);
});
$('ghibli-go').addEventListener('click',async()=>{
  if(!ghibliB64){ $('ghibli-err').textContent='Please upload your photo first!'; $('ghibli-err').style.display='block'; return; }
  $('ghibli-err').style.display='none'; $('ghibli-result-wrap').style.display='none'; $('ghibli-loading').style.display='block'; $('ghibli-go').disabled=true;
  const extra=$('ghibli-extra').value.trim();
  const prompt=`Studio Ghibli anime portrait, Spirited Away style, soft watercolor art, beautiful detailed character, ${extra?extra+', ':''}masterpiece, highly detailed, dreamy atmosphere`;
  try{
    const r=await fetch('/api/generate-image',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,style:'',image:ghibliB64})});
    const d=await r.json(); $('ghibli-loading').style.display='none';
    if(d.image){ $('ghibli-result').src='data:image/png;base64,'+d.image; $('ghibli-result-wrap').style.display='block'; clearEmpty(); const row=document.createElement('div'); row.className='msg-row ai'; const b=document.createElement('div'); b.className='msg ai'; b.style.padding='8px'; const cap=document.createElement('div'); cap.textContent='🌿 Your Ghibli portrait'; cap.style.cssText='font-size:12px;color:var(--muted);margin-bottom:8px'; const img=document.createElement('img'); img.src='data:image/png;base64,'+d.image; img.style.cssText='max-width:100%;border-radius:12px;cursor:pointer'; img.title='Click to download'; img.addEventListener('click',()=>dlImg(d.image,'mythic-ghibli.png')); b.appendChild(cap); b.appendChild(img); row.appendChild(b); msgs.appendChild(row); scrollBottom(); }
    else{ $('ghibli-err').textContent=d.error||'Failed. Try again.'; $('ghibli-err').style.display='block'; }
  }catch(e){ $('ghibli-loading').style.display='none'; $('ghibli-err').textContent='Error: '+e.message; $('ghibli-err').style.display='block'; }
  finally{ $('ghibli-go').disabled=false; }
});
function dlImg(b64,name){ const a=document.createElement('a'); a.href='data:image/png;base64,'+b64; a.download=name; document.body.appendChild(a); a.click(); a.remove(); }
$('ghibli-dl').addEventListener('click',()=>{ const src=$('ghibli-result').src; if(src) dlImg(src.split(',')[1],'mythic-ghibli.png'); });

// Weather modal
const weatherModal=$('weather-modal');
$('weather-close').addEventListener('click',()=>weatherModal.classList.remove('show'));
weatherModal.addEventListener('click',e=>{ if(e.target===weatherModal) weatherModal.classList.remove('show'); });
$('weather-city').addEventListener('keydown',e=>{ if(e.key==='Enter') $('weather-go').click(); });
async function fetchW(payload){
  $('weather-result').style.display='none'; $('weather-err').style.display='none';
  $('weather-loading').style.display='block'; $('weather-go').disabled=true;
  try{
    const r=await fetch('/api/weather',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const d=await r.json(); $('weather-loading').style.display='none';
    if(d.weather){ const w=d.weather; $('weather-result').innerHTML=`<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px"><div style="font-size:44px">${w.icon}</div><div><div style="font-size:17px;font-weight:700">${w.location}</div><div style="font-size:13px;color:var(--muted)">${w.condition}</div></div></div><div style="display:grid;grid-template-columns:1fr 1fr;gap:8px"><div style="background:var(--panel);border-radius:8px;padding:10px"><div style="font-size:11px;color:var(--muted)">TEMPERATURE</div><div style="font-size:22px;font-weight:700">${w.temp}°C</div><div style="font-size:11px;color:var(--muted)">Feels like ${w.feels_like}°C</div></div><div style="background:var(--panel);border-radius:8px;padding:10px"><div style="font-size:11px;color:var(--muted)">HUMIDITY</div><div style="font-size:22px;font-weight:700">${w.humidity}%</div><div style="font-size:11px;color:var(--muted)">Wind ${w.wind_speed} km/h</div></div></div>`; $('weather-result').style.display='block'; inp.value=`${w.icon} ${w.location}: ${w.temp}°C, ${w.condition}. Humidity: ${w.humidity}%, Wind: ${w.wind_speed} km/h.`; resize(); }
    else{ $('weather-err').textContent=d.error||'Not found.'; $('weather-err').style.display='block'; }
  }catch(e){ $('weather-loading').style.display='none'; $('weather-err').textContent='Error: '+e.message; $('weather-err').style.display='block'; }
  finally{ $('weather-go').disabled=false; }
}
$('weather-go').addEventListener('click',()=>{ const loc=$('weather-city').value.trim(); if(loc) fetchW({location:loc}); });
$('weather-loc-btn').addEventListener('click',()=>{ if(!navigator.geolocation){alert('Geolocation not supported');return;} navigator.geolocation.getCurrentPosition(p=>fetchW({lat:p.coords.latitude,lon:p.coords.longitude}),e=>alert('Location error: '+e.message)); });

// Init
(async()=>{ const convs=await loadList(); if(convs.length>0) openConv(convs[0].id); else showEmpty(); })();

// Auto fullscreen on first interaction
let autoFsDone=false;
async function autoFs(){ if(autoFsDone) return; autoFsDone=true; try{ const el=document.documentElement; el.requestFullscreen?await el.requestFullscreen():el.webkitRequestFullscreen&&el.webkitRequestFullscreen(); }catch{} }
['click','touchstart','keydown'].forEach(e=>document.addEventListener(e,autoFs,{once:true,capture:true}));
</script>
</body>
</html>
"""

@app.route("/api/models")
def api_models():
    return jsonify({"models":[
        {"id":"mythic-1","name":"Mythic 1","desc":"Fast & light","vip":False},
        {"id":"mythic-2","name":"Mythic 2","desc":"Smart & balanced","vip":False},
        {"id":"mythic-pro","name":"Mythic Pro","desc":"Advanced reasoning","vip":False},
        {"id":"mythic-vip","name":"Mythic VIP ✨","desc":"Most powerful","vip":True},
    ],"default":"mythic-2"})

@app.route("/api/vip-unlock", methods=["POST"])
def api_vip_unlock():
    d = request.get_json(force=True) or {}
    if d.get("password") == VIP_PASSWORD:
        session["vip"] = True
        return jsonify({"success": True})
    return jsonify({"success": False}), 403

@app.route("/api/vip-status")
def api_vip_status():
    return jsonify({"vip": bool(session.get("vip"))})

@app.route("/")
@login_required
def index():
    return Response(PAGE, mimetype="text/html; charset=utf-8")

@app.route("/api/conversations", methods=["GET"])
@login_required
def api_list(): return jsonify({"conversations": list_conversations(current_username())})

@app.route("/api/conversations/<cid>", methods=["GET"])
@login_required
def api_get(cid):
    data=load_conversation(current_username(),cid)
    if not data: return jsonify({"error":"not found"}),404
    simplified=[]
    for m in data.get("messages",[]):
        role="user" if m["role"]=="user" else "ai"
        text="".join(p.get("text","") for p in m["parts"] if "text" in p)
        entry={"role":role,"text":text}
        if m.get("attachment_meta"): entry["attachment"]=m["attachment_meta"]
        simplified.append(entry)
    return jsonify({"messages":simplified,"title":data.get("title","New chat")})

@app.route("/api/conversations/<cid>", methods=["DELETE"])
@login_required
def api_del(cid):
    delete_conversation(current_username(),cid); return jsonify({"status":"deleted"})

@app.route("/api/conversations/<cid>", methods=["PATCH"])
@login_required
def api_rename(cid):
    data=request.get_json(force=True) or {}
    title=(data.get("title") or "").strip()[:120]
    if not title: return jsonify({"error":"title required"}),400
    u=current_username(); conv=load_conversation(u,cid)
    if not conv: return jsonify({"error":"not found"}),404
    conv["title"]=title; save_conversation(u,cid,conv)
    return jsonify({"status":"renamed","title":title})

def to_openai(messages, sp):
    msgs=[{"role":"system","content":sp}]
    for m in messages:
        role="user" if m["role"]=="user" else "assistant"
        text="".join(p.get("text","") for p in m["parts"] if "text" in p)
        msgs.append({"role":role,"content":text})
    return msgs

def to_ollama(messages, sp):
    msgs=[{"role":"system","content":sp}]
    for m in messages:
        role="user" if m["role"]=="user" else "assistant"
        text="".join(p.get("text","") for p in m["parts"] if "text" in p)
        entry={"role":role,"content":text}
        imgs=[p["inline_data"]["data"] for p in m["parts"] if "inline_data" in p and p["inline_data"].get("mime_type","").startswith("image/")]
        if imgs: entry["images"]=imgs
        msgs.append(entry)
    return msgs

def groq_chunks(msgs, model=None):
    if not GROQ_API_KEY: return
    m=model or GROQ_MODEL
    try:
        resp=requests.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization":f"Bearer {GROQ_API_KEY}","Content-Type":"application/json"},
            json={"model":m,"messages":msgs,"stream":True,"max_tokens":2048},stream=True,timeout=60)
        resp.encoding="utf-8"
        if resp.status_code==200:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"): continue
                d=line[5:].strip()
                if d=="[DONE]": break
                try:
                    c=json.loads(d)["choices"][0]["delta"].get("content","")
                    if c: yield c
                except: continue
        else: print(f"[Groq] {resp.status_code}: {resp.text[:200]}")
    except Exception as e: print(f"[Groq] error: {e}")

def cerebras_chunks(msgs, model=None):
    if not CEREBRAS_API_KEY: return
    m=model or CEREBRAS_MODEL
    try:
        resp=requests.post("https://api.cerebras.ai/v1/chat/completions",
            headers={"Authorization":f"Bearer {CEREBRAS_API_KEY}","Content-Type":"application/json"},
            json={"model":m,"messages":msgs,"stream":True,"max_tokens":2048},stream=True,timeout=60)
        resp.encoding="utf-8"
        if resp.status_code==200:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"): continue
                d=line[5:].strip()
                if d=="[DONE]": break
                try:
                    c=json.loads(d)["choices"][0]["delta"].get("content","")
                    if c: yield c
                except: continue
        else: print(f"[Cerebras] {resp.status_code}: {resp.text[:200]}")
    except Exception as e: print(f"[Cerebras] error: {e}")

def openrouter_chunks(msgs, model=None):
    if not OPENROUTER_API_KEY: return
    m=model or OPENROUTER_MODEL
    try:
        resp=requests.post("https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json","HTTP-Referer":"https://aarav-ai.onrender.com","X-Title":"Mythic AI"},
            json={"model":m,"messages":msgs,"stream":True,"max_tokens":2048},stream=True,timeout=60)
        resp.encoding="utf-8"
        if resp.status_code==200:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"): continue
                d=line[5:].strip()
                if d=="[DONE]": break
                try:
                    c=json.loads(d)["choices"][0]["delta"].get("content","")
                    if c: yield c
                except: continue
        else: print(f"[OpenRouter] {resp.status_code}: {resp.text[:200]}")
    except Exception as e: print(f"[OpenRouter] error: {e}")

def ollama_chunks(msgs):
    try:
        resp=requests.post(f"{OLLAMA_URL}/api/chat",json={"model":OLLAMA_MODEL,"messages":msgs,"stream":True},stream=True,timeout=120)
        resp.encoding="utf-8"
        if resp.status_code!=200: yield f"[Ollama error {resp.status_code}]"; return
        for line in resp.iter_lines(decode_unicode=True):
            if not line: continue
            try:
                obj=json.loads(line)
                if obj.get("error"): yield f"[Ollama: {obj['error']}]"; return
                c=obj.get("message",{}).get("content","")
                if c: yield c
                if obj.get("done"): break
            except: continue
    except Exception as e: yield f"[Ollama error: {e}]"

def gemini_chunks(payload):
    if not GEMINI_API_KEY: return
    try:
        resp=requests.post(GEMINI_STREAM_URL,params={"key":GEMINI_API_KEY,"alt":"sse"},json=payload,stream=True,timeout=60)
        resp.encoding="utf-8"
        if resp.status_code!=200: return
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"): continue
            d=line[len("data:"):].strip()
            if not d or d=="[DONE]": continue
            try:
                obj=json.loads(d)
                for p in obj["candidates"][0]["content"]["parts"]:
                    if "text" in p: yield p["text"]
            except: continue
    except: return

_pi=[0]
def auto_chunks(payload, messages, sp):
    oms=to_openai(messages,sp)
    providers=[]
    if PROVIDER in("auto","cerebras") and CEREBRAS_API_KEY: providers.append(("Cerebras",lambda:cerebras_chunks(oms)))
    if PROVIDER in("auto","groq") and GROQ_API_KEY: providers.append(("Groq",lambda:groq_chunks(oms)))
    if PROVIDER in("auto","openrouter") and OPENROUTER_API_KEY: providers.append(("OpenRouter",lambda:openrouter_chunks(oms)))
    if PROVIDER in("auto","gemini") and GEMINI_API_KEY: providers.append(("Gemini",lambda:gemini_chunks(payload)))
    if PROVIDER=="ollama": providers.append(("Ollama",lambda:ollama_chunks(to_ollama(messages,sp))))
    if not providers: yield "[No AI provider configured. Set at least one API key.]"; return
    n=len(providers); start=_pi[0]%n
    for i in range(n):
        idx=(start+i)%n; name,fn=providers[idx]; collected=[]
        try:
            for chunk in fn(): collected.append(chunk); yield chunk
            if collected: _pi[0]=(idx+1)%n; return
        except: pass
    yield "[All providers failed. Check logs.]"

def fetch_news(query=None, category=None):
    if not NEWS_API_KEY: return None
    params={"apiKey":NEWS_API_KEY,"country":"in","pageSize":8}
    if query: params["q"]=query; url="https://newsapi.org/v2/everything"; params.pop("country",None); params["sortBy"]="publishedAt"; params["language"]="en"
    else: url="https://newsapi.org/v2/top-headlines"
    if category: params["category"]=category
    try:
        r=requests.get(url,params=params,timeout=8)
        if r.status_code==200:
            arts=r.json().get("articles",[])
            return [{"title":a["title"],"source":a["source"]["name"],"url":a["url"]} for a in arts if a.get("title") and "[Removed]" not in a["title"]]
    except: pass
    return None

@app.route("/api/news", methods=["POST"])
@login_required
def get_news():
    d=request.get_json(force=True) or {}
    arts=fetch_news(query=d.get("query"),category=d.get("category"))
    if arts is None: return jsonify({"error":"News unavailable"}),503
    return jsonify({"articles":arts})

@app.route("/api/search", methods=["POST"])
@login_required
def web_search():
    d=request.get_json(force=True) or {}
    query=(d.get("query") or "").strip()
    if not query: return jsonify({"error":"query required"}),400
    try:
        r=requests.get("https://api.duckduckgo.com/",params={"q":query,"format":"json","no_html":1,"skip_disambig":1},headers={"User-Agent":"MythicAI/1.0"},timeout=8)
        if r.status_code!=200: return jsonify({"results":[],"query":query})
        data=r.json(); results=[]
        if data.get("Answer"): results.append({"title":"Answer","snippet":data["Answer"],"url":"","source":data.get("AnswerType","")})
        if data.get("AbstractText"): results.append({"title":data.get("Heading",query),"snippet":data["AbstractText"],"url":data.get("AbstractURL",""),"source":data.get("AbstractSource","")})
        for t in data.get("RelatedTopics",[])[:5]:
            if isinstance(t,dict) and t.get("Text"): results.append({"title":t["Text"][:80],"snippet":t["Text"],"url":t.get("FirstURL",""),"source":"DuckDuckGo"})
        return jsonify({"results":results[:6],"query":query})
    except Exception as e: return jsonify({"error":str(e)}),502

@app.route("/api/weather", methods=["POST"])
@login_required
def get_weather():
    d=request.get_json(force=True) or {}
    location=(d.get("location") or "").strip(); lat=d.get("lat"); lon=d.get("lon")
    if not location and (lat is None or lon is None): return jsonify({"error":"location or coordinates required"}),400
    try:
        if lat is not None and lon is not None:
            geo=requests.get("https://nominatim.openstreetmap.org/reverse",params={"lat":lat,"lon":lon,"format":"json"},headers={"User-Agent":"MythicAI/1.0"},timeout=8)
            addr=geo.json().get("address",{}) if geo.status_code==200 else {}
            location_name=addr.get("city") or addr.get("town") or addr.get("village") or "Your Location"
        else:
            geo=requests.get("https://geocoding-api.open-meteo.com/v1/search",params={"name":location,"count":1,"language":"en","format":"json"},timeout=8)
            if geo.status_code!=200 or not geo.json().get("results"): return jsonify({"error":f"City '{location}' not found"}),404
            res=geo.json()["results"][0]; lat=res["latitude"]; lon=res["longitude"]; location_name=res["name"]+", "+res.get("country","")
        wr=requests.get("https://api.open-meteo.com/v1/forecast",params={"latitude":lat,"longitude":lon,"current":"temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m","wind_speed_unit":"kmh","timezone":"auto"},timeout=8)
        if wr.status_code!=200: return jsonify({"error":"Weather unavailable"}),502
        cur=wr.json()["current"]; code=cur.get("weather_code",0)
        wmo={0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",45:"Foggy",48:"Icy fog",51:"Light drizzle",53:"Drizzle",55:"Heavy drizzle",61:"Light rain",63:"Rain",65:"Heavy rain",71:"Light snow",73:"Snow",75:"Heavy snow",80:"Rain showers",81:"Heavy showers",82:"Violent showers",95:"Thunderstorm",96:"Thunderstorm with hail",99:"Heavy thunderstorm"}
        icons={0:"☀️",1:"🌤",2:"⛅",3:"☁️",45:"🌫",48:"🌫",51:"🌦",53:"🌧",55:"🌧",61:"🌦",63:"🌧",65:"🌧",71:"🌨",73:"❄️",75:"❄️",80:"🌧",81:"🌧",82:"⛈",95:"⛈",96:"⛈",99:"⛈"}
        return jsonify({"weather":{"location":location_name,"temp":round(cur["temperature_2m"]),"feels_like":round(cur["apparent_temperature"]),"condition":wmo.get(code,"Unknown"),"humidity":cur["relative_humidity_2m"],"wind_speed":round(cur["wind_speed_10m"]),"icon":icons.get(code,"🌡")}})
    except Exception as e: return jsonify({"error":str(e)}),502

def pollinations_image(prompt, style=""):
    import urllib.parse, random
    quality="masterpiece, best quality, ultra detailed, sharp focus"
    full=f"{prompt}, {style} style, {quality}" if style else f"{prompt}, {quality}"
    encoded=urllib.parse.quote(full)
    seed=random.randint(1,999999)
    url=f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&model=flux&nologo=true&enhance=true&seed={seed}"
    try:
        resp=requests.get(url,timeout=90,headers={"User-Agent":"MythicAI/1.0"})
        if resp.status_code==200 and resp.headers.get("content-type","").startswith("image/") and len(resp.content)>20000:
            return base64.b64encode(resp.content).decode()
    except: pass
    return None

def nano_banana_image(prompt, style="", input_image_b64=None):
    if not NANO_BANANA_API_KEY: return None, "No Nano Banana key"
    full=f"{prompt}, {style} style" if style else prompt
    parts=[{"text":full}]
    if input_image_b64: parts.append({"inline_data":{"mime_type":"image/jpeg","data":input_image_b64}})
    payload={"contents":[{"role":"user","parts":parts}],"generationConfig":{"responseModalities":["IMAGE"]}}
    try:
        resp=requests.post(NANO_BANANA_URL,params={"key":NANO_BANANA_API_KEY},json=payload,timeout=60)
        if resp.status_code==200:
            for part in resp.json()["candidates"][0]["content"]["parts"]:
                inline=part.get("inline_data") or part.get("inlineData")
                if inline and inline.get("data"): return inline["data"], None
            return None, "No image in response"
        return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e: return None, str(e)

@app.route("/api/generate-image", methods=["POST"])
@login_required
def generate_image():
    data=request.get_json(force=True) or {}
    prompt=(data.get("prompt") or "").strip()
    style=(data.get("style") or "").strip()
    input_image_b64=data.get("image")
    if not prompt: return jsonify({"error":"prompt required"}),400
    img, err = nano_banana_image(prompt, style, input_image_b64)
    if img: return jsonify({"image":img,"provider":"nano-banana"})
    print(f"[NanoBanana] failed: {err}, trying Pollinations...")
    img=pollinations_image(prompt,style)
    if img:
        result={"image":img,"provider":"pollinations"}
        if input_image_b64: result["note"]="Your photo could not be used with the fallback generator."
        return jsonify(result)
    return jsonify({"error":f"Image generation failed. NanoBanana: {err}"}),502

@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    data=request.get_json(force=True) or {}
    user_msg=(data.get("message") or "").strip()
    news_ctx=(data.get("news_context") or "").strip()
    conv_id=data.get("conversation_id")
    attachment=data.get("attachment")
    user_name=(data.get("user_name") or "").strip()[:60]
    regenerate=bool(data.get("regenerate"))
    model_id=(data.get("model") or DEFAULT_MODEL_ID).strip()

    # VIP gate
    if model_id in VIP_MODELS and not session.get("vip"):
        return jsonify({"error":"vip_required"}),403

    # Get provider and model name
    model_cfg = MYTHIC_MODELS.get(model_id, MYTHIC_MODELS[DEFAULT_MODEL_ID])
    sel_provider, sel_model = model_cfg[0], model_cfg[1]
    if regenerate and not conv_id: return jsonify({"error":"conversation_id required"}),400
    if not regenerate and not user_msg and not attachment: return jsonify({"error":"message or attachment required"}),400
    if attachment:
        try:
            raw=base64.b64decode(attachment.get("dataBase64",""),validate=True)
            if len(raw)>MAX_UPLOAD_BYTES: return jsonify({"error":"File too large (max 8MB)"}),400
        except: return jsonify({"error":"Invalid attachment"}),400
    username=current_username()
    conv=load_conversation(username,conv_id) if conv_id else None
    if conv is None:
        if regenerate: return jsonify({"error":"conversation not found"}),404
        conv_id=str(uuid.uuid4()); conv={"title":make_title(user_msg),"messages":[]}
    messages=conv.setdefault("messages",[])
    if regenerate:
        if messages and messages[-1]["role"]=="model": messages.pop()
        if not messages or messages[-1]["role"]!="user": return jsonify({"error":"nothing to regenerate"}),400
    else:
        parts=[]
        if news_ctx and user_msg: parts.append({"text":f"[Context:]\n{news_ctx}\n\n[Question:] {user_msg}"})
        elif user_msg: parts.append({"text":user_msg})
        att_meta=None
        if attachment:
            mime=attachment.get("mimeType","application/octet-stream")
            parts.append({"inline_data":{"mime_type":mime,"data":attachment["dataBase64"]}})
            att_meta={"name":attachment.get("name","file"),"mimeType":mime}
        entry={"role":"user","parts":parts}
        if att_meta: entry["attachment_meta"]=att_meta
        messages.append(entry)
    sp=SYSTEM_PROMPT
    if user_name: sp+=f' The user\'s name is "{user_name}". Use it naturally.'
    gemini_contents=[{"role":m["role"],"parts":m["parts"]} for m in messages]
    payload={"contents":gemini_contents,"systemInstruction":{"parts":[{"text":sp+GEMINI_SEARCH_ADDENDUM}]},"tools":[{"google_search":{}}]}
    def generate():
        full=[]
        oms=to_openai(messages,sp)
        if PROVIDER=="ollama": src=ollama_chunks(to_ollama(messages,sp))
        elif sel_provider=="groq": src=groq_chunks(oms, sel_model)
        elif sel_provider=="cerebras": src=cerebras_chunks(oms, sel_model)
        elif sel_provider=="openrouter": src=openrouter_chunks(oms, sel_model)
        else: src=auto_chunks(payload,messages,sp)
        for chunk in src: full.append(chunk); yield chunk
        messages.append({"role":"model","parts":[{"text":"".join(full)}]})
        save_conversation(username,conv_id,conv)
    resp=Response(stream_with_context(generate()),mimetype="text/plain; charset=utf-8")
    resp.headers["X-Conversation-Id"]=conv_id
    return resp

if __name__=="__main__":
    print("Starting Mythic AI at http://localhost:5000")
    app.run(host="0.0.0.0",port=5000,debug=False)
