"""
ODN Systems AI Chatbot — Flask Backend
Provider: Groq (free, fast, no quota issues)
Model: llama-3.3-70b-versatile
"""
 
import os
import json
import sys
import time
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from groq import Groq, RateLimitError, AuthenticationError
from dotenv import load_dotenv
 
load_dotenv()
 
app  = Flask(__name__)
CORS(app)
 
# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
KNOWLEDGE_FILE    = "knowledge_base.json"
MODEL             = "llama-3.3-70b-versatile"   # Best free Groq model
MAX_SNIPPET_CHARS = 3500
MAX_RETRIES       = 3
RETRY_DELAY       = 5
 
# ── Validate API key ──────────────────────────────────────────────────────────
if not GROQ_API_KEY or "paste" in GROQ_API_KEY.lower():
    print("\n" + "="*55)
    print("  ❌  GROQ API KEY NOT SET")
    print("="*55)
    print("  1. Go to: https://console.groq.com/keys")
    print("  2. Click 'Create API Key'")
    print("  3. Copy the key (starts with gsk_...)")
    print("  4. Open your .env file and add:")
    print("     GROQ_API_KEY=gsk_your_key_here")
    print("="*55 + "\n")
    sys.exit(1)
 
client = Groq(api_key=GROQ_API_KEY)
 
# ── Load Knowledge Base ───────────────────────────────────────────────────────
KB_PAGES = []
KB_FULL  = ""
 
def load_knowledge_base():
    global KB_PAGES, KB_FULL
 
    if not os.path.exists(KNOWLEDGE_FILE):
        print(f"\n⚠  '{KNOWLEDGE_FILE}' not found — run: python scraper.py\n")
        KB_FULL = (
            "ODN Systems is a professional IT solutions company in Greater Noida, "
            "India. Services include cybersecurity, network infrastructure, software "
            "deployment, data solutions, and IT consulting."
        )
        return
 
    with open(KNOWLEDGE_FILE, encoding="utf-8") as f:
        kb = json.load(f)
 
    KB_PAGES = kb.get("pages", [])
    KB_FULL  = kb.get("full_text", "")
    print(f"✅ Knowledge base loaded — {len(KB_PAGES)} pages, {len(KB_FULL):,} chars")
 
load_knowledge_base()
 
# ── Smart Context Retrieval ───────────────────────────────────────────────────
STOP_WORDS = {
    "a","an","the","is","it","in","on","to","of","and","or","for",
    "what","how","do","does","can","you","i","me","my","we","our",
    "tell","about","please","give","show","want","know","get","are","was"
}
 
def extract_keywords(text: str) -> set:
    words = re.findall(r"[a-z]{3,}", text.lower())
    return {w for w in words if w not in STOP_WORDS}
 
def get_relevant_context(question: str) -> str:
    """Score each KB page by keyword overlap and return top 3 relevant pages."""
    if not KB_PAGES:
        return KB_FULL[:MAX_SNIPPET_CHARS]
 
    q_kws = extract_keywords(question)
    if not q_kws:
        return "\n\n".join(
            p.get("content", "")[:700] for p in KB_PAGES[:2]
        )[:MAX_SNIPPET_CHARS]
 
    scored = []
    for page in KB_PAGES:
        page_blob = " ".join([
            page.get("title", ""),
            " ".join(page.get("headings", [])) if "headings" in page else "",
            page.get("content", "")
        ]).lower()
        score = len(q_kws & extract_keywords(page_blob))
        if score > 0:
            scored.append((score, page))
 
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [p for _, p in scored[:3]]
 
    if not top:
        return KB_FULL[:MAX_SNIPPET_CHARS]
 
    chunks    = []
    remaining = MAX_SNIPPET_CHARS
    for page in top:
        snippet = f"[{page.get('title', 'Page')}]\n{page.get('content', '')}"
        chunks.append(snippet[:remaining])
        remaining -= len(snippet)
        if remaining <= 0:
            break
 
    return "\n\n---\n\n".join(chunks)
 
# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the official AI assistant for ODN Systems, a professional IT solutions and services company based in Greater Noida, India.
 
Your job: Answer visitor questions helpfully, accurately, and professionally using the ODN Systems knowledge provided in each message.
 
Personality:
- Warm, confident, and professional
- Concise: 2-4 sentences for simple questions, up to 6 for detailed ones
- Sound like a helpful team member, not a robotic FAQ
 
Rules:
- Use the provided knowledge context to answer questions about ODN Systems
- For general IT questions, use your knowledge while linking it to ODN's expertise
- If something is not in the knowledge base say: "For more details, please contact our team at +91-83759 19422 or visit our Contact page."
- NEVER invent specific prices, SLAs, or contract terms
- End service-specific answers by inviting the visitor to get in touch
 
Contact & Hours:
- Phone: +91-83759 19422
- Address: Breeze 201, Sector 1, Bisrakh Jalalpur, Greater Noida - 201306, UP, India
- Hours: Monday-Saturday 9 AM - 5 PM | Sunday Closed
- Website: https://odnsystems.com"""
 
# ── Chat Route ────────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or not data.get("message", "").strip():
        return jsonify({"error": "No message provided"}), 400
 
    user_message = data["message"].strip()
    history      = data.get("history", [])
 
    # Get relevant KB context for this specific question
    context = get_relevant_context(user_message)
 
    # Build Groq message list (same format as OpenAI)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
 
    # Add last 6 turns of history
    for turn in history[-6:]:
        role = turn.get("role", "")
        text = turn.get("content", "").strip()
        if role in ("user", "assistant") and text:
            messages.append({"role": role, "content": text})
 
    # Final message with fresh context injected
    messages.append({
        "role": "user",
        "content": (
            f"[Relevant ODN Systems Knowledge]\n{context}\n\n"
            f"[Visitor Question]\n{user_message}"
        )
    })
 
    # ── Call Groq with retry ──────────────────────────────────────────────────
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"   Calling Groq ({MODEL}) attempt {attempt}…")
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.45,
                max_tokens=400,
            )
            reply = response.choices[0].message.content.strip()
            print(f"   ✅ Success")
            return jsonify({"reply": reply})
 
        except AuthenticationError:
            print("❌ Groq: Invalid API key")
            return jsonify({
                "reply": "⚠️ Chatbot API key is invalid. Please contact the site administrator."
            }), 200
 
        except RateLimitError:
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"   ⚠ Rate limit — waiting {wait}s before retry…")
                time.sleep(wait)
            else:
                print("   ✗ Rate limit: all retries exhausted")
                return jsonify({
                    "reply": (
                        "I'm receiving a lot of requests right now. "
                        "Please wait a moment and try again, "
                        "or call us at +91-83759 19422."
                    )
                }), 200
 
        except Exception as e:
            print(f"   ✗ Groq error (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                return jsonify({
                    "reply": "Something went wrong. Please try again in a moment."
                }), 200
 
# ── Static Routes ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")
 
@app.route("/api/health")
def health():
    return jsonify({
        "status":   "ok",
        "provider": "Groq",
        "model":    MODEL,
        "kb_pages": len(KB_PAGES),
        "key_set":  bool(GROQ_API_KEY)
    })
 
# ── Start ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🤖  ODN Systems Chatbot")
    print("=" * 45)
    print(f"   Provider :  Groq (Free)")
    print(f"   Model    :  {MODEL}")
    print(f"   Port     :  5000")
    print(f"   URL      :  http://127.0.0.1:5000")
    print("=" * 45 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
 