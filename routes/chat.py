# routes/chat.py
from flask import Blueprint, request, jsonify
from groq import Groq
import os
from dotenv import load_dotenv
from extensions import limiter
from utils.auth import login_required

load_dotenv()

chat_bp = Blueprint('chat', __name__)

# تحقق من الـ key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️ GROQ_API_KEY not found in .env")
else:
    print(f"✅ GROQ_API_KEY loaded: {GROQ_API_KEY[:8]}...")

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are NephroAI, an expert medical assistant specialized in Chronic Kidney Disease (CKD).
- Answer in the same language as the user (Arabic, French, or English)
- Be concise, professional, and medically accurate
- Always recommend consulting a nephrologist for serious concerns
- You can explain lab values, symptoms, treatments, and prevention
"""

@chat_bp.route('/chat', methods=['POST'])
@login_required
@limiter.limit("15 per minute; 200 per day")
def chat_with_groq():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data received"}), 400

        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        if not GROQ_API_KEY:
            return jsonify({"error": "GROQ_API_KEY not configured"}), 503

        completion = client.chat.completions.create(
            messages=[
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": user_message},
            ],
            model="llama-3.1-8b-instant",
            max_tokens=1024,
            temperature=0.7,
        )

        reply = completion.choices[0].message.content
        return jsonify({"reply": reply})

    except Exception as e:
        print(f"❌ CHATBOT ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 500