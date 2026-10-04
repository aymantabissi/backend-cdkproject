# utils/auth.py
import jwt
import os
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify
from bson import ObjectId

SECRET = os.getenv("JWT_SECRET", "kidney_ai_secret")

# ─── Generate token ───────────────────────────────────────────────────────────
def generate_token(user_id, email: str, role: str = "patient") -> str:
    payload = {
        "user_id": str(user_id),
        "email":   email,
        "role":    role,
        "exp":     datetime.utcnow() + timedelta(hours=1),  
    }
    return jwt.encode(payload, SECRET, algorithm="HS256")


# ─── Verify token ─────────────────────────────────────────────────────────────
def verify_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET, algorithms=["HS256"])
    except Exception:
        return None


# ─── Middleware: login required ───────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        payload = verify_token(token)
        if not payload:
            return jsonify({"error": "Unauthorized — please login"}), 401
        request.user = payload   # نحطو في request باش الـ route يقدر يوصلو
        return f(*args, **kwargs)
    return decorated


# ─── Middleware: role required ────────────────────────────────────────────────
def role_required(*roles):
    """
    استعمال:
      @role_required("admin")
      @role_required("admin", "doctor")
    """
    def decorator(f):
        @wraps(f)
        @login_required
        def decorated(*args, **kwargs):
            user_role = getattr(request, "user", {}).get("role", "")
            if user_role not in roles:
                return jsonify({
                    "error": f"Access denied — requires role: {' or '.join(roles)}"
                }), 403
            return f(*args, **kwargs)
        return decorated
    return decorator