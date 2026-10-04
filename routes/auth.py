# routes/auth.py
from flask import Blueprint, request, jsonify
from db.mongo import users_collection
import bcrypt
import os
from utils.auth import generate_token, login_required
from extensions import limiter
from flask_limiter.util import get_remote_address

auth = Blueprint("auth", __name__)

VALID_ROLES = {"admin", "doctor", "patient"}

def db_available():
    return users_collection is not None

# ─── REGISTER ─────────────────────────────────────────────────────────────────
@auth.route("/register", methods=["POST"])
@limiter.limit("5 per minute", key_func=get_remote_address)
def register():
    if not db_available():
        return jsonify({"error": "Database not available"}), 503

    data = request.json
    if not data or not data.get("email") or not data.get("password"):
        return jsonify({"error": "Email and password required"}), 400

    # التسجيل العام ديما patient — admin/doctor كيتصاوبو غير من الـ admin panel
    role = "patient"

    if users_collection.find_one({"email": data["email"]}):
        return jsonify({"error": "User already exists"}), 400

    hashed_pw = bcrypt.hashpw(data["password"].encode(), bcrypt.gensalt())

    user = users_collection.insert_one({
        "name":     data.get("name", ""),
        "email":    data["email"],
        "password": hashed_pw,
        "role":     role,
    })

    token = generate_token(user.inserted_id, data["email"], role)
    return jsonify({
        "token": token,
        "email": data["email"],
        "name":  data.get("name", ""),
        "role":  role,
    })


# ─── LOGIN ────────────────────────────────────────────────────────────────────
@auth.route("/login", methods=["POST"])
@limiter.limit("10 per minute", key_func=get_remote_address)
def login():
    if not db_available():
        return jsonify({"error": "Database not available"}), 503

    data = request.json
    if not data or not data.get("email") or not data.get("password"):
        return jsonify({"error": "Email and password required"}), 400

    user = users_collection.find_one({"email": data["email"]})
    if not user:
        return jsonify({"error": "User not found"}), 404

    if not user.get("password"):
        return jsonify({"error": "This account uses Google login"}), 400
    if not bcrypt.checkpw(data["password"].encode(), user["password"]):
        return jsonify({"error": "Wrong password"}), 401

    role  = user.get("role", "patient")
    token = generate_token(user["_id"], user["email"], role)
    return jsonify({
        "token": token,
        "email": user["email"],
        "name":  user.get("name", ""),
        "role":  role,
    })


# ─── GET CURRENT USER ─────────────────────────────────────────────────────────
@auth.route("/me", methods=["GET"])
def me():
    from utils.auth import verify_token
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    payload = verify_token(token)
    if not payload:
        return jsonify({"error": "Unauthorized"}), 401
    user = users_collection.find_one(
        {"email": payload["email"]},
        {"password": 0, "_id": 0}
    )
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user)


# ─── CHANGE PASSWORD ──────────────────────────────────────────────────────────
@auth.route("/change-password", methods=["POST"])
@login_required
@limiter.limit("5 per minute")
def change_password():
    if not db_available():
        return jsonify({"error": "Database not available"}), 503

    data         = request.json or {}
    old_password = data.get("old_password", "")
    new_password = data.get("new_password", "")

    if not old_password or not new_password:
        return jsonify({"error": "Old and new password required"}), 400
    if len(new_password) < 8:
        return jsonify({"error": "New password must be at least 8 characters"}), 400
    if old_password == new_password:
        return jsonify({"error": "New password must be different"}), 400

    user = users_collection.find_one({"email": request.user["email"]})
    if not user:
        return jsonify({"error": "User not found"}), 404
    if not user.get("password"):
        return jsonify({"error": "Google accounts have no password to change"}), 400
    if not bcrypt.checkpw(old_password.encode(), user["password"]):
        return jsonify({"error": "Wrong current password"}), 401

    hashed_pw = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt())
    users_collection.update_one({"_id": user["_id"]}, {"$set": {"password": hashed_pw}})
    return jsonify({"success": True, "message": "Password updated"})


# ─── GOOGLE LOGIN ─────────────────────────────────────────────────────────────
@auth.route("/google", methods=["POST"])
def google_login():
    if not db_available():
        return jsonify({"error": "Database not available"}), 503

    data       = request.json
    credential = data.get("credential")
    if not credential:
        return jsonify({"error": "No credential provided"}), 400

    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests
        # كنتحققو من الـ signature ديال Google و audience = GOOGLE_CLIENT_ID
        info = id_token.verify_oauth2_token(
            credential, google_requests.Request(), os.getenv("GOOGLE_CLIENT_ID")
        )

        email = info.get("email")
        name  = info.get("name", "")

        if not email:
            return jsonify({"error": "No email found in token"}), 400

        user = users_collection.find_one({"email": email})
        if not user:
            result = users_collection.insert_one({
                "name":     name,
                "email":    email,
                "password": None,
                "role":     "patient",
                "provider": "google",
            })
            role    = "patient"
            user_id = result.inserted_id
        else:
            role    = user.get("role", "patient")
            user_id = user["_id"]

        token = generate_token(user_id, email, role)
        return jsonify({
            "token": token,
            "email": email,
            "name":  name,
            "role":  role,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 401