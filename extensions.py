# extensions.py — rate limiter partagé entre app.py et les blueprints
from flask import request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from utils.auth import verify_token


def user_or_ip():
    # user connecté → limite par email, sinon par IP
    token   = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    payload = verify_token(token) if token else None
    if payload and payload.get("email"):
        return payload["email"]
    return get_remote_address()


limiter = Limiter(key_func=user_or_ip, storage_uri="memory://")
