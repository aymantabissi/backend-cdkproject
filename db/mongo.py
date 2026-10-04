# db/mongo.py
from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME   = os.getenv("DB_NAME", "kidney_ai")

try:
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
    )
    client.admin.command("ping")
    db                 = client[DB_NAME]
    users_collection   = db["users"]
    history_collection = db["predictions"]
    print("✅ MongoDB connected!")
except Exception as e:
    print(f"⚠️ MongoDB not connected: {e}")
    db                 = None
    users_collection   = None
    history_collection = None