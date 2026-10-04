from pymongo import MongoClient
from dotenv import load_dotenv
import os

# load .env
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

try:
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000
    )

    # test connection
    client.admin.command("ping")

    print("✅ MongoDB Connected Successfully!")

except Exception as e:
    print("❌ MongoDB Connection Failed:")
    print(e)