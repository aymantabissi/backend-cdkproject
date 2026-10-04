from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db     = client["kidney_ai"]

email = "YOUR_EMAIL_HERE"  # ← بدّل بالـ email ديالك

result = db.users.update_one(
    {"email": email},
    {"$set": {"role": "admin"}}
)

print(f"Modified: {result.modified_count}")
print("Done! Now logout and login again.")