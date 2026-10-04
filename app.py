# kidney-backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import joblib
import numpy as np
import os
import math
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
# Railway/Render kaykhdmo b proxy — bach request.remote_addr ykon l'IP l7a9i9i
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
# FRONTEND_URL = domain(s) ديال Vercel مفرقين بفاصلة، مثلا: https://nephroai.vercel.app
ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"] + [
    o.strip().rstrip("/") for o in os.getenv("FRONTEND_URL", "").split(",") if o.strip()
]
CORS(app, origins=ALLOWED_ORIGINS)

@app.after_request
def apply_headers(response):
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin-allow-popups"
    return response

# ─── Rate limit + auth decorators ─────────────────────────────────────────────
from extensions import limiter
from utils.auth import role_required
limiter.init_app(app)

# ─── MongoDB — optional ───────────────────────────────────────────────────────
try:
    from db.mongo import db, users_collection, history_collection
    MONGO_ENABLED = db is not None
    print("✅ MongoDB ready!" if MONGO_ENABLED else "⚠️ MongoDB disabled")
except Exception as e:
    db = users_collection = history_collection = None
    MONGO_ENABLED = False
    print(f"⚠️ MongoDB disabled: {e}")

# ─── Auth routes ──────────────────────────────────────────────────────────────
try:
    from routes.auth import auth
    app.register_blueprint(auth, url_prefix="/auth")
except Exception as e:
    print(f"⚠️ Auth routes disabled: {e}")

# ─── Chat routes ──────────────────────────────────────────────────────────────
try:
    from routes.chat import chat_bp
    app.register_blueprint(chat_bp)
except Exception as e:
    print(f"⚠️ Chat routes disabled: {e}")

# ─── Load ML Models ───────────────────────────────────────────────────────────
MODEL_PATH  = "model/kidney_model.pkl"
SCALER_PATH = "model/scaler.pkl"
FMAP_PATH   = "model/feature_map.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model not found! Run: python train.py first")

model         = joblib.load(MODEL_PATH)
scaler        = joblib.load(SCALER_PATH)
feature_map   = joblib.load(FMAP_PATH)
FORM_FEATURES = list(feature_map.keys())
print("✅ Gradient Boosting Model loaded!")

# ─── Load XGBoost Model ───────────────────────────────────────────────────────
XGB_PATH = "model/xgb_model.pkl"
if os.path.exists(XGB_PATH):
    xgb_model = joblib.load(XGB_PATH)
    print("✅ XGBoost Model loaded!")
else:
    xgb_model = None
    print("⚠️ XGBoost not found — run: python train.py")

# ─── Load Meta ────────────────────────────────────────────────────────────────
META_PATH = "model/meta.pkl"
if os.path.exists(META_PATH):
    meta         = joblib.load(META_PATH)
    XGB_ACCURACY = meta.get("xgb_accuracy", 87.05)
    print(f"✅ Meta loaded — XGBoost Accuracy: {XGB_ACCURACY}%")
else:
    XGB_ACCURACY = 87.05
    print("⚠️ meta.pkl not found — using default accuracy")

# ─── JWT Secret ───────────────────────────────────────────────────────────────
JWT_SECRET = os.getenv("JWT_SECRET", "kidney_ai_secret")

# ─── Helper: decode token ─────────────────────────────────────────────────────
def decode_token(token):
    if not token:
        return {}
    import jwt as pyjwt
    try:
        decoded = pyjwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        # normalize — support both "user_id" and "id"
        if "user_id" in decoded and "id" not in decoded:
            decoded["id"] = decoded["user_id"]
        return decoded
    except Exception:
        return {}
# ─── Helper: parse form input ─────────────────────────────────────────────────
def parse_input(data: dict):
    """Rj3 (row, missing). Ila chi champ khawi wla machi ra9m, kaydkhl f missing."""
    row, missing = [], []
    for feat in FORM_FEATURES:
        try:
            val = float(data.get(feat, ""))
            if math.isnan(val):
                raise ValueError
            row.append(val)
        except (ValueError, TypeError):
            missing.append(feat)
    return row, missing

def missing_response(missing):
    return jsonify({
        "error":   f"Missing or invalid fields: {', '.join(missing)}",
        "missing": missing,
    }), 400

def save_to_history(data, result, token=""):
    if not MONGO_ENABLED or history_collection is None:
        return
    try:
        from datetime import datetime
        decoded = decode_token(token)
        email   = decoded.get("email", "")
        name    = decoded.get("name", decoded.get("given_name", ""))
        history_collection.insert_one({
            **data, **result,
            "user_email": email,
            "user_name":  name,
            "date":       datetime.utcnow().isoformat(),
        })
    except Exception as ex:
        print(f"⚠️ History save error: {ex}")

# ─── Helper: check admin ──────────────────────────────────────────────────────
def is_admin(token):
    decoded = decode_token(token)
    print(f"🔍 is_admin check — role: '{decoded.get('role')}' email: '{decoded.get('email')}'")
    return decoded.get("role") == "admin"

def get_token():
    return request.headers.get("Authorization", "").replace("Bearer ", "").strip()

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "message":      "NephroAI Backend is running 🚀",
        "mongodb":      MONGO_ENABLED,
        "models": {
            "gradient_boosting": True,
            "xgboost":           xgb_model is not None,
        },
        "xgb_accuracy": XGB_ACCURACY,
    })

@app.route("/predict", methods=["POST"])
@role_required("doctor", "admin")
@limiter.limit("30 per minute")
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        row, missing = parse_input(data)
        if missing:
            return missing_response(missing)
        X_scaled = scaler.transform(np.array([row]))
        proba    = model.predict_proba(X_scaled)[0]

        risk_percent = round(float(proba[1]) * 100, 1)
        prediction   = 1 if proba[1] >= 0.50 else 0
        confidence   = round(float(proba[prediction]) * 100, 1)

        result = {
            "prediction":   prediction,
            "confidence":   confidence,
            "risk_percent": risk_percent,
            "label":        "High Risk" if prediction == 1 else "Low Risk",
            "model_type":   "Gradient Boosting",
        }

        save_to_history(data, result, get_token())
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/xgboost", methods=["POST"])
@role_required("doctor", "admin")
@limiter.limit("30 per minute")
def predict_xgboost():
    if xgb_model is None:
        return jsonify({"error": "XGBoost model not loaded — run: python train.py"}), 503
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        row, missing = parse_input(data)
        if missing:
            return missing_response(missing)
        X_scaled = scaler.transform(np.array([row]))
        proba    = xgb_model.predict_proba(X_scaled)[0]

        risk_percent = round(float(proba[1]) * 100, 1)
        prediction   = 1 if proba[1] >= 0.50 else 0
        confidence   = round(float(proba[prediction]) * 100, 1)

        result = {
            "prediction":   prediction,
            "confidence":   confidence,
            "risk_percent": risk_percent,
            "label":        "High Risk" if prediction == 1 else "Low Risk",
            "model_type":   "XGBoost",
        }

        save_to_history(data, result, get_token())
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
@role_required("doctor", "admin")
@limiter.limit("30 per minute")
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty file"}), 400

        import pandas as pd
        df = pd.read_csv(file)
        if df.empty:
            return jsonify({"error": "Empty CSV file"}), 400

        raw         = df.iloc[0].to_dict()
        REVERSE_MAP = {v: k for k, v in feature_map.items()}
        inputs      = {}
        for dataset_col, form_key in REVERSE_MAP.items():
            val = raw.get(dataset_col, raw.get(form_key, ""))
            inputs[form_key] = str(val) if val != "" else ""

        row, missing = parse_input(inputs)
        if missing:
            return missing_response(missing)
        X_scaled = scaler.transform(np.array([row]))
        proba    = xgb_model.predict_proba(X_scaled)[0] if xgb_model else model.predict_proba(X_scaled)[0]

        risk_percent = round(float(proba[1]) * 100, 1)
        prediction   = 1 if proba[1] >= 0.50 else 0
        confidence   = round(float(proba[prediction]) * 100, 1)

        result = {
            "prediction":   prediction,
            "confidence":   confidence,
            "risk_percent": risk_percent,
            "label":        "High Risk" if prediction == 1 else "Low Risk",
            "model_type":   "XGBoost" if xgb_model else "Gradient Boosting",
        }

        save_to_history(inputs, result, get_token())
        return jsonify({**result, "inputs": inputs})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/history", methods=["GET"])
@role_required("admin")
def history():
    if not MONGO_ENABLED or history_collection is None:
        return jsonify({"error": "MongoDB not connected"}), 503
    try:
        records = list(
            history_collection.find({}, {"_id": 0})
            .sort("date", -1).limit(50)
        )
        return jsonify(records)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/my-history", methods=["GET"])
def my_history():
    if not MONGO_ENABLED or history_collection is None:
        return jsonify({"error": "MongoDB not connected"}), 503
    try:
        token   = get_token()
        decoded = decode_token(token)
        email   = decoded.get("email", "")
        role    = decoded.get("role", "patient")

        print(f"📋 my-history — email: '{email}' role: '{role}'")

        if role == "admin":
            # admin يشوف كل الـ predictions
            records = list(
                history_collection.find({}, {"_id": 0})
                .sort("date", -1).limit(100)
            )
        elif email:
            # doctor و patient يشوفو predictions ديالهم فقط
            records = list(
                history_collection.find({"user_email": email}, {"_id": 0})
                .sort("date", -1).limit(50)
            )
        else:
            records = []

        print(f"📋 Found {len(records)} records for {role} — {email}")
        return jsonify(records)

    except Exception as e:
        print(f"❌ my-history error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/stats", methods=["GET"])
def stats():
    try:
        import pandas as pd

        csv_path = "Chronic_Kidney_Disease_Balanced.csv"
        if not os.path.exists(csv_path):
            csv_path = "Chronic_Kidney_Dsease_data.csv"

        df = pd.read_csv(csv_path)

        total     = len(df)
        high_risk = int(df["Diagnosis"].sum())
        low_risk  = total - high_risk

        bins   = [0, 20, 30, 40, 50, 60, 70, 200]
        labels = ["0-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71+"]
        df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels)
        age_dist = (
            df.groupby(["age_group", "Diagnosis"], observed=True)
              .size().unstack(fill_value=0).reset_index()
        )
        age_distribution = [
            {"age": str(r["age_group"]), "high": int(r.get(1, 0)), "low": int(r.get(0, 0))}
            for _, r in age_dist.iterrows()
        ]

        risk_factors = [
            {"factor": "High Creatinine",  "impact": round(float((df["SerumCreatinine"] > 3).mean() * 100), 1)},
            {"factor": "Diabetes History", "impact": round(float(df[df["FamilyHistoryDiabetes"] == 1]["Diagnosis"].mean() * 100), 1)},
            {"factor": "Hypertension",     "impact": round(float(df[df["FamilyHistoryHypertension"] == 1]["Diagnosis"].mean() * 100), 1)},
            {"factor": "High BUN Levels",  "impact": round(float(df[df["BUNLevels"] > 20]["Diagnosis"].mean() * 100), 1)},
            {"factor": "UTI History",      "impact": round(float(df[df["UrinaryTractInfections"] == 1]["Diagnosis"].mean() * 100), 1)},
            {"factor": "Low Hemoglobin",   "impact": round(float(df[df["HemoglobinLevels"] < 12]["Diagnosis"].mean() * 100), 1)},
        ]

        df["creat_group"] = pd.cut(
            df["SerumCreatinine"],
            bins=[0, 1, 2, 3, 5, 20],
            labels=["Normal(<1)", "Mild(1-2)", "Moderate(2-3)", "High(3-5)", "Critical(5+)"]
        )
        creat_dist = (
            df.groupby(["creat_group", "Diagnosis"], observed=True)
              .size().unstack(fill_value=0).reset_index()
        )
        creatinine_distribution = [
            {"range": str(r["creat_group"]), "high": int(r.get(1, 0)), "low": int(r.get(0, 0))}
            for _, r in creat_dist.iterrows()
        ]

        df["gfr_group"] = pd.cut(
            df["GFR"],
            bins=[0, 15, 30, 60, 90, 200],
            labels=["Failure(<15)", "Severe(15-30)", "Moderate(30-60)", "Mild(60-90)", "Normal(90+)"]
        )
        gfr_dist = (
            df.groupby(["gfr_group", "Diagnosis"], observed=True)
              .size().unstack(fill_value=0).reset_index()
        )
        gfr_distribution = [
            {"stage": str(r["gfr_group"]), "high": int(r.get(1, 0)), "low": int(r.get(0, 0))}
            for _, r in gfr_dist.iterrows()
        ]

        gender_dist = (
            df.groupby(["Gender", "Diagnosis"], observed=True)
              .size().unstack(fill_value=0).reset_index()
        )
        gender_distribution = [
            {
                "gender": "Female" if int(r["Gender"]) == 0 else "Male",
                "high":   int(r.get(1, 0)),
                "low":    int(r.get(0, 0)),
            }
            for _, r in gender_dist.iterrows()
        ]

        return jsonify({
            "total_records":           total,
            "high_risk":               high_risk,
            "low_risk":                low_risk,
            "model_accuracy":          XGB_ACCURACY,
            "features_count":          12,
            "age_distribution":        age_distribution,
            "risk_factors":            risk_factors,
            "creatinine_distribution": creatinine_distribution,
            "gfr_distribution":        gfr_distribution,
            "gender_distribution":     gender_distribution,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── Admin Routes ─────────────────────────────────────────────────────────────

@app.route("/admin/users", methods=["GET"])
def admin_get_users():
    if not MONGO_ENABLED or users_collection is None:
        return jsonify({"error": "MongoDB not connected"}), 503
    try:
        if not is_admin(get_token()):
            return jsonify({"error": "Admin access required"}), 403

        users = list(users_collection.find({}, {"_id": 0, "password": 0}))
        return jsonify(users)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/admin/users/role", methods=["PUT"])
def admin_update_role():
    if not MONGO_ENABLED or users_collection is None:
        return jsonify({"error": "MongoDB not connected"}), 503
    try:
        if not is_admin(get_token()):
            return jsonify({"error": "Admin access required"}), 403

        data     = request.get_json()
        email    = data.get("email")
        new_role = data.get("role")

        if not email or not new_role:
            return jsonify({"error": "email and role required"}), 400
        if new_role not in ["patient", "doctor", "admin"]:
            return jsonify({"error": "Invalid role"}), 400

        result = users_collection.update_one(
            {"email": email},
            {"$set": {"role": new_role}}
        )

        if result.modified_count == 0:
            return jsonify({"error": "User not found"}), 404

        return jsonify({"success": True, "message": f"Role updated to {new_role}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/admin/users/<path:email>", methods=["DELETE"])
def admin_delete_user(email):
    if not MONGO_ENABLED or users_collection is None:
        return jsonify({"error": "MongoDB not connected"}), 503
    try:
        token   = get_token()
        decoded = decode_token(token)

        if not is_admin(token):
            return jsonify({"error": "Admin access required"}), 403
        if decoded.get("email") == email:
            return jsonify({"error": "Cannot delete your own account"}), 400

        result = users_collection.delete_one({"email": email})
        if result.deleted_count == 0:
            return jsonify({"error": "User not found"}), 404

        return jsonify({"success": True, "message": "User deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/admin/users/create", methods=["POST"])
def admin_create_user():
    if not MONGO_ENABLED or users_collection is None:
        return jsonify({"error": "MongoDB not connected"}), 503
    try:
        if not is_admin(get_token()):
            return jsonify({"error": "Admin access required"}), 403

        data = request.get_json()
        name     = data.get("name", "")
        email    = data.get("email", "")
        password = data.get("password", "")
        role     = data.get("role", "patient")

        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400
        if role not in ["patient", "doctor", "admin"]:
            return jsonify({"error": "Invalid role"}), 400
        if users_collection.find_one({"email": email}):
            return jsonify({"error": "User already exists"}), 400

        import bcrypt
        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        users_collection.insert_one({
            "name":     name,
            "email":    email,
            "password": hashed_pw,
            "role":     role,
        })

        return jsonify({"success": True, "message": f"User {email} created as {role}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/doctor/patients/create", methods=["POST"])
def doctor_create_patient():
    if not MONGO_ENABLED or users_collection is None:
        return jsonify({"error": "MongoDB not connected"}), 503
    try:
        token   = get_token()
        decoded = decode_token(token)
        role    = decoded.get("role", "")

        if role not in ["doctor", "admin"]:
            return jsonify({"error": "Doctor or Admin access required"}), 403

        data     = request.get_json()
        name     = data.get("name", "")
        email    = data.get("email", "")
        password = data.get("password", "")

        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400
        if users_collection.find_one({"email": email}):
            return jsonify({"error": "User already exists"}), 400

        import bcrypt
        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        users_collection.insert_one({
            "name":       name,
            "email":      email,
            "password":   hashed_pw,
            "role":       "patient",
            "created_by": decoded.get("email", ""),
        })

        return jsonify({"success": True, "message": f"Patient {email} created"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/admin/stats", methods=["GET"])
def admin_stats():
    if not MONGO_ENABLED:
        return jsonify({"error": "MongoDB not connected"}), 503
    try:
        if not is_admin(get_token()):
            return jsonify({"error": "Admin access required"}), 403

        from datetime import datetime, timedelta

        uc = users_collection
        hc = history_collection

        total_users    = uc.count_documents({})                if uc is not None else 0
        total_doctors  = uc.count_documents({"role": "doctor"})  if uc is not None else 0
        total_patients = uc.count_documents({"role": "patient"}) if uc is not None else 0
        total_admins   = uc.count_documents({"role": "admin"})   if uc is not None else 0
        total_preds    = hc.count_documents({})                if hc is not None else 0

        today       = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        preds_today = hc.count_documents({
            "date": {"$gte": today.isoformat()}
        }) if hc is not None else 0

        week_ago   = (datetime.utcnow() - timedelta(days=7)).isoformat()
        preds_week = hc.count_documents({
            "date": {"$gte": week_ago}
        }) if hc is not None else 0

        high_risk = hc.count_documents({"label": "High Risk"}) if hc is not None else 0
        low_risk  = hc.count_documents({"label": "Low Risk"})  if hc is not None else 0

        return jsonify({
            "total_users":    total_users,
            "total_doctors":  total_doctors,
            "total_patients": total_patients,
            "total_admins":   total_admins,
            "total_preds":    total_preds,
            "preds_today":    preds_today,
            "preds_week":     preds_week,
            "high_risk":      high_risk,
            "low_risk":       low_risk,
        })
    except Exception as e:
        print(f"❌ admin_stats error: {str(e)}")
        return jsonify({"error": str(e)}), 500
# ─── Test DB ──────────────────────────────────────────────────────────────────
@app.route("/test-db")
def test_db():
    if not MONGO_ENABLED:
        return jsonify({"status": "disabled", "message": "MongoDB not connected"})
    try:
        db.command("ping")
        return jsonify({"status": "success", "message": "MongoDB is connected ✅"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))