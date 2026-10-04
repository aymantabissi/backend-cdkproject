# kidney-backend/train.py
# شغّل مرة وحدة: python train.py
# يسيف الـ model في model/kidney_model.pkl
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Utilisation d'imblearn pour le suréchantillonnage de la classe minoritaire
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("❌ Erreur : La bibliothèque 'imblearn' est absente.")
    print("👉 Exécutez : pip install imbalanced-learn")
    exit()

# Configuration des colonnes (Mapping Frontend -> Dataset)
FEATURE_MAP = {
    "age":           "Age",
    "bp":            "SystolicBP",
    "creatinine":    "SerumCreatinine",
    "urea":          "BUNLevels",
    "hemoglobin":    "HemoglobinLevels",
    "sodium":        "SerumElectrolytesSodium",
    "potassium":     "SerumElectrolytesPotassium",
    "protein":       "ProteinInUrine",
    "glucose":       "FastingBloodSugar",
    "rbc":           "UrinaryTractInfections",
    "diabetes":      "FamilyHistoryDiabetes",
    "hypertension":  "FamilyHistoryHypertension",
}

DATA_FEATURES = list(FEATURE_MAP.values())

def train():
    # 1. Chargement des données
    csv_path = "Chronic_Kidney_Dsease_data.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Fichier introuvable : {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"✅ Dataset chargé : {df.shape[0]} lignes")

    # 2. Préparation des variables
    X = df[DATA_FEATURES].copy()
    y = df["Diagnosis"]

    # 3. Traitement des valeurs manquantes (médiane)
    X = X.fillna(X.median())

    # 4. Division des données (Stratified pour garder les proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Mise à l'échelle (Standardisation)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 6. Équilibrage des données avec SMOTE (Crucial pour votre dataset)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"📊 Données équilibrées : {len(y_train_res)} échantillons d'entraînement")

    # 7. Entraînement du Gradient Boosting
    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train_res, y_train_res)

    # 8. Évaluation
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "="*30)
    print(f"✅ Gradient Boosting (avec SMOTE) entraîné !")
    print(f"📊 Précision globale : {acc * 100:.2f}%")
    print("="*30)
    print(classification_report(y_test, y_pred, target_names=['Low Risk (0)', 'High Risk (1)']))

    # 9. Sauvegarde
    os.makedirs("model", exist_ok=True)
    joblib.dump(model,  "model/kidney_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(FEATURE_MAP, "model/feature_map.pkl")
    print("\n✅ Modèle sauvegardé avec succès dans le dossier /model/")

# التعديل المهم هنا: حيدنا الفراغ اللي كان مورا main
if __name__ == "__main__":
    train()

"""
# kidney-backend/train.py
# Commande : python train.py
# Ce script entraîne un modèle Gradient Boosting avec SMOTE pour équilibrer les classes.

"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Utilisation d'imblearn pour le suréchantillonnage de la classe minoritaire
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("❌ Erreur : La bibliothèque 'imblearn' est absente.")
    print("👉 Exécutez : pip install imbalanced-learn")
    exit()

# Configuration des colonnes (Mapping Frontend -> Dataset)
FEATURE_MAP = {
    "age":           "Age",
    "bp":            "SystolicBP",
    "creatinine":    "SerumCreatinine",
    "urea":          "BUNLevels",
    "hemoglobin":    "HemoglobinLevels",
    "sodium":        "SerumElectrolytesSodium",
    "potassium":     "SerumElectrolytesPotassium",
    "protein":       "ProteinInUrine",
    "glucose":       "FastingBloodSugar",
    "rbc":           "UrinaryTractInfections",
    "diabetes":      "FamilyHistoryDiabetes",
    "hypertension":  "FamilyHistoryHypertension",
}

DATA_FEATURES = list(FEATURE_MAP.values())

def train():
    # 1. Chargement des données
    csv_path = "Chronic_Kidney_Dsease_data.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Fichier introuvable : {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"✅ Dataset chargé : {df.shape[0]} lignes")

    # 2. Préparation des variables
    X = df[DATA_FEATURES].copy()
    y = df["Diagnosis"]

    # 3. Traitement des valeurs manquantes (médiane)
    X = X.fillna(X.median())

    # 4. Division des données (Stratified pour garder les proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Mise à l'échelle (Standardisation)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 6. Équilibrage des données avec SMOTE (Crucial pour votre dataset)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"📊 Données équilibrées : {len(y_train_res)} échantillons d'entraînement")

    # 7. Entraînement du Gradient Boosting
    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train_res, y_train_res)

    # 8. Évaluation
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "="*30)
    print(f"✅ Gradient Boosting (avec SMOTE) entraîné !")
    print(f"📊 Précision globale : {acc * 100:.2f}%")
    print("="*30)
    print(classification_report(y_test, y_pred, target_names=['Low Risk (0)', 'High Risk (1)']))

    # 9. Sauvegarde
    os.makedirs("model", exist_ok=True)
    joblib.dump(model,  "model/kidney_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(FEATURE_MAP, "model/feature_map.pkl")
    print("\n✅ Modèle sauvegardé avec succès dans le dossier /model/")

# التعديل المهم هنا: حيدنا الفراغ اللي كان مورا main
if __name__ == "__main__":
    train()


"""


## kidney-backend/train.py

import os
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import joblib

from xgboost import XGBClassifier

# =========================
# CONFIG
# =========================

FEATURE_MAP = {
    "age":          "Age",
    "bp":           "SystolicBP",
    "creatinine":   "SerumCreatinine",
    "urea":         "BUNLevels",
    "hemoglobin":   "HemoglobinLevels",
    "sodium":       "SerumElectrolytesSodium",
    "potassium":    "SerumElectrolytesPotassium",
    "protein":      "ProteinInUrine",
    "glucose":      "FastingBloodSugar",
    "rbc":          "UrinaryTractInfections",
    "diabetes":     "FamilyHistoryDiabetes",
    "hypertension": "FamilyHistoryHypertension",
}

DATA_FEATURES = list(FEATURE_MAP.values())

# =========================
# EVALUATE
# =========================

def evaluate_model(name, model, X_train, y_train, X_test, y_test, sample_weight=None):
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="macro")

    print("\n" + "="*60)
    print(f"🤖 Model: {name}")
    print(f"📊 Accuracy: {acc * 100:.2f}%  |  Macro F1: {f1 * 100:.2f}%")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=["Low Risk (0)", "High Risk (1)"]))

    return model, acc, f1

# =========================
# TRAIN
# =========================

def train():
    # 1. Load dataset
    csv_path = "Chronic_Kidney_Disease_Balanced.csv"
    if not os.path.exists(csv_path):
        print("❌ Dataset introuvable — ضع الـ CSV في نفس مجلد train.py")
        return

    df = pd.read_csv(csv_path)
    print(f"✅ Dataset loaded: {df.shape}")

    # 2. Features / target
    X = df[DATA_FEATURES].copy()
    y = df["Diagnosis"].astype(int)

    # 3. Missing values
    X = X.fillna(X.median())

    # 4. Class distribution
    n_low  = int((y == 0).sum())
    n_high = int((y == 1).sum())
    print(f"\n📊 Class distribution:")
    print(f"   Low Risk  (0): {n_low}")
    print(f"   High Risk (1): {n_high}")
    print(f"   Ratio: {n_high/n_low:.2f}x  ({'Balanced ✅' if abs(n_high - n_low) < 100 else 'Imbalanced ⚠️'})")

    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 7. Dataset balanced — ما محتاجناش SMOTEENN
    print("\n✅ Dataset already balanced — skipping resampling")
    X_train_res = X_train_scaled
    y_train_res = y_train

    print(f"📊 Training set: {len(y_train_res)} samples")
    print(f"   Low Risk:  {int((y_train_res == 0).sum())}")
    print(f"   High Risk: {int((y_train_res == 1).sum())}")

    # sample_weight للـ Gradient Boosting
    sample_weights = compute_sample_weight("balanced", y_train_res)

    # =========================
    # MODELS
    # =========================

    models = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=0.1,
                solver="lbfgs"
            ),
            None
        ),

        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                class_weight="balanced",
                min_samples_leaf=2,
                random_state=42
            ),
            None
        ),

        "Gradient Boosting": (
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                min_samples_leaf=5,
                random_state=42
            ),
            sample_weights
        ),

        "XGBoost": (
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                scale_pos_weight=1,  # 1 لأن dataset balanced
                random_state=42
            ),
            None
        ),
    }

    # =========================
    # TRAIN ALL
    # =========================

    best_model    = None
    best_score_f1 = 0
    best_name     = ""
    results       = {}

    for name, (model, sw) in models.items():
        trained_model, acc, f1 = evaluate_model(
            name, model,
            X_train_res, y_train_res,
            X_test_scaled, y_test,
            sample_weight=sw
        )
        results[name] = {"accuracy": acc, "f1": f1}

        if f1 > best_score_f1:
            best_score_f1 = f1
            best_model    = trained_model
            best_name     = name

    # =========================
    # SUMMARY
    # =========================

    print("\n📊 FINAL COMPARISON")
    print("="*55)
    print(f"{'Model':<25} {'Accuracy':>10} {'Macro F1':>10}")
    print("-"*55)
    for k, v in results.items():
        marker = " ← BEST" if k == best_name else ""
        print(f"{k:<25} {v['accuracy']*100:>9.2f}%  {v['f1']*100:>9.2f}%{marker}")
    print("="*55)
    print(f"\n🏆 Best Model (by Macro F1): {best_name} ({best_score_f1*100:.2f}%)")

    # =========================
    # SAVE
    # =========================

    os.makedirs("model", exist_ok=True)

    joblib.dump(scaler,      "model/scaler.pkl")
    joblib.dump(FEATURE_MAP, "model/feature_map.pkl")
    print("\n✅ scaler & feature_map saved")

    joblib.dump(models["Gradient Boosting"][0], "model/kidney_model.pkl")
    print("✅ Gradient Boosting saved → model/kidney_model.pkl")

    joblib.dump(best_model, "model/xgb_model.pkl")
    print("✅ XGBoost saved → model/xgb_model.pkl")

    # حفظ accuracy الحقيقية ديال XGBoost
    xgb_accuracy = results["XGBoost"]["accuracy"] * 100
    xgb_f1       = results["XGBoost"]["f1"] * 100

    joblib.dump({
        "scale_pos_weight": 1,
        "best_model":       best_name,
        "xgb_accuracy":     round(xgb_accuracy, 2),
        "xgb_f1":           round(xgb_f1, 2),
        "results":          results,
    }, "model/meta.pkl")
    print(f"✅ meta saved")
    print(f"\n📊 XGBoost — Accuracy: {xgb_accuracy:.2f}% | Macro F1: {xgb_f1:.2f}%")

if __name__ == "__main__":
    train()