import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from modules.data_loader import load_dataset_1, load_dataset_2
from modules.feature_engineering import aggregate_activity_data, engineer_profile_features
from modules.temp_join_engine import create_temp_join


# ======================================================
# STEP 1 — Load and Prepare Data
# ======================================================

print("Loading datasets...")

df1 = load_dataset_1()
df2 = load_dataset_2()

df1_feat = engineer_profile_features(df1)
df2_agg = aggregate_activity_data(df2)
df = create_temp_join(df1_feat, df2_agg)

print("Data preparation complete.")


# ======================================================
# STEP 2 — Feature Selection
# ======================================================

features = [
    "Age",
    "BMI",
    "Level_of_Hemoglobin",
    "Genetic_Pedigree_Coefficient",
    "Smoking",
    "salt_content_in_the_diet",
    "alcohol_consumption_per_day",
    "Level_of_Stress",
    "avg_steps"
]

X = df[features]
y = df["Blood_Pressure_Abnormality"]


# ======================================================
# STEP 3 — Train/Test Split
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train/Test split complete.")


# ======================================================
# STEP 4 — Preprocessing
# ======================================================

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_imputed = imputer.fit_transform(X_train)
X_train_scaled = scaler.fit_transform(X_train_imputed)

X_test_imputed = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test_imputed)


# ======================================================
# STEP 5 — Train Model
# ======================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

print("Training model...")
model.fit(X_train_scaled, y_train)


# ======================================================
# STEP 6 — Evaluate Model
# ======================================================

preds = model.predict(X_test_scaled)
probs = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, preds)
auc = roc_auc_score(y_test, probs)

print("Model Evaluation Complete.")
print("Accuracy:", round(accuracy, 4))
print("AUC:", round(auc, 4))


# ======================================================
# STEP 7 — Save Model Components
# ======================================================

if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump(model, "models/bp_model.pkl")
joblib.dump(imputer, "models/imputer.pkl")
joblib.dump(scaler, "models/scaler.pkl")

metrics = {
    "accuracy": round(accuracy, 4),
    "auc": round(auc, 4)
}

with open("models/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Model, preprocessing components, and metrics saved successfully.")
print("Training pipeline completed.")