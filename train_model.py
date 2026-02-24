import pandas as pd
import joblib
import json
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
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
# STEP 3 — Train/Test Split (Enterprise Standard)
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  # preserves class distribution
)

print("Train/Test split complete.")


# ======================================================
# STEP 4 — Build Pipeline
# ======================================================

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    ))
])

print("Training model...")

pipeline.fit(X_train, y_train)


# ======================================================
# STEP 5 — Evaluate Model
# ======================================================

preds = pipeline.predict(X_test)
probs = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, preds)
auc = roc_auc_score(y_test, probs)

print("Model Evaluation Complete.")
print("Accuracy:", round(accuracy, 4))
print("AUC:", round(auc, 4))


# ======================================================
# STEP 6 — Save Model & Metrics
# ======================================================

if not os.path.exists("models"):
    os.makedirs("models")

# Save trained model
joblib.dump(pipeline, "models/bp_model.pkl")

# Save metrics
metrics = {
    "accuracy": round(accuracy, 4),
    "auc": round(auc, 4)
}

with open("models/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Model and metrics saved successfully.")
print("Training pipeline completed.")