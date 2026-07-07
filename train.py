"""
Smart Lender -- model training

Loads data/loan_data.csv, cleans/encodes it, trains four classifiers
(Decision Tree, Random Forest, KNN, XGBoost), evaluates each, and saves the
best-performing one (plus the fitted encoders/scaler) to model/model.pkl so
app.py can load it for real-time predictions.

Run:  python train.py
"""
import pickle

import matplotlib
matplotlib.use("Agg")  # no GUI backend needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost is not installed -- skipping the XGBoost model.")
    print("Install it with: pip install xgboost")

CATEGORICAL_COLS = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area",
]
NUMERIC_COLS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History",
]

# ---------------------------------------------------------------------------
# 1. Load + clean
# ---------------------------------------------------------------------------
df = pd.read_csv("data/loan_data.csv")

# Fill missing categoricals with the mode, numerics with the median
for col in CATEGORICAL_COLS:
    df[col] = df[col].fillna(df[col].mode()[0])
for col in NUMERIC_COLS:
    df[col] = df[col].fillna(df[col].median())

# ---------------------------------------------------------------------------
# 2. Encode categoricals
# ---------------------------------------------------------------------------
encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

target_encoder = LabelEncoder()
df["Loan_Status"] = target_encoder.fit_transform(df["Loan_Status"])  # N=0, Y=1

feature_cols = CATEGORICAL_COLS + NUMERIC_COLS
X = df[feature_cols]
y = df["Loan_Status"]

# ---------------------------------------------------------------------------
# 3. Train/test split + scaling (KNN needs scaled features; tree models don't
#    strictly need it, but scaling doesn't hurt them here since we keep a
#    single shared feature matrix for simplicity)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. Train + evaluate each model
# ---------------------------------------------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=9),
}
if HAS_XGB:
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        eval_metric="logloss", random_state=42,
    )

results = {}
for name, model in models.items():
    # KNN benefits from scaled input; tree-based models are trained on raw
    # (unscaled) features so importances stay interpretable.
    if name == "KNN":
        model.fit(X_train_scaled, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    else:
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

    results[name] = {"model": model, "train_acc": train_acc, "test_acc": test_acc}
    print(f"{name:15s}  train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

# ---------------------------------------------------------------------------
# 5. Pick the best model by test accuracy
# ---------------------------------------------------------------------------
best_name = max(results, key=lambda n: results[n]["test_acc"])
best_model = results[best_name]["model"]
uses_scaled_input = best_name == "KNN"
print(f"\nBest model: {best_name} (test_acc={results[best_name]['test_acc']:.3f})")

y_pred = best_model.predict(X_test_scaled if uses_scaled_input else X_test)
print("\nClassification report for best model:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# ---------------------------------------------------------------------------
# 6. Save a confusion-matrix chart (Matplotlib/Seaborn requirement) + save model
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_,
)
plt.title(f"{best_name} — Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png", dpi=120)
print("Saved model/confusion_matrix.png")

with open("model/model.pkl", "wb") as f:
    pickle.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "uses_scaled_input": uses_scaled_input,
            "scaler": scaler,
            "encoders": encoders,
            "target_encoder": target_encoder,
            "feature_cols": feature_cols,
        },
        f,
    )
print("Saved model/model.pkl")
