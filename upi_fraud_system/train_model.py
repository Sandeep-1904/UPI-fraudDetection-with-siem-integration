"""
train_model.py
==============
Full ML pipeline:
  1. Load upi_transactions.csv
  2. Label-encode categorical features (location, device_type)
  3. Engineer 5 derived risk indicator features
  4. 80/20 train-test split (stratified)
  5. SMOTE on training split only
  6. Train Random Forest  (200 estimators, max_depth=12, class_weight=balanced)
  7. Train Gradient Boosting (100 estimators, lr=0.1)
  8. Evaluate on held-out test set (AUC-ROC, classification report)
  9. Serialize models + encoders to models/ via joblib
"""

import os, joblib, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing     import LabelEncoder
from sklearn.model_selection   import train_test_split
from sklearn.metrics           import (classification_report, roc_auc_score,
                                        confusion_matrix)
from imblearn.over_sampling    import SMOTE

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)

# ══════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════
print("Loading dataset...")
df = pd.read_csv("upi_transactions.csv")
print(f"  Rows: {len(df):,}  |  Fraud: {df.is_fraud.sum()}  ({df.is_fraud.mean()*100:.1f}%)")

# ══════════════════════════════════════════════════════
# 2. LABEL-ENCODE CATEGORICALS
# ══════════════════════════════════════════════════════
print("\nEncoding categorical features...")
le_location = LabelEncoder()
le_device   = LabelEncoder()

df["location_enc"]    = le_location.fit_transform(df["location"])
df["device_type_enc"] = le_device.fit_transform(df["device_type"])

joblib.dump(le_location, "models/le_location.pkl")
joblib.dump(le_device,   "models/le_device.pkl")
print("  Encoders saved → models/le_location.pkl, models/le_device.pkl")

# ══════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING  (5 derived binary indicators)
# ══════════════════════════════════════════════════════
print("\nEngineering risk indicator features...")

df["is_odd_hour"]      = ((df["hour_of_day"] < 6) | (df["hour_of_day"] > 22)).astype(int)
df["is_large_amount"]  = (df["amount"] > 50_000).astype(int)
df["is_high_frequency"]= (df["txn_frequency_1hr"] > 10).astype(int)
df["risk_device"]      = df["device_type"].str.contains(
                             "Unknown|Emulator|Rooted|VM", case=False
                         ).astype(int)
df["risk_location"]    = df["location"].str.contains(
                             "Unknown|Foreign|Spoofed|TOR|VPN", case=False
                         ).astype(int)

FEATURE_COLS = [
    # raw numeric
    "amount", "hour_of_day", "day_of_week",
    "is_new_receiver", "txn_frequency_1hr", "failed_pin_attempts", "vpn_used",
    # encoded categorical
    "location_enc", "device_type_enc",
    # 5 engineered indicators
    "is_odd_hour", "is_large_amount", "is_high_frequency",
    "risk_device", "risk_location",
]

X = df[FEATURE_COLS].values
y = df["is_fraud"].values
print(f"  Feature matrix: {X.shape}")

# ══════════════════════════════════════════════════════
# 4. TRAIN / TEST SPLIT  (80 / 20, stratified)
# ══════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"  Train fraud: {y_train.sum()}  |  Test fraud: {y_test.sum()}")

# ══════════════════════════════════════════════════════
# 5. SMOTE  (training split only)
# ══════════════════════════════════════════════════════
print("\nApplying SMOTE to training split only...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"  Before SMOTE → Fraud: {y_train.sum():,}  |  Normal: {(y_train==0).sum():,}")
print(f"  After  SMOTE → Fraud: {y_train_bal.sum():,}  |  Normal: {(y_train_bal==0).sum():,}")

# ══════════════════════════════════════════════════════
# 6. TRAIN RANDOM FOREST
# ══════════════════════════════════════════════════════
print("\nTraining Random Forest (200 estimators, max_depth=12)...")
rf = RandomForestClassifier(
    n_estimators  = 200,
    max_depth     = 12,
    class_weight  = "balanced",
    random_state  = 42,
    n_jobs        = -1,
)
rf.fit(X_train_bal, y_train_bal)

rf_prob = rf.predict_proba(X_test)[:, 1]
rf_pred = rf.predict(X_test)
rf_auc  = roc_auc_score(y_test, rf_prob)

print(f"\n  Random Forest — AUC-ROC: {rf_auc:.4f}")
print(classification_report(y_test, rf_pred, target_names=["Normal","Fraud"]))

joblib.dump(rf, "models/random_forest.pkl")
print("  Saved → models/random_forest.pkl")

# ══════════════════════════════════════════════════════
# 7. TRAIN GRADIENT BOOSTING
# ══════════════════════════════════════════════════════
print("\nTraining Gradient Boosting (100 estimators, lr=0.1)...")
gb = GradientBoostingClassifier(
    n_estimators  = 100,
    learning_rate = 0.1,
    max_depth     = 5,
    random_state  = 42,
)
gb.fit(X_train_bal, y_train_bal)

gb_prob = gb.predict_proba(X_test)[:, 1]
gb_pred = gb.predict(X_test)
gb_auc  = roc_auc_score(y_test, gb_prob)

print(f"\n  Gradient Boosting — AUC-ROC: {gb_auc:.4f}")
print(classification_report(y_test, gb_pred, target_names=["Normal","Fraud"]))

joblib.dump(gb, "models/gradient_boosting.pkl")
print("  Saved → models/gradient_boosting.pkl")

# ══════════════════════════════════════════════════════
# 8. SAVE FEATURE LIST
# ══════════════════════════════════════════════════════
joblib.dump(FEATURE_COLS, "models/feature_cols.pkl")
print("\n  Feature list saved → models/feature_cols.pkl")

# ══════════════════════════════════════════════════════
# 9. SUMMARY
# ══════════════════════════════════════════════════════
print("\n" + "="*48)
print("  Training Complete")
print("="*48)
print(f"  Random Forest   AUC-ROC : {rf_auc:.4f}  {'PASS ✓' if rf_auc >= 0.95 else 'CHECK'}")
print(f"  Gradient Boost  AUC-ROC : {gb_auc:.4f}  {'PASS ✓' if gb_auc >= 0.95 else 'CHECK'}")
print("\n  Saved files:")
for f in ["models/random_forest.pkl","models/gradient_boosting.pkl",
          "models/le_location.pkl","models/le_device.pkl","models/feature_cols.pkl"]:
    print(f"    {f}")
