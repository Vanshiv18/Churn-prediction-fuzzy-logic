"""
Section 9 - Application of Machine-Learning Models
"""
import pandas as pd, numpy as np, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("telco_model_ready.csv")
# drop helper columns not meant for modelling (keep only encoded numeric features)
df = df.drop(columns=["TenureGroup", "ChargeBand"], errors="ignore")

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

results = {}

def evaluate(name, model, Xtr, Xte):
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    probs = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else None
    res = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs) if probs is not None else None,
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    results[name] = res
    print(f"\n=== {name} ===")
    for k, v in res.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")
    print("Confusion matrix [[TN,FP],[FN,TP]]:", res["confusion_matrix"])

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    fig, ax = plt.subplots(figsize=(4.2, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"cm_{safe_name}.png", dpi=140)
    plt.close()
    return model

log_reg = evaluate("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42), X_train_s, X_test_s)
dtree   = evaluate("Decision Tree", DecisionTreeClassifier(max_depth=6, random_state=42), X_train, X_test)
rforest = evaluate("Random Forest", RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42), X_train, X_test)

# Feature importance from Random Forest -> "strongest predictors" business question
importances = pd.Series(rforest.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 predictors (Random Forest importance):")
print(importances.head(10))

with open("ml_results.json", "w") as f:
    json.dump(results, f, indent=2)
importances.head(10).to_csv("top_predictors.csv")

# save train/test splits for ANN stage to reuse identical split
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("\nSaved ml_results.json, top_predictors.csv, and train/test splits.")
