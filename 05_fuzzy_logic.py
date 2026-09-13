"""
Section 15 - Fuzzy Logic-Based Retention Decision Support System
Implemented from first principles (triangular/trapezoidal membership functions,
Mamdani-style rule evaluation, centroid defuzzification) so every step is
transparent and explainable in the report - no external fuzzy library needed.

Inputs:
  1. Churn Risk       - from the trained ANN's predicted probability (0-1)
  2. Customer Value   - proxy = MonthlyCharges (0-120 range in this dataset)
  3. Tenure (months)  - 0-72 range in this dataset

Output:
  Retention Priority score (0-100) -> mapped to Low / Medium / High
"""
import numpy as np
import pandas as pd
import joblib
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 1. Membership functions ----------

def trimf(x, a, b, c):
    """Triangular membership function."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)

def trapmf(x, a, b, c, d):
    """Trapezoidal membership function."""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)

# Churn risk (0-1 probability): Low / Medium / High
def churn_low(x):    return trapmf(x, 0.0, 0.0, 0.20, 0.40)
def churn_medium(x): return trimf(x, 0.30, 0.50, 0.70)
def churn_high(x):   return trapmf(x, 0.60, 0.80, 1.0, 1.0)

# Customer value (monthly charges, ~18-120 in this dataset): Low / Medium / High
def value_low(x):    return trapmf(x, 0, 0, 35, 60)
def value_medium(x): return trimf(x, 45, 70, 95)
def value_high(x):   return trapmf(x, 80, 100, 120, 120)

# Tenure (months, 0-72): New / Medium / Long-term
def tenure_new(x):    return trapmf(x, 0, 0, 6, 18)
def tenure_medium(x): return trimf(x, 12, 30, 48)
def tenure_long(x):   return trapmf(x, 36, 55, 72, 72)

# ---------- 2. Output membership functions (for centroid defuzzification) ----------
# Retention priority score scale 0-100
priority_domain = np.linspace(0, 100, 101)

def out_low(x):    return trapmf(x, 0, 0, 20, 45)
def out_medium(x): return trimf(x, 30, 50, 70)
def out_high(x):   return trapmf(x, 55, 80, 100, 100)

# ---------- 3. Rule base (9 rules - churn x value, tenure used as a modifier) ----------
# Each rule returns a firing strength per output category (Low/Medium/High)

def apply_rules(risk, value, tenure):
    cl, cm, ch = churn_low(risk), churn_medium(risk), churn_high(risk)
    vl, vm, vh = value_low(value), value_medium(value), value_high(value)
    tn, tm, tl = tenure_new(tenure), tenure_medium(tenure), tenure_long(tenure)

    rules_fired = {"Low": [], "Medium": [], "High": []}

    # Rule 1: IF risk LOW AND value LOW THEN priority LOW
    rules_fired["Low"].append(min(cl, vl))
    # Rule 2: IF risk HIGH AND value HIGH THEN priority HIGH
    rules_fired["High"].append(min(ch, vh))
    # Rule 3: IF risk HIGH AND tenure LONG THEN priority HIGH
    rules_fired["High"].append(min(ch, tl))
    # Rule 4: IF risk MEDIUM AND value HIGH THEN priority HIGH
    rules_fired["High"].append(min(cm, vh))
    # Rule 5: IF risk HIGH AND value LOW THEN priority MEDIUM
    rules_fired["Medium"].append(min(ch, vl))
    # Rule 6: IF risk LOW AND value HIGH THEN priority MEDIUM
    rules_fired["Medium"].append(min(cl, vh))
    # Rule 7: IF risk MEDIUM AND value LOW THEN priority MEDIUM
    rules_fired["Medium"].append(min(cm, vl))
    # Rule 8: IF risk MEDIUM AND value MEDIUM THEN priority MEDIUM
    rules_fired["Medium"].append(min(cm, vm))
    # Rule 9: IF risk LOW AND tenure NEW THEN priority LOW (new, low-risk customer needs no urgency)
    rules_fired["Low"].append(min(cl, tn))

    strength = {k: max(v) if v else 0.0 for k, v in rules_fired.items()}
    return strength

def defuzzify(strength):
    """Centroid (center-of-gravity) defuzzification over the output domain."""
    agg = np.zeros_like(priority_domain, dtype=float)
    for i, x in enumerate(priority_domain):
        agg[i] = max(
            min(strength["Low"], out_low(x)),
            min(strength["Medium"], out_medium(x)),
            min(strength["High"], out_high(x)),
        )
    if agg.sum() == 0:
        return 0.0
    return float(np.sum(priority_domain * agg) / np.sum(agg))

def retention_priority(risk, value, tenure):
    strength = apply_rules(risk, value, tenure)
    score = defuzzify(strength)
    if score < 35:
        label = "Low"
    elif score < 65:
        label = "Medium"
    else:
        label = "High"
    return score, label

# ---------- 4. Apply to the test set using ANN churn-risk predictions ----------
import tensorflow as tf
from tensorflow import keras
ann = keras.models.load_model("final_ann.keras")
scaler = joblib.load("final_scaler.joblib")

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()
X_test_s = scaler.transform(X_test)

churn_prob = ann.predict(X_test_s, verbose=0).ravel()

results = pd.DataFrame({
    "ActualChurn": y_test,
    "ChurnRiskProb": churn_prob,
    "MonthlyCharges": X_test["MonthlyCharges"].values,
    "Tenure": X_test["tenure"].values,
})

scores, labels = [], []
for _, row in results.iterrows():
    s, l = retention_priority(row["ChurnRiskProb"], row["MonthlyCharges"], row["Tenure"])
    scores.append(s); labels.append(l)

results["RetentionScore"] = scores
results["RetentionPriority"] = labels

results.to_csv("fuzzy_retention_results.csv", index=False)

print("Retention priority distribution on test set:")
print(results["RetentionPriority"].value_counts())

print("\nMean churn probability by assigned priority:")
print(results.groupby("RetentionPriority")["ChurnRiskProb"].mean())

print("\nSample decisions:")
print(results.sample(8, random_state=1).round(3).to_string(index=False))

# Visualise: retention priority vs actual churn outcome (validation of fuzzy logic)
plt.figure(figsize=(6, 4))
order = ["Low", "Medium", "High"]
actual_churn_rate = results.groupby("RetentionPriority")["ActualChurn"].mean().reindex(order) * 100
actual_churn_rate.plot(kind="bar", color=["#2ecc71", "#f39c12", "#e74c3c"])
plt.title("Actual Churn Rate by Fuzzy Retention Priority (Test Set)")
plt.ylabel("Actual Churn Rate (%)"); plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plot9_fuzzy_validation.png", dpi=140)
plt.close()

print("\nActual churn rate by assigned priority (validates the fuzzy system):")
print(actual_churn_rate.round(1))
