"""
Additional charts to make the report more thorough:
- More EDA (distributions, boxplots, additional segment breakdowns)
- ROC curves for all models
- Precision-Recall curves
- Fuzzy membership function visualisations
- Fuzzy decision surface heatmap
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import joblib
from tensorflow import keras

# THEME COLORS
PRIMARY = "#1B4965"
ACCENT = "#EE964B"
GREEN = "#2ecc71"
RED = "#e74c3c"
ORANGE = "#f39c12"
sns.set_style("whitegrid")
plt.rcParams["axes.edgecolor"] = "#444444"

df = pd.read_csv("telco_clean.csv")
df["ChurnLabel"] = df["Churn"].map({1: "Yes", 0: "No"})

# ================= EXTRA EDA =================

# Tenure distribution histogram
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="tenure", hue="ChurnLabel", multiple="stack", palette=[GREEN, RED], bins=30)
plt.title("Distribution of Customer Tenure (Months)", fontsize=13, color=PRIMARY, fontweight="bold")
plt.xlabel("Tenure (months)"); plt.ylabel("Number of Customers")
plt.tight_layout(); plt.savefig("plotX1_tenure_hist.png", dpi=140); plt.close()

# Monthly charges distribution histogram
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="MonthlyCharges", hue="ChurnLabel", multiple="stack", palette=[GREEN, RED], bins=30)
plt.title("Distribution of Monthly Charges", fontsize=13, color=PRIMARY, fontweight="bold")
plt.xlabel("Monthly Charges"); plt.ylabel("Number of Customers")
plt.tight_layout(); plt.savefig("plotX2_charges_hist.png", dpi=140); plt.close()

# Boxplot: monthly charges by churn
plt.figure(figsize=(5, 4))
sns.boxplot(data=df, x="ChurnLabel", y="MonthlyCharges", hue="ChurnLabel", palette=[GREEN, RED], legend=False)
plt.title("Monthly Charges by Churn Status", fontsize=13, color=PRIMARY, fontweight="bold")
plt.xlabel("Churn"); plt.ylabel("Monthly Charges")
plt.tight_layout(); plt.savefig("plotX3_charges_boxplot.png", dpi=140); plt.close()

# Churn by Internet Service
plt.figure(figsize=(6, 4))
ct5 = pd.crosstab(df["InternetService"], df["ChurnLabel"], normalize="index") * 100
ct5.plot(kind="bar", stacked=True, color=[GREEN, RED])
plt.title("Churn Rate by Internet Service Type", fontsize=13, color=PRIMARY, fontweight="bold")
plt.ylabel("% of Customers"); plt.xlabel("Internet Service"); plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout(); plt.savefig("plotX4_churn_by_internet.png", dpi=140); plt.close()

# Churn by Senior Citizen
plt.figure(figsize=(5, 4))
df["SeniorLabel"] = df["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})
ct6 = pd.crosstab(df["SeniorLabel"], df["ChurnLabel"], normalize="index") * 100
ct6.plot(kind="bar", stacked=True, color=[GREEN, RED])
plt.title("Churn Rate: Senior vs Non-Senior Citizens", fontsize=13, color=PRIMARY, fontweight="bold")
plt.ylabel("% of Customers"); plt.xlabel(""); plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout(); plt.savefig("plotX5_churn_by_senior.png", dpi=140); plt.close()

# Churn by Dependents
plt.figure(figsize=(5, 4))
df["DependentsLabel"] = df["Dependents"].map({1: "Has Dependents", 0: "No Dependents"})
ct7 = pd.crosstab(df["DependentsLabel"], df["ChurnLabel"], normalize="index") * 100
ct7.plot(kind="bar", stacked=True, color=[GREEN, RED])
plt.title("Churn Rate by Dependents Status", fontsize=13, color=PRIMARY, fontweight="bold")
plt.ylabel("% of Customers"); plt.xlabel(""); plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout(); plt.savefig("plotX6_churn_by_dependents.png", dpi=140); plt.close()

print("Extra EDA charts saved.")
print("Churn by internet service:\n", ct5["Yes"].round(1))
print("Churn by senior citizen:\n", ct6["Yes"].round(1))
print("Churn by dependents:\n", ct7["Yes"].round(1))

# ================= ROC & PRECISION-RECALL CURVES =================
df_model = pd.read_csv("telco_model_ready.csv").drop(columns=["TenureGroup", "ChargeBand"], errors="ignore")
X_train = pd.read_csv("X_train.csv"); X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test  = pd.read_csv("y_test.csv").values.ravel()

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_s, y_train)
dtree = DecisionTreeClassifier(max_depth=6, random_state=42).fit(X_train, y_train)
rforest = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42).fit(X_train, y_train)

ann_scaler = joblib.load("final_scaler.joblib")
ann_model = keras.models.load_model("final_ann.keras")
X_test_ann_s = ann_scaler.transform(X_test)

probs = {
    "Logistic Regression": log_reg.predict_proba(X_test_s)[:, 1],
    "Decision Tree": dtree.predict_proba(X_test)[:, 1],
    "Random Forest": rforest.predict_proba(X_test)[:, 1],
    "ANN (Keras)": ann_model.predict(X_test_ann_s, verbose=0).ravel(),
}

colors = {"Logistic Regression": "#1B4965", "Decision Tree": "#EE964B",
          "Random Forest": "#2ecc71", "ANN (Keras)": "#e74c3c"}

plt.figure(figsize=(6.5, 5.5))
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=colors[name], linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random Guess")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All Models Compared", fontsize=13, color=PRIMARY, fontweight="bold")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout(); plt.savefig("plotX7_roc_curves.png", dpi=140); plt.close()

plt.figure(figsize=(6.5, 5.5))
for name, p in probs.items():
    prec, rec, _ = precision_recall_curve(y_test, p)
    plt.plot(rec, prec, label=name, color=colors[name], linewidth=2)
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curves — All Models Compared", fontsize=13, color=PRIMARY, fontweight="bold")
plt.legend(loc="lower left", fontsize=9)
plt.tight_layout(); plt.savefig("plotX8_pr_curves.png", dpi=140); plt.close()

print("\nROC-AUC scores:")
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    print(f"{name}: {auc(fpr, tpr):.4f}")

# ================= FUZZY MEMBERSHIP FUNCTION PLOTS =================
def trimf(x, a, b, c):
    return np.maximum(np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)), 0)
def trapmf(x, a, b, c, d):
    return np.maximum(np.minimum(np.minimum((x - a) / (b - a + 1e-9), 1), (d - x) / (d - c + 1e-9)), 0)

x_risk = np.linspace(0, 1, 200)
low_r = trapmf(x_risk, 0, 0, 0.20, 0.40)
med_r = trimf(x_risk, 0.30, 0.50, 0.70)
high_r = trapmf(x_risk, 0.60, 0.80, 1.0, 1.0)

x_val = np.linspace(0, 120, 200)
low_v = trapmf(x_val, 0, 0, 35, 60)
med_v = trimf(x_val, 45, 70, 95)
high_v = trapmf(x_val, 80, 100, 120, 120)

x_ten = np.linspace(0, 72, 200)
new_t = trapmf(x_ten, 0, 0, 6, 18)
med_t = trimf(x_ten, 12, 30, 48)
long_t = trapmf(x_ten, 36, 55, 72, 72)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
axes[0].plot(x_risk, low_r, label="Low", color=GREEN, linewidth=2)
axes[0].plot(x_risk, med_r, label="Medium", color=ORANGE, linewidth=2)
axes[0].plot(x_risk, high_r, label="High", color=RED, linewidth=2)
axes[0].set_title("Churn Risk Membership Functions", color=PRIMARY, fontweight="bold")
axes[0].set_xlabel("ANN Predicted Churn Probability"); axes[0].set_ylabel("Membership Degree")
axes[0].legend()

axes[1].plot(x_val, low_v, label="Low", color=GREEN, linewidth=2)
axes[1].plot(x_val, med_v, label="Medium", color=ORANGE, linewidth=2)
axes[1].plot(x_val, high_v, label="High", color=RED, linewidth=2)
axes[1].set_title("Customer Value Membership Functions", color=PRIMARY, fontweight="bold")
axes[1].set_xlabel("Monthly Charges"); axes[1].set_ylabel("Membership Degree")
axes[1].legend()

axes[2].plot(x_ten, new_t, label="New", color=GREEN, linewidth=2)
axes[2].plot(x_ten, med_t, label="Medium", color=ORANGE, linewidth=2)
axes[2].plot(x_ten, long_t, label="Long-term", color=RED, linewidth=2)
axes[2].set_title("Tenure Membership Functions", color=PRIMARY, fontweight="bold")
axes[2].set_xlabel("Tenure (months)"); axes[2].set_ylabel("Membership Degree")
axes[2].legend()

plt.tight_layout(); plt.savefig("plotX9_fuzzy_membership.png", dpi=140); plt.close()
print("\nFuzzy membership function plots saved.")

# ================= FUZZY DECISION SURFACE (heatmap) =================
import sys
sys.path.insert(0, ".")
# Re-implement the retention_priority scoring inline for the surface plot
def churn_low(x):    return trapmf(np.array([x]), 0.0, 0.0, 0.20, 0.40)[0]
def churn_medium(x): return trimf(np.array([x]), 0.30, 0.50, 0.70)[0]
def churn_high(x):   return trapmf(np.array([x]), 0.60, 0.80, 1.0, 1.0)[0]
def value_low(x):    return trapmf(np.array([x]), 0, 0, 35, 60)[0]
def value_medium(x): return trimf(np.array([x]), 45, 70, 95)[0]
def value_high(x):   return trapmf(np.array([x]), 80, 100, 120, 120)[0]
def tenure_new(x):    return trapmf(np.array([x]), 0, 0, 6, 18)[0]
def tenure_long(x):   return trapmf(np.array([x]), 36, 55, 72, 72)[0]

priority_domain = np.linspace(0, 100, 101)
def out_low(x):    return trapmf(np.array([x]), 0, 0, 20, 45)[0]
def out_medium(x): return trimf(np.array([x]), 30, 50, 70)[0]
def out_high(x):   return trapmf(np.array([x]), 55, 80, 100, 100)[0]

def retention_score(risk, value, tenure=30):
    cl, cm, ch = churn_low(risk), churn_medium(risk), churn_high(risk)
    vl, vm, vh = value_low(value), value_medium(value), value_high(value)
    tn, tl = tenure_new(tenure), tenure_long(tenure)
    strength = {
        "Low": max(min(cl, vl), min(cl, tn)),
        "Medium": max(min(ch, vl), min(cl, vh), min(cm, vl), min(cm, vm)),
        "High": max(min(ch, vh), min(ch, tl), min(cm, vh)),
    }
    agg = np.zeros_like(priority_domain)
    for i, x in enumerate(priority_domain):
        agg[i] = max(min(strength["Low"], out_low(x)), min(strength["Medium"], out_medium(x)), min(strength["High"], out_high(x)))
    return float(np.sum(priority_domain * agg) / np.sum(agg)) if agg.sum() > 0 else 0.0

risk_grid = np.linspace(0, 1, 40)
value_grid = np.linspace(0, 120, 40)
surface = np.zeros((len(value_grid), len(risk_grid)))
for i, v in enumerate(value_grid):
    for j, r in enumerate(risk_grid):
        surface[i, j] = retention_score(r, v, tenure=30)

plt.figure(figsize=(7, 5.5))
c = plt.pcolormesh(risk_grid, value_grid, surface, cmap="RdYlGn_r", shading="auto")
plt.colorbar(c, label="Retention Priority Score (0-100)")
plt.xlabel("Churn Risk (ANN Probability)"); plt.ylabel("Customer Value (Monthly Charges)")
plt.title("Fuzzy Decision Surface (Tenure = 30 months)", fontsize=13, color=PRIMARY, fontweight="bold")
plt.tight_layout(); plt.savefig("plotX10_fuzzy_surface.png", dpi=140); plt.close()
print("Fuzzy decision surface plot saved.")
