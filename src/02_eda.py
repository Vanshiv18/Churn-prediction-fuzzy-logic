"""
Section 8 - Exploratory Data Analysis
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
df = pd.read_csv("telco_clean.csv")
df["ChurnLabel"] = df["Churn"].map({1: "Yes", 0: "No"})

overall_rate = df["Churn"].mean() * 100
print(f"Overall churn rate: {overall_rate:.2f}%")

# 1. Churn distribution
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="ChurnLabel", palette=["#2ecc71", "#e74c3c"])
plt.title(f"Overall Churn Distribution (Churn rate = {overall_rate:.1f}%)")
plt.xlabel("Churn"); plt.ylabel("Number of Customers")
plt.tight_layout(); plt.savefig("plot1_churn_distribution.png", dpi=140); plt.close()

# 2. Churn by contract type
plt.figure(figsize=(6, 4))
ct = pd.crosstab(df["Contract"], df["ChurnLabel"], normalize="index") * 100
ct.plot(kind="bar", stacked=True, color=["#2ecc71", "#e74c3c"])
plt.title("Churn Rate by Contract Type")
plt.ylabel("% of Customers"); plt.xlabel("Contract Type"); plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout(); plt.savefig("plot2_churn_by_contract.png", dpi=140); plt.close()

# 3. Churn by tenure group
plt.figure(figsize=(6, 4))
order = ["0-1yr", "1-3yr", "3yr+"]
ct2 = pd.crosstab(df["TenureGroup"], df["ChurnLabel"], normalize="index").reindex(order) * 100
ct2.plot(kind="bar", stacked=True, color=["#2ecc71", "#e74c3c"])
plt.title("Churn Rate by Tenure Group")
plt.ylabel("% of Customers"); plt.xlabel("Tenure Group"); plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout(); plt.savefig("plot3_churn_by_tenure.png", dpi=140); plt.close()

# 4. Churn by monthly charge band
plt.figure(figsize=(6, 4))
ct3 = pd.crosstab(df["ChargeBand"], df["ChurnLabel"], normalize="index").reindex(["Low", "Medium", "High"]) * 100
ct3.plot(kind="bar", stacked=True, color=["#2ecc71", "#e74c3c"])
plt.title("Churn Rate by Monthly-Charge Band")
plt.ylabel("% of Customers"); plt.xlabel("Charge Band"); plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout(); plt.savefig("plot4_churn_by_chargeband.png", dpi=140); plt.close()

# 5. Numerical relationship: MonthlyCharges vs TotalCharges, coloured by churn
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x="tenure", y="MonthlyCharges", hue="ChurnLabel",
                 alpha=0.4, palette=["#2ecc71", "#e74c3c"])
plt.title("Tenure vs Monthly Charges by Churn Status")
plt.tight_layout(); plt.savefig("plot5_tenure_vs_charges.png", dpi=140); plt.close()

# 6. Correlation heatmap of numeric variables
plt.figure(figsize=(5, 4))
num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn"]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap - Numeric Variables")
plt.tight_layout(); plt.savefig("plot6_correlation_heatmap.png", dpi=140); plt.close()

# Extra: churn by payment method (feeds narrative)
plt.figure(figsize=(7, 4))
ct4 = pd.crosstab(df["PaymentMethod"], df["ChurnLabel"], normalize="index") * 100
ct4.plot(kind="bar", stacked=True, color=["#2ecc71", "#e74c3c"])
plt.title("Churn Rate by Payment Method")
plt.ylabel("% of Customers"); plt.xticks(rotation=20, ha="right")
plt.legend(title="Churn")
plt.tight_layout(); plt.savefig("plot7_churn_by_payment.png", dpi=140); plt.close()

print("\nChurn rate by contract:\n", ct["Yes"].round(1))
print("\nChurn rate by tenure group:\n", ct2["Yes"].round(1))
print("\nChurn rate by charge band:\n", ct3["Yes"].round(1))
print("\nChurn rate by payment method:\n", ct4["Yes"].round(1))
print("\nAll EDA plots saved.")
