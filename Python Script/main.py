import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# 1. Load Dataset
file_path = "customer_churn_dataset.xlsx"
df = pd.read_excel(file_path)

# 2. Remove duplicate rows
original_count = len(df)
df.drop_duplicates(inplace=True)
print(f"Removed {original_count - len(df)} duplicate rows. Remaining rows: {len(df)}")

# 3. Print original missing values
print("Original missing values:\n", df.isnull().sum())

# 4. Fill missing string/numeric columns
df.fillna({
    "Name": "Unknown",
    "Email": "unknown@email.com",
    "Subscription_Type": "Unknown",
    "Monthly_Usage": 0,
    "Total_Payments": 0,
    "Complaint_Count": 0,
    "Customer_Satisfaction_Score": 0,
    "Region": "Unknown",
    "Churned": 0
}, inplace=True)

# 5. Handle dates: fill missing with '0000-00-00'
date_cols = ["Subscription_Start_Date", "Subscription_End_Date", "Last_Payment_Date"]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")          
    df[col] = df[col].dt.strftime('%Y-%m-%d')                   
    df[col] = df[col].fillna('0000-00-00')                      

# 6. Calculate Customer_Lifetime in days (only valid dates)
start_dates = pd.to_datetime(df["Subscription_Start_Date"], errors='coerce')
end_dates = pd.to_datetime(df["Subscription_End_Date"], errors='coerce')
df["Customer_Lifetime"] = (end_dates - start_dates).dt.days.fillna(0).clip(lower=0)

# Ensure numeric columns
numeric_cols = ["Monthly_Usage", "Total_Payments", "Complaint_Count", "Customer_Satisfaction_Score", "Churned", "Customer_Lifetime"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# 7. Feature Selection
features = ["Monthly_Usage", "Total_Payments", "Complaint_Count",
            "Customer_Satisfaction_Score", "Customer_Lifetime",
            "Subscription_Type", "Region"]
X = df[features]
y = df["Churned"]

# 8a. Logistic Regression
X_log = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X_log_scaled = scaler.fit_transform(X_log)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_log_scaled, y)
y_pred_log = log_model.predict(X_log_scaled)
y_prob_log = log_model.predict_proba(X_log_scaled)[:, 1]

# 8b. Random Forest
X_rf = pd.get_dummies(X)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_rf, y)
y_pred_rf = rf_model.predict(X_rf)
y_prob_rf = rf_model.predict_proba(X_rf)[:, 1]

# 9. Prepare Predictions DataFrame
results_df = pd.DataFrame({
    "Customer_ID": df["Customer_ID"],
    "Actual": y,
    "Predicted_Logistic": y_pred_log,
    "Probability_Logistic(%)": np.round(y_prob_log * 100, 2),
    "Predicted_RF": y_pred_rf,
    "Probability_RF(%)": np.round(y_prob_rf * 100, 2)
})

# 10. Calculate Metrics
metrics = [
    {
        "Model": "Logistic Regression",
        "Accuracy": accuracy_score(y, y_pred_log),
        "Precision": precision_score(y, y_pred_log),
        "Recall": recall_score(y, y_pred_log),
        "F1": f1_score(y, y_pred_log),
        "ROC_AUC": roc_auc_score(y, y_prob_log)
    },
    {
        "Model": "Random Forest",
        "Accuracy": accuracy_score(y, y_pred_rf),
        "Precision": precision_score(y, y_pred_rf),
        "Recall": recall_score(y, y_pred_rf),
        "F1": f1_score(y, y_pred_rf),
        "ROC_AUC": roc_auc_score(y, y_prob_rf)
    }
]
metrics_df = pd.DataFrame(metrics)

# 11. Print Summary
print("\nMissing values after cleaning:\n", df.isnull().sum())
print(f"\nAverage Monthly Usage: {df['Monthly_Usage'].mean():.2f}")
print(f"Total Churned Customers: {df['Churned'].sum()}")
print(f"Total Active Customers: {len(df) - df['Churned'].sum()}")
print(f"Average Customer Lifetime (days): {df['Customer_Lifetime'].mean():.2f}\n")

print("Logistic Regression Results:")
print(confusion_matrix(y, y_pred_log))
print(classification_report(y, y_pred_log))
print(f"ROC AUC: {roc_auc_score(y, y_prob_log)}\n")

print("Random Forest Results:")
print(confusion_matrix(y, y_pred_rf))
print(classification_report(y, y_pred_rf))
print(f"ROC AUC: {roc_auc_score(y, y_prob_rf)}\n")

print("--- Full Predictions (1st 20 rows) ---")
print(results_df.head(20))

# 12. Save all datasets separately
df.to_excel("customer_cleaned_dataset.xlsx", index=False)
results_df.to_excel("customer_churn_predictions.xlsx", index=False)
metrics_df.to_excel("customer_churn_metrics.xlsx", index=False)

print("\nAll datasets saved separately: Cleaned, Predictions, Metrics!")
