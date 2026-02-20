# ==============================
# Credit Card Fraud Detection (~30k sample)
# ==============================

import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 1. Load Dataset
# ==============================
data = pd.read_csv("creditcard.csv")

# ==============================
# 2. Create ~30k Sample
# ==============================
# Separate fraud and non-fraud
fraud = data[data['Class'] == 1]          # all fraud cases (492 rows)
non_fraud = data[data['Class'] == 0]

# Sample non-fraud rows to reach ~30k total
n_non_fraud = 30000 - len(fraud)         # 30,000 - 492 ≈ 29,508
non_fraud_sample = non_fraud.sample(n=n_non_fraud, random_state=42)

# Combine fraud + sampled non-fraud
data_sample = pd.concat([fraud, non_fraud_sample]).reset_index(drop=True)

# Shuffle dataset
data_sample = data_sample.sample(frac=1, random_state=42).reset_index(drop=True)

# Check result
print("Shape of sampled dataset:", data_sample.shape)
print("Class distribution in sample:")
print(data_sample['Class'].value_counts())

# ==============================
# 3. Separate Features & Target
# ==============================
X = data_sample.drop('Class', axis=1)
Y = data_sample['Class']

# ==============================
# 4. Scale Features
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# 5. Train-Test Split
# ==============================
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42, stratify=Y
)

print("\nBefore SMOTE - Training class distribution:")
print(Y_train.value_counts())

# ==============================
# 6. Apply SMOTE on Training Set Only
# ==============================
smote = SMOTE(random_state=42)
X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)

print("\nAfter SMOTE - Training class distribution:")
print(Y_train_res.value_counts())

# ==============================
# 7. Logistic Regression Model
# ==============================
log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train_res, Y_train_res)

# Predictions
log_pred = log_model.predict(X_test)
log_pred_proba = log_model.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression Evaluation ===")
print("Accuracy:", accuracy_score(Y_test, log_pred))
print("ROC-AUC:", roc_auc_score(Y_test, log_pred_proba))
print("\nClassification Report:\n", classification_report(Y_test, log_pred))
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, log_pred))

# ==============================
# 8. Optimized Random Forest Model
# ==============================
rf_model = RandomForestClassifier(
    n_estimators=50,      # fewer trees → faster
    max_depth=8,          # limit depth → faster
    min_samples_split=10, # prevent overfitting
    n_jobs=-1,            # use all CPU cores
    random_state=42
)
rf_model.fit(X_train_res, Y_train_res)

# Predictions
rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print("\n=== Random Forest Evaluation ===")
print("Accuracy:", accuracy_score(Y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(Y_test, rf_pred_proba))
print("\nClassification Report:\n", classification_report(Y_test, rf_pred))
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, rf_pred))

# ==============================
# 9. Feature Importance Plot
# ==============================
feat_importances = pd.Series(rf_model.feature_importances_, index=data_sample.columns[:-1])
feat_importances = feat_importances.sort_values(ascending=True)

plt.figure(figsize=(10, 8))

sns.barplot(
    x=feat_importances,
    y=feat_importances.index,
    hue=feat_importances.index,
    palette="viridis",
    dodge=False,
    legend=False
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()






