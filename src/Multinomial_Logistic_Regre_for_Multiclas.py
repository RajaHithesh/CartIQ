# ============================================================================
# PROJECT : CartIQ - Grocery Market Basket Intelligence
# MODULE  : Multinomial Logistic Regression (Softmax)
# TARGET  : department (21 classes)
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

# ----------------------------------------------------------------------------
# 1. LOAD THE FINAL PREPROCESSED DATASET
# ----------------------------------------------------------------------------
DATASET_PATH = "D:/ML/CartIQ/dataset/final_preprocess_M2.csv"
OUTPUT_DIR   = "D:/ML/CartIQ/outputs/Multinomial_Logistic_Regre_for_Multiclass_M2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATASET_PATH)
print("=" * 76)
print("MULTINOMIAL LOGISTIC REGRESSION RESULTS")
print("=" * 76)
print(f"\nDataset: {DATASET_PATH}")
print(f"Target: department")
print(f"Number of classes: {df['department'].nunique()}")

# ----------------------------------------------------------------------------
# 2. SELECT FEATURES AND TARGET
# ----------------------------------------------------------------------------
FEATURES = ["order_number", "order_dow", "order_hour_of_day",
            "days_since_prior_order", "add_to_cart_order", "product_id", "aisle"]
TARGET   = "department"

X = df[FEATURES]
y = df[TARGET]

# ----------------------------------------------------------------------------
# 3. LABEL ENCODE THE TARGET
# ----------------------------------------------------------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save the label encoder for later use in the web app
joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

print("\nCLASS MAPPING")
print("-" * 40)
for i, cls in enumerate(le.classes_):
    print(f"{i} = {cls}")

# ----------------------------------------------------------------------------
# 4. TRAIN TEST SPLIT
# ----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=42)
print("\nTrain size:", X_train.shape[0], "| Test size:", X_test.shape[0])

# ----------------------------------------------------------------------------
# 5. TRAIN MULTINOMIAL LOGISTIC REGRESSION
#    sklearn 1.9: lbfgs auto-selects softmax for multiclass (no multi_class param)
# ----------------------------------------------------------------------------
model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# ----------------------------------------------------------------------------
# 6. PREDICTIONS
# ----------------------------------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# ----------------------------------------------------------------------------
# 7. EVALUATION
# ----------------------------------------------------------------------------
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

eps = 1e-9
ce_loss = -np.mean(np.log(y_prob[np.arange(len(y_test)), y_test] + eps))

print("\nMODEL PERFORMANCE")
print("-" * 40)
print(f"Accuracy: {acc:.4f}")
print(f"Accuracy (%): {acc * 100:.2f}%")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Cross-Entropy Loss: {ce_loss:.4f}")

print("\nCLASSIFICATION REPORT")
print("-" * 40)
print(classification_report(y_test, y_pred, zero_division=0))

# ----------------------------------------------------------------------------
# 8. SAVE OUTPUTS
# ----------------------------------------------------------------------------
# 8.1 Model metrics
with open(os.path.join(OUTPUT_DIR, "model_metrics.txt"), "w") as f:
    f.write("MULTINOMIAL LOGISTIC REGRESSION RESULTS\n")
    f.write("=" * 76 + "\n")
    f.write(f"\nDataset: {DATASET_PATH}\n")
    f.write(f"Target: department\n")
    f.write(f"Number of classes: {df['department'].nunique()}\n")
    f.write("\nCLASS MAPPING\n")
    f.write("-" * 40 + "\n")
    for i, cls in enumerate(le.classes_):
        f.write(f"{i} = {cls}\n")
    f.write("\nMODEL PERFORMANCE\n")
    f.write("-" * 40 + "\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Accuracy (%): {acc * 100:.2f}%\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall: {rec:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Cross-Entropy Loss: {ce_loss:.4f}\n")
    f.write("\nCLASSIFICATION REPORT\n")
    f.write("-" * 40 + "\n")
    f.write(classification_report(y_test, y_pred, zero_division=0))

# 8.2 Dataset information
with open(os.path.join(OUTPUT_DIR, "dataset_information.txt"), "w") as f:
    f.write(f"Dataset Shape: {df.shape}\n")
    f.write(f"Features: {FEATURES}\n")
    f.write(f"Target: {TARGET}\n")
    f.write(f"Classes: {df['department'].nunique()}\n")

# 8.3 Predictions
pred_df = pd.DataFrame({
    "Actual": le.inverse_transform(y_test),
    "Predicted": le.inverse_transform(y_pred)
})
pred_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
pred_df.to_csv(os.path.join(OUTPUT_DIR, "actual_vs_predicted.csv"), index=False)

# 8.4 Softmax probabilities
prob_df = pd.DataFrame(y_prob, columns=[f"class_{i}" for i in range(y_prob.shape[1])])
prob_df.to_csv(os.path.join(OUTPUT_DIR, "softmax_probabilities.csv"), index=False)

# 8.5 Coefficients and intercepts
coef_df = pd.DataFrame(model.coef_, columns=FEATURES)
coef_df.to_csv(os.path.join(OUTPUT_DIR, "model_coefficients.csv"), index=False)

intercept_df = pd.DataFrame({"Class": range(len(model.intercept_)), "Intercept": model.intercept_})
intercept_df.to_csv(os.path.join(OUTPUT_DIR, "model_intercepts.csv"), index=False)

# 8.6 Class distribution
class_dist = pd.Series(y_encoded).value_counts().sort_index()
class_dist_df = pd.DataFrame({"Class": class_dist.index, "Count": class_dist.values})
class_dist_df.to_csv(os.path.join(OUTPUT_DIR, "class_distribution.csv"), index=False)

# 8.7 Save the trained model
joblib.dump(model, os.path.join(OUTPUT_DIR, "multinomial_logistic_regression.pkl"))

# ----------------------------------------------------------------------------
# 9. VISUALIZATIONS
# ----------------------------------------------------------------------------
# 9.1 Class distribution
plt.figure(figsize=(10, 6))
plt.bar(class_dist.index, class_dist.values, color="#2DD4BF")
plt.xlabel("Department Class")
plt.ylabel("Count")
plt.title("Class Distribution - Department")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150)
plt.close()

# 9.2 Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(12, 10))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix - Multinomial Logistic Regression")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

print("\n" + "=" * 76)
print("PROGRAM COMPLETED")
print("=" * 76)
print("\nAll outputs saved to:")
print(OUTPUT_DIR)