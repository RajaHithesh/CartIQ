# ============================================================================
# PROJECT : CartIQ - Grocery Market Basket Intelligence
# MODULE  : Binary Logistic Regression (manual sigmoid + gradient descent)
# TARGET  : reordered (binary 0/1)
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ----------------------------------------------------------------------------
# 1. LOAD THE FINAL PREPROCESSED DATASET
# ----------------------------------------------------------------------------
DATASET_PATH = "D:/ML/CartIQ/dataset/final_preprocess_M2.csv"
OUTPUT_DIR   = "D:/ML/CartIQ/outputs/Logistic_Regression_Binary_Classify_M2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATASET_PATH)
print("=" * 76)
print("BINARY LOGISTIC REGRESSION - MANUAL SIGMOID + GRADIENT DESCENT")
print("=" * 76)
print("Dataset Shape:", df.shape)

# ----------------------------------------------------------------------------
# 2. RECOVER BINARY TARGET FROM SCALED reordered
# ----------------------------------------------------------------------------
df["reordered_binary"] = (df["reordered"] > 0).astype(int)
print("\nreordered distribution:")
print(df["reordered_binary"].value_counts())

# ----------------------------------------------------------------------------
# 3. SELECT FEATURES AND TARGET
#    Pre-checkout attributes only (avoids leakage from same-order aggregates)
# ----------------------------------------------------------------------------
FEATURES = ["order_number", "days_since_prior_order", "order_hour_of_day"]
TARGET   = "reordered_binary"

X = df[FEATURES].values
y = df[TARGET].values

# ----------------------------------------------------------------------------
# 4. TRAIN TEST SPLIT
# ----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("\nTrain size:", X_train.shape[0], "| Test size:", X_test.shape[0])

# ----------------------------------------------------------------------------
# 5. SIGMOID FUNCTION
# ----------------------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 5.1 Sigmoid plot
z = np.linspace(-10, 10, 200)
plt.figure(figsize=(8, 6))
plt.plot(z, sigmoid(z), color="#0D9488", linewidth=2.5)
plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
plt.axvline(0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.title("Sigmoid Function")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_sigmoid.png"), dpi=150)
plt.close()

# ----------------------------------------------------------------------------
# 6. GRADIENT DESCENT FOR LOGISTIC REGRESSION
# ----------------------------------------------------------------------------
def logistic_gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    X_b = np.c_[np.ones(m), X]
    theta = np.zeros(n + 1)
    losses = []
    for i in range(iterations):
        z = X_b @ theta
        h = sigmoid(z)
        gradient = (1 / m) * X_b.T @ (h - y)
        theta = theta - learning_rate * gradient
        loss = -(1 / m) * np.sum(y * np.log(h + 1e-9) + (1 - y) * np.log(1 - h + 1e-9))
        losses.append(loss)
    return theta, losses

theta, losses = logistic_gradient_descent(X_train, y_train, learning_rate=0.01, iterations=1000)
print("\nLearned Theta:", theta)

# 6.1 Cross-Entropy Loss plot
plt.figure(figsize=(8, 6))
plt.plot(losses, color="#FB923C", linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Cross-Entropy Loss")
plt.title("Cross-Entropy Loss Curve")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_cross_entropy_loss.png"), dpi=150)
plt.close()

# ----------------------------------------------------------------------------
# 7. PREDICTIONS AND EVALUATION
# ----------------------------------------------------------------------------
X_test_b = np.c_[np.ones(X_test.shape[0]), X_test]
y_prob = sigmoid(X_test_b @ theta)
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")

# 7.1 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Reorder", "Reorder"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Binary Logistic Regression")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_confusion_matrix.png"), dpi=150)
plt.close()

# 7.2 Decision Boundary Hyperplane
#    2D projection: order_number (x) vs days_since_prior_order (y),
#    order_hour_of_day fixed at its mean.
plt.figure(figsize=(8, 6))
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], alpha=0.4,
            label="No Reorder", color="#FB923C")
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], alpha=0.4,
            label="Reorder", color="#0D9488")
x1_range = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100)
x3_mean = X_test[:, 2].mean()
x2_boundary = -(theta[0] + theta[1] * x1_range + theta[3] * x3_mean) / theta[2]
plt.plot(x1_range, x2_boundary, "r--", linewidth=2, label="Decision Boundary")
plt.xlabel("order_number")
plt.ylabel("days_since_prior_order")
plt.title("Decision Boundary Hyperplane")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_decision_boundary_hyperplane.png"), dpi=150)
plt.close()

# ----------------------------------------------------------------------------
# 8. SAVE PREDICTIONS
# ----------------------------------------------------------------------------
pred_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Probability": y_prob
})
pred_df.to_csv(os.path.join(OUTPUT_DIR, "prediction_results.csv"), index=False)

print("\n" + "=" * 76)
print("PROGRAM COMPLETED")
print("=" * 76)
print("\nAll output images are stored in:")
print(OUTPUT_DIR)