# ============================================================================
# PROJECT : CartIQ - Grocery Market Basket Intelligence
# MODULE  : Linear Regression - Closed Form Normal Equation vs Gradient Descent
# TARGET  : basket_size (basket value proxy)
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------------------------------------------------
# 1. LOAD THE FINAL PREPROCESSED DATASET
# ----------------------------------------------------------------------------
DATASET_PATH = "D:/ML/CartIQ/dataset/final_preprocess_M2.csv"
OUTPUT_DIR   = "D:/ML/CartIQ/outputs/Linear_Regression_CFNE_GD_Compare_M2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATASET_PATH)
print("=" * 76)
print("LINEAR REGRESSION - CLOSED FORM NORMAL EQUATION vs GRADIENT DESCENT")
print("=" * 76)
print("Dataset Shape:", df.shape)

# ----------------------------------------------------------------------------
# 2. SELECT FEATURES AND TARGET
# ----------------------------------------------------------------------------
FEATURES = ["order_number", "order_dow", "order_hour_of_day", "days_since_prior_order"]
TARGET   = "basket_size"

X = df[FEATURES].values
y = df[TARGET].values

# Add bias (intercept) column of ones
X_b = np.c_[np.ones(X.shape[0]), X]

# ----------------------------------------------------------------------------
# 3. TRAIN TEST SPLIT
# ----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_b, y, test_size=0.20, random_state=42)
print("\nTrain size:", X_train.shape[0], "| Test size:", X_test.shape[0])

# ----------------------------------------------------------------------------
# 4. CLOSED FORM NORMAL EQUATION
#    theta = (X^T X)^-1 X^T y
# ----------------------------------------------------------------------------
theta_cfne = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
y_pred_cfne = X_test @ theta_cfne

mse_cfne = mean_squared_error(y_test, y_pred_cfne)
r2_cfne  = r2_score(y_test, y_pred_cfne)

print("\nCLOSED FORM NORMAL EQUATION")
print("-" * 40)
print("Theta:", theta_cfne)
print(f"MSE: {mse_cfne:.6f} | R2: {r2_cfne:.6f}")

# ----------------------------------------------------------------------------
# 5. GRADIENT DESCENT
# ----------------------------------------------------------------------------
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)
    losses = []
    for i in range(iterations):
        gradient = (1 / m) * X.T @ (X @ theta - y)
        theta = theta - learning_rate * gradient
        loss = (1 / (2 * m)) * np.sum((X @ theta - y) ** 2)
        losses.append(loss)
    return theta, losses

theta_gd, losses = gradient_descent(X_train, y_train, learning_rate=0.01, iterations=1000)
y_pred_gd = X_test @ theta_gd

mse_gd = mean_squared_error(y_test, y_pred_gd)
r2_gd  = r2_score(y_test, y_pred_gd)

print("\nGRADIENT DESCENT")
print("-" * 40)
print("Theta:", theta_gd)
print(f"MSE: {mse_gd:.6f} | R2: {r2_gd:.6f}")

# ----------------------------------------------------------------------------
# 6. COMPARISON TABLE
# ----------------------------------------------------------------------------
comparison = pd.DataFrame({
    "Method": ["Closed Form Normal Equation", "Gradient Descent"],
    "MSE": [mse_cfne, mse_gd],
    "R2": [r2_cfne, r2_gd]
})
comparison.to_csv(os.path.join(OUTPUT_DIR, "image_information.csv"), index=False)
print("\n" + comparison.to_string(index=False))

# ----------------------------------------------------------------------------
# 7. VISUALIZATIONS
# ----------------------------------------------------------------------------
# 7.1 Actual vs Predicted (both methods)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_cfne, alpha=0.5, label="Normal Equation", color="#10B981")
plt.scatter(y_test, y_pred_gd, alpha=0.5, label="Gradient Descent", color="#FB923C")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=2)
plt.xlabel("Actual basket_size")
plt.ylabel("Predicted basket_size")
plt.title("Actual vs Predicted - CFNE vs GD")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=150)
plt.close()

# 7.2 Gradient Descent Loss Curve
plt.figure(figsize=(8, 6))
plt.plot(losses, color="#0D9488")
plt.xlabel("Iteration")
plt.ylabel("Loss (MSE)")
plt.title("Gradient Descent Loss Curve")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gradient_descent_loss.png"), dpi=150)
plt.close()

# 7.3 Residual Comparison
res_cfne = y_test - y_pred_cfne
res_gd   = y_test - y_pred_gd
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_cfne, res_cfne, alpha=0.5, label="Normal Equation", color="#10B981")
plt.scatter(y_pred_gd, res_gd, alpha=0.5, label="Gradient Descent", color="#FB923C")
plt.axhline(0, color="red", linestyle="--", linewidth=2)
plt.xlabel("Predicted basket_size")
plt.ylabel("Residuals")
plt.title("Residual Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residual_comparison.png"), dpi=150)
plt.close()

print("\n" + "=" * 76)
print("PROGRAM COMPLETED")
print("=" * 76)
print("\nAll output images are stored in:")
print(OUTPUT_DIR)