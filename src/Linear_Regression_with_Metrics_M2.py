# ============================================================================
# PROJECT : CartIQ - Grocery Market Basket Intelligence
# MODULE  : Linear Regression with Metrics (M2)
# TARGET  : basket_size (basket value proxy)
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------------------
# 1. LOAD THE FINAL PREPROCESSED DATASET
# ----------------------------------------------------------------------------
DATASET_PATH = "D:/ML/CartIQ/dataset/final_preprocess_M2.csv"
OUTPUT_DIR   = "D:/ML/CartIQ/outputs/Linear_Regression_with_Metrics_M2"
IMAGE_DIR    = os.path.join(OUTPUT_DIR, "images")

os.makedirs(IMAGE_DIR, exist_ok=True)

df = pd.read_csv(DATASET_PATH)
print("=" * 76)
print("LINEAR REGRESSION WITH METRICS")
print("=" * 76)
print("Dataset Shape:", df.shape)

# ----------------------------------------------------------------------------
# 2. SELECT FEATURES AND TARGET
# ----------------------------------------------------------------------------
FEATURES = ["order_number", "order_dow", "order_hour_of_day", "days_since_prior_order"]
TARGET   = "basket_size"

X = df[FEATURES]
y = df[TARGET]

print("\nFeatures:", FEATURES)
print("Target  :", TARGET)

# ----------------------------------------------------------------------------
# 3. TRAIN TEST SPLIT
# ----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("\nTrain size:", X_train.shape[0], "| Test size:", X_test.shape[0])

# ----------------------------------------------------------------------------
# 4. TRAIN LINEAR REGRESSION MODEL
# ----------------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\nIntercept:", model.intercept_)
print("Coefficients:", dict(zip(FEATURES, model.coef_)))

# ----------------------------------------------------------------------------
# 5. PREDICTIONS
# ----------------------------------------------------------------------------
y_pred = model.predict(X_test)

# ----------------------------------------------------------------------------
# 6. EVALUATION METRICS
# ----------------------------------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "-" * 40)
print("MODEL PERFORMANCE")
print("-" * 40)
print(f"MAE  : {mae:.6f}")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R2   : {r2:.6f}")

# ----------------------------------------------------------------------------
# 7. SAVE RESULTS
# ----------------------------------------------------------------------------
# 7.1 Predictions
predictions_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
predictions_df.to_csv(os.path.join(OUTPUT_DIR, "linear_regression_predictions.csv"), index=False)

# 7.2 Coefficients
coeff_df = pd.DataFrame({
    "Feature": ["Intercept"] + FEATURES,
    "Coefficient": [model.intercept_] + list(model.coef_)
})
coeff_df.to_csv(os.path.join(OUTPUT_DIR, "linear_regression_coefficients.csv"), index=False)

# 7.3 Metrics
metrics_df = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2"],
    "Value": [mae, mse, rmse, r2]
})
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "linear_regression_metrics.csv"), index=False)

# 7.4 Equation
with open(os.path.join(OUTPUT_DIR, "linear_regression_equation.txt"), "w") as f:
    f.write("Linear Regression Equation\n")
    f.write("=" * 40 + "\n")
    eq = f"basket_size = {model.intercept_:.4f}"
    for feat, coef in zip(FEATURES, model.coef_):
        eq += f" + ({coef:.4f} * {feat})"
    f.write(eq + "\n")

# ----------------------------------------------------------------------------
# 8. VISUALIZATIONS
# ----------------------------------------------------------------------------
# 8.1 Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="#10B981")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=2)
plt.xlabel("Actual basket_size")
plt.ylabel("Predicted basket_size")
plt.title("Actual vs Predicted (Linear Regression)")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "actual_vs_predicted.png"), dpi=150)
plt.close()

# 8.2 Feature Coefficients
plt.figure(figsize=(8, 6))
plt.bar(FEATURES, model.coef_, color="#2DD4BF")
plt.axhline(0, color="gray", linewidth=1)
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Feature Coefficients")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "feature_coefficients.png"), dpi=150)
plt.close()

# 8.3 Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5, color="#FB923C")
plt.axhline(0, color="red", linestyle="--", linewidth=2)
plt.xlabel("Predicted basket_size")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "residual_plot.png"), dpi=150)
plt.close()

print("\n" + "=" * 76)
print("PROGRAM COMPLETED")
print("=" * 76)
print("\nAll output images are stored in:")
print(OUTPUT_DIR)