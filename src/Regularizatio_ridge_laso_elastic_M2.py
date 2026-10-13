# ============================================================================
# PROJECT : CartIQ - Grocery Market Basket Intelligence
# MODULE  : Regularization - Ridge (L2), Lasso (L1), Elastic Net
# TARGET  : reordered (binary 0/1)
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------------------------------------------------------------
# 1. LOAD THE FINAL PREPROCESSED DATASET
# ----------------------------------------------------------------------------
DATASET_PATH = "D:/ML/CartIQ/dataset/final_preprocess_M2.csv"
OUTPUT_DIR   = "D:/ML/CartIQ/outputs/Regularization_ridg_laso_elastic_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATASET_PATH)
print("=" * 76)
print("REGULARIZATION - RIDGE (L2) vs LASSO (L1) vs ELASTIC NET")
print("=" * 76)
print("Dataset Shape:", df.shape)

# ----------------------------------------------------------------------------
# 2. RECOVER BINARY TARGET FROM SCALED reordered
#    (scaled value > 0  =>  original value 1)
# ----------------------------------------------------------------------------
df["reordered_binary"] = (df["reordered"] > 0).astype(int)
print("\nreordered distribution:")
print(df["reordered_binary"].value_counts())

# ----------------------------------------------------------------------------
# 3. SELECT FEATURES AND TARGET
# ----------------------------------------------------------------------------
FEATURES = ["order_number", "order_dow", "order_hour_of_day", "days_since_prior_order"]
TARGET   = "reordered_binary"

X = df[FEATURES]
y = df[TARGET]

# ----------------------------------------------------------------------------
# 4. TRAIN TEST SPLIT
# ----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("\nTrain size:", X_train.shape[0], "| Test size:", X_test.shape[0])

# ----------------------------------------------------------------------------
# 5. TRAIN REGULARIZED MODELS
# ----------------------------------------------------------------------------
models = {
    "Ridge (L2)":     Ridge(alpha=1.0),
    "Lasso (L1)":     Lasso(alpha=0.01),
    "Elastic Net":    ElasticNet(alpha=0.01, l1_ratio=0.5)
}

results = []
coefficients = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_class = (y_pred > 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred_class)
    prec = precision_score(y_test, y_pred_class)
    rec  = recall_score(y_test, y_pred_class)
    f1   = f1_score(y_test, y_pred_class)

    results.append({
        "Model": name, "Accuracy": acc,
        "Precision": prec, "Recall": rec, "F1_Score": f1
    })
    coefficients[name] = model.coef_

    print(f"\n{name}")
    print("-" * 40)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

# ----------------------------------------------------------------------------
# 6. SAVE RESULTS
# ----------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("D:/ML/CartIQ/dataset/regularisation_model_comparison_M2.csv", index=False)

coeff_df = pd.DataFrame(coefficients, index=FEATURES)
coeff_df.to_csv("D:/ML/CartIQ/dataset/regularisation_coefficients_M2.csv")

lasso_coef = coefficients["Lasso (L1)"]
lasso_selection = pd.DataFrame({"Feature": FEATURES, "Coefficient": lasso_coef})
lasso_selection.to_csv("D:/ML/CartIQ/dataset/lasso_feature_selection_M2.csv", index=False)

print("\nResults saved to dataset folder.")

# ----------------------------------------------------------------------------
# 7. VISUALIZATIONS
# ----------------------------------------------------------------------------
# 7.1 Coefficient Comparison
x_pos = np.arange(len(FEATURES))
width = 0.25
plt.figure(figsize=(10, 6))
for i, (name, coef) in enumerate(coefficients.items()):
    plt.bar(x_pos + i * width, coef, width, label=name)
plt.xticks(x_pos + width, FEATURES, rotation=30)
plt.axhline(0, color="gray", linewidth=1)
plt.ylabel("Coefficient Value")
plt.title("Coefficient Comparison - Ridge vs Lasso vs Elastic Net")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Coefficient_Comparison.png"), dpi=150)
plt.close()

# 7.2 L1 vs L2 Regularisation
theta = np.linspace(-1.5, 1.5, 400)
plt.figure(figsize=(8, 6))
plt.plot(theta, theta ** 2, label="L2 (Ridge)", color="#0D9488")
plt.plot(theta, np.abs(theta), label="L1 (Lasso)", color="#FB923C")
plt.xlabel("Coefficient value")
plt.ylabel("Penalty")
plt.title("L1 vs L2 Regularisation")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "L1_vs_L2_Regularisation.png"), dpi=150)
plt.close()

# 7.3 Lasso Sparsity Graph
plt.figure(figsize=(8, 6))
plt.bar(FEATURES, lasso_coef, color="#2DD4BF")
plt.axhline(0, color="gray", linewidth=1)
plt.ylabel("Coefficient Value")
plt.title("Lasso Sparsity - Feature Selection")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Lasso_Sparsity_Graph.png"), dpi=150)
plt.close()

# 7.4 Model Performance
metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]
plt.figure(figsize=(8, 6))
for _, row in results_df.iterrows():
    plt.plot(metrics, [row[m] for m in metrics], marker="o", label=row["Model"])
plt.xlabel("Metric")
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Model_Performance.png"), dpi=150)
plt.close()

print("\n" + "=" * 76)
print("PROGRAM COMPLETED")
print("=" * 76)
print("\nAll output images are stored in:")
print(OUTPUT_DIR)