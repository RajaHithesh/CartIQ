# ================================================================
# CARTIQ - GROCERY MARKET BASKET INTELLIGENCE
# ================================================================
# MODULE 3 : RANDOM FOREST CLASSIFIER
# ================================================================
# TASK      : Predict whether a product will be reordered
#             (reordered = 1) or not (reordered = 0).
# DATASET   : RAW dataset is used. All preprocessing is performed
#             on a copy / inside a sklearn pipeline.
#             The original RAW dataset is NEVER modified.
# ================================================================
# TECHNIQUE : Random Forest Classifier
#             n_estimators = 100
#             max_depth = 10
#             min_samples_split = 10
#             min_samples_leaf = 5
#             random_state = 42
# ================================================================

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ================================================================
# 1. PATHS
# ================================================================

DATASET_PATH = "D:/ML/CartIQ/dataset/instacart_50k_Raw.csv"

OUTPUT_FOLDER = (
    "D:/ML/CartIQ/outputs/Random_Forest_Tree_M3_Outputs"
)

METRICS_FOLDER = os.path.join(OUTPUT_FOLDER, "metrics")
PREDICTIONS_FOLDER = os.path.join(OUTPUT_FOLDER, "predictions")
CONFUSION_FOLDER = os.path.join(OUTPUT_FOLDER, "confusion_matrix")
CHARTS_FOLDER = os.path.join(OUTPUT_FOLDER, "charts")
TREE_FOLDER = os.path.join(OUTPUT_FOLDER, "random_forest_tree")
FEATURE_FOLDER = os.path.join(OUTPUT_FOLDER, "feature_importance")
MODEL_FOLDER = os.path.join(OUTPUT_FOLDER, "model")

# ================================================================
# 2. CREATE OUTPUT FOLDERS
# ================================================================

for folder in [
    METRICS_FOLDER,
    PREDICTIONS_FOLDER,
    CONFUSION_FOLDER,
    CHARTS_FOLDER,
    TREE_FOLDER,
    FEATURE_FOLDER,
    MODEL_FOLDER,
]:
    os.makedirs(folder, exist_ok=True)

print("Output folders created successfully.")

# ================================================================
# 3. CHECK DATASET
# ================================================================

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"\nDataset not found at:\n{DATASET_PATH}"
    )

print("\nDataset found successfully.")

# ================================================================
# 4. LOAD RAW DATASET
# ================================================================

data = pd.read_csv(DATASET_PATH)

print("\nRAW dataset loaded successfully.")
print("Rows    :", data.shape[0])
print("Columns :", data.shape[1])

# ================================================================
# 5. CREATE A WORKING COPY
# ================================================================
#
# The RAW dataset is never modified. All work is done on a copy.
# ================================================================

df = data.copy()

# ================================================================
# 6. FIND TARGET COLUMN
# ================================================================

possible_targets = ["reordered", "Reordered", "Reorder", "REORDERED"]

target_column = None

for column in possible_targets:
    if column in df.columns:
        target_column = column
        break

if target_column is None:
    raise ValueError(
        "\nTarget column 'reordered' was not found."
        "\n\nAvailable columns:\n"
        + "\n".join(df.columns.astype(str))
    )

print("\nTarget column :", target_column)

# ================================================================
# 7. SEPARATE FEATURES AND TARGET
# ================================================================
#
# ID / name columns (order_id, user_id, product_id, product_name)
# are identifiers, not predictors. They are excluded from features.
# ================================================================

id_columns = [
    "order_id",
    "user_id",
    "product_id",
    "product_name",
]

feature_columns = [
    column
    for column in df.columns
    if column != target_column and column not in id_columns
]

X = df[feature_columns].copy()
y = df[target_column].copy()

print("\nFeatures used :", len(feature_columns))
for column in feature_columns:
    print("  -", column)

print("\nTarget classes :", sorted(y.unique()))

# ================================================================
# 8. SPLIT NUMERIC AND CATEGORICAL COLUMNS
# ================================================================

numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

categorical_columns = (
    X.select_dtypes(include=["str", "category"]).columns.tolist()
)

print("\nNumeric columns     :", numeric_columns)
print("Categorical columns :", categorical_columns)

# ================================================================
# 9. PREPROCESSING TRANSFORMERS
# ================================================================
#
# Numeric      -> SimpleImputer (median)
# Categorical  -> SimpleImputer (most_frequent) + OneHotEncoder
#
# Everything runs INSIDE the pipeline, so the RAW dataset is
# never modified.
# ================================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# ================================================================
# 10. COLUMN TRANSFORMER
# ================================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_columns),
        ("cat", categorical_transformer, categorical_columns),
    ]
)

# ================================================================
# 11. TRAIN TEST SPLIT
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("\nTrain samples :", X_train.shape[0])
print("Test samples  :", X_test.shape[0])

# ================================================================
# 12. RANDOM FOREST CLASSIFIER
# ================================================================

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

# ================================================================
# 13. BUILD PIPELINE
# ================================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", random_forest),
    ]
)

# ================================================================
# 14. TRAIN THE MODEL
# ================================================================

print("\n" + "=" * 80)
print("                 TRAINING RANDOM FOREST")
print("=" * 80)

model.fit(X_train, y_train)

print("\nModel training completed successfully.")

# ================================================================
# 15. MAKE PREDICTIONS
# ================================================================

y_pred = model.predict(X_test)

# ================================================================
# 16. EVALUATION METRICS
# ================================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 80)
print("                 RANDOM FOREST PERFORMANCE")
print("=" * 80)

print(f"\nAccuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

# ================================================================
# 17. SAVE METRICS CSV
# ================================================================

metrics_df = pd.DataFrame(
    {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [accuracy, precision, recall, f1],
    }
)

metrics_path = os.path.join(
    METRICS_FOLDER,
    "random_forest_metrics.csv",
)

metrics_df.to_csv(metrics_path, index=False)

print("\nMetrics saved at:")
print(metrics_path)

# ================================================================
# 18. CLASSIFICATION REPORT
# ================================================================

report_dict = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report_dict).transpose()

report_path = os.path.join(
    METRICS_FOLDER,
    "classification_report.csv",
)

report_df.to_csv(report_path)

print("\nClassification report saved at:")
print(report_path)

# ================================================================
# 19. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"],
)

cm_path = os.path.join(
    CONFUSION_FOLDER,
    "confusion_matrix.csv",
)

cm_df.to_csv(cm_path)

print("\nConfusion matrix saved at:")
print(cm_path)

# ================================================================
# 20. CONFUSION MATRIX HEATMAP
# ================================================================

plt.figure(figsize=(7, 6))

plt.imshow(cm, interpolation="nearest", cmap="Blues")

plt.title("Confusion Matrix - Random Forest")
plt.colorbar()

tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Predicted 0", "Predicted 1"])
plt.yticks(tick_marks, ["Actual 0", "Actual 1"])

thresh = cm.max() / 2.0

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()

cm_image_path = os.path.join(
    CONFUSION_FOLDER,
    "confusion_matrix.png",
)

plt.savefig(cm_image_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nConfusion matrix heatmap saved at:")
print(cm_image_path)

# ================================================================
# 21. PERFORMANCE GRAPH
# ================================================================

performance_values = [accuracy, precision, recall, f1]
performance_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

plt.figure(figsize=(9, 6))

plt.bar(performance_labels, performance_values, color="#2E86AB")

for index, value in enumerate(performance_values):
    plt.text(
        index,
        value + 0.01,
        f"{value * 100:.2f}%",
        ha="center",
        va="bottom",
    )

plt.ylim(0, 1.1)
plt.ylabel("Score")
plt.title("Random Forest Performance Metrics")
plt.tight_layout()

performance_path = os.path.join(
    CHARTS_FOLDER,
    "performance_graph.png",
)

plt.savefig(performance_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nPerformance graph saved at:")
print(performance_path)

# ================================================================
# 22. ACTUAL VS PREDICTED CHART
# ================================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    y_test.values,
    y_pred,
    alpha=0.4,
    color="#E07A5F",
    edgecolors="white",
    linewidths=0.5,
)

plt.plot(
    [0, 1],
    [0, 1],
    "--",
    color="#3D405B",
    label="Perfect Prediction",
)

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted - Random Forest")
plt.legend()
plt.tight_layout()

avp_path = os.path.join(
    CHARTS_FOLDER,
    "actual_vs_predicted.png",
)

plt.savefig(avp_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nActual vs Predicted chart saved at:")
print(avp_path)

# ================================================================
# 23. CLASS DISTRIBUTION CHART
# ================================================================

class_counts = y.value_counts()

plt.figure(figsize=(8, 6))

plt.bar(
    class_counts.index.astype(str),
    class_counts.values,
    color=["#81B29A", "#E07A5F"],
)

plt.xlabel("Reordered Class")
plt.ylabel("Number of Products")
plt.title("Reordered Class Distribution")
plt.tight_layout()

class_distribution_path = os.path.join(
    CHARTS_FOLDER,
    "class_distribution.png",
)

plt.savefig(class_distribution_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nClass distribution chart saved at:")
print(class_distribution_path)

# ================================================================
# 24. GET FEATURE NAMES AFTER ENCODING
# ================================================================

feature_names = (
    model.named_steps["preprocessor"].get_feature_names_out()
)

# ================================================================
# 25. FEATURE IMPORTANCE
# ================================================================

feature_importances = random_forest.feature_importances_

feature_importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": feature_importances,
    }
)

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False,
)

# ================================================================
# 26. SAVE FEATURE IMPORTANCE CSV
# ================================================================

feature_importance_path = os.path.join(
    FEATURE_FOLDER,
    "feature_importance.csv",
)

feature_importance_df.to_csv(feature_importance_path, index=False)

print("\nFeature importance saved at:")
print(feature_importance_path)

# ================================================================
# 27. FEATURE IMPORTANCE GRAPH
# ================================================================

top_features = (
    feature_importance_df.head(15).sort_values(by="Importance")
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"],
    color="#2E86AB",
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Random Forest Feature Importances")
plt.tight_layout()

feature_chart_path = os.path.join(
    FEATURE_FOLDER,
    "feature_importance.png",
)

plt.savefig(feature_chart_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nFeature importance graph saved at:")
print(feature_chart_path)

# ================================================================
# 28. RANDOM FOREST TREE VISUALIZATION
# ================================================================
#
# A random forest contains 100 trees. The FIRST tree of the
# forest is visualized as a representative sample.
# ================================================================

print("\n")
print("=" * 80)
print("                 RANDOM FOREST TREE VISUALIZATION")
print("=" * 80)

first_tree = random_forest.estimators_[0]

plt.figure(figsize=(30, 18))

plot_tree(
    first_tree,
    feature_names=feature_names,
    class_names=[str(label) for label in sorted(y.unique())],
    filled=True,
    rounded=True,
    proportion=False,
    precision=2,
    fontsize=8,
)

plt.title(
    "Random Forest - First Tree (CartIQ Reorder Prediction)",
    fontsize=20,
)

plt.tight_layout()

# ================================================================
# 29. SAVE RANDOM FOREST TREE DIAGRAM
# ================================================================

tree_image_path = os.path.join(
    TREE_FOLDER,
    "random_forest_tree_1.png",
)

plt.savefig(tree_image_path, dpi=300, bbox_inches="tight")

# ================================================================
# 30. DISPLAY RANDOM FOREST TREE
# ================================================================

plt.show()

print("\nRandom Forest tree diagram displayed successfully.")
print("Random Forest tree diagram saved at:")
print(tree_image_path)

# ================================================================
# 31. SAVE TEST PREDICTIONS
# ================================================================

test_predictions = X_test.copy()

test_predictions["Actual"] = y_test.values
test_predictions["Predicted"] = y_pred

prediction_path = os.path.join(
    PREDICTIONS_FOLDER,
    "test_predictions.csv",
)

test_predictions.to_csv(prediction_path, index=False)

print("\nTest predictions saved at:")
print(prediction_path)

# ================================================================
# 32. SAVE TRAINED MODEL
# ================================================================

model_path = os.path.join(
    MODEL_FOLDER,
    "random_forest_model.pkl",
)

with open(model_path, "wb") as file:
    pickle.dump(model, file)

print("\nTrained model saved at:")
print(model_path)

# ================================================================
# 33. SAVE RANDOM FOREST PARAMETERS
# ================================================================

parameters_df = pd.DataFrame(
    {
        "Parameter": [
            "Algorithm",
            "Number of Estimators",
            "Maximum Depth",
            "Minimum Samples Split",
            "Minimum Samples Leaf",
            "Random State",
        ],
        "Value": [
            "Random Forest Classifier",
            random_forest.n_estimators,
            random_forest.max_depth,
            random_forest.min_samples_split,
            random_forest.min_samples_leaf,
            random_forest.random_state,
        ],
    }
)

parameters_path = os.path.join(
    METRICS_FOLDER,
    "random_forest_parameters.csv",
)

parameters_df.to_csv(parameters_path, index=False)

print("\nModel parameters saved at:")
print(parameters_path)

# ================================================================
# 34. FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 80)
print("        RANDOM FOREST COMPLETED SUCCESSFULLY")
print("=" * 80)

print("\nOriginal RAW dataset was NOT modified.")
print("\nOriginal dataset:")
print(DATASET_PATH)

print("\nAll outputs stored in:")
print(OUTPUT_FOLDER)

print("\n")
print("FINAL PERFORMANCE")
print("-" * 50)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\n")
print("OUTPUT FOLDERS")
print("-" * 50)

print("1. metrics")
print("2. predictions")
print("3. confusion_matrix")
print("4. charts")
print("5. random_forest_tree")
print("6. feature_importance")
print("7. model")

print("\n")
print("=" * 80)
print("                 PROGRAM FINISHED")
print("=" * 80)