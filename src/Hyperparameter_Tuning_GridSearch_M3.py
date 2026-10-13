# ================================================================
# CARTIQ - GROCERY MARKET BASKET INTELLIGENCE
# ================================================================
# MODULE 3 : HYPERPARAMETER TUNING (GridSearchCV)
# ================================================================
# PURPOSE   : Tune the Module 3 tree-based classifiers instead of
#             using fixed, hand-picked hyperparameters.
#
# MODELS    : Decision Tree, Random Forest, AdaBoost
#
# TASK      : Predict whether a product will be reordered
#             (reordered = 1) or not (reordered = 0).
#
# DATASET   : RAW dataset is used. All preprocessing happens inside
#             a sklearn pipeline. The original RAW dataset is NEVER
#             modified.
#
# METHOD    : GridSearchCV with StratifiedKFold (5 folds),
#             scoring = "f1",  refit = True
#
# NOTE      : This script writes ONLY to its own output folder.
#             No existing M3 output, chart, message or .pkl file is
#             overwritten. Old results stay intact for comparison.
# ================================================================

import os
import json
import time
import pickle

import matplotlib
matplotlib.use("Agg")          # headless-safe backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

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
    "D:/ML/CartIQ/outputs/Hyperparameter_Tuning_GridSearch_M3_Outputs"
)

TUNING_FOLDER = os.path.join(OUTPUT_FOLDER, "tuning")
METRICS_FOLDER = os.path.join(OUTPUT_FOLDER, "metrics")
CHARTS_FOLDER = os.path.join(OUTPUT_FOLDER, "charts")
MESSAGES_FOLDER = os.path.join(OUTPUT_FOLDER, "messages")
MODEL_FOLDER = os.path.join(OUTPUT_FOLDER, "model")
BEST_PARAMS_FOLDER = os.path.join(OUTPUT_FOLDER, "best_params")

for folder in [
    TUNING_FOLDER,
    METRICS_FOLDER,
    CHARTS_FOLDER,
    MESSAGES_FOLDER,
    MODEL_FOLDER,
    BEST_PARAMS_FOLDER,
]:
    os.makedirs(folder, exist_ok=True)

print("Output folders created successfully.")
print("Output folder :", OUTPUT_FOLDER)

# ================================================================
# 2. LOG (everything printed is also saved to a text file)
# ================================================================


class Tee:
    """Write to the console and to a log file at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


LOG_PATH = os.path.join(MESSAGES_FOLDER, "Tuning_Log.txt")

log_file = open(LOG_PATH, "w", encoding="utf-8")

import sys  # noqa: E402

original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, log_file)

# ================================================================
# 3. CHECK DATASET
# ================================================================

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"\nDataset not found at:\n{DATASET_PATH}")

print("\nDataset found successfully.")

# ================================================================
# 4. LOAD RAW DATASET
# ================================================================

data = pd.read_csv(DATASET_PATH)

print("\nRAW dataset loaded successfully.")
print("Rows    :", data.shape[0])
print("Columns :", data.shape[1])

# ================================================================
# 5. CREATE A WORKING COPY (RAW is never modified)
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
        "\n\nAvailable columns:\n" + "\n".join(df.columns.astype(str))
    )

print("\nTarget column :", target_column)

# ================================================================
# 7. SEPARATE FEATURES AND TARGET
# ================================================================
#
# Same convention as every other Module 3 script: identifier columns
# are not predictors and are removed.
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
# 9. PREPROCESSING TRANSFORMERS (identical to Module 3)
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
# 11. TRAIN TEST SPLIT (same split as Module 3 -> comparable results)
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
# 12. CROSS-VALIDATION STRATEGY
# ================================================================

CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

print("\nCross-validation : StratifiedKFold(n_splits=5, shuffle=True)")
print("Scoring metric   : f1 (positive class = reordered = 1)")

# ================================================================
# 13. MODEL DEFINITIONS
# ================================================================
#
# baseline  -> the fixed hyperparameters used in the original M3 scripts
# grid      -> the parameter values GridSearchCV will try
# threads   -> how to split the CPU work (avoid oversubscription)
#
# Random Forest already uses every core per fit, so its search runs
# one configuration at a time. Decision Tree and AdaBoost are
# single-threaded, so their searches run configurations in parallel.
# ================================================================

MODELS = {
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "baseline": {
            "criterion": "gini",
            "max_depth": 6,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
        },
        "grid": {
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [4, 6, 8, 10, None],
            "classifier__min_samples_split": [2, 10],
            "classifier__min_samples_leaf": [1, 5],
        },
        "threads": {"search_n_jobs": -1, "model_n_jobs": 1},
        "heatmap": (
            "classifier__max_depth",
            "classifier__criterion",
        ),
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
        "baseline": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
        },
        "grid": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [10, None],
            "classifier__min_samples_leaf": [1, 5],
        },
        "threads": {"search_n_jobs": 1, "model_n_jobs": -1},
        "heatmap": (
            "classifier__n_estimators",
            "classifier__max_depth",
        ),
    },
    "AdaBoost": {
        "estimator": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            random_state=42,
        ),
        "baseline": {
            "estimator__max_depth": 2,
            "n_estimators": 100,
        },
        "grid": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__estimator__max_depth": [1, 2],
        },
        "threads": {"search_n_jobs": -1, "model_n_jobs": 1},
        "heatmap": (
            "classifier__n_estimators",
            "classifier__estimator__max_depth",
        ),
    },
}

# ================================================================
# 14. HELPER FUNCTIONS
# ================================================================


def pretty_params(params):
    """Compact, readable one-line version of a parameter dictionary."""
    parts = []
    for key, value in sorted(params.items()):
        name = key.replace("classifier__", "")
        parts.append(f"{name}={value}")
    return ", ".join(parts)


def short_axis(value):
    """Axis label helper - None becomes the word 'None'."""
    return "None" if value is None else str(value)


def natural_key(text):
    """Sort '4, 6, 8, 10, None' naturally instead of alphabetically."""
    try:
        return (0, float(text), text)
    except (TypeError, ValueError):
        return (1, 0.0, str(text))


def evaluate(name, pipeline, X_test, y_test):
    """Full test-set metrics for one fitted pipeline."""
    predictions = pipeline.predict(X_test)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions),
        "Predictions": predictions,
    }


def save_heatmap(results, param_x, param_y, path, title):
    """Mean CV F1 heatmap over the two most important parameters.

    When the grid has extra parameters, the best score for each
    (x, y) cell is shown, so the heatmap stays easy to read.
    """
    table = results.copy()

    table["__x"] = table["param_" + param_x].map(short_axis)
    table["__y"] = table["param_" + param_y].map(short_axis)

    pivot = table.pivot_table(
        index="__y",
        columns="__x",
        values="mean_test_score",
        aggfunc="max",
    )

    pivot = pivot.reindex(
        sorted(pivot.index, key=natural_key)
    )
    pivot = pivot.reindex(
        sorted(pivot.columns, key=natural_key),
        axis=1,
    )

    matrix = pivot.to_numpy(dtype=float)

    figure, axes = plt.subplots(figsize=(8, 5.5))

    image = axes.imshow(matrix, cmap="YlGnBu", aspect="auto")

    axes.set_xticks(np.arange(len(pivot.columns)))
    axes.set_xticklabels(
        [str(c) for c in pivot.columns],
        rotation=0,
    )
    axes.set_yticks(np.arange(len(pivot.index)))
    axes.set_yticklabels([str(i) for i in pivot.index])

    axes.set_xlabel(param_x.replace("classifier__", ""))
    axes.set_ylabel(param_y.replace("classifier__", ""))
    axes.set_title(title, fontsize=12)

    # Write the score inside every cell.
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isnan(value):
                continue
            axes.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black" if value < 0.80 else "white",
            )

    figure.colorbar(image, ax=axes, label="Mean CV F1")

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    return pivot


def save_top_configs(results, path, title, top_n=10):
    """Horizontal bar chart of the best configurations found."""
    top = results.sort_values("mean_test_score", ascending=False).head(top_n)

    labels = [pretty_params(p) for p in top["params"]]

    figure, axes = plt.subplots(figsize=(11, 6))

    positions = np.arange(len(top))

    axes.barh(positions, top["mean_test_score"], color="#14B8A6")

    axes.set_yticks(positions)
    axes.set_yticklabels(labels, fontsize=8)
    axes.invert_yaxis()

    axes.set_xlabel("Mean CV F1")
    axes.set_title(title, fontsize=12)

    for position, value in zip(positions, top["mean_test_score"]):
        axes.text(
            value,
            position,
            f" {value:.4f}",
            va="center",
            fontsize=8,
        )

    axes.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# ================================================================
# 15. RUN GRID SEARCH FOR EVERY MODEL
# ================================================================

summary_rows = []
all_results = {}
baseline_metrics = {}
tuned_metrics = {}
confusion_pairs = {}
best_params_by_model = {}

for name, config in MODELS.items():
    print("\n" + "=" * 80)
    print(f"HYPERPARAMETER TUNING - {name.upper()}")
    print("=" * 80)

    key = name.replace(" ", "_")

    # ------------------------------------------------------------
    # 15.1 Baseline (original fixed hyperparameters)
    # ------------------------------------------------------------

    print("\n[1/4] Training baseline with the ORIGINAL fixed settings...")
    print(f"      baseline params : {pretty_params(config['baseline'])}")

    baseline_model = config["estimator"].set_params(**config["baseline"])

    baseline_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", baseline_model),
        ]
    )

    started = time.time()
    baseline_pipeline.fit(X_train, y_train)
    baseline_seconds = time.time() - started

    baseline_result = evaluate(
        f"{name} (before tuning)",
        baseline_pipeline,
        X_test,
        y_test,
    )

    baseline_metrics[name] = baseline_result

    print(f"      baseline F1     : {baseline_result['F1']:.4f}")
    print(f"      baseline time   : {baseline_seconds:.2f}s")

    # ------------------------------------------------------------
    # 15.2 GridSearchCV
    # ------------------------------------------------------------

    search_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", config["estimator"]),
        ]
    )

    grid_size = int(np.prod([len(v) for v in config["grid"].values()]))

    print("\n[2/4] Running GridSearchCV...")
    print(f"      parameters      : {list(config['grid'].keys())}")
    print(f"      configurations  : {grid_size}")
    print(f"      folds           : {CV.get_n_splits()}")
    print(f"      total fits      : {grid_size * CV.get_n_splits()}")

    search = GridSearchCV(
        estimator=search_pipeline,
        param_grid=config["grid"],
        scoring="f1",
        cv=CV,
        refit=True,
        n_jobs=config["threads"]["search_n_jobs"],
        verbose=1,
        return_train_score=True,
        error_score="raise",
    )

    started = time.time()
    search.fit(X_train, y_train)
    search_seconds = time.time() - started

    print(f"\n      search time     : {search_seconds:.2f}s")
    print(f"      best CV F1      : {search.best_score_:.4f}")
    print(f"      best params     : {pretty_params(search.best_params_)}")

    # ------------------------------------------------------------
    # 15.3 Tidy results table + CSV
    # ------------------------------------------------------------

    results = pd.DataFrame(search.cv_results_)

    results["param_summary"] = [pretty_params(p) for p in results["params"]]

    results = results.sort_values("rank_test_score")

    display_columns = [
        "rank_test_score",
        "param_summary",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "mean_fit_time",
    ]

    results_table = results[display_columns].copy()

    results_table.columns = [
        "Rank",
        "Parameters",
        "Mean CV F1",
        "Std CV F1",
        "Mean Train F1",
        "Mean Fit Seconds",
    ]

    results_path = os.path.join(
        TUNING_FOLDER,
        f"GridSearch_All_Results_{key}.csv",
    )

    results_table.to_csv(results_path, index=False)

    print(f"      full results    : {results_path}")

    all_results[name] = results

    # ------------------------------------------------------------
    # 15.4 Refit best model on the full training set + test metrics
    # ------------------------------------------------------------

    print("\n[3/4] Evaluating the tuned model on the held-out test set...")

    tuned_pipeline = search.best_estimator_

    tuned_result = evaluate(
        f"{name} (after tuning)",
        tuned_pipeline,
        X_test,
        y_test,
    )

    tuned_metrics[name] = tuned_result
    best_params_by_model[name] = search.best_params_

    confusion_pairs[name] = (
        confusion_matrix(y_test, baseline_result["Predictions"]),
        confusion_matrix(y_test, tuned_result["Predictions"]),
    )

    print(f"      tuned F1        : {tuned_result['F1']:.4f}")
    print(
        f"      change in F1    : "
        f"{tuned_result['F1'] - baseline_result['F1']:+.4f}"
    )

    # ------------------------------------------------------------
    # 15.5 Charts + saved artefacts
    # ------------------------------------------------------------

    print("\n[4/4] Saving charts and artefacts...")

    save_heatmap(
        results,
        config["heatmap"][0],
        config["heatmap"][1],
        os.path.join(CHARTS_FOLDER, f"GridSearch_Heatmap_{key}.png"),
        f"{name} - mean CV F1 over the parameter grid",
    )

    save_top_configs(
        results,
        os.path.join(CHARTS_FOLDER, f"Top10_Configs_{key}.png"),
        f"{name} - top 10 configurations by mean CV F1",
    )

    report = classification_report(
        y_test,
        tuned_result["Predictions"],
        digits=4,
    )

    with open(
        os.path.join(TUNING_FOLDER, f"Classification_Report_Tuned_{key}.txt"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(f"CARTIQ - {name} AFTER HYPERPARAMETER TUNING\n")
        handle.write("=" * 60 + "\n\n")
        handle.write("Best parameters:\n")
        for parameter, value in sorted(search.best_params_.items()):
            handle.write(f"  {parameter} = {value}\n")
        handle.write(f"\nBest cross-validated F1 : {search.best_score_:.4f}\n")
        handle.write(f"Test F1                 : {tuned_result['F1']:.4f}\n")
        handle.write("\n")
        handle.write(report)

    with open(
        os.path.join(BEST_PARAMS_FOLDER, f"Best_Params_{key}.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "model": name,
                "baseline_params": config["baseline"],
                "best_params": search.best_params_,
                "baseline_test_f1": round(baseline_result["F1"], 6),
                "best_cv_f1": round(float(search.best_score_), 6),
                "tuned_test_f1": round(tuned_result["F1"], 6),
                "improvement_f1": round(
                    tuned_result["F1"] - baseline_result["F1"], 6
                ),
            },
            handle,
            indent=2,
        )

    with open(
        os.path.join(MODEL_FOLDER, f"Tuned_{key}_Best_Pipeline.pkl"),
        "wb",
    ) as handle:
        pickle.dump(tuned_pipeline, handle)

    summary_rows.append(
        {
            "Model": name,
            "Configurations Tested": grid_size,
            "Folds": CV.get_n_splits(),
            "Total Fits": grid_size * CV.get_n_splits(),
            "Baseline Params": pretty_params(config["baseline"]),
            "Best Params": pretty_params(search.best_params_),
            "Baseline Test F1": round(baseline_result["F1"], 4),
            "Best CV F1": round(float(search.best_score_), 4),
            "Tuned Test F1": round(tuned_result["F1"], 4),
            "F1 Improvement": round(
                tuned_result["F1"] - baseline_result["F1"], 4
            ),
            "Search Seconds": round(search_seconds, 2),
        }
    )

    print(f"      charts saved    : GridSearch_Heatmap_{key}.png, "
          f"Top10_Configs_{key}.png")
    print(f"      model saved     : Tuned_{key}_Best_Pipeline.pkl")

# ================================================================
# 16. COMPARISON TABLE (before vs after)
# ================================================================

print("\n" + "=" * 80)
print("BEFORE vs AFTER HYPERPARAMETER TUNING")
print("=" * 80)

summary = pd.DataFrame(summary_rows)

summary_path = os.path.join(
    METRICS_FOLDER,
    "Tuning_Before_vs_After.csv",
)

summary.to_csv(summary_path, index=False)

print("\n", summary.to_string(index=False), sep="")

# One tidy row per model per stage, easy to read in the app.
stage_rows = []

for name in MODELS:
    for stage, metrics in [
        ("Before tuning (fixed params)", baseline_metrics[name]),
        ("After tuning (GridSearchCV)", tuned_metrics[name]),
    ]:
        stage_rows.append(
            {
                "Model": name,
                "Stage": stage,
                "Accuracy": round(metrics["Accuracy"], 4),
                "Precision": round(metrics["Precision"], 4),
                "Recall": round(metrics["Recall"], 4),
                "F1": round(metrics["F1"], 4),
            }
        )

stage_table = pd.DataFrame(stage_rows)

stage_table.to_csv(
    os.path.join(METRICS_FOLDER, "Tuned_Test_Metrics_All_Models.csv"),
    index=False,
)

# ================================================================
# 17. CHART - BEFORE vs AFTER F1
# ================================================================

names = list(MODELS.keys())

before_values = [baseline_metrics[n]["F1"] for n in names]
after_values = [tuned_metrics[n]["F1"] for n in names]

positions = np.arange(len(names))
width = 0.36

figure, axes = plt.subplots(figsize=(9, 5.5))

bars_before = axes.bar(
    positions - width / 2,
    before_values,
    width,
    label="Before tuning (fixed params)",
    color="#94A3B8",
)

bars_after = axes.bar(
    positions + width / 2,
    after_values,
    width,
    label="After tuning (GridSearchCV)",
    color="#14B8A6",
)

axes.set_xticks(positions)
axes.set_xticklabels(names)
axes.set_ylabel("Test F1 score")
axes.set_title("Hyperparameter tuning - test F1 before vs after", fontsize=12)
axes.set_ylim(0, 1)
axes.legend()
axes.grid(axis="y", alpha=0.3)

for bars in (bars_before, bars_after):
    for bar in bars:
        height = bar.get_height()
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

plt.tight_layout()
plt.savefig(
    os.path.join(CHARTS_FOLDER, "Before_vs_After_F1.png"),
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# ================================================================
# 18. CHART - CONFUSION MATRICES BEFORE vs AFTER
# ================================================================

figure, axes_grid = plt.subplots(
    len(names),
    2,
    figsize=(9, 4 * len(names)),
)

for row, name in enumerate(names):
    before_matrix, after_matrix = confusion_pairs[name]

    for column, (matrix, stage) in enumerate(
        [(before_matrix, "Before tuning"), (after_matrix, "After tuning")]
    ):
        axes = axes_grid[row][column]

        axes.imshow(matrix, cmap="Blues")

        axes.set_title(f"{name}\n{stage}", fontsize=10)
        axes.set_xlabel("Predicted")
        axes.set_ylabel("Actual")

        axes.set_xticks([0, 1])
        axes.set_xticklabels(["0", "1"])
        axes.set_yticks([0, 1])
        axes.set_yticklabels(["0", "1"])

        for i in range(2):
            for j in range(2):
                axes.text(
                    j,
                    i,
                    f"{matrix[i, j]:,}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black" if matrix[i, j] < matrix.max() else "white",
                )

plt.tight_layout()
plt.savefig(
    os.path.join(CHARTS_FOLDER, "Confusion_Matrices_Before_vs_After.png"),
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# ================================================================
# 19. WRITTEN SUMMARY
# ================================================================

info_lines = []
info_lines.append("CARTIQ - HYPERPARAMETER TUNING (GRIDSEARCHCV)")
info_lines.append("=" * 52)
info_lines.append("")
info_lines.append("Module           : Module 3 tree-based classifiers")
info_lines.append("Models tuned     : Decision Tree, Random Forest, AdaBoost")
info_lines.append("Dataset          : " + DATASET_PATH)
info_lines.append("Rows used        : " + str(df.shape[0]))
info_lines.append("Features used    : " + ", ".join(feature_columns))
info_lines.append("Train / Test     : 80% / 20% (stratified, random_state=42)")
info_lines.append("Cross-validation : StratifiedKFold, 5 folds, shuffle=True")
info_lines.append("Scoring metric   : F1 (positive class = reordered = 1)")
info_lines.append("Search method    : GridSearchCV (exhaustive grid search)")
info_lines.append("")
info_lines.append("Why tuning was needed")
info_lines.append("-" * 52)
info_lines.append(
    "The original Module 3 scripts used fixed, hand-picked "
    "hyperparameters."
)
info_lines.append(
    "GridSearchCV now searches the parameter grid and selects the "
    "combination with"
)
info_lines.append("the best cross-validated F1 score.")
info_lines.append("")

for name in names:
    config = MODELS[name]
    grid_size = int(np.prod([len(v) for v in config["grid"].values()]))
    metrics = tuned_metrics[name]

    info_lines.append(f"{name}")
    info_lines.append("-" * 52)
    info_lines.append(
        "  Baseline params     : " + pretty_params(config["baseline"])
    )
    info_lines.append(
        "  Best params         : " + pretty_params(best_params_by_model[name])
    )
    info_lines.append(f"  Configurations      : {grid_size}")
    info_lines.append(f"  Total fits          : {grid_size * 5}")
    info_lines.append(
        "  Baseline test F1    : "
        f"{baseline_metrics[name]['F1']:.4f}"
    )
    info_lines.append(
        "  Best CV F1          : "
        + f"{all_results[name]['mean_test_score'].max():.4f}"
    )
    info_lines.append(f"  Tuned test F1       : {metrics['F1']:.4f}")
    info_lines.append(
        "  F1 improvement      : "
        f"{metrics['F1'] - baseline_metrics[name]['F1']:+.4f}"
    )
    info_lines.append("")

info_lines.append("Important note")
info_lines.append("-" * 52)
info_lines.append(
    "Hyperparameters are tuned ONLY on the training data through"
)
info_lines.append(
    "cross-validation. The 20% test set is untouched during the search"
)
info_lines.append(
    "and is used once, at the end, to report the final scores."
)
info_lines.append("")
info_lines.append(
    "Original Module 3 outputs and .pkl files were not modified by"
)
info_lines.append("this script - it writes to its own output folder only.")

info_path = os.path.join(
    MESSAGES_FOLDER,
    "Hyperparameter_Tuning_Info.txt",
)

with open(info_path, "w", encoding="utf-8") as handle:
    handle.write("\n".join(info_lines))

print("\nSaved:", info_path)
print("Saved:", summary_path)

print("\n" + "=" * 80)
print("HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY")
print("=" * 80)
print("\nOutput folder:", OUTPUT_FOLDER)

sys.stdout = original_stdout
log_file.close()
