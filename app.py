import os
import pickle
import re

import pandas as pd
from flask import Flask, render_template, request, send_from_directory, abort

# ================================================================
# PATHS
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "instacart_50k_Raw.csv")

app = Flask(__name__)

# ================================================================
# TRAINED MODELS (Module 3 pipelines, trained on the raw dataset)
# ================================================================

MODEL_OPTIONS = {
    "random_forest": {
        "label": "Random Forest (best F1 — 80.0%)",
        "path": os.path.join(
            OUTPUTS_DIR,
            "Random_Forest_Tree_M3_Outputs",
            "model",
            "random_forest_model.pkl",
        ),
    },
    "adaboost": {
        "label": "AdaBoost (best accuracy — 73.3%)",
        "path": os.path.join(
            OUTPUTS_DIR,
            "AdaBoost_Classfier_M3_Outputs",
            "model",
            "adaboost_model.pkl",
        ),
    },
    "decision_tree": {
        "label": "Decision Tree (baseline — 72.2%)",
        "path": os.path.join(
            OUTPUTS_DIR,
            "Decision_Tree_Classifier_M3_Outputs",
            "model",
            "decision_tree_model.pkl",
        ),
    },
}

_model_cache = {}


def load_model(key):
    """Load a trained pipeline once and cache it in memory."""
    if key not in _model_cache:
        with open(MODEL_OPTIONS[key]["path"], "rb") as file:
            _model_cache[key] = pickle.load(file)
    return _model_cache[key]


# ================================================================
# FORM OPTIONS (unique categorical values from the raw dataset)
# ================================================================

_options = pd.read_csv(DATASET_PATH, usecols=["aisle", "department"])
AISLES = sorted(_options["aisle"].dropna().unique().tolist())
DEPARTMENTS = sorted(_options["department"].dropna().unique().tolist())
del _options

DAY_NAMES = [
    "Sunday", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday",
]

# ================================================================
# SERVE CHARTS AND FILES FROM THE OUTPUTS FOLDER
# ================================================================


@app.route("/outputs/<path:filename>")
def outputs_file(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


# ================================================================
# PAGES
# ================================================================


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


# ---- Dataset preview (first N rows shown; full file via the button) ----

PREVIEW_ROWS = 500
_preview_cache = None


def dataset_preview(n=PREVIEW_ROWS):
    """Read only the first n rows of the raw CSV (cached) so the page stays light.

    Rows come back as plain JSON-safe Python values (int / float / str / None)
    so the page can hand them to JavaScript and render them in small pages.
    """
    global _preview_cache
    if _preview_cache is None:
        df = pd.read_csv(DATASET_PATH, nrows=n)
        with open(DATASET_PATH, "r", encoding="utf-8", errors="replace") as fh:
            total = max(sum(1 for _ in fh) - 1, 0)  # minus the header line
        columns = [str(c) for c in df.columns]
        frame = df.astype(object).where(pd.notna(df), None)
        _preview_cache = (columns, frame.values.tolist(), total)
    return _preview_cache


@app.route("/dataset")
def dataset():
    columns, rows, total_rows = dataset_preview()
    return render_template(
        "dataset.html",
        preview_columns=columns,
        preview_rows=rows,
        preview_shown=len(rows),
        preview_limit=PREVIEW_ROWS,
        total_rows=total_rows,
    )


@app.route("/dataset/csv")
def dataset_csv():
    """Open the full raw CSV inline, or download it with ?download=1."""
    as_attachment = "download" in request.args
    return send_from_directory(
        os.path.join(BASE_DIR, "dataset"),
        "instacart_50k_Raw.csv",
        mimetype="text/csv",
        as_attachment=as_attachment,
        download_name="instacart_50k_Raw.csv",
    )


PREPROCESS_STAGES = [
    {
        "step": "01",
        "title": "Cleaning",
        "icon": "🧼",
        "tone": "tint-peach",
        "file": "clean_del_mean_model_M2.csv",
        "script": "clean_del_mean_model_missing_imputer_M2.py",
        "summary": "Duplicates removed, missing values filled — median for numbers, mode for text — and every text value tidied with strip() and lower().",
        "changes": [
            "duplicates removed",
            "median / mean / model imputation compared",
            "text → trimmed & lowercased",
            "nulls become 0 (59 columns with indicator flags)",
        ],
    },
    {
        "step": "02",
        "title": "Encoding",
        "icon": "🔤",
        "tone": "tint-lilac",
        "file": "clean_label_encode_M2.csv",
        "script": "clean_label_encode_M2.py (+ one-hot, ordinal, target, embedding)",
        "summary": "Text categories become numbers. Label encoding drives the main pipeline, while four other encoders are benchmarked side by side.",
        "changes": [
            "text → numbers",
            "label · one-hot · ordinal · target · embedding",
            "eval_set, aisle & department encoded",
        ],
    },
    {
        "step": "03",
        "title": "Scaling",
        "icon": "⚖️",
        "tone": "tint-mint",
        "file": "clean_minmax_stand_norma_M2.csv",
        "script": "clean_minmax_stand_norma_M2.py",
        "summary": "Every numeric feature is rescaled so that no single column dominates. The file keeps the originals beside their z-score, Min-Max and L2 versions so the methods can be compared.",
        "changes": [
            "z-score · min-max · L2",
            "mean ≈ 0, standard deviation ≈ 1",
            "44 columns explored",
        ],
    },
]


FINAL_STAGE = {
    "step": "04",
    "title": "Final preprocessed file",
    "icon": "✅",
    "tone": "tint-lemon",
    "file": "final_preprocess_M2.csv",
    "script": "final_preprocess_M2.py",
    "summary": "One clean, fully numeric, scaled matrix — the single output that every model in the project reads.",
    "changes": [
        "50,000 rows × 14 columns",
        "0 missing values",
        "100% numeric & scaled",
        "saved as dataset/final_preprocess_M2.csv",
    ],
}

ALLOWED_STAGE_FILES = {s["file"] for s in PREPROCESS_STAGES} | {FINAL_STAGE["file"]}

_stage_preview_cache = {}


def stage_preview(filename, n=6, max_cols=14):
    """Preview the first n rows of a stage's CSV (cached, column-capped)."""
    key = (filename, n, max_cols)
    if key not in _stage_preview_cache:
        path = os.path.join(BASE_DIR, "dataset", filename)
        df = pd.read_csv(path, nrows=n)
        total_cols = df.shape[1]
        shown = [str(c) for c in df.columns[:max_cols]]
        data = df[shown].fillna("").values.tolist()
        _stage_preview_cache[key] = (shown, data, total_cols)
    return _stage_preview_cache[key]


@app.route("/preprocessing")
def preprocessing():
    def decorate(stage):
        columns, rows, total_cols = stage_preview(stage["file"])
        return {
            **stage,
            "columns": columns,
            "rows": rows,
            "total_cols": total_cols,
            "hidden_cols": max(total_cols - len(columns), 0),
        }

    stages = [decorate(stage) for stage in PREPROCESS_STAGES]
    _, _, total_rows = dataset_preview()
    return render_template(
        "preprocessing.html",
        stages=stages,
        final_stage=decorate(FINAL_STAGE),
        total_rows=total_rows,
    )


@app.route("/preprocessing/file/<path:filename>")
def preprocessing_file(filename):
    """Open or download one of the stage CSVs (?download=1 forces download)."""
    if filename not in ALLOWED_STAGE_FILES:
        abort(404)
    as_attachment = "download" in request.args
    return send_from_directory(
        os.path.join(BASE_DIR, "dataset"),
        filename,
        mimetype="text/csv",
        as_attachment=as_attachment,
        download_name=filename,
    )


M1_FOLDERS = ["EDA_Analysis_outputs", "Boxplots_correlation"]


# ---- Visualisation: group the Module 1 charts into readable stories ----

VIZ_CATEGORIES = [
    ("reorder", "Reorder relationships", "🔁",
     "How each feature behaves once a customer reorders — the target we predict."),
    ("distributions", "Distributions", "📊",
     "Histograms that reveal the shape, skew and spread of every numeric feature."),
    ("boxplots", "Boxplots & spread", "📦",
     "Median, range and outliers per feature — overall and split by reorder."),
    ("correlation", "Correlation & missing values", "🔥",
     "How the features move together, and exactly where the dataset has gaps."),
    ("counts", "Categorical counts", "🏷️",
     "How often each aisle, department and eval_set value appears in the baskets."),
    ("patterns", "Outliers & patterns", "🧭",
     "Pairplots and scatter views used to spot structure and extreme points."),
    ("other", "Other diagnostics", "🖼️",
     "The remaining charts produced by the Module 1 analysis scripts."),
]


def viz_category(filename):
    """Map a chart filename to one of the visualisation categories."""
    name = filename.lower()
    if "correlation" in name or "missing" in name or "heatmap" in name:
        return "correlation"
    if "histogram" in name:
        return "distributions"
    if "boxplot" in name:
        return "boxplots"
    if "countplot" in name:
        return "counts"
    if "target" in name or "vs_reordered" in name:
        return "reorder"
    if any(word in name for word in ("pairplot", "scatter", "sctter", "outlier")):
        return "patterns"
    return "other"


# The charts the page opens with — the ones that set the scene.
VIZ_HIGHLIGHTS = [
    {
        "file": "EDA_Analysis_outputs/Correlation_Heatmap.png",
        "icon": "🔥",
        "tone": "tint-sky",
        "title": "Correlation heatmap",
        "fact": "No two features are strongly linked — the basket signals are almost independent, "
                "so no single column can predict a reorder on its own. That is the honest starting point.",
    },
    {
        "file": "Boxplots_correlation/boxplot_basket_size_vs_reordered.png",
        "icon": "📦",
        "tone": "tint-peach",
        "title": "Boxplot · basket size vs reorder",
        "fact": "Reordered and non-reordered baskets share almost the same box — basket size alone "
                "says very little about whether an item will be bought again.",
    },
    {
        "file": "EDA_Analysis_outputs/Target_Distribution.png",
        "icon": "🎯",
        "tone": "tint-mint",
        "title": "Target balance",
        "fact": "About 59% of lines are reorders — a mild imbalance, which is exactly why the project "
                "reports F1 and recall, not accuracy alone.",
    },
    {
        "file": "EDA_Analysis_outputs/Missing_Values_Heatmap_M1.png",
        "icon": "🕳️",
        "tone": "tint-lilac",
        "title": "Missing-value map",
        "fact": "Only days_since_prior_order has gaps (3,209 of them) — and those gaps are first-time "
                "orders, not spoiled data.",
    },
]


# Plain-language facts anyone can take away from the charts.
VIZ_FACTS = [
    ("🧺", "50,000 basket lines",
     "Across 4,799 customers, 11,650 products and 21 departments — the average basket holds about 15.8 items."),
    ("🔁", "59.1% are reorders",
     "Most lines are repeat purchases, so the target is imbalanced and accuracy alone would mislead."),
    ("🕳️", "Only one column is incomplete",
     "days_since_prior_order is empty 3,209 times — those are customers placing their very first order."),
    ("📊", "Baskets are right-skewed",
     "Most baskets are small, but a long tail of very large baskets pulls the average upward."),
    ("🕐", "Shopping follows daylight",
     "Orders cluster in daytime hours with a late-morning and afternoon peak — very little happens overnight."),
    ("🌳", "Features barely correlate",
     "Because the signals are nearly independent, tree-based models such as Random Forest beat the linear ones."),
]


def viz_highlights():
    """Only return highlight charts that actually exist on disk."""
    found = []
    for item in VIZ_HIGHLIGHTS:
        path = os.path.join(OUTPUTS_DIR, *item["file"].split("/"))
        if os.path.exists(path):
            found.append(item)
    return found


@app.route("/visualization")
def visualization():
    """Show every Module 1 exploratory chart, grouped into readable stories."""
    by_name = {m["name"]: m for m in _collect_modules()}
    modules = [by_name[folder] for folder in M1_FOLDERS if folder in by_name]

    charts = []
    for module in modules:
        for img in module["images"]:
            if img.get("group") != "charts":
                continue
            charts.append({**img, "cat": viz_category(img["name"])})

    meta = {key: (title, icon, blurb) for key, title, icon, blurb in VIZ_CATEGORIES}
    order = {key: i for i, (key, *_rest) in enumerate(VIZ_CATEGORIES)}

    sections = []
    for key in sorted(meta, key=lambda k: order[k]):
        items = sorted(
            (c for c in charts if c["cat"] == key),
            key=lambda c: c["name"].lower(),
        )
        if not items:
            continue
        title, icon, blurb = meta[key]
        sections.append({
            "key": key,
            "title": title,
            "icon": icon,
            "blurb": blurb,
            "charts": items,
            "count": len(items),
        })

    return render_template(
        "visualization.html",
        sections=sections,
        modules=modules,
        highlights=viz_highlights(),
        facts=VIZ_FACTS,
        total_images=len(charts),
        total_messages=sum(m["n_messages"] for m in modules),
    )


@app.route("/models")
def models():
    """List every model by name, with a jump/filter bar at the top."""
    by_name = {m["name"]: m for m in _collect_modules()}
    folders = [folder for spec in MODEL_GROUPS for folder in spec["folders"]]

    entries = []
    for folder in folders:
        module = by_name.get(folder)
        if not module:
            continue
        # Drop the "Module N · " prefix so only the model name shows.
        short = re.sub(r"^Module\s*\d+\s*[·\-]\s*", "", module["title"]).strip()
        entries.append({
            "key": folder,
            "short": short,
            "module": module,
            "n_images": module["n_images"],
            "n_messages": module["n_messages"],
        })

    return render_template(
        "models.html",
        models=entries,
        total_models=len(entries),
        comparison=CLASSIFICATION_COMPARISON,
        regression=REGRESSION_COMPARISON,
        unsupervised=UNSUPERVISED_COMPARISON,
        tuning=TUNING_COMPARISON,
        multinomial_note=MULTINOMIAL_NOTE,
        best_model=BEST_MODEL,
        baseline_accuracy=BASELINE_ACCURACY,
        baseline_f1=BASELINE_F1,
    )


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    result = None
    error = None

    form = {
        "model": "random_forest",
        "order_number": "5",
        "order_dow": "2",
        "order_hour_of_day": "10",
        "days_since_prior_order": "7",
        "add_to_cart_order": "3",
        "basket_size": "10",
        "eval_set": "train",
        "aisle": "fresh fruits",
        "department": "produce",
    }

    if request.method == "POST":
        for key in form:
            form[key] = request.form.get(key, form[key])

        try:
            model_key = form["model"]
            if model_key not in MODEL_OPTIONS:
                raise ValueError("Unknown model selected.")

            row = pd.DataFrame(
                [
                    {
                        "order_number": int(form["order_number"]),
                        "order_dow": int(form["order_dow"]),
                        "order_hour_of_day": int(form["order_hour_of_day"]),
                        "days_since_prior_order": (
                            float(form["days_since_prior_order"])
                            if form["days_since_prior_order"].strip()
                            else None
                        ),
                        "add_to_cart_order": int(form["add_to_cart_order"]),
                        "basket_size": int(form["basket_size"]),
                        "eval_set": form["eval_set"],
                        "aisle": form["aisle"],
                        "department": form["department"],
                    }
                ]
            )

            model = load_model(model_key)
            prediction_value = int(model.predict(row)[0])
            probability = float(model.predict_proba(row)[0][1])

            result = {
                "reordered": prediction_value == 1,
                "probability": round(probability * 100, 1),
                "model_label": MODEL_OPTIONS[model_key]["label"],
            }
        except Exception as exc:
            error = str(exc)

    return render_template(
        "prediction.html",
        model_options=MODEL_OPTIONS,
        aisles=AISLES,
        departments=DEPARTMENTS,
        day_names=DAY_NAMES,
        form=form,
        result=result,
        error=error,
    )


# ================================================================
# RESULTS GALLERY (every chart + every generated message)
# ================================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp"}
TEXT_EXTS = {".txt", ".md", ".log", ".json"}

# Internal note/log/json files (.txt like dataset_information.txt) are NOT shown
# on the website. Files stay on disk; only the display is switched off.
# Set this to True if they should appear again.
SHOW_TEXT_NOTES = False
TABLE_EXTS = {".csv", ".tsv"}

# Friendly titles / one-line descriptions / icon for each module folder.
MODULE_META = {
    "EDA_Analysis_outputs": (
        "Module 1 · Exploratory Data Analysis",
        "Histograms, counts, correlation, missing-value and outlier views of the raw basket data.",
        "🔎",
    ),
    "Boxplots_correlation": (
        "Module 1 · Boxplots & Correlation",
        "Boxplot comparisons against the reorder target plus the correlation heatmap.",
        "📦",
    ),
    "Linear_Regression_with_Metrics_M2": (
        "Module 2 · Linear Regression",
        "Predicting basket size with every regression metric and diagnostic plot.",
        "📈",
    ),
    "Linear_Regression_CFNE_GD_Compare_M2": (
        "Module 2 · Normal Equation vs Gradient Descent",
        "Closed-form vs iterative solutions compared on loss and residuals.",
        "🧮",
    ),
    "Logistic_Regression_Binary_Classify_M2": (
        "Module 2 · Binary Logistic Regression",
        "Sigmoid, cross-entropy loss, confusion matrix and decision boundary.",
        "🔗",
    ),
    "Multinomial_Logistic_Regre_for_Multiclass_M2": (
        "Module 2 · Multinomial Logistic Regression",
        "21-class department prediction with softmax probabilities.",
        "🧩",
    ),
    "Regularization_ridg_laso_elastic_outputs": (
        "Module 2 · Ridge / Lasso / Elastic Net",
        "L1 vs L2 shrinkage, sparsity and per-model performance.",
        "⚖️",
    ),
    "Decision_Tree_Classifier_M3_Outputs": (
        "Module 3 · Decision Tree",
        "Interpretable reorder baseline: tree diagram, importance and metrics.",
        "🌳",
    ),
    "Random_Forest_Tree_M3_Outputs": (
        "Module 3 · Random Forest",
        "Best F1 ensemble for reorder prediction, with a sample tree.",
        "🌲",
    ),
    "AdaBoost_Classfier_M3_Outputs": (
        "Module 3 · AdaBoost",
        "Best-accuracy boosting model with base-tree and estimator weights.",
        "🚀",
    ),
    "Hyperparameter_Tuning_GridSearch_M3_Outputs": (
        "Module 3 · Hyperparameter Tuning (GridSearchCV)",
        "Exhaustive grid search with 5-fold cross-validation over the tree models, "
        "reporting the best parameters and the before/after F1 change.",
        "🎛️",
    ),
    "K_Means_Elbow_Silhouetee_M4_Outputs": (
        "Module 4 · K-Means & K-Means++",
        "Elbow and silhouette selection with 2-D/3-D/PCA cluster views.",
        "🌀",
    ),
    "Hierarchical_Clust_Agglome_Divisive_M4_Outputs": (
        "Module 4 · Hierarchical Clustering",
        "Agglomerative vs divisive dendrograms and cluster assignments.",
        "🪜",
    ),
    "DBSCAN_Raw_dataset_M4_outputs": (
        "Module 4 · DBSCAN (raw dataset)",
        "Density clustering on the raw behaviour features.",
        "🔵",
    ),
    "DBSCAN_Pre_Processed_dataset_M4_outputs": (
        "Module 4 · DBSCAN (pre-processed)",
        "Density clustering on the standardized feature set.",
        "🔷",
    ),
    "PCA_M4_Outputs": (
        "Module 4 · PCA",
        "Scree plot, cumulative variance, 2-D projection and PC1 loadings.",
        "📉",
    ),
    "t-SNE_M4_outputs": (
        "Module 4 · t-SNE",
        "Non-linear 2-D embedding of the numeric features.",
        "🌌",
    ),
    "CV_DT_All_Classifi_Regres_Metrics_M5_outputs": (
        "Module 5 · Cross-Validated Decision Tree",
        "5-fold stratified CV with ROC-AUC, PR-AUC and full metric reports.",
        "✅",
    ),
}

MODULE_ORDER = list(MODULE_META.keys())

# How the ML Models page groups module folders into course modules.
MODEL_GROUPS = [
    {
        "title": "Module 2 — Regression &amp; regularisation",
        "icon": "📈",
        "blurb": "Basket-size regression, both solvers compared, and regularised reorder classifiers.",
        "folders": [
            "Linear_Regression_with_Metrics_M2",
            "Linear_Regression_CFNE_GD_Compare_M2",
            "Logistic_Regression_Binary_Classify_M2",
            "Multinomial_Logistic_Regre_for_Multiclass_M2",
            "Regularization_ridg_laso_elastic_outputs",
        ],
    },
    {
        "title": "Module 3 — Tree ensembles",
        "icon": "🌳",
        "blurb": "Decision Tree, Random Forest and AdaBoost reorder classifiers.",
        "folders": [
            "Hyperparameter_Tuning_GridSearch_M3_Outputs",
            "Decision_Tree_Classifier_M3_Outputs",
            "Random_Forest_Tree_M3_Outputs",
            "AdaBoost_Classfier_M3_Outputs",
        ],
    },
    {
        "title": "Module 4 — Clustering &amp; dimensionality reduction",
        "icon": "🧩",
        "blurb": "K-Means, hierarchical and density clustering, with PCA and t-SNE.",
        "folders": [
            "K_Means_Elbow_Silhouetee_M4_Outputs",
            "Hierarchical_Clust_Agglome_Divisive_M4_Outputs",
            "DBSCAN_Raw_dataset_M4_outputs",
            "DBSCAN_Pre_Processed_dataset_M4_outputs",
            "PCA_M4_Outputs",
            "t-SNE_M4_outputs",
        ],
    },
    {
        "title": "Module 5 — Cross-validated decision tree",
        "icon": "✅",
        "blurb": "5-fold stratified CV with every classification and regression metric.",
        "folders": [
            "CV_DT_All_Classifi_Regres_Metrics_M5_outputs",
        ],
    },
]

# --------------------------------------------------------------------------
# Verified model comparison shown on the ML Models page.
#
# Every number below was read from the real output files, for example
# outputs/Random_Forest_Tree_M3_Outputs/metrics/random_forest_metrics.csv
#
# Target   : reordered (binary 0/1)
# Test set : 10,000 rows, 80/20 stratified split, random_state = 42
#            (identical split for every model, so the scores compare fairly)
# --------------------------------------------------------------------------

# A model that always answers "reordered" - the lazy baseline to beat.
BASELINE_ACCURACY = 0.5905
BASELINE_F1 = 0.7425

BEST_MODEL = {
    "name": "Random Forest",
    "headline": "F1 0.8093 — best of all 11 reorder models",
    "reasons": [
        "Highest F1 of every model that predicts reorder (0.8093 tuned, 0.8000 untuned).",
        "Highest recall (0.8667 tuned, 0.9153 untuned), so very few real reorders are missed.",
        "Handles the raw dataset's 6 numeric + 3 categorical columns without scaling, and "
        "learns non-linear interactions that logistic regression cannot.",
        "Already saved as a .pkl and served on the Prediction page, so it is deployable today.",
    ],
}

# Sorted by F1 score, best first. accuracy=None means that run reported
# cross-validated metrics only (the M5 script).
CLASSIFICATION_COMPARISON = [
    {"model": "Random Forest (tuned, GridSearchCV)", "kind": "Ensemble",
     "accuracy": 0.7588, "precision": 0.7590, "recall": 0.8667, "f1": 0.8093,
     "note": "BEST F1 — recommended model", "best": True},
    {"model": "Random Forest (original fixed params)", "kind": "Ensemble",
     "accuracy": 0.7298, "precision": 0.7105, "recall": 0.9153, "f1": 0.8000,
     "note": "Highest recall (91.5%)", "best": False},
    {"model": "AdaBoost (tuned, GridSearchCV)", "kind": "Boosting",
     "accuracy": 0.7374, "precision": 0.7388, "recall": 0.8589, "f1": 0.7944,
     "note": "Best accuracy", "best": False},
    {"model": "AdaBoost (original fixed params)", "kind": "Boosting",
     "accuracy": 0.7328, "precision": 0.7339, "recall": 0.8589, "f1": 0.7915,
     "note": "", "best": False},
    {"model": "Decision Tree (tuned, GridSearchCV)", "kind": "Single tree",
     "accuracy": 0.7270, "precision": 0.7214, "recall": 0.8759, "f1": 0.7912,
     "note": "", "best": False},
    {"model": "Decision Tree (original fixed params)", "kind": "Single tree",
     "accuracy": 0.7223, "precision": 0.7241, "recall": 0.8557, "f1": 0.7844,
     "note": "", "best": False},
    {"model": "Lasso (L1)", "kind": "Regularised linear",
     "accuracy": 0.6976, "precision": 0.6990, "recall": 0.8449, "f1": 0.7650,
     "note": "", "best": False},
    {"model": "Elastic Net", "kind": "Regularised linear",
     "accuracy": 0.6967, "precision": 0.7072, "recall": 0.8183, "f1": 0.7587,
     "note": "", "best": False},
    {"model": "Ridge (L2)", "kind": "Regularised linear",
     "accuracy": 0.6933, "precision": 0.7117, "recall": 0.7961, "f1": 0.7516,
     "note": "", "best": False},
    {"model": "Binary Logistic Regression", "kind": "Linear",
     "accuracy": 0.6907, "precision": 0.7149, "recall": 0.7805, "f1": 0.7462,
     "note": "", "best": False},
    {"model": "CV Decision Tree (M5, 5-fold CV)", "kind": "Tree + cross-validation",
     "accuracy": None, "precision": 0.6911, "recall": 0.6905, "f1": 0.6908,
     "note": "ROC-AUC 0.6810 · PR-AUC 0.7003", "best": False},
]

# --------------------------------------------------------------------------
# Table 2 - Regression models (target: basket_size, continuous)
# Read from outputs/Linear_Regression_with_Metrics_M2/linear_regression_metrics.csv
# --------------------------------------------------------------------------

REGRESSION_COMPARISON = [
    {"model": "Linear Regression", "mae": 0.7503, "mse": 0.9464, "rmse": 0.9728,
     "r2": 0.0108, "note": "R² near 0 — basket size is not predictable from these features"},
    {"model": "Normal Equation (closed form)", "mae": None, "mse": 0.9464, "rmse": None,
     "r2": None, "note": "Identical result to gradient descent"},
    {"model": "Gradient Descent (lr = 0.01, 1000 iterations)", "mae": None, "mse": 0.9464,
     "rmse": None, "r2": None, "note": "Identical result to the normal equation"},
]

# --------------------------------------------------------------------------
# Table 3 - Unsupervised learning and dimensionality reduction.
# No labels -> these models cannot predict 'reordered'.
# --------------------------------------------------------------------------

UNSUPERVISED_COMPARISON = [
    {"model": "K-Means", "task": "Clustering",
     "result": "K = 3 · Silhouette 0.3219 · Inertia 114,111.44 · reorder-match 59.05%"},
    {"model": "K-Means++", "task": "Clustering",
     "result": "K = 3 · Silhouette 0.3217 · Inertia 114,111.61 · reorder-match 59.05%"},
    {"model": "Hierarchical — Agglomerative", "task": "Clustering",
     "result": "K = 3 · Silhouette 0.2475 · Calinski-Harabasz 155.74 · Davies-Bouldin 1.3813"},
    {"model": "Hierarchical — Divisive", "task": "Clustering",
     "result": "K = 3 · Silhouette 0.3039 · Calinski-Harabasz 177.08 · Davies-Bouldin 1.2075"},
    {"model": "DBSCAN (raw dataset)", "task": "Density clustering",
     "result": "7 clusters · eps 0.80 · min_samples 10 · 0% noise · 46,791 rows used"},
    {"model": "DBSCAN (pre-processed)", "task": "Density clustering",
     "result": "7 clusters · eps 0.80 · min_samples 10 · 0% noise · 50,000 rows used"},
    {"model": "PCA", "task": "Dimensionality reduction",
     "result": "13 components keep 95% of variance · PC1 = 12.38%"},
    {"model": "t-SNE", "task": "Dimensionality reduction",
     "result": "2-D embedding · perplexity 30 · KL divergence 3.1552"},
]

# --------------------------------------------------------------------------
# Table 4 - Hyperparameter tuning, before vs after (GridSearchCV)
# Read from outputs/Hyperparameter_Tuning_GridSearch_M3_Outputs/metrics/
# --------------------------------------------------------------------------

TUNING_COMPARISON = [
    {"model": "Decision Tree", "configs": 40, "fits": 200, "before": 0.7844, "after": 0.7912,
     "improvement": 0.0067,
     "params": "criterion=gini, max_depth=8, min_samples_split=2, min_samples_leaf=1"},
    {"model": "Random Forest", "configs": 8, "fits": 40, "before": 0.8000, "after": 0.8093,
     "improvement": 0.0093,
     "params": "n_estimators=100, max_depth=None, min_samples_leaf=1"},
    {"model": "AdaBoost", "configs": 6, "fits": 30, "before": 0.7915, "after": 0.7944,
     "improvement": 0.0029,
     "params": "n_estimators=200, base estimator max_depth=2"},
]

# Different question: 21-class department prediction (not comparable with the tables above).
MULTINOMIAL_NOTE = (
    "Multinomial Logistic Regression predicts one of 21 departments, not reorder — "
    "accuracy 0.3358 · precision 0.2096 · recall 0.3358 · F1 0.2153. "
    "A random guess would score about 4.8%, so it is far better than chance, but it answers "
    "a different question and is not ranked with the models above."
)

# keep acronyms looking right when prettifying file names
_ACRONYMS = {
    "pca": "PCA", "dbscan": "DBSCAN", "tsne": "t-SNE", "sne": "t-SNE",
    "roc": "ROC", "auc": "ROC-AUC", "pr": "PR", "pc": "PC", "pc1": "PC1",
    "kmeans": "K-Means", "cv": "CV", "gd": "GD", "cfne": "CFNE",
    "l1": "L1", "l2": "L2", "id": "ID", "vs": "vs", "3d": "3-D", "2d": "2-D",
}


def _prettify(filename):
    """Turn 'confusion_matrix.png' style names into a readable caption."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = re.sub(r"^\d+[_\- ]*", "", stem)
    text = re.sub(r"[_\-]+", " ", stem).strip()
    words = []
    for word in text.split():
        words.append(_ACRONYMS.get(word.lower(), word))
    caption = " ".join(words)
    return caption[:1].upper() + caption[1:] if caption else stem


def _fmt_cell(value):
    """Format a single table cell for display."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,}"
    if isinstance(value, float):
        if float(value).is_integer() and abs(value) < 1e12:
            return f"{int(value):,}"
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return str(value)


def _read_text(path, limit=7000):
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        data = handle.read(limit + 1)
    return data[:limit], len(data) > limit


def _count_data_rows(path, size_cap=30 * 1024 * 1024):
    """Count CSV data rows (header excluded) for small/medium files."""
    try:
        if os.path.getsize(path) > size_cap:
            return None
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            total = sum(1 for _ in handle)
        return max(total - 1, 0)
    except OSError:
        return None


def _read_table(path, max_rows=12):
    try:
        frame = pd.read_csv(path, nrows=max_rows)
    except Exception:
        return [], [], True
    headers = [str(col) for col in frame.columns]
    rows = [[_fmt_cell(v) for v in record] for record in frame.itertuples(index=False, name=None)]
    return headers, rows, len(rows) >= max_rows


def _collect_modules():
    """Walk every module output folder and gather images, messages and downloads."""
    modules = []

    if not os.path.isdir(OUTPUTS_DIR):
        return modules

    present = [
        name
        for name in os.listdir(OUTPUTS_DIR)
        if os.path.isdir(os.path.join(OUTPUTS_DIR, name)) and name != "ui_previews"
    ]
    ordered = [n for n in MODULE_ORDER if n in present]
    ordered += [n for n in sorted(present) if n not in MODULE_ORDER]

    for name in ordered:
        folder = os.path.join(OUTPUTS_DIR, name)
        images, messages, downloads = [], [], []

        for root, _dirs, names in os.walk(folder):
            for file_name in sorted(names):
                full = os.path.join(root, file_name)
                rel = os.path.relpath(full, OUTPUTS_DIR).replace("\\", "/")
                ext = os.path.splitext(file_name)[1].lower()
                sub = os.path.relpath(root, folder).replace("\\", "/")
                group = "charts" if sub == "." else sub.split("/")[0]

                if ext in IMAGE_EXTS:
                    images.append(
                        {
                            "rel": rel,
                            "name": file_name,
                            "caption": _prettify(file_name),
                            "group": group,
                        }
                    )
                elif ext in TEXT_EXTS:
                    if not SHOW_TEXT_NOTES:
                        continue
                    text, truncated = _read_text(full)
                    messages.append(
                        {
                            "rel": rel,
                            "name": file_name,
                            "group": group,
                            "kind": "text",
                            "text": text,
                            "truncated": truncated,
                        }
                    )
                elif ext in TABLE_EXTS:
                    headers, rows, truncated = _read_table(full)
                    messages.append(
                        {
                            "rel": rel,
                            "name": file_name,
                            "group": group,
                            "kind": "table",
                            "headers": headers,
                            "rows": rows,
                            "total": _count_data_rows(full),
                            "shown": len(rows),
                            "truncated": truncated,
                        }
                    )
                else:
                    downloads.append(
                        {
                            "rel": rel,
                            "name": file_name,
                            "group": group,
                            "ext": (ext.lstrip(".") or "file"),
                        }
                    )

        if not (images or messages or downloads):
            continue

        title, blurb, icon = MODULE_META.get(
            name, (name.replace("_", " "), "", "📦")
        )
        modules.append(
            {
                "name": name,
                "title": title,
                "blurb": blurb,
                "icon": icon,
                "images": images,
                "messages": messages,
                "downloads": downloads,
                "n_images": len(images),
                "n_messages": len(messages),
                "n_downloads": len(downloads),
            }
        )

    return modules


@app.route("/reports")
def reports():
    """Show every generated chart and every generated message, per module."""
    modules = _collect_modules()

    total_images = sum(m["n_images"] for m in modules)
    total_messages = sum(m["n_messages"] for m in modules)
    total_downloads = sum(m["n_downloads"] for m in modules)

    return render_template(
        "reports.html",
        modules=modules,
        total_images=total_images,
        total_messages=total_messages,
        total_downloads=total_downloads,
        total_modules=len(modules),
    )


# ================================================================
# WARM-UP
# Build the cached previews now so the first page view is instant.
# ================================================================

def _warm_caches():
    try:
        dataset_preview()
        for stage in PREPROCESS_STAGES:
            stage_preview(stage["file"])
        stage_preview(FINAL_STAGE["file"])
    except Exception:
        pass  # never block startup on a cache miss


_warm_caches()


if __name__ == "__main__":
    app.run(debug=True)
