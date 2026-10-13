# ============================================================
# CARTIQ - GROCERY MARKET BASKET INTELLIGENCE
# DBSCAN CLUSTERING
# CartIQ RAW Dataset (instacart_50k_Raw.csv)
# Features:
#   1. order_hour_of_day
#   2. days_since_prior_order
#   3. basket_size
#
# IMPORTANT:
# 1. This script uses the RAW (uncleaned) dataset.
# 2. Original raw dataset is NOT modified.
# 3. All processing is performed on a copy.
# 4. All outputs are stored in ONE folder.
#
# NOTE:
# Unlike the class reference (where both DBSCAN scripts
# used the preprocessed dataset), this "Raw" variant
# genuinely reads the raw CSV and handles missing values
# with dropna before StandardScaler.
# ============================================================


import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


warnings.filterwarnings("ignore")


# ============================================================
# 1. INPUT AND OUTPUT PATHS
# ============================================================

INPUT_FILE = r"D:/ML/CartIQ/dataset/instacart_50k_Raw.csv"

OUTPUT_FOLDER = (
    r"D:/ML/CartIQ/outputs/DBSCAN_Raw_dataset_M4_outputs"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 75)
print("DBSCAN CLUSTERING - CARTIQ RAW DATASET")
print("=" * 75)


if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nDataset not found:\n{INPUT_FILE}"
    )


# READ ONLY - original dataset is not modified
data = pd.read_csv(INPUT_FILE)

print("\nOriginal raw dataset shape:")
print(data.shape)

print("\nAvailable columns:")
print(list(data.columns))


# ============================================================
# 3. SELECT REQUIRED ATTRIBUTES
# ============================================================

FEATURES = [
    "order_hour_of_day",
    "days_since_prior_order",
    "basket_size"
]


missing_columns = [
    column for column in FEATURES
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        "\nThe following required columns are missing:\n"
        f"{missing_columns}"
    )


# ============================================================
# 4. CREATE A SEPARATE COPY
# ============================================================

# IMPORTANT:
# This creates a completely separate DataFrame.
# The original dataset remains untouched.

dbscan_data = data[FEATURES].copy()


# ============================================================
# 5. CONVERT FEATURES TO NUMERIC
# ============================================================

for column in FEATURES:
    dbscan_data[column] = pd.to_numeric(
        dbscan_data[column],
        errors="coerce"
    )


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")
print(dbscan_data.isnull().sum())


# ============================================================
# 7. REMOVE INVALID ROWS FROM THE COPY ONLY
# ============================================================

before_cleaning = len(dbscan_data)

dbscan_data = dbscan_data.dropna().reset_index(drop=True)

after_cleaning = len(dbscan_data)

removed_rows = before_cleaning - after_cleaning

print("\nRows before cleaning:", before_cleaning)
print("Rows after cleaning :", after_cleaning)
print("Rows removed        :", removed_rows)


# ============================================================
# 8. SAVE SELECTED FEATURES
# ============================================================

selected_features_file = os.path.join(
    OUTPUT_FOLDER,
    "DBSCAN_selected_features.csv"
)

dbscan_data.to_csv(selected_features_file, index=False)


# ============================================================
# 9. STANDARDIZATION
# ============================================================

# DBSCAN is distance-based.
# Standardization prevents one feature from dominating
# the distance calculation.

scaler = StandardScaler()

X_scaled = scaler.fit_transform(dbscan_data[FEATURES])


# ============================================================
# 10. SAVE STANDARDIZED DATA
# ============================================================

standardized_data = pd.DataFrame(
    X_scaled,
    columns=FEATURES
)

standardized_file = os.path.join(
    OUTPUT_FOLDER,
    "DBSCAN_standardized_features.csv"
)

standardized_data.to_csv(standardized_file, index=False)


# ============================================================
# 11. DBSCAN PARAMETERS
# ============================================================

EPS = 0.80
MIN_SAMPLES = 10


print("\nDBSCAN Parameters")
print("-" * 40)
print("eps        :", EPS)
print("min_samples:", MIN_SAMPLES)


# ============================================================
# 12. APPLY DBSCAN
# ============================================================

dbscan_model = DBSCAN(
    eps=EPS,
    min_samples=MIN_SAMPLES
)

cluster_labels = dbscan_model.fit_predict(X_scaled)


# ============================================================
# 13. CREATE CLUSTERED DATASET
# ============================================================

clustered_data = dbscan_data.copy()

clustered_data["DBSCAN_Cluster"] = cluster_labels


# ============================================================
# 14. CLUSTER INFORMATION
# ============================================================

unique_labels = sorted(np.unique(cluster_labels))

actual_clusters = [
    label for label in unique_labels
    if label != -1
]

number_of_clusters = len(actual_clusters)

noise_count = int(np.sum(cluster_labels == -1))

clustered_count = len(cluster_labels) - noise_count

total_records = len(cluster_labels)

if total_records > 0:
    noise_percentage = (
        noise_count / total_records
    ) * 100

    clustered_percentage = (
        clustered_count / total_records
    ) * 100
else:
    noise_percentage = 0
    clustered_percentage = 0


print("\n" + "=" * 75)
print("DBSCAN RESULTS")
print("=" * 75)

print("\nTotal Records       :", total_records)
print("Number of Clusters  :", number_of_clusters)
print("Noise Points        :", noise_count)
print("Clustered Points    :", clustered_count)
print(
    "Noise Percentage    :",
    round(noise_percentage, 2), "%"
)
print(
    "Clustered Percentage:",
    round(clustered_percentage, 2), "%"
)


# ============================================================
# 15. SAVE CLUSTERED DATA
# ============================================================

clustered_file = os.path.join(
    OUTPUT_FOLDER,
    "DBSCAN_clustered_data.csv"
)

clustered_data.to_csv(clustered_file, index=False)


# ============================================================
# 16. CLUSTER DISTRIBUTION
# ============================================================

cluster_counts = (
    clustered_data["DBSCAN_Cluster"]
    .value_counts()
    .sort_index()
)

print("\nCluster distribution:")
print(cluster_counts)


cluster_distribution = pd.DataFrame({
    "Cluster": cluster_counts.index,
    "Number_of_Records": cluster_counts.values
})

cluster_distribution_file = os.path.join(
    OUTPUT_FOLDER,
    "DBSCAN_cluster_distribution.csv"
)

cluster_distribution.to_csv(
    cluster_distribution_file,
    index=False
)


# ============================================================
# 17. CLUSTER SUMMARY
# ============================================================

summary_records = []

for cluster_id in unique_labels:

    cluster_subset = clustered_data[
        clustered_data["DBSCAN_Cluster"] == cluster_id
    ]

    if cluster_id == -1:
        cluster_name = "Noise"
    else:
        cluster_name = "Cluster_" + str(cluster_id)

    summary_records.append({
        "Cluster": cluster_name,
        "Cluster_ID": cluster_id,
        "Number_of_Records": len(cluster_subset),
        "Average_Order_Hour_Of_Day":
            round(cluster_subset["order_hour_of_day"].mean(), 3),
        "Average_Days_Since_Prior_Order":
            round(cluster_subset["days_since_prior_order"].mean(), 3),
        "Average_Basket_Size":
            round(cluster_subset["basket_size"].mean(), 3)
    })


cluster_summary = pd.DataFrame(summary_records)

cluster_summary_file = os.path.join(
    OUTPUT_FOLDER,
    "DBSCAN_cluster_summary.csv"
)

cluster_summary.to_csv(cluster_summary_file, index=False)


# ============================================================
# 18. PERFORMANCE RESULTS
# ============================================================

performance_results = pd.DataFrame({
    "Metric": [
        "Algorithm",
        "Input Dataset",
        "Original Dataset Rows",
        "Rows Used",
        "Rows Removed",
        "Number of Features",
        "Features Used",
        "Scaling Method",
        "DBSCAN eps",
        "DBSCAN min_samples",
        "Number of Clusters",
        "Noise Points",
        "Clustered Points",
        "Noise Percentage",
        "Clustered Percentage"
    ],
    "Value": [
        "DBSCAN",
        os.path.basename(INPUT_FILE),
        before_cleaning,
        after_cleaning,
        removed_rows,
        len(FEATURES),
        ", ".join(FEATURES),
        "StandardScaler",
        EPS,
        MIN_SAMPLES,
        number_of_clusters,
        noise_count,
        clustered_count,
        round(noise_percentage, 2),
        round(clustered_percentage, 2)
    ]
})

performance_file = os.path.join(
    OUTPUT_FOLDER,
    "DBSCAN_performance_results.csv"
)

performance_results.to_csv(performance_file, index=False)


# ============================================================
# 19. PCA FOR 2-D VISUALIZATION
# ============================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

pca_coordinates = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "DBSCAN_Cluster": cluster_labels
})

pca_file = os.path.join(
    OUTPUT_FOLDER,
    "DBSCAN_PCA_coordinates.csv"
)

pca_coordinates.to_csv(pca_file, index=False)


# ============================================================
# 20. GRAPH 1 - ORDER HOUR VS DAYS SINCE PRIOR ORDER
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    dbscan_data["order_hour_of_day"],
    dbscan_data["days_since_prior_order"],
    c=cluster_labels,
    s=25,
    alpha=0.7
)

plt.xlabel("Order Hour Of Day")
plt.ylabel("Days Since Prior Order")

plt.title(
    "DBSCAN Clustering (Raw): Order Hour vs Days Since Prior Order"
)

plt.grid(True, alpha=0.3)
plt.tight_layout()

graph1 = os.path.join(
    OUTPUT_FOLDER,
    "01_DBSCAN_OrderHour_vs_DaysSincePrior.png"
)

plt.savefig(graph1, dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 21. GRAPH 2 - ORDER HOUR VS BASKET SIZE
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    dbscan_data["order_hour_of_day"],
    dbscan_data["basket_size"],
    c=cluster_labels,
    s=25,
    alpha=0.7
)

plt.xlabel("Order Hour Of Day")
plt.ylabel("Basket Size")

plt.title(
    "DBSCAN Clustering (Raw): Order Hour vs Basket Size"
)

plt.grid(True, alpha=0.3)
plt.tight_layout()

graph2 = os.path.join(
    OUTPUT_FOLDER,
    "02_DBSCAN_OrderHour_vs_BasketSize.png"
)

plt.savefig(graph2, dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 22. GRAPH 3 - DAYS SINCE PRIOR ORDER VS BASKET SIZE
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    dbscan_data["days_since_prior_order"],
    dbscan_data["basket_size"],
    c=cluster_labels,
    s=25,
    alpha=0.7
)

plt.xlabel("Days Since Prior Order")
plt.ylabel("Basket Size")

plt.title(
    "DBSCAN Clustering (Raw): Days Since Prior Order vs Basket Size"
)

plt.grid(True, alpha=0.3)
plt.tight_layout()

graph3 = os.path.join(
    OUTPUT_FOLDER,
    "03_DBSCAN_DaysSincePrior_vs_BasketSize.png"
)

plt.savefig(graph3, dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 23. GRAPH 4 - PCA CLUSTER VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
    s=25,
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("DBSCAN Cluster Visualization using PCA (Raw)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

graph4 = os.path.join(
    OUTPUT_FOLDER,
    "04_DBSCAN_PCA_Clusters.png"
)

plt.savefig(graph4, dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 24. GRAPH 5 - CLUSTER DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    cluster_distribution["Cluster"].astype(str),
    cluster_distribution["Number_of_Records"]
)

plt.xlabel("DBSCAN Cluster")
plt.ylabel("Number of Records")
plt.title("DBSCAN Cluster Distribution (Raw Dataset)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

graph5 = os.path.join(
    OUTPUT_FOLDER,
    "05_DBSCAN_Cluster_Distribution.png"
)

plt.savefig(graph5, dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 25. SAVE EXECUTION INFORMATION
# ============================================================

execution_information = pd.DataFrame({
    "Item": [
        "Input Dataset",
        "Output Folder",
        "Original Dataset Modified",
        "Features Used",
        "Preprocessing",
        "Scaling",
        "Algorithm",
        "eps",
        "min_samples"
    ],
    "Details": [
        INPUT_FILE,
        OUTPUT_FOLDER,
        "NO",
        ", ".join(FEATURES),
        "Raw dataset - dropna on selected features",
        "StandardScaler",
        "DBSCAN",
        EPS,
        MIN_SAMPLES
    ]
})

execution_file = os.path.join(
    OUTPUT_FOLDER,
    "execution_information.csv"
)

execution_information.to_csv(execution_file, index=False)


# ============================================================
# 26. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("ALL OUTPUTS GENERATED SUCCESSFULLY")
print("=" * 75)

print("\nOutput folder:")
print(OUTPUT_FOLDER)

print("\nFiles generated:")

for filename in sorted(os.listdir(OUTPUT_FOLDER)):
    print("   ", filename)

print("\n" + "=" * 75)
print("DBSCAN RAW-DATASET ANALYSIS COMPLETED")
print("Original raw dataset was NOT modified.")
print("=" * 75)
