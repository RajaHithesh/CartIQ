# ================================================================
# CARTIQ - GROCERY MARKET BASKET INTELLIGENCE
# ================================================================
# MODULE 4 : K-MEANS CLUSTERING
#             (ELBOW METHOD + SILHOUETTE SCORE)
# ================================================================
# TASK      : Group shopping behaviour into clusters using
#             K-Means and K-Means++ initialisation.
# DATASET   : final_preprocess_M2.csv (all numeric, scaled).
# ================================================================
# TECHNIQUES:
#   1. Elbow Method (Inertia)
#   2. Silhouette Score
#   3. K-Means (random init) vs K-Means++ (smart init)
#   4. Best K selected by maximum Silhouette Score
# ================================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ================================================================
# 1. PATHS
# ================================================================

DATASET_PATH = "D:/ML/CartIQ/dataset/final_preprocess_M2.csv"

OUTPUT_FOLDER = (
    "D:/ML/CartIQ/outputs/K_Means_Elbow_Silhouetee_M4_Outputs"
)

ELBOW_FOLDER = os.path.join(OUTPUT_FOLDER, "elbow")
SILHOUETTE_FOLDER = os.path.join(OUTPUT_FOLDER, "silhouette")
CLUSTERED_FOLDER = os.path.join(OUTPUT_FOLDER, "clustered_data")
METRICS_FOLDER = os.path.join(OUTPUT_FOLDER, "metrics")
KMEANS_FOLDER = os.path.join(OUTPUT_FOLDER, "clustering_3d")
KMEANS_PP_FOLDER = os.path.join(OUTPUT_FOLDER, "clustering_3d")
PCA_FOLDER = os.path.join(OUTPUT_FOLDER, "clustering_pca")

# ================================================================
# 2. CREATE OUTPUT FOLDERS
# ================================================================

for folder in [
    ELBOW_FOLDER,
    SILHOUETTE_FOLDER,
    CLUSTERED_FOLDER,
    METRICS_FOLDER,
    KMEANS_FOLDER,
    PCA_FOLDER,
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
# 4. LOAD PREPROCESSED DATASET
# ================================================================

data = pd.read_csv(DATASET_PATH)

print("\nPreprocessed dataset loaded successfully.")
print("Rows    :", data.shape[0])
print("Columns :", data.shape[1])

# ================================================================
# 5. DISPLAY COLUMNS
# ================================================================

print("\nAvailable columns:")
print("-" * 75)

for i, column in enumerate(data.columns, 1):
    print(i, ".", column)

# ================================================================
# 6. COLUMN-NAME NORMALIZATION
# ================================================================


def normalize_column_name(column):
    return (
        str(column)
        .strip()
        .lower()
        .replace("_", "")
        .replace(" ", "")
        .replace("-", "")
    )


normalized_columns = {
    normalize_column_name(column): column
    for column in data.columns
}

# ================================================================
# 7. REQUIRED FEATURES
# ================================================================
#
# Four shopping-behaviour features are used for clustering:
#   - order_hour_of_day : when the order was placed
#   - days_since_prior_order : how long since the last order
#   - add_to_cart_order : position of the product in the basket
#   - basket_size : total number of products in the basket
# ================================================================

required_features = [
    "order_hour_of_day",
    "days_since_prior_order",
    "add_to_cart_order",
    "basket_size",
]

# ================================================================
# 8. FIND FEATURES
# ================================================================

feature_columns = []

for feature in required_features:
    normalized_feature = normalize_column_name(feature)

    if normalized_feature in normalized_columns:
        feature_columns.append(
            normalized_columns[normalized_feature]
        )
    else:
        raise ValueError(
            f"\nRequired column '{feature}' was not found."
        )

# ================================================================
# 9. DISPLAY SELECTED FEATURES
# ================================================================

print("\n" + "=" * 75)
print("FEATURES USED FOR CLUSTERING")
print("=" * 75)

for feature in feature_columns:
    print("-", feature)

# ================================================================
# 10. SELECT FOUR FEATURES
# ================================================================

X = data[feature_columns].copy()

# ================================================================
# 11. CONVERT TO NUMERIC
# ================================================================

for column in feature_columns:
    X[column] = pd.to_numeric(X[column], errors="coerce")

# ================================================================
# 12. HANDLE INFINITE VALUES
# ================================================================

X = X.replace([np.inf, -np.inf], np.nan)

# ================================================================
# 13. HANDLE MISSING VALUES
# ================================================================

print("\nMissing values before imputation:")
print(X.isnull().sum())

X = X.fillna(X.median())

print("\nMissing values after imputation:")
print(X.isnull().sum())

# ================================================================
# 14. STANDARDISATION
# ================================================================
#
# Even though this is a preprocessed dataset, K-Means is
# distance-based. Therefore, the four clustering variables
# are standardised before clustering.
# ================================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeature standardisation completed.")

# ================================================================
# 15. K RANGE
# ================================================================

K_VALUES = range(2, 11)

# ================================================================
# 16. K-MEANS - ELBOW AND SILHOUETTE
# ================================================================

print("\n" + "=" * 75)
print("K-MEANS ANALYSIS")
print("=" * 75)

kmeans_inertia = []
kmeans_silhouette = []

for k in K_VALUES:
    model = KMeans(
        n_clusters=k,
        init="random",
        n_init=10,
        random_state=42,
    )

    labels = model.fit_predict(X_scaled)

    inertia = model.inertia_

    silhouette = silhouette_score(X_scaled, labels)

    kmeans_inertia.append(inertia)
    kmeans_silhouette.append(silhouette)

    print(
        f"K = {k:2d} | "
        f"Inertia = {inertia:.4f} | "
        f"Silhouette = {silhouette:.4f}"
    )

# ================================================================
# 17. K-MEANS++ - ELBOW AND SILHOUETTE
# ================================================================

print("\n" + "=" * 75)
print("K-MEANS++ ANALYSIS")
print("=" * 75)

kmeans_pp_inertia = []
kmeans_pp_silhouette = []

for k in K_VALUES:
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42,
    )

    labels = model.fit_predict(X_scaled)

    inertia = model.inertia_

    silhouette = silhouette_score(X_scaled, labels)

    kmeans_pp_inertia.append(inertia)
    kmeans_pp_silhouette.append(silhouette)

    print(
        f"K = {k:2d} | "
        f"Inertia = {inertia:.4f} | "
        f"Silhouette = {silhouette:.4f}"
    )

# ================================================================
# 18. SELECT BEST K
# ================================================================
#
# Maximum Silhouette Score is used to select K automatically.
# The Elbow graph should also be inspected.
# ================================================================

best_k_kmeans = list(K_VALUES)[np.argmax(kmeans_silhouette)]

best_k_kmeans_pp = list(K_VALUES)[np.argmax(kmeans_pp_silhouette)]

print("\n" + "=" * 75)
print("SELECTED NUMBER OF CLUSTERS")
print("=" * 75)

print("K-Means best K       :", best_k_kmeans)
print("K-Means++ best K     :", best_k_kmeans_pp)

# ================================================================
# 19. ELBOW GRAPH - K-MEANS
# ================================================================

plt.figure(figsize=(10, 6))

plt.plot(
    list(K_VALUES),
    kmeans_inertia,
    marker="o",
    color="#2E86AB",
    linewidth=2,
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method - K-Means")
plt.xticks(list(K_VALUES))
plt.grid(True, alpha=0.3)
plt.tight_layout()

elbow_kmeans_path = os.path.join(
    ELBOW_FOLDER,
    "elbow_kmeans.png",
)

plt.savefig(elbow_kmeans_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nElbow graph (K-Means) saved at:")
print(elbow_kmeans_path)

# ================================================================
# 20. ELBOW GRAPH - K-MEANS++
# ================================================================

plt.figure(figsize=(10, 6))

plt.plot(
    list(K_VALUES),
    kmeans_pp_inertia,
    marker="o",
    color="#E07A5F",
    linewidth=2,
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method - K-Means++")
plt.xticks(list(K_VALUES))
plt.grid(True, alpha=0.3)
plt.tight_layout()

elbow_kmeans_pp_path = os.path.join(
    ELBOW_FOLDER,
    "elbow_kmeans_plus_plus.png",
)

plt.savefig(elbow_kmeans_pp_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nElbow graph (K-Means++) saved at:")
print(elbow_kmeans_pp_path)

# ================================================================
# 21. SILHOUETTE GRAPH - K-MEANS
# ================================================================

plt.figure(figsize=(10, 6))

plt.plot(
    list(K_VALUES),
    kmeans_silhouette,
    marker="s",
    color="#81B29A",
    linewidth=2,
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score - K-Means")
plt.xticks(list(K_VALUES))
plt.grid(True, alpha=0.3)
plt.tight_layout()

silhouette_kmeans_path = os.path.join(
    SILHOUETTE_FOLDER,
    "silhouette_kmeans.png",
)

plt.savefig(silhouette_kmeans_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nSilhouette graph (K-Means) saved at:")
print(silhouette_kmeans_path)

# ================================================================
# 22. SILHOUETTE GRAPH - K-MEANS++
# ================================================================

plt.figure(figsize=(10, 6))

plt.plot(
    list(K_VALUES),
    kmeans_pp_silhouette,
    marker="s",
    color="#E9C46A",
    linewidth=2,
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score - K-Means++")
plt.xticks(list(K_VALUES))
plt.grid(True, alpha=0.3)
plt.tight_layout()

silhouette_kmeans_pp_path = os.path.join(
    SILHOUETTE_FOLDER,
    "silhouette_kmeans_plus_plus.png",
)

plt.savefig(silhouette_kmeans_pp_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nSilhouette graph (K-Means++) saved at:")
print(silhouette_kmeans_pp_path)

# ================================================================
# 23. FINAL K-MEANS MODEL
# ================================================================

kmeans_model = KMeans(
    n_clusters=best_k_kmeans,
    init="random",
    n_init=10,
    random_state=42,
)

kmeans_labels = kmeans_model.fit_predict(X_scaled)

# ================================================================
# 24. FINAL K-MEANS++ MODEL
# ================================================================

kmeans_pp_model = KMeans(
    n_clusters=best_k_kmeans_pp,
    init="k-means++",
    n_init=10,
    random_state=42,
)

kmeans_pp_labels = kmeans_pp_model.fit_predict(X_scaled)

# ================================================================
# 25. SAVE CLUSTERED DATA - K-MEANS
# ================================================================

clustered_kmeans = X.copy()

clustered_kmeans["KMeans_Cluster"] = kmeans_labels

clustered_kmeans_path = os.path.join(
    CLUSTERED_FOLDER,
    "kmeans_clustered_data.csv",
)

clustered_kmeans.to_csv(clustered_kmeans_path, index=False)

print("\nClustered data (K-Means) saved at:")
print(clustered_kmeans_path)

# ================================================================
# 26. SAVE CLUSTERED DATA - K-MEANS++
# ================================================================

clustered_kmeans_pp = X.copy()

clustered_kmeans_pp["KMeansPlusPlus_Cluster"] = kmeans_pp_labels

clustered_kmeans_pp_path = os.path.join(
    CLUSTERED_FOLDER,
    "kmeans_plus_plus_clustered_data.csv",
)

clustered_kmeans_pp.to_csv(clustered_kmeans_pp_path, index=False)

print("\nClustered data (K-Means++) saved at:")
print(clustered_kmeans_pp_path)

# ================================================================
# 27. SAVE METRICS - K-MEANS
# ================================================================

km_silhouette = silhouette_score(X_scaled, kmeans_labels)

kmeans_metrics_df = pd.DataFrame(
    {
        "Metric": [
            "Algorithm",
            "Number of Clusters",
            "Inertia",
            "Silhouette Score",
        ],
        "Value": [
            "K-Means",
            best_k_kmeans,
            kmeans_model.inertia_,
            km_silhouette,
        ],
    }
)

kmeans_metrics_path = os.path.join(
    METRICS_FOLDER,
    "kmeans_metrics.csv",
)

kmeans_metrics_df.to_csv(kmeans_metrics_path, index=False)

print("\nMetrics (K-Means) saved at:")
print(kmeans_metrics_path)

# ================================================================
# 28. SAVE METRICS - K-MEANS++
# ================================================================

pp_silhouette = silhouette_score(X_scaled, kmeans_pp_labels)

kmeans_pp_metrics_df = pd.DataFrame(
    {
        "Metric": [
            "Algorithm",
            "Number of Clusters",
            "Inertia",
            "Silhouette Score",
        ],
        "Value": [
            "K-Means++",
            best_k_kmeans_pp,
            kmeans_pp_model.inertia_,
            pp_silhouette,
        ],
    }
)

kmeans_pp_metrics_path = os.path.join(
    METRICS_FOLDER,
    "kmeans_plus_plus_metrics.csv",
)

kmeans_pp_metrics_df.to_csv(kmeans_pp_metrics_path, index=False)

print("\nMetrics (K-Means++) saved at:")
print(kmeans_pp_metrics_path)

# ================================================================
# 29. 3-D K-MEANS CLUSTERING GRAPH
# ================================================================
#
# Four features cannot be directly plotted in a normal 3-D
# graph. The first three features are shown here.
#
# All FOUR features are used for actual clustering.
# ================================================================

fig = plt.figure(figsize=(11, 8))

ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    X[feature_columns[0]],
    X[feature_columns[1]],
    X[feature_columns[2]],
    c=kmeans_labels,
    s=20,
    alpha=0.7,
)

ax.set_xlabel(feature_columns[0])
ax.set_ylabel(feature_columns[1])
ax.set_zlabel(feature_columns[2])
ax.set_title("K-Means Clustering")

fig.colorbar(scatter, ax=ax, label="Cluster")

plt.tight_layout()

plt.savefig(
    os.path.join(
        KMEANS_FOLDER,
        "kmeans_3D_clustering.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\n3-D clustering graph (K-Means) saved.")

# ================================================================
# 30. 3-D K-MEANS++ CLUSTERING GRAPH
# ================================================================

fig = plt.figure(figsize=(11, 8))

ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    X[feature_columns[0]],
    X[feature_columns[1]],
    X[feature_columns[2]],
    c=kmeans_pp_labels,
    s=20,
    alpha=0.7,
)

ax.set_xlabel(feature_columns[0])
ax.set_ylabel(feature_columns[1])
ax.set_zlabel(feature_columns[2])
ax.set_title("K-Means++ Clustering")

fig.colorbar(scatter, ax=ax, label="Cluster")

plt.tight_layout()

plt.savefig(
    os.path.join(
        KMEANS_PP_FOLDER,
        "kmeans_plus_plus_3D_clustering.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\n3-D clustering graph (K-Means++) saved.")

# ================================================================
# 31. PCA 2-D VISUALISATION
# ================================================================
#
# PCA is ONLY used here to visualize the four-dimensional
# clustering results in two dimensions.
#
# PCA is NOT used to perform the clustering.
# ================================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# ================================================================
# 32. PCA GRAPH - K-MEANS
# ================================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    s=20,
    alpha=0.7,
    cmap="viridis",
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering (PCA 2-D View)")
plt.colorbar(label="Cluster")
plt.tight_layout()

pca_kmeans_path = os.path.join(
    PCA_FOLDER,
    "kmeans_PCA_clustering.png",
)

plt.savefig(pca_kmeans_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nPCA graph (K-Means) saved at:")
print(pca_kmeans_path)

# ================================================================
# 33. PCA GRAPH - K-MEANS++
# ================================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_pp_labels,
    s=20,
    alpha=0.7,
    cmap="viridis",
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means++ Clustering (PCA 2-D View)")
plt.colorbar(label="Cluster")
plt.tight_layout()

pca_kmeans_pp_path = os.path.join(
    PCA_FOLDER,
    "kmeans_plus_plus_PCA_clustering.png",
)

plt.savefig(pca_kmeans_pp_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nPCA graph (K-Means++) saved at:")
print(pca_kmeans_pp_path)

# ================================================================
# 34. FIND TARGET COLUMN FOR MATCHING
# ================================================================
#
# The clusters are compared against the actual reordered
# behaviour to measure how well the clusters match reality.
# ================================================================

target_column = None

for candidate in ["reordered", "Reordered", "Reorder", "REORDERED"]:
    normalized_candidate = normalize_column_name(candidate)

    if normalized_candidate in normalized_columns:
        target_column = normalized_columns[normalized_candidate]
        break

if target_column is None:
    raise ValueError(
        "\nTarget column 'reordered' was not found."
    )

print("\nTarget column for matching :", target_column)

# ================================================================
# 35. RECOVER BINARY TARGET
# ================================================================
#
# In final_preprocess_M2.csv the target was standard-scaled.
# Values above 0 correspond to reordered = 1, values below 0
# correspond to reordered = 0.
# ================================================================

target_binary = (data[target_column] > 0).astype(int)

print("\nRecovered binary target distribution:")
print(target_binary.value_counts().sort_index())

# ================================================================
# 36. CLUSTER MATCHING ACCURACY
# ================================================================
#
# Each cluster is assigned the majority target value inside it.
# Accuracy = fraction of rows where the cluster majority
# matches the actual reordered value.
# ================================================================


def cluster_matching_accuracy(labels, target):
    cluster_df = pd.DataFrame(
        {
            "Cluster": labels,
            "Target": target.values,
        }
    )

    majority_map = (
        cluster_df.groupby("Cluster")["Target"]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )

    predicted = labels.copy()

    for cluster, majority_value in majority_map.items():
        predicted[labels == cluster] = majority_value

    accuracy = float(np.mean(predicted == target.values))

    return accuracy, majority_map


km_accuracy, km_majority = cluster_matching_accuracy(
    kmeans_labels,
    target_binary,
)

pp_accuracy, pp_majority = cluster_matching_accuracy(
    kmeans_pp_labels,
    target_binary,
)

print("\n" + "=" * 75)
print("CLUSTER MATCHING ACCURACY")
print("=" * 75)

print(f"\nK-Means   matching accuracy : {km_accuracy * 100:.2f}%")
print(f"K-Means++ matching accuracy : {pp_accuracy * 100:.2f}%")

# ================================================================
# 37. SAVE CLUSTER MATCHING ACCURACY
# ================================================================

matching_accuracy_df = pd.DataFrame(
    {
        "Algorithm": ["K-Means", "K-Means++"],
        "Number of Clusters": [best_k_kmeans, best_k_kmeans_pp],
        "Matching Accuracy": [km_accuracy, pp_accuracy],
    }
)

matching_accuracy_path = os.path.join(
    METRICS_FOLDER,
    "reordered_cluster_matching_accuracy.csv",
)

matching_accuracy_df.to_csv(matching_accuracy_path, index=False)

print("\nCluster matching accuracy saved at:")
print(matching_accuracy_path)

# ================================================================
# 38. FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 80)
print("        K-MEANS CLUSTERING COMPLETED SUCCESSFULLY")
print("=" * 80)

print("\nDataset used:")
print(DATASET_PATH)

print("\nAll outputs stored in:")
print(OUTPUT_FOLDER)

print("\n")
print("FINAL RESULTS")
print("-" * 50)

print(f"K-Means best K       : {best_k_kmeans}")
print(f"K-Means inertia      : {kmeans_model.inertia_:.4f}")
print(f"K-Means silhouette   : {km_silhouette:.4f}")
print(f"K-Means matching acc : {km_accuracy * 100:.2f}%")

print(f"\nK-Means++ best K     : {best_k_kmeans_pp}")
print(f"K-Means++ inertia    : {kmeans_pp_model.inertia_:.4f}")
print(f"K-Means++ silhouette : {pp_silhouette:.4f}")
print(f"K-Means++ match acc  : {pp_accuracy * 100:.2f}%")

print("\n")
print("OUTPUT FOLDERS")
print("-" * 50)

print("1. elbow")
print("2. silhouette")
print("3. clustered_data")
print("4. metrics")
print("5. clustering_3d")
print("6. clustering_pca")

print("\n")
print("=" * 80)
print("                 PROGRAM FINISHED")
print("=" * 80)