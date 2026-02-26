import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "volatility_mean", "volatility_max", "volatility_5d", "downside_deviation",
    "var_95", "max_drawdown", "mean_return", "std_return",
    "return_skew", "return_kurtosis", "return_consistency", "sharpe_ratio",
    "rsi_mean", "bb_width_mean", "macd_volatility",
    "avg_volume", "volume_volatility", "amihud_illiquidity", "trading_frequency",
    "momentum_20d", "momentum_60d", "trend_strength", "current_price",
]

CLUSTERING_FEATURES = [
    # Core volatility (most important)
    "std_return", "volatility_mean", "volatility_max",
    # Downside risk (critical)
    "max_drawdown", "downside_deviation", "var_95",
    # Risk-adjusted return
    "sharpe_ratio",
    # Distribution shape
    "return_skew", "return_kurtosis",
    # Technical indicators
    "rsi_mean", "bb_width_mean",
    # Momentum
    "momentum_20d", "momentum_60d",
    # Liquidity risk
    "trading_frequency", "amihud_illiquidity",
]

# Automatic cluster label assignment

# Features used to score each cluster; higher value = higher risk/return
# Risk tier labels — ordered from lowest to highest risk
_RISK_TIERS = [
    "Low Risk",           # very low volatility, small drawdown, high liquidity
    "Low-Moderate Risk",  # below-average vol, modest drawdown
    "Moderate Risk",      # average vol, balanced return profile
    "Moderate-High Risk", # above-average vol, negative momentum possible
    "High Risk",          # high vol, large drawdown, poor sharpe
    "Very High Risk",     # extreme vol/drawdown, likely speculative
]

# Composite risk score weights
_RISK_WEIGHTS = {
    "std_return":         0.20,
    "volatility_mean":    0.20,
    "max_drawdown":       0.20,   # absolute value used
    "downside_deviation": 0.15,
    "var_95":             0.15,   # absolute value used
    "amihud_illiquidity": 0.10,
}


def assign_cluster_labels(df_clustered, method_col="KMeans_Cluster", features=None):

    if features is None:
        features = CLUSTERING_FEATURES

    available = [f for f in features if f in df_clustered.columns]
    profile = df_clustered.groupby(method_col)[available].mean()

    # Use absolute values for drawdown / VaR (they are negative)
    profile_abs = profile.copy()
    for col in ["max_drawdown", "var_95"]:
        if col in profile_abs.columns:
            profile_abs[col] = profile_abs[col].abs()

    # Normalise each feature to [0, 1] across clusters
    norm = (profile_abs - profile_abs.min()) / \
           (profile_abs.max() - profile_abs.min() + 1e-9)

    # Compute weighted composite risk score per cluster
    composite = pd.Series(0.0, index=profile.index)
    total_weight = 0.0
    for feat, w in _RISK_WEIGHTS.items():
        if feat in norm.columns:
            composite += norm[feat] * w
            total_weight += w
    if total_weight > 0:
        composite /= total_weight  # re-normalise to [0, 1]

    # Map composite score → risk tier using equal-width bands
    n_tiers = len(_RISK_TIERS)
    label_map = {}
    for cluster_id, score in composite.items():
        tier_idx = min(int(score * n_tiers), n_tiers - 1)
        label_map[cluster_id] = _RISK_TIERS[tier_idx]

    # Disambiguate duplicates by appending the cluster id suffix
    seen = {}
    for cid, lbl in label_map.items():
        seen[lbl] = seen.get(lbl, 0) + 1
    count = {}
    final_map = {}
    for cid, lbl in label_map.items():
        if seen[lbl] > 1:
            count[lbl] = count.get(lbl, 0) + 1
            final_map[cid] = "{} ({})".format(lbl, count[lbl])
        else:
            final_map[cid] = lbl

    label_col = method_col.replace("_Cluster", "_Risk_Label")
    df_clustered[label_col] = df_clustered[method_col].map(final_map)

    print("\nAuto-assigned risk labels [{}]:".format(method_col))
    print("  {:<6}  {:<28}  {:>12}  {}".format(
        "Cluster", "Risk Label", "Risk Score", "n"))
    print("  " + "-" * 60)
    for cid in sorted(final_map.keys()):
        lbl   = final_map[cid]
        score = composite[cid]
        n     = int((df_clustered[method_col] == cid).sum())
        print("  {:<6}  {:<28}  {:>12.4f}  {}".format(cid, lbl, score, n))

    return final_map, label_col


# Data Loading & Preprocessing

def load_data(filepath):
    df = pd.read_csv(filepath)
    print("Loaded {} stocks x {} columns".format(len(df), df.shape[1]))
    return df


def preprocess(df, features=None, log_cols=None):
    if features is None:
        features = CLUSTERING_FEATURES
    if log_cols is None:
        log_cols = ["amihud_illiquidity"]

    X = df[features].copy()

    for col in log_cols:
        if col in X.columns:
            X[col] = np.log1p(X[col].clip(lower=0))

    X = X.fillna(X.median())

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    meta = df[["Stock_code", "Sector", "Name"]].reset_index(drop=True)
    print("Preprocessed — shape: {}".format(X_scaled.shape))
    return X_scaled, meta, scaler


# Dimensionality Reduction

def run_pca(X_scaled, n_components=10):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.cumsum()
    print("PCA: {} components explain {:.1f}% variance".format(
        n_components, explained[-1] * 100))
    return X_pca, pca


def run_tsne(X_scaled, n_components=2, perplexity=10):
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=42,
        n_iter=1000,
    )
    X_tsne = tsne.fit_transform(X_scaled)
    print("t-SNE embedding shape: {}".format(X_tsne.shape))
    return X_tsne

# Clustering Algorithms

def kmeans_cluster(X, n_clusters=4, random_state=42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state,
                   n_init=100, max_iter=500)
    labels = model.fit_predict(X)
    print("KMeans — k={}, inertia={:.2f}".format(n_clusters, model.inertia_))
    return labels


def hierarchical_cluster(X, n_clusters=4, linkage_method="ward"):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X)
    print("Hierarchical — k={}, linkage={}".format(n_clusters, linkage_method))
    return labels


def dbscan_cluster(X, eps=1.5, min_samples=3):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print("DBSCAN — eps={}, clusters={}, noise points={}".format(
        eps, n_clusters, n_noise))
    return labels


# Elbow & Silhouette Search

def find_optimal_k(X, k_range=None):
    if k_range is None:
        k_range = range(2, 10)

    results = {
        "k": [],
        "inertia": [],
        "silhouette": [],
        "davies_bouldin": [],
        "calinski_harabasz": [],
    }

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=50, max_iter=500)
        labels = model.fit_predict(X)
        results["k"].append(k)
        results["inertia"].append(model.inertia_)
        results["silhouette"].append(silhouette_score(X, labels))
        results["davies_bouldin"].append(davies_bouldin_score(X, labels))
        results["calinski_harabasz"].append(calinski_harabasz_score(X, labels))

    best_k = results["k"][int(np.argmax(results["silhouette"]))]
    print("Optimal k by silhouette score: {}".format(best_k))
    return results, best_k


# Evaluation

def evaluate_clustering(X, labels, name=""):
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        print("Not enough clusters for evaluation.")
        return {}

    metrics = {
        "method": name,
        "n_clusters": len(set(labels[mask])),
        "silhouette": round(silhouette_score(X[mask], labels[mask]), 4),
        "davies_bouldin": round(davies_bouldin_score(X[mask], labels[mask]), 4),
        "calinski_harabasz": round(calinski_harabasz_score(X[mask], labels[mask]), 4),
    }
    print("  [{}] Silhouette={:.4f} | DB={:.4f} | CH={:.1f}".format(
        name,
        metrics["silhouette"],
        metrics["davies_bouldin"],
        metrics["calinski_harabasz"],
    ))
    return metrics

# Plotting Utilities
def plot_pca_variance(pca, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    cumvar = pca.explained_variance_ratio_.cumsum() * 100
    ax.bar(range(1, len(cumvar) + 1), pca.explained_variance_ratio_ * 100,
           alpha=0.6, color="steelblue", label="Individual")
    ax.plot(range(1, len(cumvar) + 1), cumvar, "o-", color="firebrick",
            label="Cumulative")
    ax.axhline(90, ls="--", color="grey", lw=1, label="90% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("PCA - Explained Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_elbow(results, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(results["k"], results["inertia"], "o-", color="steelblue")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Curve")
    ax.grid(True, alpha=0.3)


def plot_silhouette_k(results, best_k, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(results["k"], results["silhouette"], "o-", color="darkorange")
    ax.axvline(best_k, ls="--", color="grey", label="Best k={}".format(best_k))
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs k")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_clusters_2d(X_2d, labels, meta, title="Cluster Plot",
                     label_points=True, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))

    unique_labels = sorted(set(labels))
    # Use plt.get_cmap (compatible with matplotlib >= 3.7 & Python 3.8)
    try:
        palette = cm.get_cmap("tab10", len(unique_labels))
    except AttributeError:
        palette = plt.get_cmap("tab10", len(unique_labels))

    colors = {
        lbl: ("grey" if lbl == -1 else palette(i))
        for i, lbl in enumerate(unique_labels)
    }

    for lbl in unique_labels:
        mask = labels == lbl
        cluster_name = "Noise" if lbl == -1 else "Cluster {}".format(lbl)
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            color=colors[lbl], label=cluster_name,
            s=80, alpha=0.85, edgecolors="white", lw=0.5,
        )

    if label_points:
        for i, code in enumerate(meta["Stock_code"]):
            ax.annotate(
                code, (X_2d[i, 0], X_2d[i, 1]),
                fontsize=7, alpha=0.75, ha="center", va="bottom",
            )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.2)


def plot_dendrogram(X, meta, method="ward", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))
    Z = linkage(X, method=method)
    dendrogram(
        Z,
        labels=meta["Stock_code"].tolist(),
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=0.7 * max(Z[:, 2]),
        ax=ax,
    )
    ax.set_title("Dendrogram ({} linkage)".format(method),
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Stock")
    ax.set_ylabel("Distance")


def plot_cluster_profiles(df_clustered, features, method_col="KMeans_Cluster",
                          ax=None):
    profile = df_clustered.groupby(method_col)[features].mean()
    profile_norm = (profile - profile.min()) / \
                   (profile.max() - profile.min() + 1e-9)

    if ax is None:
        _, ax = plt.subplots(
            figsize=(14, max(4, len(features) // 3)))

    sns.heatmap(
        profile_norm.T,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Normalised Mean"},
    )
    ax.set_title("Cluster Feature Profiles - {}".format(method_col),
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Feature")


def plot_sector_distribution(df_clustered, method_col="KMeans_Cluster"):
    cross = pd.crosstab(df_clustered[method_col], df_clustered["Sector"])
    cross.plot(kind="bar", stacked=True, figsize=(10, 5),
               colormap="Set3", edgecolor="grey", linewidth=0.5)
    plt.title("Sector Distribution per Cluster - {}".format(method_col),
              fontweight="bold")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()


def compare_methods(metrics_list):
    df = pd.DataFrame(metrics_list)
    df = df.set_index("method")
    print("\n-- Method Comparison ------------------------------------------")
    print(df.to_string())
    return df