import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def apply_pca(X, variance_threshold=0.9):
    """
    Apply PCA to reduce dimensionality.
    """
    pca = PCA(n_components=variance_threshold)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def compute_elbow(X, k_range=range(1, 11)):
    """
    Compute inertia values for elbow method.
    """
    wcss = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        wcss.append(model.inertia_)

    return wcss


def compute_silhouette(X, k_range=range(3, 7)):
    """
    Compute silhouette scores for different k.
    """
    scores = {}

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = score

    return scores


def run_kmeans(X, n_clusters=3):
    """
    Train final KMeans model.
    """
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return model, labels