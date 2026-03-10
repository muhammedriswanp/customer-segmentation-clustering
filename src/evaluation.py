from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.decomposition import PCA

def get_davies_bouldin(X, labels, model_name="Model"):
    db_score = davies_bouldin_score(X, labels)
    print(f"  {model_name} DB Index: {db_score:.4f}  (lower is better)")
    return db_score

    