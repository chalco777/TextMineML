#!/usr/bin/env python3
"""
04_clusteing.py
-------------
Aplica K-Means, DBSCAN y HDBSCAN sobre todas las matrices de vectores
creadas en 03_vectorize.py y guarda etiquetas + métricas comparativas.

Uso:
    python scripts/04_cluster.py
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import scipy.sparse as sp

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

VEC_DIR = Path("../results/vectorizers")
OUT_DIR = Path("../results/clustering")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def evaluate(X, labels):
    # Etiquetas -1 = ruido (DBSCAN/HDBSCAN)
    core = labels[labels != -1]
    if len(set(core)) < 2:          # silhouette necesita ≥2 clústeres válidos
        return {"silhouette": None, "db_score": None}
    return {
        "silhouette": float(silhouette_score(X, labels, metric="cosine")),
        "db_score":   float(davies_bouldin_score(X, labels))
    }

def run_kmeans(X, k):
    km = KMeans(n_clusters=k, random_state=0)
    labels = km.fit_predict(X)
    metrics = {"inertia": float(km.inertia_)} | evaluate(X, labels)
    return labels, metrics

def run_dbscan(X, eps, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = db.fit_predict(X)
    metrics = evaluate(X, labels)
    metrics |= {"n_noise": int((labels == -1).sum())}
    return labels, metrics

def run_hdbscan(X, min_cluster_size=5):
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        prediction_data=True
    )
    labels = hdb.fit_predict(X)
    metrics = evaluate(X, labels)
    metrics |= {"n_noise": int((labels == -1).sum())}
    return labels, metrics

def main():
    summary = []
    for vec_path in VEC_DIR.rglob("X_*.pkl"):
        method = vec_path.parent.name      # tfidf, w2v, sbert
        print(f"\n=== {method.upper()} ===")
        X = joblib.load(vec_path)
        #TF-IDF se guarda como sparse; W2V/SBERT ya son ndarray
        
        if sp.issparse(X):
          X_dense = X.toarray()
        else:
          X_dense = X

        # ----- K-Means k=4..10 (ejemplo) -----
        for k in range(4, 11):
            labels, metrics = run_kmeans(X_dense, k)
            out_csv = OUT_DIR / f"labels_{method}_kmeans{k}.csv"
            pd.DataFrame({"label": labels}).to_csv(out_csv, index=False)
            summary.append({"method": method, "algo": f"kmeans{k}", **metrics})

        # ----- DBSCAN rejilla eps -----
        for eps in (0.3, 0.4, 0.5, 0.6):
            labels, metrics = run_dbscan(X_dense, eps)
            out_csv = OUT_DIR / f"labels_{method}_dbscan{eps}.csv"
            pd.DataFrame({"label": labels}).to_csv(out_csv, index=False)
            summary.append({"method": method, "algo": f"dbscan{eps}", **metrics})

        # ----- HDBSCAN -----
        if HDBSCAN_AVAILABLE:
            labels, metrics = run_hdbscan(X_dense)
            out_csv = OUT_DIR / f"labels_{method}_hdbscan.csv"
            pd.DataFrame({"label": labels}).to_csv(out_csv, index=False)
            summary.append({"method": method, "algo": "hdbscan", **metrics})

    # ---- Guardar tabla de métricas ----
    pd.DataFrame(summary).to_csv(OUT_DIR / "clustering_metrics.csv", index=False)
    with open(OUT_DIR / "clustering_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
