#!/usr/bin/env python3
"""
clustering.py
-------------
Lee todos los archivos X_*.pkl que encuentre en --vecdir y aplica:

  • K-Means  (k = 4…10)
  • DBSCAN   (eps = 0.3, 0.4, 0.5, 0.6)
  • HDBSCAN  (si está instalado)

Guarda:
  ├─ labels_*  (.csv)  → etiquetas por documento
  └─ clustering_metrics.csv / .json → métrica comparativa por ejecución

Ejemplo de uso (lo llama tu wrapper):
    python scripts/clustering.py \
           --vecdir results/basic/vectorizers \
           --outdir results/basic/clustering
"""

import argparse, json
from pathlib import Path

import joblib, pandas as pd, scipy.sparse as sp
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


# ────────────────────────── UTILIDADES ──────────────────────────────────────
def evaluate(X, labels):
    """Silhouette & Davies-Bouldin (None si queda un solo clúster válido)."""
    core = labels[labels != -1]
    if len(set(core)) < 2:
        return {"silhouette": None, "db_score": None}
    return {
        "silhouette": float(silhouette_score(X, labels, metric="cosine")),
        "db_score":   float(davies_bouldin_score(X, labels))
    }


def run_kmeans(X, k):
    km      = KMeans(n_clusters=k, random_state=0)
    labels  = km.fit_predict(X)
    metrics = {"inertia": float(km.inertia_)} | evaluate(X, labels)
    return labels, metrics


def run_dbscan(X, eps, min_samples=5):
    db      = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels  = db.fit_predict(X)
    metrics = evaluate(X, labels) | {"n_noise": int((labels == -1).sum())}
    return labels, metrics


def run_hdbscan(X, min_cluster_size=5):
    hdb     = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                              metric="euclidean", prediction_data=True)
    labels  = hdb.fit_predict(X)
    metrics = evaluate(X, labels) | {"n_noise": int((labels == -1).sum())}
    return labels, metrics


# ───────────────────────────── MAIN ─────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vecdir", type=Path, required=True,
                    help="Carpeta con los .pkl generados por vectorize.py")
    ap.add_argument("--outdir", type=Path, required=True,
                    help="Carpeta destino para etiquetas y métricas")
    args = ap.parse_args()

    vec_dir: Path = args.vecdir
    out_dir: Path = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for vec_path in vec_dir.rglob("X_*.pkl"):
        method = vec_path.parent.name      # tfidf, w2v, sbert
        print(f"\n=== {method.upper()} ===")
        X_raw = joblib.load(vec_path)
        X     = X_raw.toarray() if sp.issparse(X_raw) else X_raw

        # -------- K-MEANS (k = 4…10) ----------------------------------------
        for k in range(4, 11):
            labels, metrics = run_kmeans(X, k)
            pd.DataFrame({"label": labels}).to_csv(
                out_dir / f"labels_{method}_kmeans{k}.csv", index=False)
            summary.append({"method": method, "algo": f"kmeans{k}", **metrics})

        # -------- DBSCAN (eps grid) ----------------------------------------
        for eps in (0.3, 0.4, 0.5, 0.6):
            labels, metrics = run_dbscan(X, eps)
            pd.DataFrame({"label": labels}).to_csv(
                out_dir / f"labels_{method}_dbscan{eps}.csv", index=False)
            summary.append({"method": method, "algo": f"dbscan{eps}", **metrics})

        # -------- HDBSCAN ---------------------------------------------------
        if HDBSCAN_AVAILABLE:
            labels, metrics = run_hdbscan(X)
            pd.DataFrame({"label": labels}).to_csv(
                out_dir / f"labels_{method}_hdbscan.csv", index=False)
            summary.append({"method": method, "algo": "hdbscan", **metrics})

    # -------- Guardar tabla resumen ----------------------------------------
    pd.DataFrame(summary).to_csv(out_dir / "clustering_metrics.csv", index=False)
    with (out_dir / "clustering_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
