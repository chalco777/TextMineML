#!/usr/bin/env python3
"""
Lanza vectorización + clustering en modo BASIC o EXTENDED

Ejemplos:
  python scripts/run_pipeline.py basic
  python scripts/run_pipeline.py extended
"""
import subprocess, sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[1]   # = …/TextMineML
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"
TABLE   = ROOT / "results/tables/articulos_preprocessed_final.csv"

MODES = {
    "basic": {
        "cols": ["title_processed", "abstract_processed"],
        "res_dir": RESULTS / "basic"
    },
    "extended": {
        "cols": ["title_processed", "abstract_processed",
                 "publisher_processed", "authors_processed"],   # <── columna correcta
        "res_dir": RESULTS / "extended"
    }
}

def run(mode: str):
    cfg      = MODES[mode]
    vec_dir  = cfg["res_dir"] / "vectorizers"
    clu_dir  = cfg["res_dir"] / "clustering"
    cfg["res_dir"].mkdir(parents=True, exist_ok=True)

    # 1. Vectorización (TF-IDF + W2V + SBERT)
    subprocess.run([
        "python", SCRIPTS / "04_vectorize.py",     # ← nombre real del script
        "-i", TABLE,
        "-o", vec_dir,
        "-c", *cfg["cols"]
    ], check=True)

    # 2. Clustering (KMeans + DBSCAN + HDBSCAN)
    subprocess.run([
        "python", SCRIPTS / "05_clustering.py",    # ← nombre real del script
        "--vecdir", vec_dir,
        "--outdir", clu_dir
    ], check=True)

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in MODES:
        sys.exit(f"Uso: {Path(__file__).name} <basic|extended>")
    run(sys.argv[1])
