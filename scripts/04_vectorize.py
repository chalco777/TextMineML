#!/usr/bin/env python3
"""
vectorize.py
------------
Genera representaciones de documentos en tres sabores:
  • TF-IDF            → X_tfidf.pkl   (sparse)
  • Word2Vec (fastText) → X_w2v.pkl   (ndarray)
  • SBERT             → X_sbert.pkl  (ndarray)

Ejemplo:
  python scripts/vectorize.py \
         -i results/tables/articulos_preprocessed_final.csv \
         -o results/basic/vectorizers \
         -c title_processed abstract_processed
"""

import argparse
from pathlib import Path

import joblib, pandas as pd
from tqdm import tqdm

# ────────────────────────── CONSTANTES ──────────────────────────────────────
DEFAULT_MODELDIR       = Path("../models")          # buscar aquí primero
DEFAULT_W2V_BASENAME   = "wiki-news-300d-1M-subword.vec"
DEFAULT_SBERT_DIR      = "all-mpnet-base-v2"
DEFAULT_MAX_FEATURES   = 3000
DEFAULT_BATCH          = 16
# ─────────────────────────────────────────────────────────────────────────────


def load_texts(csv_path: Path, cols: list[str]) -> list[str]:
    """Concatena las columnas indicadas y devuelve la lista de documentos."""
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    return df[cols].fillna("").agg(" ".join, axis=1).tolist()


# ───────────────────────── TF-IDF ───────────────────────────────────────────
def vectorize_tfidf(texts: list[str], max_feat: int, outdir: Path):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vct = TfidfVectorizer(max_features=max_feat, ngram_range=(1, 2), min_df=2)
    X   = vct.fit_transform(texts)

    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(X, outdir / "X_tfidf.pkl")
    (outdir / "tfidf_features.txt").write_text(
        "\n".join(vct.get_feature_names_out()), encoding="utf-8")
    print(f"✔ TF-IDF guardado → {outdir/'X_tfidf.pkl'}  ({X.shape[0]}×{X.shape[1]})")


# ──────────────────────── WORD2VEC / FASTTEXT ───────────────────────────────
def load_w2v_model(emb_path: Path | None = None):
    """Carga embeddings desde disco; si faltan, descarga fastText y los guarda."""
    import gensim.downloader as api
    from gensim.models import KeyedVectors

    if emb_path is None:
        emb_path = DEFAULT_MODELDIR / DEFAULT_W2V_BASENAME
    emb_path.parent.mkdir(parents=True, exist_ok=True)

    if emb_path.exists():
        print(f"→ Cargando embeddings locales: {emb_path}")
        return KeyedVectors.load_word2vec_format(emb_path, binary=False)

    print("→ Embeddings no encontrados; descargando fastText ‘wiki-news-300d’…")
    kv = api.load("fasttext-wiki-news-subwords-300")
    print(f"→ Guardando copia en {emb_path}")
    kv.save(str(emb_path))
    return kv


def vectorize_w2v(texts: list[str], emb_path: Path | None, outdir: Path):
    import numpy as np
    from gensim.parsing.preprocessing import preprocess_string

    kv  = load_w2v_model(emb_path)
    dim = kv.vector_size

    vectors = []
    for doc in tqdm(texts, desc="Promediando vectores"):
        toks  = preprocess_string(doc)
        vecs  = [kv[w] for w in toks if w in kv]
        vectors.append(np.mean(vecs, axis=0) if vecs else np.zeros(dim))

    X = np.vstack(vectors)
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(X, outdir / "X_w2v.pkl")
    print(f"✔ Word2Vec guardado → {outdir/'X_w2v.pkl'}  ({X.shape})")


# ───────────────────────────── SBERT ─────────────────────────────────────────
def vectorize_sbert(texts: list[str], batch: int, outdir: Path):
    from sentence_transformers import SentenceTransformer

    model_path = DEFAULT_MODELDIR / DEFAULT_SBERT_DIR
    print(f"→ Cargando modelo SBERT local: {model_path}")
    model = SentenceTransformer(str(model_path))
    X     = model.encode(texts, batch_size=batch, show_progress_bar=True)

    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(X, outdir / "X_sbert.pkl")
    print(f"✔ SBERT guardado → {outdir/'X_sbert.pkl'}  ({X.shape})")


# ───────────────────────────── MAIN ─────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=Path, required=True,
                    help="CSV preprocesado de entrada")
    ap.add_argument("-o", "--outdir", type=Path, required=True,
                    help="Directorio base de salida (se crearán subcarpetas)")
    ap.add_argument("-c", "--cols", nargs="+", default=["title_processed",
                                                        "abstract_processed"],
                    help="Columnas a concatenar por documento")
    ap.add_argument("-f", "--features", type=int, default=DEFAULT_MAX_FEATURES,
                    help="Máx. features TF-IDF")
    ap.add_argument("--embeddings", type=Path, default=None,
                    help="Ruta a embeddings .vec/.bin (Word2Vec)")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                    help="Batch para SBERT")
    args = ap.parse_args()

    texts = load_texts(args.input, args.cols)

    # TF-IDF
    print("\n── TF-IDF ──")
    vectorize_tfidf(texts, args.features, args.outdir / "tfidf")

    # Word2Vec
    print("\n── Word2Vec ──")
    vectorize_w2v(texts, args.embeddings, args.outdir / "w2v")

    # SBERT
    print("\n── SBERT ──")
    vectorize_sbert(texts, args.batch, args.outdir / "sbert")


if __name__ == "__main__":
    main()
