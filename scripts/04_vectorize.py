#!/usr/bin/env python3
# 03_vectorize.py  (versión con caché local en ../models/)
# ─────────────────────────────────────────────────────────────────────────────
# Véanse los comentarios iniciales del script original (no cambiaron).
# ─────────────────────────────────────────────────────────────────────────────

import argparse
from pathlib import Path
import joblib
import pandas as pd
from tqdm import tqdm

# =============== PARÁMETROS POR DEFECTO =====================================
DEFAULT_INPUT   = Path("../results/tables/articulos_preprocessed_final_sin_scientific_stop_words.csv")
DEFAULT_OUTDIR  = Path("../results/vectorizers/")
DEFAULT_MODELDIR = Path("models/")          # ← NUEVO
DEFAULT_W2V_BASENAME = "wiki-news-300d-1M-subword.vec"     # se guarda como KeyedVectors
DEFAULT_SBERT_DIR = "all-mpnet-base-v2"                    # ruta a tu SBERT local
DEFAULT_MAX_FEATURES = 3000
DEFAULT_BATCH = 16
# ============================================================================


# Funcion para cargar nuestro csv preprocesado
# concatenamos titulo y abstract en una sola fila y generamos una lista con todos los documentos ya limpios
def load_texts(input_csv: Path) -> list[str]:
    df = pd.read_csv(input_csv, sep=";", encoding="utf-8")
    texts = (
        df["title_processed"].fillna("") + " " +
        df["abstract_processed"].fillna("")
    ).tolist()
    return texts


# ───────────────────────── TF-IDF ────────────────────────────────────────────

def vectorize_tfidf(texts: list[str], max_features: int, outdir: Path):
    from sklearn.feature_extraction.text import TfidfVectorizer
    #genera unigramas y bigramas, ignora términso que salgan en menos de dos documentos
    vct = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2
    )
    # fit aprende IDF global y vocabularios
    # transform produce la matriz dispersa doc*terminos
    X = vct.fit_transform(texts)

    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(X, outdir / "X_tfidf.pkl")
    (outdir / "tfidf_features.txt").write_text(
        "\n".join(vct.get_feature_names_out()), encoding="utf-8"
    )
    print(f"✔ TF-IDF guardado: {outdir/'X_tfidf.pkl'}  ({X.shape[0]}×{X.shape[1]})")


# ───────────────────────── WORD2VEC ──────────────────────────────────────────
#Carga los embeddings de Word2Vec, ya sea desde un archivo local o descargándolos
def load_w2v_model(embeddings_path: Path | None = None):
    """
    Intenta cargar embeddings locales; si no existen, los descarga y los
    almacena en ../models/ para usos futuros.
    """
    import gensim.downloader as api
    from gensim.models import KeyedVectors

    # Ruta por defecto si el usuario no especifica nada
    if embeddings_path is None:
        embeddings_path = DEFAULT_MODELDIR / DEFAULT_W2V_BASENAME
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    if embeddings_path.exists():
        print(f"→ Cargando embeddings locales: {embeddings_path}")
        return KeyedVectors.load_word2vec_format(embeddings_path, binary=False)

    # Si no existen, descargamos fastText multilingüe (300-d)
    print("→ Embeddings no encontrados; descargando modelo "
          "'fasttext-wiki-news-subwords-300' (puede tardar)…")
    kv = api.load("fasttext-wiki-news-subwords-300")

    # Guardamos para próximos lanzamientos
    print(f"→ Guardando copia en {embeddings_path} …")
    kv.save(str(embeddings_path))

    return kv


def vectorize_w2v(texts: list[str], embeddings_path: Path | None, outdir: Path):
    import numpy as np
    from gensim.parsing.preprocessing import preprocess_string

    kv = load_w2v_model(embeddings_path)
    dim = kv.vector_size

    vectors = []
    for doc in tqdm(texts, desc="Promediando vectores"):
        tokens = preprocess_string(doc)
        #en cada documento iteramos sobre sus tokens y vemos si tienen una representacion en el vocabulario kv del modelo
        vecs = [kv[w] for w in tokens if w in kv]
        # se promedian los vectores de los tokens del documento, de tal modo que queda un embedding por documento por documento (representación del documento)
        vectors.append(np.mean(vecs, axis=0) if vecs else np.zeros(dim))
    # se aplian los vectores en una matriz 2D
    # cada fila es un documento y cada columna es una dimensión del embedding
    X = np.vstack(vectors)

    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(X, outdir / "X_w2v.pkl")
    print(f"✔ Word2Vec guardado: {outdir/'X_w2v.pkl'}  ({X.shape})")



# ───────────────────────── SBERT ─────────────────────────────────────────────
def vectorize_sbert(texts: list[str], batch: int, outdir: Path):
    from sentence_transformers import SentenceTransformer

    model_path = DEFAULT_MODELDIR / DEFAULT_SBERT_DIR
    print(f"→ Cargando modelo SBERT local: {model_path}")
    # Acá se carga el modelo BERT afinado para embeddings
    model = SentenceTransformer(str(model_path))   # ← ruta al disco
    # tokeniza cada texto y genera un embedding de 768 dimensiones para cada documento
    X = model.encode(texts, batch_size=batch, show_progress_bar=True)

    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(X, outdir / "X_sbert.pkl")
    print(f"✔ SBERT guardado: {outdir/'X_sbert.pkl'}  ({X.shape})")


# ───────────────────────── MAIN ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Vectoriza texto con TF-IDF, Word2Vec y SBERT")
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT, type=Path,
                        help="CSV preprocesado de entrada")
    parser.add_argument("-o", "--outdir", default=DEFAULT_OUTDIR, type=Path,
                        help="Directorio base de salida")
    parser.add_argument("-f", "--features", type=int, default=DEFAULT_MAX_FEATURES,
                        help="Máximo de características (TF-IDF)")
    parser.add_argument("--embeddings", type=Path, default=None,
                        help="Ruta EMBEDDINGS .kv/.bin; si falta, se usa ../models/")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help="Tamaño de lote SBERT")
    args = parser.parse_args()

    texts = load_texts(args.input)

    # TF-IDF
    print("\n── TF-IDF ──")
    #vectorize_tfidf(texts, args.features, args.outdir / "tfidf")

    # Word2Vec
    print("\n── Word2Vec ──")
    #vectorize_w2v(texts, args.embeddings, args.outdir / "w2v")

    # SBERT
    print("\n── SBERT ──")
    vectorize_sbert(texts, args.batch, args.outdir / "sbert")



if __name__ == "__main__":
    main()
