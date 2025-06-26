# TextMineML

Este proyecto implementa un pipeline de minería de texto usando artículos académicos recolectados desde diversas fuentes, con enfoque en preprocesamiento, vectorización y clustering.

## 🔁 Flujo de trabajo actual

1. **Recolección y preprocesamiento**
   - Se extraen artículos mediante web scraping.
   - Los textos se limpian (minúsculas, eliminación de signos de puntuación, stopwords, etc.) con `02_preprocess.py`.
   - Los resultados se guardan en `results/tables/`.

2. **Vectorización**
   Usamos tres estrategias complementarias para representar los textos como vectores numéricos:
   - **TF-IDF**
   - **Word2Vec (preentrenado)**
   - **SBERT (preentrenado)**

   El script `03_vectorize.py` genera y guarda las matrices en `results/vectorizers/`.

## 📁 Estructura del proyecto

    TextMineML/
    ├── models/ # Modelos preentrenados almacenados localmente
    ├── notebooks/ # Notebooks para exploración y pruebas
    ├── results/ # Resultados de procesamiento y vectorización
    ├── scripts/ # Scripts del pipeline
    └── README.md

## 🧠 Modelos preentrenados

### 1. Word2Vec: `wiki-news-300d-1M-subword.vec`

- 📥 **Descarga manual**:
  - URL: [https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip)
  - Descomprimir y mover a `models/wiki-news-300d-1M-subword.vec`.

- 📦 Formato: texto plano compatible con Gensim (`KeyedVectors.load_word2vec_format`)

### 2. SBERT: `all-mpnet-base-v2`

### 2. SBERT: `all-mpnet-base-v2`

- 📥 **Descarga desde Hugging Face**  
  https://huggingface.co/sentence-transformers/all-mpnet-base-v2/tree/main

- **Clonación manual del repositorio oficial**  
  ```bash
  cd models/
  git lfs install
  git clone https://huggingface.co/sentence-transformers/all-mpnet-base-v2
  ```