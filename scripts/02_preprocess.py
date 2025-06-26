#!/usr/bin/env python3
"""
02_preprocess_no_link_date.py
-----------------------------
Preprocesa texto en authors, publisher, title_clean y abstract_clean,
elimina mayúsculas, signos de puntuación y stop-words, y descarta
las columnas link y date. Produce un CSV con:

    authors_processed;
    publisher_processed;
    title_clean_processed;
    abstract_clean_processed

Uso rápido (rutas por defecto):
    python scripts/02_preprocess_no_link_date.py

Uso con rutas personalizadas:
    python scripts/02_preprocess_no_link_date.py \
        -i ../otra_ruta/entrada.csv \
        -o ../otra_ruta/salida.csv
"""

import argparse
import re
from pathlib import Path

import nltk #toolkit de procesamiento de lenguaje natural
import pandas as pd
from nltk.corpus import stopwords #lista de stop-words en varios idiomas
import unicodedata #quitar acentos, transforma todo a caracteres unicode

# Rutas por defecto
DEFAULT_INPUT = Path("../results/tables/articulos_preprocesados_con_editorial.csv")
DEFAULT_OUTPUT = Path("../results/tables/articulos_preprocessed_cleaned_no_link_date.csv")


def strip_accents(text: str) -> str:
    """
    Convierte caracteres acentuados a su equivalente ASCII:
    'á'→'a', 'ñ'→'n', etc.
    """
    text_norm = unicodedata.normalize("NFKD", text) # descompone cada caracter en su forma base mas la carga diacritica
    return "".join(ch for ch in text_norm if not unicodedata.combining(ch)) # me quedo solo con los caracteres que no son combinaciones (diacríticos)


def preprocess_text(text: str, stop_words: set[str]) -> str:
    """Minúsculas, quita acentos, limpia signos, tokeniza y filtra stop-words."""
    text = text.lower()
    text = strip_accents(text)                     # quita acentos usando nuestra función
    text = re.sub(r"[^a-z0-9\s]", " ", text)       # signos y caracteres especiales, los reemplaza por espacio
    tokens = text.split()                          # determina que los tokens serán separados por espacios
    tokens = [tok for tok in tokens if tok not in stop_words and len(tok) > 1] # filtra stop-words y tokens de un solo carácter
    return " ".join(tokens)


def main(input_path: Path, output_path: Path) -> None:
    # Asegurar stop-words
    try:
        stop_words = set(stopwords.words("spanish"))
    except LookupError:
        nltk.download("stopwords")
        stop_words = set(stopwords.words("spanish"))

    # Leer CSV
    df = pd.read_csv(input_path, sep=";", quotechar='"', encoding="utf-8")

    # Aplico el preprocesamiento a las columnas requeridas
    df["authors_processed"] = (
        df["authors"].astype(str).apply(lambda x: preprocess_text(x, stop_words))
    )
    df["publisher_processed"] = (
        df["publisher"].astype(str).apply(lambda x: preprocess_text(x, stop_words))
    )
    df["title_clean_processed"] = (
        df["title_clean"].astype(str).apply(lambda x: preprocess_text(x, stop_words))
    )
    df["abstract_clean_processed"] = (
        df["abstract_clean"].astype(str).apply(lambda x: preprocess_text(x, stop_words))
    )

    # Acá de modo importante verifico que no queden mayúsculas en las columnas procesadas. 
    upp_cols = ["authors_processed", "publisher_processed",
                "title_clean_processed", "abstract_clean_processed"]
    for col in upp_cols:
        upper_count = df[col].str.contains(r"[A-ZÁÉÍÓÚÑÜ]").sum()
        print(f"Mayúsculas residuales en {col}: {upper_count}")

    # Seleccionar columnas finales y eliminar duplicados
    final_cols = [
        "authors_processed",
        "publisher_processed",
        "title_clean_processed",
        "abstract_clean_processed",
    ]
    df_final = df[final_cols].drop_duplicates().reset_index(drop=True)

    # Crear carpeta de salida si no existe
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Guardar resultado
    df_final.to_csv(output_path, index=False, sep=";", encoding="utf-8")
    print(f"✔ CSV generado en: {output_path.resolve()}")
    print(f"Total de filas únicas: {len(df_final)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesa texto y elimina columnas link/date"
    )
    parser.add_argument(
        "-i", "--input",
        default=DEFAULT_INPUT,
        help=f"CSV de entrada (por defecto: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT,
        help=f"CSV de salida (por defecto: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()
    main(Path(args.input), Path(args.output))
