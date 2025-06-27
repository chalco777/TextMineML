#!/usr/bin/env python3

import pandas as pd
import re
import spacy
import unicodedata
from pathlib import Path
from spacy.lang.en.stop_words import STOP_WORDS

# Lista extendida de stopwords personalizadas (en inglés académico/científico)
custom_stopwords = {
    "abstract","acknowledgment","acknowledgments","aim","aims",
    "analysis","analyses","article","author","authors","background",
    "conclusion","conclusions","design","discussion","finding","findings","funding","goal","goals",
    "introduction","journal","limitation","limitations",
    "material","materials","method","methods",
    "objective","objectives","outcome","outcomes",
    "paper","purpose","randomized","report","research","result","results",
    "review","setting","significance","significant",
    "study","studies","summary","use"
}

# Unión de las stopwords de spaCy con las personalizadas
stopwords_custom = STOP_WORDS.union(custom_stopwords)

# Cargar modelo de spaCy para inglés
nlp = spacy.load("en_core_web_sm")

# Función para quitar acentos (aunque no es común en inglés, puede haber palabras extranjeras)
def strip_accents(text: str) -> str:
    text_norm = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text_norm if not unicodedata.combining(ch))

# Preprocesamiento completo
def preprocess_text(text: str) -> str:
    if pd.isna(text) or text.strip() == "":
        return ""
    
    text = text.lower()
    text = strip_accents(text)
    text = re.sub(r"[^a-z\s]", " ", text)

    doc = nlp(text)
    tokens = [
            token.lemma_
            for token in doc
            if token.lemma_ not in stopwords_custom          
            and token.is_alpha                            
            and len(token.lemma_) > 1                    #filtra 1-letra
        ]
    return " ".join(tokens)

# === CONFIGURA TU ARCHIVO DE ENTRADA Y SALIDA ===
INPUT_CSV = "../results/tables/articulos_upch_traducidos_con_editorial_editados_manual.csv"
OUTPUT_CSV = "../results/tables/articulos_preprocessed_final.csv"

# Cargar datos
df = pd.read_csv(INPUT_CSV, sep=";", quotechar='"', encoding="utf-8")

# Aplicar procesamiento
df["title_processed"]     = df["title"].apply(preprocess_text)
df["abstract_processed"]  = df["abstract"].apply(preprocess_text)
df["publisher_processed"] = df["publisher"].apply(preprocess_text)
df["authors_processed"]   = df["authors"].apply(preprocess_text)

# Eliminar columnas innecesarias
df_final = df[["title_processed", "abstract_processed", "publisher_processed", "authors_processed"]]

# Eliminar duplicados (opcional pero útil)
df_final = df_final.drop_duplicates().reset_index(drop=True)

# Guardar el resultado
#Path("results").mkdir(exist_ok=True)
df_final.to_csv(OUTPUT_CSV, index=False, sep=";", encoding="utf-8")

print(f"Preprocesamiento completo. Archivo guardado en 'results/{OUTPUT_CSV}' con {len(df_final)} filas.")
