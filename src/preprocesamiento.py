import os
from pathlib import Path
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def quitar_cabecera(texto):

    partes = re.split(r'\n\s*\n', texto, maxsplit=1)
    return partes[1] if len(partes) > 1 else texto

def quitar_firma(texto):

    if "\n--\n" in texto:
        return texto.split("\n--\n")[0]
    
    # si hay email en últimas 6 líneas, corta desde ahí
    lines = texto.splitlines()
    tail = "\n".join(lines[-6:])
    if re.search(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", tail):
        for i, ln in enumerate(lines[::-1], 1):
            if re.search(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", ln):
                cut_index = len(lines) - i
                return "\n".join(lines[:cut_index])
    return texto

def limpiar_texto(raw_text):

    texto = quitar_cabecera(raw_text)
    texto = quitar_firma(texto)
    return texto


def normalizar_texto(texto):
    """
    - Minúsculas
    - Eliminar URLs, emails y caracteres no alfabéticos y números
    """
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+', '', texto)
    texto = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', texto)
    texto = re.sub(r'In article <[^>]+>, [\w\.-]+@[\w\.-]+\.\w+ writes:', '', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^a-z\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def tokenizar_texto(texto):
    """Tokeniza y elimina stopwords y tokens muy cortos (Utilizando NLTK)"""
    tokens = nltk.word_tokenize(texto)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return tokens

def lematizar_tokens(tokens):
    """Lematiza los tokens"""
    return [lemmatizer.lemmatize(t) for t in tokens]



def eliminar_outliers(ruta_preprocesada="../data/preprocessed/", percentil_inf=0.5, percentil_sup=99.5):
    ruta_preprocesada = Path(ruta_preprocesada)
    longitudes = []

    for fichero in ruta_preprocesada.glob("*.txt"):
        with open(fichero, "r", encoding="utf-8") as f:
            tokens = f.read().split()
            longitudes.append(len(tokens))

    min_len = np.percentile(longitudes, percentil_inf)
    max_len = np.percentile(longitudes, percentil_sup)
    print(f"Eliminando documentos fuera del rango [{min_len}, {max_len}] palabras")

    eliminados = 0
    for fichero in ruta_preprocesada.glob("*.txt"):
        with open(fichero, "r", encoding="utf-8") as f:
            tokens = f.read().split()
        if len(tokens) < min_len or len(tokens) > max_len:
            fichero.unlink()
            eliminados += 1

    print(f"Documentos eliminados: {eliminados}")


def preprocesar_carpeta(ruta_raw="../data/raw/", ruta_destino="../data/preprocessed/"):
    ruta_raw = Path(ruta_raw)
    ruta_destino = Path(ruta_destino)
    ruta_destino.mkdir(parents=True, exist_ok=True)
    longitudes = []
    for carpeta in ruta_raw.iterdir():
        if carpeta.is_dir():
            for fichero in carpeta.iterdir():
                if fichero.is_file():
                    with open(fichero, "r", encoding="latin1") as f:
                        texto = f.read()
                    texto_limpio = limpiar_texto(texto)
                    texto_normalizado = normalizar_texto(texto_limpio)
                    texto_tokens = tokenizar_texto(texto_normalizado)
                    texto_lemmatizado = lematizar_tokens(texto_tokens)
                    texto_final = " ".join(texto_lemmatizado)

                    longitudes.append(len(texto_lemmatizado))
                    ruta_destino.mkdir(parents=True, exist_ok=True)
                    with open(ruta_destino / (fichero.name + ".txt"), "w", encoding="utf-8") as f_out:
                        f_out.write(texto_final)
    
    eliminar_outliers(ruta_destino)
    print("Preprocesamiento completado")


if __name__ == "__main__":
    preprocesar_carpeta()
    
