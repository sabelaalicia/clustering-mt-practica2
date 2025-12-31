from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import _stop_words
import os
import glob

def crear_textos(root):
    archivos = sorted(glob.glob(os.path.join(root, "*.txt")))
    textos = []
    nombres = []
    for ruta in archivos:
        with open(ruta, "r", encoding="utf-8") as f:
            textos.append(f.read())
        nombres.append(os.path.splitext(os.path.basename(ruta))[0])
    return textos, nombres

def vectorizacion_TF(textos):
    stop_words_custom = list(_stop_words.ENGLISH_STOP_WORDS)+['writes', 'article']
    vectorizer = CountVectorizer(stop_words=stop_words_custom)
    matrix_tf = vectorizer.fit_transform(textos)
    return matrix_tf, vectorizer

def vectorizacion_TFIDF(textos):
    stop_words_custom = list(_stop_words.ENGLISH_STOP_WORDS)+['writes', 'article']
    vectorizer = TfidfVectorizer(stop_words=stop_words_custom)
    matrix_tfidf = vectorizer.fit_transform(textos)
    return matrix_tfidf, vectorizer
