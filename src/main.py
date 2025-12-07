from representacion import crear_textos, vectorizacion_TF, vectorizacion_TFIDF
import os
from utils import plot_palabras_mas_frecuentes


def main():

    import os
    print("Directorio actual:", os.getcwd())

    ruta = "C:\\Users\\sabel\\OneDrive\\Escritorio\\Master\\MT_conv2\\Practica 2\\clustering-mt-practica2\\data\\preprocessed"

    textos = crear_textos(ruta)
    print(f"Documentos cargados: {len(textos)}")

    X_tf, vectorizer_tf = vectorizacion_TF(textos)
    print(f"Matriz TF generada: {X_tf.shape}")

    X_tfidf, vectorizer_tfidf = vectorizacion_TFIDF(textos)
    print(f"Matriz TF-IDF generada: {X_tfidf.shape}")

    plot_palabras_mas_frecuentes(X_tf, vectorizer_tf, vector_type="TF", top_n=10, guardar=True)
    plot_palabras_mas_frecuentes(X_tfidf, vectorizer_tfidf, vector_type="TFIDF", top_n=10, guardar=True)


if __name__ == "__main__":
    main()
