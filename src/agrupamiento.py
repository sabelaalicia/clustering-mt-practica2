import os
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering

def guardar_resultados_agrupamiento(resultados_dict, nombre_salida="resultados_agrupamiento", carpeta="results"):
    os.makedirs(carpeta, exist_ok=True)
    ruta_archivo = os.path.join(carpeta, f"{nombre_salida}.npz")

    # Guardar cada array con su clave
    np.savez(ruta_archivo, **resultados_dict)

    print(f"[OK] Diccionario de resultados guardado en: {ruta_archivo}")


def run_metodo(X, metodo="kmeans", n_clusters=7):
    if metodo == "kmeans":
        modelo = KMeans(n_clusters=n_clusters, random_state=42)
        labels = modelo.fit_predict(X)

    elif metodo == "agglomerative":
        if hasattr(X, "toarray"):
            X = X.toarray()
        modelo = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = modelo.fit_predict(X)

    else:
        raise ValueError("Método no reconocido: usa 'kmeans' o 'agglomerative'")

    return labels


def run_agrupamiento(X_tf, X_tfidf, nombres_archivos):
    results = {}

    # TF
    results["tf_kmeans"] = {"labels": run_metodo(X_tf, metodo="kmeans", n_clusters=7),
                             "nombres": nombres_archivos}
    results["tf_agglomerative"] = {"labels": run_metodo(X_tf, metodo="agglomerative", n_clusters=7),
                                   "nombres": nombres_archivos}

    # TF-IDF
    results["tfidf_kmeans"] = {"labels": run_metodo(X_tfidf, metodo="kmeans", n_clusters=7),
                               "nombres": nombres_archivos}
    results["tfidf_agglomerative"] = {"labels": run_metodo(X_tfidf, metodo="agglomerative", n_clusters=7),
                                      "nombres": nombres_archivos}

    # Guardar todo junto en un .npz
    guardar_resultados_agrupamiento(results)

    return results

