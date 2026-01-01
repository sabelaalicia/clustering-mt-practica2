from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def run_elbow_kmeans(X, max_k=30, titulo="Método Elbow", guardar=False, ruta_guardado=None):
    inertias = []
    Ks = np.arange(1, max_k + 1)

    for k in Ks:
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8,5))
    plt.plot(Ks, inertias, marker='o', linewidth=2, markersize=8, color="#20a386")
    plt.xticks(Ks)
    plt.grid(alpha=0.3, linestyle="--")

    plt.title(titulo, fontsize=14, fontweight="bold")
    plt.xlabel("Número de clusters (k)", fontsize=12)
    plt.ylabel("Inercia (SSE)", fontsize=12)

    if guardar:
        plt.savefig(ruta_guardado, bbox_inches="tight")
        print(f"Gráfica Elbow guardada en: {ruta_guardado}")

    plt.show()

    return Ks, inertias
