import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.decomposition import PCA
from wordcloud import WordCloud


def plot_palabras_mas_frecuentes(X, vectorizer, vector_type="TF", top_n=10, guardar=True):

    freq = X.toarray().sum(axis=0)
    espacios = "          "
    palabras_originales = vectorizer.get_feature_names_out()
    palabras = [f"{espacios}{p}{espacios}" for p in palabras_originales]
    df_freq = pd.DataFrame({'palabra': palabras, 'frecuencia': freq})
    df_freq = df_freq.sort_values('frecuencia', ascending=False).head(top_n)
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 10)) 
    
    sns.barplot(x='frecuencia', y='palabra', data=df_freq, palette='viridis')
    plt.xlabel("Peso", fontsize=30) 
    plt.ylabel("Palabra", fontsize=30)
    
    plt.xticks(fontsize=30) 
    plt.yticks(fontsize=30)
    
    plt.tight_layout()
    
    if guardar:
        path_guardar = os.path.join("docs", "images")
        os.makedirs(path_guardar, exist_ok=True)
        nombre_archivo = f"{vector_type}_top{top_n}_palabras.pdf" 
        ruta_completa = os.path.join(path_guardar, nombre_archivo)
        plt.savefig(ruta_completa, bbox_inches='tight')
    
    plt.show()
    

def plot_wordcloud(X, vectorizer, vector_type="TF", guardar=True):

    freq = X.toarray().sum(axis=0)
    palabras_originales = vectorizer.get_feature_names_out()
    
    # Crear diccionario palabra:frecuencia
    palabra_freq = dict(zip(palabras_originales, freq))
    
    # Crear wordcloud
    wordcloud = WordCloud(width=1200, height=700, background_color='white', 
                         colormap='viridis', relative_scaling=0.5).generate_from_frequencies(palabra_freq)
    
    plt.figure(figsize=(16, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    if guardar:
        path_guardar = os.path.join("docs", "images")
        os.makedirs(path_guardar, exist_ok=True)
        nombre_archivo = f"{vector_type}_wordcloud.pdf"
        ruta_completa = os.path.join(path_guardar, nombre_archivo)
        plt.savefig(ruta_completa, bbox_inches='tight', dpi=300)
    
    plt.show()

def plot_matrices_confusion(resultados_eval, carpeta_salida="results", guardar=True):

    os.makedirs(carpeta_salida, exist_ok=True)
    ruta_img = os.path.join(carpeta_salida, "matrices_confusion.pdf")

    metodos = list(resultados_eval.keys())
    n = len(metodos)

    # Definir figura 2x2 (sirve para 4 métodos)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, metodo in zip(axes, metodos):
        matriz = resultados_eval[metodo]["matriz_confusion"]

        sns.heatmap(
            matriz,
            annot=False,
            cmap="viridis",
            ax=ax,
            cbar=False
        )
        ax.set_title(f"Matriz de confusión - {metodo}")
        ax.set_xlabel("Cluster asignado")
        ax.set_ylabel("Cluster real")

    # Ocultar ejes sobrantes si hay menos de 4 métodos
    for i in range(len(metodos), 4):
        fig.delaxes(axes[i])

    plt.tight_layout()
        
    if guardar:
        path_guardar = os.path.join("docs", "images")
        os.makedirs(path_guardar, exist_ok=True)
        nombre_archivo = "matrices_confusion.pdf" 
        ruta_completa = os.path.join(path_guardar, nombre_archivo)
        plt.savefig(ruta_completa, bbox_inches='tight')

    plt.show()

def plot_scatter_pca(matrices, labels_list, nombres=None, titles=None, guardar=True, carpeta_salida="docs/images"):

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    cmap = plt.get_cmap("tab10")

    for i in range(4):
        X = matrices[i].toarray()
        labels = np.asarray(labels_list[i])

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)
        evr = pca.explained_variance_ratio_
        evr1 = evr[0] * 100
        evr2 = evr[1] * 100
        evr_total = evr1 + evr2

        ax = axes[i]
        for j, lab in enumerate(np.unique(labels)):
            ax.scatter(coords[labels == lab, 0], coords[labels == lab, 1], s=40, alpha=0.85,
                       color=cmap(j % 10), label=f"Cluster {lab}")


        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        title_base = titles[i] if titles else f"Scatter PCA - Plot {i+1}"
        title_full = f"{title_base} (PC1 {evr1:.1f}%, PC2 {evr2:.1f}%, total {evr_total:.1f}%)"
        ax.set_title(title_full)
        ax.legend(title="Clusters", loc='best', fontsize='small')

    plt.tight_layout(pad=3.0)
    if guardar:
        os.makedirs(carpeta_salida, exist_ok=True)
        plt.savefig(os.path.join(carpeta_salida, "scatter_4plots.pdf"), bbox_inches='tight')

    plt.show()
