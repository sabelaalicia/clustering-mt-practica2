import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
        nombre_archivo = f"{vector_type}_top{top_n}_palabras.png" 
        ruta_completa = os.path.join(path_guardar, nombre_archivo)
        plt.savefig(ruta_completa, bbox_inches='tight')
    
    plt.show()
    
def plot_matrices_confusion(resultados_eval, carpeta_salida="results", guardar=True):

    os.makedirs(carpeta_salida, exist_ok=True)
    ruta_img = os.path.join(carpeta_salida, "matrices_confusion.png")

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
        nombre_archivo = "matrices_confusion.png" 
        ruta_completa = os.path.join(path_guardar, nombre_archivo)
        plt.savefig(ruta_completa, bbox_inches='tight')

    plt.show()
    
