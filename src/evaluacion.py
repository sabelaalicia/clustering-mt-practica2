import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def filtrar_goldstandard(ruta_gold="data/goldstandard.csv", archivos_procesados=None):
    df_gold = pd.read_csv(ruta_gold)
    df_gold_filtrado = df_gold[df_gold["fichero"].astype(str).isin(archivos_procesados)].copy()

    return df_gold_filtrado

def cargar_goldstandard(ruta_csv):

    df = pd.read_csv(ruta_csv)
    etiquetas = df["etiqueta_real"].values
    le = LabelEncoder()
    etiquetas_codificadas = le.fit_transform(etiquetas)

    return etiquetas_codificadas, le


def evaluar_un_metodo(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    matriz = confusion_matrix(y_true, y_pred)

    return precision, recall, f1, matriz


def plot_pca_tsne(matrix, labels, titulo="Visualización PCA + t-SNE", n_componentes_pca=50, n_componentes_tsne=2, random_state=42, guardar=True, guardar_ruta=None):
    """
    Aplica PCA seguido de t-SNE y visualiza los resultados.
    
    Args:
        matrix: Matriz de representación (TF, TF-IDF, etc.) de forma (n_samples, n_features)
        labels: Etiquetas de los documentos (gold standard, k-means, aglomerativo, etc.)
        titulo: Título del gráfico
        n_componentes_pca: Número de componentes para PCA (por defecto 50)
        n_componentes_tsne: Número de componentes para t-SNE (por defecto 2)
        random_state: Semilla aleatoria para reproducibilidad
        guardar_ruta: Ruta donde guardar la imagen (ej: "results/pca_tsne.png"). Si es None, solo muestra.
    
    Returns:
        coordenadas_tsne: Array con las coordenadas t-SNE
        labels: Las etiquetas originales
    """
    print(f"Aplicando PCA a {n_componentes_pca} dimensiones...")
    pca = PCA(n_components=n_componentes_pca, random_state=random_state)
    matriz_pca = pca.fit_transform(matrix)
    varianza_explicada = pca.explained_variance_ratio_.sum()
    print(f"Varianza explicada por PCA: {varianza_explicada:.4f}")
    
    print(f"Aplicando t-SNE a {n_componentes_tsne} dimensiones...")
    tsne = TSNE(n_components=n_componentes_tsne, random_state=random_state, perplexity=30, n_iter_without_progress=1000)
    coordenadas_tsne = tsne.fit_transform(matriz_pca)
    print("Reducción de dimensionalidad completada.")
    
    # Crear visualización
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coordenadas_tsne[:, 0], coordenadas_tsne[:, 1], 
                         c=labels, cmap='viridis', alpha=0.6, edgecolors='k', s=50)
    
    plt.colorbar(scatter, label='Etiquetas')
    plt.title(titulo)
    plt.xlabel('Componente t-SNE 1')
    plt.ylabel('Componente t-SNE 2')
    plt.grid(True, alpha=0.3)
    
    if guardar:
        os.makedirs(os.path.dirname(guardar_ruta), exist_ok=True)
        plt.savefig(guardar_ruta, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {guardar_ruta}")
    
    plt.show()
    
    return coordenadas_tsne, labels


def run_evaluacion(resultados_dict, ruta_goldstandard="data/goldstandard.csv", carpeta_resultados="results", archivos_procesados=None):
    df_gold = filtrar_goldstandard(ruta_gold=ruta_goldstandard, archivos_procesados=archivos_procesados)

    y_true = df_gold["etiqueta_real"].values

    le = LabelEncoder()
    y_true_cod = le.fit_transform(y_true)

    resultados_eval = {}

    os.makedirs(carpeta_resultados, exist_ok=True)
    ruta_txt = os.path.join(carpeta_resultados, "resultados_evaluacion.txt")

    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write("=== RESULTADOS DE EVALUACIÓN ===\n\n")
        
        for nombre, info in resultados_dict.items():
            y_pred = info["labels"]
            nombres_pred = info["nombres"]
            pred_dict = {os.path.splitext(n)[0]: label for n, label in zip(nombres_pred, y_pred)}
            y_pred_ordenado = np.array([pred_dict[str(n)] for n in df_gold["fichero"]])
            # Calcular métricas
            precision, recall, f1, matriz = evaluar_un_metodo(y_true_cod, y_pred_ordenado)

            resultados_eval[nombre] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "matriz_confusion": matriz
            }

            f.write(f"--- Método: {nombre} ---\n")
            f.write(f"Precisión (macro): {precision:.4f}\n")
            f.write(f"Cobertura (macro): {recall:.4f}\n")
            f.write(f"F1 (macro): {f1:.4f}\n")
            f.write("Matriz de confusión:\n")
            f.write(np.array2string(matriz))
            f.write("\n\n")

    print(f"Resultados de evaluación guardados en: {ruta_txt}")
    return resultados_eval
