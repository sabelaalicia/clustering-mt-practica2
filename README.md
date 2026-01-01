# Clustering en Minería de Textos - Práctica 2

Este proyecto implementa técnicas de agrupamiento (clustering) de documentos de texto, correspondiente a la Tarea 2 de la asignatura Minería de Textos.

## Descripción

El proyecto realiza el análisis y agrupamiento de documentos médicos utilizando técnicas de procesamiento de lenguaje natural y machine learning. Incluye:

- **Preprocesamiento**: Limpieza y normalización de textos
- **Vectorización**: Representación TF y TF-IDF
- **Agrupamiento**: K-Means y Agglomerative Clustering
- **Evaluación**: Métricas de precisión, recall, F1 y matrices de confusión
- **Visualización**: PCA, t-SNE, word clouds y gráficos de frecuencias

## Estructura del Proyecto

```
clustering-mt-practica2/
│
├── data/
│   ├── raw/                    # Datos originales
│   ├── preprocessed/           # Textos preprocesados
│   └── goldstandard.csv        # Etiquetas de referencia
│
├── src/
│   ├── main.py                 # Script principal de ejecución
│   ├── preprocesamiento.py     # Limpieza y normalización
│   ├── representacion.py       # Vectorización TF/TF-IDF
│   ├── agrupamiento.py         # K-Means y Agglomerative
│   ├── evaluacion.py           # Métricas y visualizaciones
│   ├── utils.py                # Utilidades de visualización
│   ├── opcional.py             # Método Elbow
│   └── crear_goldstandard.py   # Generación de etiquetas
│
├── docs/
│   ├── images/                 # Gráficos generados
│   └── Tarea2-MineriaTextos.pdf
│
├── notebooks/
│   └── analisis_estadistico.ipynb
│
├── results/                    # Resultados de clustering
│
└── requirements.txt            # Dependencias Python
```

## Requisitos

- Python 3.8 o superior
- pip

## Instalación

### 1. Clonar o descargar el proyecto

```bash
git clone <url-repositorio>
cd clustering-mt-practica2
```

### 2. Crear un entorno virtual en Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Ejecución

Una vez activado el entorno virtual:

```bash
python src/main.py
```

Este script ejecutará:
1. Carga y vectorización de textos (TF y TF-IDF)
2. Generación de word clouds y gráficos de frecuencias
3. Clustering con K-Means y Agglomerative (k=7)
4. Evaluación y visualización PCA/t-SNE
5. Método Elbow para determinar número óptimo de clusters
6. Clustering con k=9 y cálculo de métricas (precisión, recall, F1, matriz de confusión)

## Resultados

Los resultados se guardan automáticamente en:

- **`results/`**: Métricas de evaluación y arrays de etiquetas (.npz, .txt)
- **`docs/images/`**: Gráficos (word clouds, matrices de confusión, PCA, t-SNE, Elbow)
