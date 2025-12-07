import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_palabras_mas_frecuentes(X, vectorizer, vector_type="TF", top_n=10, guardar=True):

    freq = X.toarray().sum(axis=0)
    # === MODIFICACIÓN CLAVE AQUÍ ===
    # Define la cadena de 10 espacios
    espacios = "          " # 10 espacios en blanco
    
    # Obtiene y modifica las palabras: añade 10 espacios antes y después de cada palabra.
    palabras_originales = vectorizer.get_feature_names_out()
    palabras = [f"{espacios}{p}{espacios}" for p in palabras_originales]
    # ===============================
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
        # Cambiamos el nombre del archivo para reflejar que es la versión mejorada o simplemente sobrescribimos
        nombre_archivo = f"{vector_type}_top{top_n}_palabras.png" 
        ruta_completa = os.path.join(path_guardar, nombre_archivo)
        plt.savefig(ruta_completa, bbox_inches='tight')
        print(f"Imagen guardada en {ruta_completa}")
    
    plt.show()

# Nota: El nombre de la función se cambió a 'plot_palabras_mas_frecuentes_mejorado' para evitar confusiones.