import os
import pandas as pd

def crear_goldstandard(ruta_raw="data/raw/", salida_csv="data/goldstandard.csv"):
    rows = []
    for carpeta in os.listdir(ruta_raw):
        ruta_carpeta = os.path.join(ruta_raw, carpeta)
        if os.path.isdir(ruta_carpeta):
            for fichero in os.listdir(ruta_carpeta):
                rows.append({
                    "fichero": fichero,
                    "etiqueta_real": carpeta
                })

    df = pd.DataFrame(rows)
    df.to_csv(salida_csv, index=False)
    print(f"Goldstandard creado en {salida_csv}, con {len(rows)} documentos.")

if __name__ == "__main__":
    crear_goldstandard()

