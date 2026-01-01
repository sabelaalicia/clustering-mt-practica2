from representacion import crear_textos, vectorizacion_TF, vectorizacion_TFIDF
from utils import plot_palabras_mas_frecuentes, plot_matrices_confusion, plot_wordcloud
from agrupamiento import run_agrupamiento, run_metodo
from evaluacion import run_evaluacion, plot_pca_tsne, filtrar_goldstandard
from utils import plot_scatter_pca
from sklearn.preprocessing import LabelEncoder
from opcional import run_elbow_kmeans


def main():

    ####  Vectorización  ####

    #ruta = "C:\\Users\\sabel\\OneDrive\\Escritorio\\Master\\MT_conv2\\Practica 2\\clustering-mt-practica2\\data\\preprocessed"
    ruta ="C:\\Users\\Sabela\\clustering-mt-practica2\\data\\preprocessed"
    
    
    textos, nombres_archivos = crear_textos(ruta)
    print(f"Documentos cargados: {len(textos)}")

    X_tf, vectorizer_tf = vectorizacion_TF(textos)
    print(f"Matriz TF generada: {X_tf.shape}")

    X_tfidf, vectorizer_tfidf = vectorizacion_TFIDF(textos)
    print(f"Matriz TF-IDF generada: {X_tfidf.shape}")

    plot_palabras_mas_frecuentes(X_tf, vectorizer_tf, vector_type="TF", top_n=10, guardar=False)
    plot_palabras_mas_frecuentes(X_tfidf, vectorizer_tfidf, vector_type="TFIDF", top_n=10, guardar=False)
    plot_wordcloud(X_tf, vectorizer_tf, vector_type="TF", guardar=True)
    plot_wordcloud(X_tfidf, vectorizer_tfidf, vector_type="TFIDF", guardar=True)

    ####   Agrupamiento   ####
    results = run_agrupamiento(X_tf, X_tfidf, nombres_archivos)

    matrices = [X_tf, X_tf, X_tfidf, X_tfidf]

    labels_list = [results['tf_kmeans']['labels'],
                   results['tf_agglomerative']['labels'],
                   results['tfidf_kmeans']['labels'],
                   results['tfidf_agglomerative']['labels']]

    titles = ['TF - KMeans','TF - Agglomerative','TFIDF - KMeans','TFIDF - Agglomerative']

    plot_scatter_pca(matrices, labels_list, nombres=nombres_archivos, titles=titles, guardar=False)

    ####   Evaluaciones   ####
    evals = run_evaluacion(results, ruta_goldstandard="data/goldstandard.csv",archivos_procesados=nombres_archivos)
    plot_matrices_confusion(evals, guardar=True)
    
    #TF con goldstandard
    df_gold = filtrar_goldstandard("data/goldstandard.csv", nombres_archivos)
    etiquetas_gold = LabelEncoder().fit_transform(df_gold["etiqueta_real"])
    plot_pca_tsne(
        matrix=X_tf.toarray(),
        labels= etiquetas_gold,
        titulo="TF + Gold Standard",
        guardar_ruta="docs/images/pca_tsne_tf_gold.pdf"
    )

    #TF-IDF con goldstandard
    plot_pca_tsne(
        matrix=X_tfidf.toarray(),
        labels= etiquetas_gold,
        titulo="TF-IDF + Gold Standard",
        guardar_ruta="docs/images/pca_tsne_tfidf_gold.pdf"
    )

    #TF con K-Means
    plot_pca_tsne(
        matrix=X_tf.toarray(), 
        labels=results['tf_kmeans']['labels'],
        titulo="TF + K-Means",
        guardar_ruta="docs/images/pca_tsne_tf_kmeans.pdf"
    )
    #TF-IDF con K-Means
    plot_pca_tsne(
        matrix=X_tfidf.toarray(),
        labels=results['tfidf_kmeans']['labels'],
        titulo="TF-IDF + K-Means",
        guardar_ruta="docs/images/pca_tsne_tfidf_kmeans.pdf"
    )
    #TF con Agglomerative
    plot_pca_tsne(
        matrix=X_tf.toarray(),
        labels=results['tf_agglomerative']['labels'],
        titulo="TF + Agglomerative",
        guardar_ruta="docs/images/pca_tsne_tf_agglomerative.pdf"
    )
    #TF-IDF con Agglomerative
    plot_pca_tsne(
        matrix=X_tfidf.toarray(),
        labels=results['tfidf_agglomerative']['labels'],
        titulo="TF-IDF + Agglomerative",
        guardar_ruta="docs/images/pca_tsne_tfidf_agglomerative.pdf"
    )
    
    
    ####   Opcional: Método Elbow   ####
    print("Ejecutando método Elbow para TF...")
    run_elbow_kmeans(X_tf, max_k=12, titulo="Elbow TF", guardar=True, ruta_guardado="docs/images/elbow_tf.png")

    print("Ejecutando método Elbow para TF-IDF...")
    run_elbow_kmeans(X_tfidf, max_k=12, titulo="Elbow TF-IDF", guardar=True, ruta_guardado="docs/images/elbow_tfidf.png")

    labels = run_metodo(X_tf, metodo="kmeans", n_clusters=9)
    print(labels)

    ####   Evaluación para k=9   ####
    resultados_k9 = {
        "tf_kmeans_k9": {
            "labels": labels,
            "nombres": nombres_archivos
        }
    }

    eval_k9 = run_evaluacion(resultados_k9,ruta_goldstandard="data/goldstandard.csv",archivos_procesados=nombres_archivos)

    metrica_k9 = eval_k9["tf_kmeans_k9"]

    print("\n=== Métricas k=9 (TF + KMeans) ===")
    print(f"Precisión (macro): {metrica_k9['precision']:.4f}")
    print(f"Recall (macro): {metrica_k9['recall']:.4f}")
    print(f"F1 (macro): {metrica_k9['f1']:.4f}")
    print("Matriz de confusión:\n", metrica_k9["matriz_confusion"])

    

if __name__ == "__main__":
    main()
