from sklearn.feature_extraction.text import CountVectorizer

def vectorizacion_TF(texts):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    return X