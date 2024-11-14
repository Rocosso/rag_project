import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.settings.logging_config import setup_logger


class TFIDFRetriever:
    def __init__(self, documents: list, logger: logging = setup_logger()):
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        # Procesar solo el contenido de cada documento para la vectorización
        doc_contents = [doc["content"] for doc in documents]
        self.doc_vectors = self.vectorizer.fit_transform(doc_contents)
        self.logger = logger

    def retrieve(self, query: str, top_k: int = 3):
        query_vec = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = np.argsort(cosine_similarities)[-top_k:][::-1]

        self.logger.info("\n=== Validation Results (Cosine Similarity) Success ===")
        # Devuelve el contenido de los documentos en lugar de una descripción del archivo
        return [self.documents[i]["content"] for i in top_indices]
