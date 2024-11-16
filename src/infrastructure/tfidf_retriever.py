import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.settings.logging_config import setup_logger


class TFIDFRetriever:
    """
        Un recuperador que utiliza TF-IDF vectorization y similitud coseno
        para encontrar los documentos más relevantes en respuesta a una consulta.

        Attributes:
            documents (List[Dict[str, Any]]): Lista de documentos, donde cada documento es un diccionario
                que contiene al menos una clave "content" con el texto del documento.
            vectorizer (TfidfVectorizer): Vectorizador TF-IDF para transformar el texto en vectores.
            doc_vectors (sparse matrix): Matriz de vectores TF-IDF de los documentos.
            logger (logging.Logger): Instancia de logger para registrar información y mensajes de depuración.
        """
    def __init__(self, documents: list, logger: logging = setup_logger()):
        """
        Inicializa el recuperador TFIDFRetriever con los documentos proporcionados y configura el vectorizador TF-IDF.

        Args:
            documents (List[Dict[str, Any]]): Los documentos que serán indexados y recuperados.
            logger (logging.Logger, optional): Logger para registrar mensajes. Si no se proporciona, se configura un logger predeterminado.
        """
        self.documents = documents
        self.logger = logger if logger is not None else setup_logger()

        # Inicializar el vectorizador TF-IDF
        self.vectorizer = TfidfVectorizer()

        # Procesar solo el contenido de cada documento para la vectorización
        doc_contents = [doc["content"] for doc in documents]

        # Ajustar el vectorizador al contenido de los documentos y transformarlos en vectores TF-IDF
        self.doc_vectors = self.vectorizer.fit_transform(doc_contents)

        self.logger.info("TF-IDF Vectorization Success")

    def retrieve(self, query: str, top_k: int = 3):
        """
        Recupera los top_k documentos más similares a la consulta proporcionada basándose en la similitud coseno.

        Args:
            query (str): La consulta de búsqueda.
            top_k (int, optional): El número de documentos más relevantes a recuperar. Por defecto es 3.

        Returns:
            List[str]: Lista de contenidos de los top_k documentos más similares.
        """
        # Transformar la consulta en un vector TF-IDF utilizando el vectorizador ajustado
        query_vec = self.vectorizer.transform([query])

        # Calcular las similitudes coseno entre el vector de la consulta y todos los vectores de los documentos
        cosine_similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()

        # Obtener los índices de los top_k documentos con mayor similitud
        top_indices = np.argsort(cosine_similarities)[-top_k:][::-1]

        self.logger.info("\n=== Validation Results (Cosine Similarity) Success ===")

        # Devolver el contenido de los top_k documentos más similares
        return [self.documents[i]["content"] for i in top_indices]
