import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.commons.still_not_in_use_decorator import still_not_in_use
from src.settings.logging_config import setup_logger
from src.settings.settings import Settings

settings = Settings()


class DenseRetrieverClustering:
    def __init__(self, documents: list, logger: setup_logger, index_path: str = settings.faiss_database_path):
        self.documents = documents
        self.logger=logger
        self.index_path = index_path

        # modelo de embeddings más efectivo
        self.model = SentenceTransformer('all-mpnet-base-v2')

        # Crear embeddings para todos los documentos
        doc_contents = [doc["content"] for doc in documents]
        self.doc_embeddings = self.model.encode(doc_contents, convert_to_numpy=True)

        # Normaliza los embeddings de los documentos de Distancia Euclidiana a Cosine Similarity
        faiss.normalize_L2(self.doc_embeddings)

        # Dimensión de los embeddings
        self.dimension = self.doc_embeddings.shape[1]

        # Intentar cargar el índice FAISS desde el disco
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                self.logger.info("Index FAISS succesfully load.")
            except Exception as e:
                self.logger.error(f"Error loading index  FAISS: {e}")
                self.logger.info("Building a new FAISS.")
                self._build_and_save_index()
        else:
            self.logger.info("Índice FAISS not found. Building a new FAISS Index.")
            self._build_and_save_index()

    def _build_and_save_index(self):
        """Construye el índice FAISS y lo guarda en disco."""
        # Configurar el índice FAISS con IndexHNSWFlat
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 conexiones por nodo
        self.index.hnsw.efConstruction = 40  # Puedes ajustar este valor según tus necesidades

        # Añadir los embeddings al índice FAISS
        self.index.add(self.doc_embeddings)
        self.logger.info(f"Total de vectores en el índice: {self.index.ntotal}")

        # Guardar el índice en disco
        faiss.write_index(self.index, self.index_path)
        self.logger.info(f"Índice FAISS guardado en {self.index_path}.")

    def retrieve(self, query: str, top_k: int = 1, documents: list = None, logger: setup_logger = None):

        # Genera embedding para la consulta
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Normaliza los embeddings de la pregunta de Distancia Euclideana a Cosine Similarity
        faiss.normalize_L2(query_embedding)

        # Realiza la búsqueda en el índice FAISS
        distances, indices = self.index.search(query_embedding, top_k)

        # Devuelve los documentos correspondientes a los índices más cercanos
        results = [(self.documents[idx]["title"], self.documents[idx]["content"], distances[0][i]) for i, idx in
                   enumerate(indices[0])]

        # Validar la búsqueda usando similitud coseno sin FAISS
        self.validate_retrieval(query_embedding, top_k)

        return results

    def validate_retrieval(self, query_embedding, top_k: int = 1):

        similarities = cosine_similarity(query_embedding, self.doc_embeddings).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        self.logger.info("\n=== Validation Results (Cosine Similarity) ===")
        for i, idx in enumerate(top_indices):
            self.logger.info(f"Rank {i + 1}:")
            self.logger.info(f"Title: {self.documents[idx]['title']}")
            self.logger.info(f"Content: {self.documents[idx]['content'][:200]}...")
            self.logger.info(f"Cosine Similarity Score: {similarities[idx]}")
            self.logger.info("==========================================")

    @still_not_in_use
    def add_documents(self, new_documents: list):
        """Añade nuevos documentos al índice FAISS y los guarda en disco."""
        # Añadir nuevos documentos a la lista existente
        self.documents.extend(new_documents)

        # Generar embeddings para los nuevos documentos
        new_doc_contents = [doc["content"] for doc in new_documents]
        new_doc_embeddings = self.model.encode(new_doc_contents, convert_to_numpy=True).astype('float32')

        # Normalizar los nuevos embeddings
        faiss.normalize_L2(new_doc_embeddings)

        # Añadir los nuevos embeddings al índice FAISS
        self.index.add(new_doc_embeddings)
        self.logger.info(f"Se añadieron {len(new_documents)} nuevos vectores al índice.")

        # Actualizar los embeddings completos (opcional, si se necesita para validación)
        self.doc_embeddings = np.vstack((self.doc_embeddings, new_doc_embeddings))

        # Guardar el índice actualizado en disco
        faiss.write_index(self.index, self.index_path)
        self.logger.info(f"Índice FAISS actualizado y guardado en {self.index_path}.")

