import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.settings.logging_config import setup_logger

class DenseRetrieverClustering:
    def __init__(self, documents: list, logger: setup_logger):
        self.documents = documents

        # modelo de embeddings más efectivo
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.logger=logger



    def retrieve(self, query: str, top_k: int = 3, documents: list = None, logger: setup_logger = None):
        # Crear embeddings para todos los documentos
        doc_contents = [doc["content"] for doc in documents]
        self.doc_embeddings = self.model.encode(doc_contents, convert_to_numpy=True)

        # Configura el índice FAISS con IndexHNSWFlat
        d = self.doc_embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(d, 32)  # 64 conexiones por nodo, ajusta según el tamaño del dataset

        # Añade los embeddings al índice FAISS
        self.index.add(self.doc_embeddings)

        # Genera embedding para la consulta
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Realiza la búsqueda en el índice FAISS
        distances, indices = self.index.search(query_embedding, top_k)

        self.logger.info(f"\n\ndistances: {distances}")
        self.logger.info(f"indices: {indices}")
        self.logger.info(f"Total documents: {len(self.documents)}")
        self.logger.info(f"indices[0]: {indices[0]}\n\n\n\n")

        # Devuelve los documentos correspondientes a los índices más cercanos
        results = [(self.documents[idx]["title"], self.documents[idx]["content"], distances[0][i]) for i, idx in
                   enumerate(indices[0])]

        # Validar la búsqueda usando similitud coseno sin FAISS
        self.validate_retrieval(query_embedding, top_k)

        return results

    def validate_retrieval(self, query_embedding, top_k: int = 3):
        similarities = cosine_similarity(query_embedding, self.doc_embeddings).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        self.logger.info("\n=== Validation Results (Cosine Similarity) ===")
        for i, idx in enumerate(top_indices):
            self.logger.info(f"Rank {i + 1}:")
            self.logger.info(f"Title: {self.documents[idx]['title']}")
            self.logger.info(f"Content: {self.documents[idx]['content'][:200]}...")
            self.logger.info(f"Cosine Similarity Score: {similarities[idx]}")
            self.logger.info("==========================================")
