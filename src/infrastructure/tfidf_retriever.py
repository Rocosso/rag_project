from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.domain.retriever_interface import RetrieverInterface
import numpy as np


class TFIDFRetriever(RetrieverInterface):
    def __init__(self, documents: list):
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(documents)

    def retrieve(self, query: str, top_k: int = 3) -> list:
        query_vec = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = np.argsort(cosine_similarities)[-top_k:][::-1]
        return [(self.documents[i], cosine_similarities[i]) for i in top_indices]
