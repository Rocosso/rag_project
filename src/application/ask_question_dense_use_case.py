from src.domain.generator_interface import GeneratorInterface
from src.domain.retriever_interface import RetrieverInterface
from src.commons.load_documents import load_documents_from_directory
from src.settings.logging_config import setup_logger

class AskQuestionDenseUseCase:
    def __init__(self,
                 retriever: RetrieverInterface,
                 generator: GeneratorInterface,
                 logger: setup_logger,
                 ):
        self.retriever = retriever
        self.generator = generator
        self.logger = logger

    def execute(self, query: str, documents: list) -> str:

        # Utiliza DenseRetriever para recuperar los documentos
        retrieved_docs = self.retriever.retrieve(query=query, top_k=1, documents=documents, logger=self.logger)

        # Imprime los documentos recuperados
        self.logger.info("Retrieved Documents (Dense): %s", retrieved_docs)

        # Combina el contenido de los documentos recuperados como contexto
        context = " ".join([doc[1] for doc in retrieved_docs])
        self.logger.info("Context for Dense Generation:", context)

        # Genera la respuesta utilizando el contexto
        response = self.generator.generate(context)
        return response

    def execute_with_lda(self, question: str) -> str:
        self.documents = load_documents_from_directory("data/LDA_separator")
        return self.execute(question, self.documents)

    def execute_with_kmeans(self, question: str) -> str:
        self.documents = load_documents_from_directory("data/Cluster_kmeans_separator")
        return self.execute(question, self.documents)

