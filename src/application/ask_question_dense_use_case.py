from distutils.util import execute
from venv import logger

from src.containers.container_llama import documents
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
        self.logger.info(f"Retrieved Documents (Dense):  {retrieved_docs}")

        # Combina el contenido de los documentos recuperados como contexto
        context = " ".join([doc[1] for doc in retrieved_docs])
        self.logger.info(f"Context for Dense Generation: {context}")

        # Genera la respuesta utilizando el contexto
        response = self.generator.generate(context=context, question=query, conversation_history="")
        return response

    def execute_with_lda(self, question: str) -> str:
        self.documents = load_documents_from_directory("data/LDA_separator")
        return self.execute(question, self.documents)

    def execute_with_kmeans(self, question: str) -> str:
        self.documents = load_documents_from_directory("data/Cluster_kmeans_separator")
        return self.execute(question, self.documents)

    def self_retrieval_augmented_generation(self, question: str, type_selector: str) -> str:
        # Validate the type_selector input
        retrieval_methods = {
            "kmeans": self.execute_with_kmeans,
            "lda": self.execute_with_lda
        }
        if type_selector not in retrieval_methods:
            raise ValueError(f"Invalid type_selector '{type_selector}'. Must be 'kmeans' or 'lda'.")

        retrieve_documents = retrieval_methods[type_selector]

        # Initialize the conversation history with the initial question
        conversation_history = [question]

        for _ in range(3):
            current_query = conversation_history[-1]

            # Retrieve documents based on the current query
            documents = retrieve_documents(current_query)

            # Generate a new response using the current query and retrieved documents
            response = self.execute(query=current_query, documents=documents)

            # Append the new response to the conversation history
            conversation_history.append(response)

        # Return the last response generated
        return conversation_history[-1]

