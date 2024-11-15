from venv import logger

from src.domain.generator_interface import GeneratorInterface
from src.domain.retriever_interface import RetrieverInterface
from src.settings.logging_config import setup_logger


class AskQuestionUseCase:
    def __init__(self, retriever: RetrieverInterface, generator: GeneratorInterface):
        self.retriever = retriever
        self.generator = generator
        self.logger = setup_logger()

    def execute(self, query: str) -> str:
        # Recupera los documentos relevantes
        retrieved_docs = self.retriever.retrieve(query)
        self.logger.info(f"Retrieved Documents: {retrieved_docs}")  # Verificar documentos recuperados

        # Combina el contenido de los documentos recuperados como contexto
        context = " ".join(retrieved_docs)
        self.logger.info(F"Context for Generation: {context}")  # Verificar contexto antes de generar respuesta

        try:
            # Genera la respuesta utilizando el contexto
            llm_response = self.generator.generate(context=context, question=query)
            if llm_response is None or llm_response == "":
                raise ValueError("LLM response failed.")

            self.logger.info("LLM Response success")
            return llm_response
        except ValueError as error:
            self.logger.error(F"Error calling LLM Api:  {error}")
        except EnvironmentError as error:
            self.logger.error(F"Error in environment calling LLM Api:  {error}")
