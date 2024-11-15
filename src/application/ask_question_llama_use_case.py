from src.domain.generator_llama_interface import GeneratorLLamaInterface
from src.domain.retriever_interface import RetrieverInterface
from src.settings.logging_config import setup_logger


class AskQuestionLlamaUseCase:
    def __init__(self, retriever: RetrieverInterface, generator: GeneratorLLamaInterface):
        self.retriever = retriever
        self.generator = generator
        self.logger = setup_logger()

    def execute(self, question: str) -> str:
        # Recupera los documentos relevantes
        retrieved_docs = self.retriever.retrieve(question)
        self.logger.info(f"Retrieved Documents: {retrieved_docs}")  # Verificar documentos recuperados

        # Combina el contenido de los documentos recuperados como contexto
        context = " ".join(retrieved_docs)
        self.logger.info(F"Context for Generation: {context}")  # Verificar contexto antes de generar respuesta

        # Construye el prompt
        prompt = self._build_prompt(question, context)

        # Genera la respuesta utilizando el contexto
        response = self.generator.generate(prompt=prompt)
        return response

    def _build_prompt(self, question: str, documents: str) -> str:
        # Combina la pregunta con los documentos para crear el prompt
        context = "\n".join(documents)
        return f"Contexto:\n{context}\n\nPregunta:\n{question}"