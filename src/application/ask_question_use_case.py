from src.domain.generator_interface import GeneratorInterface
from src.domain.retriever_interface import RetrieverInterface

class AskQuestionUseCase:
    def __init__(self, retriever: RetrieverInterface, generator: GeneratorInterface):
        self.retriever = retriever
        self.generator = generator

    def execute(self, query: str) -> str:
        # Recupera los documentos relevantes
        retrieved_docs = self.retriever.retrieve(query)
        print("Retrieved Documents:", retrieved_docs)  # Verificar documentos recuperados

        # Combina el contenido de los documentos recuperados como contexto
        context = " ".join(retrieved_docs)
        print("Context for Generation:", context)  # Verificar contexto antes de generar respuesta

        # Genera la respuesta utilizando el contexto
        response = self.generator.generate(context)
        return response