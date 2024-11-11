from src.domain.generator_interface import GeneratorInterface
from src.domain.retriever_interface import RetrieverInterface

class AskQuestionUseCase:
    def __init__(self, retriever: RetrieverInterface, generator: GeneratorInterface):
        self.retriever = retriever
        self.generator = generator

    def execute(self, query: str) -> str:
        retrieved_docs = self.retriever.retrieve(query)
        context = " ".join([doc[0] for doc in retrieved_docs])
        return self.generator.generate(context)
