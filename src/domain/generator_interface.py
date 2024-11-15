from abc import ABC, abstractmethod


class GeneratorInterface(ABC):
    @abstractmethod
    def generate(self, context: str, question: str, conversation_history: str) -> str:
        pass
