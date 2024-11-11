from abc import ABC, abstractmethod

class GeneratorInterface(ABC):
    @abstractmethod
    def generate(self, context: str) -> str:
        pass
