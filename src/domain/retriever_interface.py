from abc import ABC, abstractmethod

class RetrieverInterface(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> list:
        pass
