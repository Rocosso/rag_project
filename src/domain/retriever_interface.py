from abc import ABC, abstractmethod

from src.settings.logging_config import setup_logger


class RetrieverInterface(ABC):
    @abstractmethod
    def retrieve(self, documents: list, logger: setup_logger) -> list:
        pass
