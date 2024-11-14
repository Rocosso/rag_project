from abc import ABC, abstractmethod

from src.settings.logging_config import setup_logger


class GeneratorInterface(ABC):
    @abstractmethod
    def generate(self, context: str, logger: setup_logger) -> str:
        pass
