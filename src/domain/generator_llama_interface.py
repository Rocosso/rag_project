from abc import ABC, abstractmethod

from src.settings.logging_config import setup_logger

class GeneratorLLamaInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
