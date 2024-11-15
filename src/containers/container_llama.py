from dependency_injector import containers, providers

from src.settings.logging_config import setup_logger
from src.settings.settings import Settings
from src.application.ask_question_llama_use_case import AskQuestionLlamaUseCase
from src.infrastructure.generator_llama import LlamaGenerator
from src.infrastructure.tfidf_retriever import TFIDFRetriever
from src.commons.load_documents import load_documents_from_directory


settings = Settings()
base_url = settings.llama_generator_base_url

# Cargar documentos desde la carpeta `data`
documents = load_documents_from_directory("data/Handler_separator")

class ContainerLLama(containers.DeclarativeContainer):
    config = providers.Configuration()
    retriever = providers.Singleton(TFIDFRetriever, documents=documents, logger=setup_logger())
    llama_generator = providers.Singleton(LlamaGenerator, base_url=base_url)
    ask_question_use_case = providers.Factory(AskQuestionLlamaUseCase, retriever=retriever, generator=llama_generator)
