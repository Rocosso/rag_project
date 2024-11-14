from dependency_injector import containers, providers

from src.settings.logging_config import setup_logger
from src.infrastructure.generator import Generator
from src.infrastructure.tfidf_retriever import TFIDFRetriever
from src.application.ask_question_use_case import AskQuestionUseCase
from src.commons.load_documents import load_documents_from_directory


# Cargar documentos desde la carpeta `data`
documents = load_documents_from_directory("data/Handler_separator")

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    retriever = providers.Singleton(TFIDFRetriever, documents=documents, logger=setup_logger())
    generator = providers.Singleton(Generator, logger=setup_logger())
    ask_question_use_case = providers.Factory(AskQuestionUseCase, retriever=retriever, generator=generator)
