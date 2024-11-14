import os
from dependency_injector import containers, providers

from src.commons.load_documents import load_documents_from_directory
from src.settings.logging_config import setup_logger
from src.infrastructure.generator import Generator
from src.infrastructure.dense_retriever_clustering import DenseRetrieverClustering
from src.application.ask_question_dense_use_case import AskQuestionDenseUseCase


documents = load_documents_from_directory("data/LDA_separator")
class DenseContainer1(containers.DeclarativeContainer):
    config = providers.Configuration()
    logger = providers.Singleton(setup_logger)
    retriever = providers.Singleton(DenseRetrieverClustering,
                                    documents=documents,
                                    logger=logger)
    generator = providers.Singleton(Generator, logger=logger)
    ask_question_dense_use_case = providers.Factory(AskQuestionDenseUseCase,
                                                    retriever=retriever,
                                                    generator=generator,
                                                    logger=logger)
