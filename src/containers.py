import os
from dependency_injector import containers, providers
from src.infrastructure.generator import Generator
from src.infrastructure.tfidf_retriever import TFIDFRetriever
from src.application.ask_question_use_case import AskQuestionUseCase


# Función para cargar todos los documentos desde archivos individuales
def load_documents_from_directory(directory: str):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read().strip()
                documents.append({"title": filename, "content": content})
    return documents

# Cargar documentos desde la carpeta `data`
documents = load_documents_from_directory("data")

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    retriever = providers.Singleton(TFIDFRetriever, documents=documents)
    generator = providers.Singleton(Generator)
    ask_question_use_case = providers.Factory(AskQuestionUseCase, retriever=retriever, generator=generator)
