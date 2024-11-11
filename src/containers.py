from dependency_injector import containers, providers
from src.infrastructure.generator import Generator
from src.infrastructure.tfidf_retriever import TFIDFRetriever
from src.application.ask_question_use_case import AskQuestionUseCase


# Función para cargar documentos desde el archivo
def load_documents(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        documents = [line.strip() for line in file if line.strip()]  # Carga cada línea como un documento
        print("Loaded Documents:", documents)  # Verificar que el contenido sea correcto
    return documents

# Cargar los documentos desde el archivo
documents = load_documents("data/documents.txt")

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    retriever = providers.Singleton(TFIDFRetriever, documents=documents)
    generator = providers.Singleton(Generator)
    ask_question_use_case = providers.Factory(AskQuestionUseCase, retriever=retriever, generator=generator)
