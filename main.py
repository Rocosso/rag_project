from fastapi import FastAPI
from src.api.ask_endpoint import router as ask_router
from src.api.ask_dense_endpoint import router as ask_dense_router
from src.api.ask_llama_endpoint import router as llama_router
from src.containers.containers import Container
from src.containers.containers_dense_1 import DenseContainer1
from src.containers.container_llama import ContainerLLama

# Configuración de la aplicación FastAPI
app = FastAPI(
    title="RAG-Based AI Response System",
    description="API for a Retrieval-Augmented Generation (RAG) AI Response System using FastAPI, Dependency Injector, and OpenAI.",
    version="1.0.0",
    docs_url="/docs",       # URL de la documentación de Swagger
    redoc_url="/redoc"      # URL de la documentación de ReDoc
)

# Inicializar los contenedores de dependencias
container = Container()
dense_container = DenseContainer1()
llama_container = ContainerLLama()
app.container = container

# Incluir los routers
app.include_router(ask_router)
app.include_router(ask_dense_router)
app.include_router(llama_router)
