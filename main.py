from fastapi import FastAPI
from src.api.ask_endpoint import router
from src.containers import Container


# Configuración de la aplicación FastAPI
app = FastAPI(
    title="RAG-Based AI Response System",
    description="API for a Retrieval-Augmented Generation (RAG) AI Response System using FastAPI, Dependency Injector, and OpenAI.",
    version="1.0.0",
    docs_url="/docs",       # URL de la documentación de Swagger
    redoc_url="/redoc"      # URL de la documentación de ReDoc
)

# Inicializa la aplicación FastAPI y el contenedor
container = Container()
app.container = container

# Incluye el router del endpoint
app.include_router(router)
