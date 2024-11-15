from pydantic import BaseModel, Field


# Modelo de solicitud
class QueryRequest(BaseModel):
    question: str = Field(..., example="¿Qué recompensas pueden obtener los clientes con la tarjeta?")
    session_id: str = Field(..., example="1234")
