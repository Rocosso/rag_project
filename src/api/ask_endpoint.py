from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from src.application.ask_question_use_case import AskQuestionUseCase
from src.containers import Container

router = APIRouter()

# Modelo de entrada para la solicitud
class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        example="What are the key benefits of renewable energy?"
    )

# Función de dependencia para obtener el caso de uso
def get_ask_question_use_case() -> AskQuestionUseCase:
    return Container.ask_question_use_case()

# Endpoint POST para recibir preguntas y responder
@router.post("/ask")
async def ask_question(request: QueryRequest, use_case: AskQuestionUseCase = Depends(get_ask_question_use_case)):
    response = use_case.execute(request.question)
    return {"response": response}

