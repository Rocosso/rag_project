from fastapi import APIRouter, Depends

from src.data_models.query_models import QueryRequest
from src.application.ask_question_use_case import AskQuestionUseCase
from src.containers.containers import Container


router = APIRouter()

# Función de dependencia para obtener el caso de uso
def get_ask_question_use_case() -> AskQuestionUseCase:
    return Container.ask_question_use_case()

# Endpoint POST para recibir preguntas y responder
@router.post("/ask")
async def ask_question(request: QueryRequest, use_case: AskQuestionUseCase = Depends(get_ask_question_use_case)):
    '''
    **Technology** documents indexed using **TF-IDF** and preproceded by handlers

        PARAM REQUEST
         Question to be asked to LLM model IA generator

        RETURN
        Response from LLM model IA generator in JSON format
        {
         "response": "¿Cuáles son las opciones de redención de puntos disponibles para los usuarios de HistoriaCard? Los usuarios de HistoriaCard pueden convertir sus puntos en efectivo, descuentos en compras, vuelos y hoteles, y experiencias únicas como eventos culturales, conciertos y cenas gourmet en restaurantes de alta categoría."
         }
    '''
    response = use_case.execute(request.question)
    return {"response": response}
