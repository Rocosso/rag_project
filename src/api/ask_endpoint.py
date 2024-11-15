from fastapi import APIRouter, Depends, HTTPException

from src.data_models.query_models import QueryRequest
from src.application.ask_question_use_case import AskQuestionUseCase
from src.containers.containers import Container
from src.infrastructure.conversation_memory import ConversationMemory


router = APIRouter()
conversation_memory = ConversationMemory()

# Función de dependencia para obtener el caso de uso
def get_ask_question_use_case() -> AskQuestionUseCase:
    return Container.ask_question_use_case()

# Endpoint POST para recibir preguntas y responder
@router.post("/ask",
             summary="Haz una pregunta , se rescataran documentos indexados para el contexto semantico",
             tags=["Indexing_TFIDF", "Contextual_generation"] )
async def ask_question(request: QueryRequest, use_case: AskQuestionUseCase = Depends(get_ask_question_use_case)):
    '''
    ### **Technology** documents indexed using **TF-IDF** and preproceded by handlers

    ### **PARAM REQUEST**
     Question to be asked to LLM model IA generator

    ### **RETURN**
    Response from LLM model IA generator in JSON format
    ```json
    {
     "response": "¿Cuáles son las opciones de redención de puntos disponibles para los usuarios de HistoriaCard? Los usuarios de HistoriaCard pueden convertir sus puntos en efectivo, descuentos en compras, vuelos y hoteles, y experiencias únicas como eventos culturales, conciertos y cenas gourmet en restaurantes de alta categoría."
     }
    ```
    '''


    try:
        session_id = request.session_id

        # Obtener el historial de la conversación
        history = conversation_memory.get_history(session_id)

        # Agregar la pregunta actual al historial
        conversation_memory.add_message(session_id, "user", request.question)

        # Ejecutar el caso de uso con el historial
        response = use_case.execute(request.question, history)

        # Agregar la respuesta al historial
        conversation_memory.add_message(session_id, "assistant", response)

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
