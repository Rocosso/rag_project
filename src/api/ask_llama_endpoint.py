from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from src.application.ask_question_llama_use_case import AskQuestionLlamaUseCase
from src.containers.container_llama import ContainerLLama
from src.data_models.query_models import QueryRequest


router = APIRouter()


@router.post("/ask_llama",
             summary="Haz una pregunta a un modelo LLAMA3.1-1B en esta maquina local",
             tags=["Local_model_LLAMA3.1-1B"] )
def ask_question(
    request: QueryRequest,
    use_case: AskQuestionLlamaUseCase = Depends(lambda: AskQuestionLlamaUseCase(
        generator=ContainerLLama().llama_generator(),
        retriever=ContainerLLama().retriever()
       )
    )
):
    '''
    ### **Technology**

    Documents indexed using **TF-IDF** and preprocessed by specific handlers.

    ---

    ### Request Parameters

    - **Question**: The question to be asked to the Llama LLM model.

    ---

    ### Response

    The Llama LLM model will return a response in JSON format:

    ```json
    {
      "response": "Response generated by the Llama model to the provided question."
    }
    '''
    try:
        response = use_case.execute(request.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
