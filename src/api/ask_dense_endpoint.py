from fastapi import APIRouter, Depends, HTTPException
from src.application.ask_question_dense_use_case import AskQuestionDenseUseCase
from src.containers.containers_dense_1 import DenseContainer1
from src.data_models.clustering_method import ClusteringMethod
from src.data_models.query_request_clustering import QueryRequestWithClustering

router = APIRouter()

def get_ask_question_dense_use_case() -> AskQuestionDenseUseCase:
    return DenseContainer1.ask_question_dense_use_case()

@router.post("/ask-dense-clustering",
             summary="Haz una pregunta con recuperación densa",
             tags=["rescue-dense all-mpnet-base-v2 embeddings", "Contextual_generation"] )
async def ask_question_dense(
    request: QueryRequestWithClustering,
    self_retrieval_augmented_generation: bool = False,
    use_case: AskQuestionDenseUseCase = Depends(get_ask_question_dense_use_case)
):
    '''
    ### **Technology** documents indexed using **all-mpnet-base-v2 embeddings** and preproceded files by LDA and KMEANS clustering

    ### **PARAM REQUEST**
    Question to be asked to LLM model IA generator

    ### **TYPE SELECTOR**
    Clustering method to be used, you can choose between **LDA** and **KMEANS** in lowercase

    ### **RETURN**
    Response from LLM model IA generator in JSON format
    ```json
    {
     "response": " Los usuarios de HistoriaCard pueden convertir sus puntos en efectivo, descuentos en compras, vuelos y hoteles, y experiencias únicas como eventos culturales, conciertos y cenas gourmet en restaurantes de alta categoría."
     }
    ```
    Please feel free to test pushing **Try it out** button and after click on blue button to see the results

    you can change the question and write it in any languaje with characters from alphabet

    '''
    type_selector = request.type_selector
    if self_retrieval_augmented_generation:
        return use_case.self_retrieval_augmented_generation(request.question, type_selector)

    if type_selector == ClusteringMethod.lda:
        response = use_case.execute_with_lda(request.question)
    elif type_selector == ClusteringMethod.kmeans:
        response = use_case.execute_with_kmeans(request.question)
    else:
        raise HTTPException(status_code=400, detail="Método de clustering no soportado.")

    if not response:
        raise HTTPException(status_code=404, detail="No se encontró una respuesta relevante.")
    return {"response": response}
