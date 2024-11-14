from pydantic import BaseModel, Field

from src.data_models.clustering_method import ClusteringMethod


class QueryRequestWithClustering(BaseModel):
    type_selector: ClusteringMethod = Field(..., description="Selecciona el método de clustering")
    question: str = Field(..., example="¿Qué recompensas pueden obtener los clientes con la tarjeta?")
