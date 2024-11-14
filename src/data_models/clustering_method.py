from enum import Enum


class ClusteringMethod(str, Enum):
    lda = "lda"
    kmeans = "kmeans"
