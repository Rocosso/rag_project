import pytest
from httpx import AsyncClient
from fastapi import status

from main import app


@pytest.mark.asyncio
async def test_ask_dense_clustering_lda():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/ask-dense-clustering",
            json={
                "type_selector": "lda",
                "question": "Cuales son los requisitos para solicitar una tarjeta de credito en linea"
            }
        )
    assert response.status_code == status.HTTP_200_OK


@pytest.mark.asyncio
async def test_ask_dense_clustering_kmeans():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/ask-dense-clustering",
            json={
                "type_selector": "kmeans",
                "question": "¿Qué beneficios obtengo al usar la tarjeta en tiendas asociadas?"
            }
        )
    assert response.status_code == status.HTTP_200_OK



@pytest.mark.asyncio
async def test_ask_dense_clustering_invalid_method():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/ask-dense-clustering",
            json={
                "type_selector": "invalid_method",
                "question": "Pregunta de prueba"
            }
        )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json() == {"detail": "Método de clustering no soportado."}

@pytest.mark.asyncio
async def test_ask_dense_clustering_missing_field():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/ask-dense-clustering",
            json={
                "type_selector": "lda"
                # Falta el campo "question"
            }
        )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
