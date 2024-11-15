import pytest
from httpx import AsyncClient
from fastapi import status

from main import app  # Importa tu aplicación FastAPI

# Si tienes una configuración especial en tu aplicación, ajústala aquí

@pytest.mark.asyncio
async def test_ask_endpoint_success():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/ask",
            json={
                "question": "¿Cuáles son las opciones de redención de puntos disponibles?"
            }
        )
    assert response.status_code == 200
    response_json = response.json()
    assert isinstance(response_json, dict)
    assert "message" in response_json
    assert response_json["response"] == ("Las opciones de redención de puntos disponibles son las siguientes:\n"
                                        "- Viajes: Puedes canjear tus puntos por vuelos, hoteles, "
                                        "alquiler de autos y paquetes vacacionales.\n"
                                        "- Productos: También puedes intercambiar tus puntos por una amplia variedad "
                                        "de productos, desde electrónica hasta artículos para el hogar.\n"
                                        "- Gift cards: Otra opción es canjear tus puntos por tarjetas de regalo de "
                                        "tiendas populares.\n- Donaciones: Si prefieres apoyar una causa benéfica, "
                                        "puedes utilizar tus puntos para realizar donaciones a organizaciones "
                                        "sin fines de lucro.")


@pytest.mark.asyncio
async def test_ask_endpoint_missing_question():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/ask", json={})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

@pytest.mark.asyncio
async def test_ask_endpoint_invalid_payload():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/ask", json={"question": 123})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

@pytest.mark.asyncio
async def test_ask_endpoint_empty_question():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/ask", json={"question": ""})
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert data["detail"] == "La pregunta no puede estar vacía."
