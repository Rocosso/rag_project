from pydantic_settings import BaseSettings
from typing import ClassVar
from pydantic import ValidationError
from dotenv import load_dotenv

from src.settings.logging_config import setup_logger


logger = setup_logger()

load_dotenv('.env')


class Settings(BaseSettings):
    llama_generator_base_url: str = "" # URL del servidor vLLM
    huggingface_hub_token: str = ""
    faiss_database_path: str = "/rag_project/src/databases/DocumentDatabase.faiss"

    ConfigDict: ClassVar = {'env_file': '.env', 'env_file_encoding': 'utf-8'}

try:
    app_settings = Settings()

except ValidationError as e:
    logger.error("Validation error occurred while initializing settings.")
    for error in e.errors():
        logger.error(f"Error in field: {error['loc'][0]} - {error['msg']}")
    raise
