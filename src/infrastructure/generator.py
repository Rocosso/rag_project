import openai
from src.config import OPENAI_API_KEY
from src.domain.generator_interface import GeneratorInterface

class Generator(GeneratorInterface):
    def __init__(self):
        openai.api_key = OPENAI_API_KEY

    def generate(self, context: str) -> str:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": context}],
            max_tokens=150
        )
        return response['choices'][0]['message']['content'].strip()
