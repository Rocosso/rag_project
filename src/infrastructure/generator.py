import openai
from src.config import OPENAI_API_KEY
from src.domain.generator_interface import GeneratorInterface

class Generator(GeneratorInterface):
    def __init__(self):
        openai.api_key = OPENAI_API_KEY

    def generate(self, context: str) -> str:
        prompt = (
            "Given the following information from documents, answer the question based on the content provided.\n\n"
            + context + "\n\nAnswer the question based on the above content.  the answer should be in the language of "
            + "the question."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response['choices'][0]['message']['content'].strip()