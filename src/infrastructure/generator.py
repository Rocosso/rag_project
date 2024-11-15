import openai
import logging
from src.config import OPENAI_API_KEY
from src.domain.generator_interface import GeneratorInterface

class Generator(GeneratorInterface):
    def __init__(self, logger=None):
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        openai.api_key = OPENAI_API_KEY


    def generate(self, context: str, question: str, conversation_history: str) -> str:
        prompt = (
                "Eres un asistente virtual experto. Utiliza la siguiente información de los documentos y la pregunta "
                "proporcionada para dar una respuesta precisa y coherente, basándote únicamente en el contenido "
                "proporcionado. No agregues información que no esté en el contexto. Evita crear nuevas preguntas en "
                "tu respuesta y limita tu respuesta a la pregunta realizada.\n\n"
                "Contexto:\n" + context + "\n\n"
                "Historial de la conversación:\n" + conversation_history + "\n\n"
                "Usuario: " + question + "\n\n"
                "Responde en el idioma de la pregunta."
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            self.logger.error(F"Error calling LLM Api:  {e}")
