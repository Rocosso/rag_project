import requests

from src.domain.generator_llama_interface import GeneratorLLamaInterface


class LlamaGenerator(GeneratorLLamaInterface):
    def __init__(self, base_url: str):
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta-llama/Llama-3.1-1B",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        try:
            response = requests.post(f"{self.base_url}/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling Llama Generator: {e}")
