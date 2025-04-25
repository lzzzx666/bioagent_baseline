from abc import ABC, abstractmethod
from openai import OpenAI

class LLMEngine:
    
    def __init__(self, llm_engine_name: str, base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY"):
        self.client = OpenAI(
            max_retries=10,
            timeout=120.0,
            api_key=api_key,
            base_url=base_url
        )
        self.llm_engine_name = llm_engine_name

    def respond(self, user_input, temperature: float = 0.7, top_p: float = 1.0) -> str:
        response = self.client.chat.completions.create( 
            model=self.llm_engine_name,
            messages=user_input,
            temperature=temperature,
            top_p=top_p,
        )
    
        return response.choices[0].message.content
