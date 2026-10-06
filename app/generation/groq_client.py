import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class GroqLLMClient:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing from .env")

        self.model = os.getenv(
            "GROQ_MODEL", "openai/gpt-oss-20b"
        )
        self.client = Groq(
            api_key=api_key,
            max_retries=3,
            timeout=60.0,
        )

    def generate_answer(self, query: str, context: str) -> str:
        if not context.strip():
            return (
                "I cannot answer this question because "
                "no relevant code context was retrieved."
            )

        system_prompt = """
You are CodeRAG, an assistant that answers questions
about Python GitHub repositories.

Rules:
1. Answer using only the supplied code context.
2. Do not invent functions, files, or behavior.
3. If the context does not contain enough evidence,
   clearly state that you cannot determine the answer.
4. Explain code behavior accurately and concisely.
5. Refer to source files and line numbers when available.
"""

        user_prompt = f"""
Question:
{query}

Retrieved code context:
{context}

Answer the question using only the context above.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=2048,
        )

        answer = response.choices[0].message.content

        if not answer or not answer.strip():
            raise RuntimeError("Groq returned an empty answer")

        return answer.strip()