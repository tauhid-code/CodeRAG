
from dotenv import load_dotenv

from app.generation.llm_client import GeminiLLMClient


def main():
    # Load the API key from .env
    load_dotenv()

    # Initialize Gemini
    llm = GeminiLLMClient()

    # Sample repository context
    context = """
    File: backend/auth.py
    Symbol: create_access_token
    Lines: 10-15

    def create_access_token(data):
        return jwt.encode(data, SECRET_KEY)
    """

    # Ask a question about the code
    query = "What does create_access_token do?"

    # Generate an answer
    answer = llm.generate_answer(
        query=query,
        context=context,
    )

    print("\nGenerated answer:")
    print(answer)

    print("\nLLM test passed!")


if __name__ == "__main__":
    main()