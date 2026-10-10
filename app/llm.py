"""
LLM factory module.

Centralizes all model creation and configuration.
Makes it easy to swap providers without changing agent code.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables once at module import time
load_dotenv()


def _validate_required_env(var_name: str) -> str:
    """
    Load a required environment variable.
    Fail fast with a clear message if missing.
    """
    value = os.getenv(var_name)
    if not value:
        raise ValueError(
            f"Missing required environment variable: {var_name}. "
            f"Copy .env.example to .env and fill in your credentials."
        )
    return value

    
def get_llm(
    temperature: float = 0,
    timeout: int = 30,
    max_retries: int = 2,
) -> ChatGroq:
    """
    Create and return a ChatGroq instance.

    Args:
        temperature: Lower = more deterministic (0 is default for routing/classification).
        timeout: Seconds to wait for a response before timing out.
        max_retries: Number of retries for transient errors (rate limits, timeouts).

    Returns:
        ChatGroq model configured with API key and model name from .env

    Raises:
        ValueError: If required environment variables are missing.
    """
    api_key = _validate_required_env("GROQ_API_KEY")
    model_name = _validate_required_env("CHAT_MODEL")

    return ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        timeout=timeout,
        max_tokens=250,
        max_retries=max_retries,
    )


if __name__ == "__main__":
    # Quick test: print model info
    llm = get_llm()
    print(f"[OK] LLM created: {llm.model_name}")
    print(f"  API Key: {'*' * 20}...")
    print(f"  Temperature: {llm.temperature}")
