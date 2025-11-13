"""LLM client for the application status agent."""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# LLM timeout and retry configuration
DEFAULT_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))  # 120 seconds default
DEFAULT_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))  # 3 retries default


def _create_llm(temperature: float = 0) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance with configured timeout and retries.
    
    Args:
        temperature: Controls randomness (0=deterministic, higher=more creative)
    
    Returns:
        Configured ChatOpenAI instance
    """
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=temperature,
        request_timeout=DEFAULT_TIMEOUT,
        max_retries=DEFAULT_MAX_RETRIES
    )


def planner_llm():
    """
    LLM for planning with structured output.
    Uses temperature=0 for deterministic, consistent JSON generation.
    """
    return _create_llm(temperature=0)


def drafter_llm():
    """
    LLM for drafting responses.
    Uses temperature=0.2 for slightly more natural, varied language.
    """
    return _create_llm(temperature=0.2)
