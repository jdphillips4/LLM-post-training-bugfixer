import time
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from bugfix import config

_ROLES = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}


def get_chat_model(model_name=None, base_url=None, api_key=None,
                   temperature=0.0, max_tokens=1024):
    """Any OpenAI-compatible endpoint: HF router, vLLM, Ollama, ..."""
    return ChatOpenAI(
        model=model_name or config.MODEL_NAME,
        base_url=base_url or config.BASE_URL,
        api_key=api_key or config.API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=180,
        max_retries=3,
    )


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


def chat(llm, messages: list[dict]) -> LLMResult:
    """messages are plain dicts: the same format used for training data."""
    lc = [_ROLES[m["role"]](content=m["content"]) for m in messages]
    t0 = time.perf_counter()
    out = llm.invoke(lc)
    dt = time.perf_counter() - t0
    usage = out.usage_metadata or {}
    return LLMResult(out.content, usage.get("input_tokens", 0),
                     usage.get("output_tokens", 0), dt)