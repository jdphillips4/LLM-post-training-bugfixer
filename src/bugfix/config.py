import os

from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("HF_TOKEN") or "EMPTY"
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")