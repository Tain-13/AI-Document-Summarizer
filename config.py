import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '.env')


def _int(name, default):
    """Read a positive whole number from .env . Bad or missing values use the default."""

    try:
        value = int(os.getenv(name, default))
    except ValueError:
        return default
    return value if value > 0 else default


MODEL = os.getenv('OPENAI_MODEL')
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
TIMEOUT = _int("TIMEOUT", 120)

CHUNK_SIZE = _int('CHUNK_SIZE',  3000)
CHUNK_OVERLAP = _int('CHUNK_OVERLAP', 300)
MAX_CHUNKS = _int('MAX_CHUNKS', 30)
MAX_CONCURRENCY = _int('MAX_CONCURRENCY', 3)
REDUCE_LIMIT = _int('REDUCE_LIMIT', 12000)

if CHUNK_OVERLAP >= CHUNK_SIZE:
    CHUNK_OVERLAP = CHUNK_SIZE // 10
