import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings
from configparser import ConfigParser
from dotenv import load_dotenv

load_dotenv()

# Read config.ini
config = ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '../config/config.ini'))

class Settings(BaseSettings):
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    UPLOAD_DIR: Path = Path(config.get('DEFAULT', 'UPLOAD_DIR', fallback='uploads'))
    CHROMA_DB_PATH: Path = Path(config.get('DEFAULT', 'CHROMA_DB_PATH', fallback='chroma_db'))
    
    # File processing
    MAX_CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = config.getint('DEFAULT', 'CHUNK_OVERLAP', fallback=250)
    MAX_FILE_SIZE: int = config.getint('DEFAULT', 'MAX_FILE_SIZE', fallback=52428800)
    
    # GROQ
    GROQ_API_KEY: str = os.getenv('GROQ_API_KEY', '')
    GROQ_MODEL: str = config.get('GROQ', 'MODEL_NAME', fallback='qwen/qwen3-32b')
    GROQ_MAX_TOKENS: int = config.getint('GROQ', 'MAX_TOKENS', fallback=800)
    GROQ_TEMPERATURE: float = config.getfloat('GROQ', 'TEMPERATURE', fallback=0.3)

    # Vector DB
    EMBEDDING_MODEL: str = config.get('VectorDB', 'EMBEDDING_MODEL', fallback='all-MiniLM-L6-v2')
    BATCH_SIZE: int = config.getint('VectorDB', 'BATCH_SIZE', fallback=32)
    
    # Legacy support
    CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", str(CHROMA_DB_PATH))
    
    class Config:
        env_file = ".env"
        extra = "allow"  # Allow extra fields from environment variables

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
