from pydantic import BaseModel
from typing import Dict

class TextChunk(BaseModel):
    content: str
    meta: Dict[str, str]
