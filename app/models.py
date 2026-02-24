from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = "default"