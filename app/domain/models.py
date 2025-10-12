from pydantic import BaseModel
from typing import Dict, List, Optional, Any


class PredictRequest(BaseModel):
    texto: str


class PredictResponse(BaseModel):
    status: str
    resultado: Dict[str, Any]
    error: Optional[str] = None