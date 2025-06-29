from pydantic import BaseModel
from typing import Optional

class PoemRequest(BaseModel):
    """Request model for poem animation generation"""
    poem: str
    output_filename: Optional[str] = None

class PoemResponse(BaseModel):
    """Response model for poem animation generation"""
    task_id: str
    status: str
    message: str
    output_path: Optional[str] = None 