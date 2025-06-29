from pydantic import BaseModel
from typing import Optional
from enum import Enum

class PromptGenerationMode(str, Enum):
    """Enum for different prompt generation modes"""
    ANALYSIS_TO_VIETNAMESE = "analysis_to_vietnamese"  # Option 1: Poem analysis -> Local Llama -> Vietnamese prompt
    DIRECT_PROMPT = "direct_prompt"                    # Option 2: Extract prompt directly from analysis
    ANALYSIS_TO_ENGLISH = "analysis_to_english"        # Option 3: Poem analysis -> Local Llama -> English prompt

class PoemRequest(BaseModel):
    """Request model for poem animation generation"""
    poem: str
    output_filename: Optional[str] = None
    prompt_generation_mode: PromptGenerationMode = PromptGenerationMode.ANALYSIS_TO_VIETNAMESE

class PoemResponse(BaseModel):
    """Response model for poem animation generation"""
    task_id: str
    status: str
    message: str
    output_path: Optional[str] = None 