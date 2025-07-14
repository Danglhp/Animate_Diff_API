from pydantic import BaseModel
from typing import Optional
from enum import Enum

class TextEncoderType(str, Enum):
    """Enum for different text encoder types"""
    PHOCLIP = "phoclip"  # Use PhoCLIP for Vietnamese text encoding
    BASE = "base"        # Use base model text encoder

class PromptGenerationMode(str, Enum):
    """Enum for different prompt generation modes"""
    ANALYSIS_TO_VIETNAMESE = "analysis_to_vietnamese"  # Option 1: Poem analysis -> Local Llama -> Vietnamese prompt
    DIRECT_PROMPT = "direct_prompt"                    # Option 2: Extract prompt directly from analysis
    ANALYSIS_TO_ENGLISH = "analysis_to_english"        # Option 3: Poem analysis -> Local Llama -> English prompt

class NegativePromptCategory(str, Enum):
    """Enum for different negative prompt categories"""
    GENERAL = "general"                    # General quality issues
    ARTISTIC = "artistic"                  # Artistic style issues
    TECHNICAL = "technical"                # Technical quality issues
    CONTENT = "content"                    # Content-specific issues
    CUSTOM = "custom"                      # Custom negative prompt

class PoemRequest(BaseModel):
    """Request model for poem animation generation"""
    poem: str
    output_filename: Optional[str] = None
    text_encoder: TextEncoderType = TextEncoderType.PHOCLIP
    prompt_generation_mode: PromptGenerationMode = PromptGenerationMode.ANALYSIS_TO_VIETNAMESE
    negative_prompt_category: NegativePromptCategory = NegativePromptCategory.GENERAL
    custom_negative_prompt: Optional[str] = None

class PoemResponse(BaseModel):
    """Response model for poem animation generation"""
    task_id: str
    status: str
    message: str
    output_path: Optional[str] = None 