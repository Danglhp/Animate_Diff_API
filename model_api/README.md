# Model API Package

This package contains all the model-related files for the Vietnamese Poem to Image Animation API, organized in a clean, modular structure for easy deployment and management.

## 📁 Clean File Structure

```
model_api/
├── __init__.py                 # Package initialization and exports
├── main.py                     # FastAPI application entry point
├── models/                     # All model components (modular structure)
│   ├── __init__.py            # Main models package exports
│   ├── phoclip_embedding/     # PhoCLIP text encoder module
│   │   └── __init__.py        # PhoCLIPEmbedding class
│   ├── poem_analyzer/         # Poem analysis module
│   │   └── __init__.py        # PoemAnalyzer class
│   ├── prompt_generator/      # Prompt generation module
│   │   └── __init__.py        # PromptGenerator class
│   └── diffusion_generator/   # AnimateDiff generation module
│       └── __init__.py        # DiffusionGenerator class
├── pipeline/                   # Main pipeline orchestration
│   ├── __init__.py
│   └── poem_to_image_pipeline.py
├── schemas/                    # Pydantic models for API
│   ├── __init__.py
│   └── api_schemas.py         # Request/Response models
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── test_api.py                 # Test script
├── test_prompt_modes.py        # Prompt modes test script
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### 3. API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /generate` - Generate animation from poem
- `GET /status/{task_id}` - Check task status
- `GET /download/{task_id}` - Download generated animation
- `GET /tasks` - List all tasks

### 4. Interactive API Documentation

Visit `http://localhost:8000/docs` for interactive API testing!

### 5. Example Usage

```python
import requests

# Generate animation from poem with different prompt modes
response = requests.post("http://localhost:8000/generate", json={
    "poem": "đẩy hoa dun lá khỏi tay trời",
    "output_filename": "my_animation",
    "prompt_generation_mode": "analysis_to_vietnamese"  # Choose from 3 modes
})

task_id = response.json()["task_id"]

# Check status
status = requests.get(f"http://localhost:8000/status/{task_id}")

# Download when completed
if status.json()["status"] == "completed":
    animation = requests.get(f"http://localhost:8000/download/{task_id}")
    with open("animation.gif", "wb") as f:
        f.write(animation.content)
```

## 🎯 Prompt Generation Modes

The API now supports **three different prompt generation modes** that users can choose from:

### Mode 1: `analysis_to_vietnamese` (Default)
**Flow**: Poem analysis → Local Llama model → Vietnamese prompt
- Analyzes the poem using Llama3.2
- Extracts emotional tone, metaphors, settings, motion
- Uses local Llama model to generate Vietnamese prompt
- Best for Vietnamese language understanding

### Mode 2: `direct_prompt`
**Flow**: Extract prompt directly from analysis
- Analyzes the poem using Llama3.2
- Extracts the prompt field directly from the analysis
- Trims excess data to fit token limits
- Fastest option, uses analysis output directly

### Mode 3: `analysis_to_english`
**Flow**: Poem analysis → Local Llama model → English prompt
- Analyzes the poem using Llama3.2
- Extracts emotional tone, metaphors, settings, motion
- Uses local Llama model to generate English prompt
- Best for international compatibility

## 🔧 Module Components

### Models (`models/`) - Modular Structure
Each model is now in its own subfolder for better organization:

- **`phoclip_embedding/`**: Vietnamese language text encoder
  - `__init__.py`: Contains `PhoCLIPEmbedding` class
- **`poem_analyzer/`**: Analyzes Vietnamese poems using Llama3.2
  - `__init__.py`: Contains `PoemAnalyzer` class
- **`prompt_generator/`**: Converts poem analysis to image generation prompts (3 modes)
  - `__init__.py`: Contains `PromptGenerator` class
- **`diffusion_generator/`**: Uses AnimateDiff for animation generation
  - `__init__.py`: Contains `DiffusionGenerator` class

### Pipeline (`pipeline/`)
- **PoemToImagePipeline**: Orchestrates all components and handles the complete workflow

### Schemas (`schemas/`)
- **PoemRequest**: API request model for poem input with prompt generation mode
- **PoemResponse**: API response model for task status
- **PromptGenerationMode**: Enum for the three prompt generation modes

## 🐳 Docker Deployment

This package can be easily containerized:

```bash
docker build -t model-api .
docker run -p 8000:8000 model-api
```

## 🧪 Testing

```bash
# Test the package structure
python test_api.py

# Test the three prompt generation modes
python test_prompt_modes.py

# Test the API endpoints
curl http://localhost:8000/health
```

## 📝 Architecture Benefits

- **Modular Design**: Each component is in its own folder with clear separation
- **Clean Structure**: Models, pipeline, and schemas are organized in subfolders
- **Easy Maintenance**: Clear structure makes updates simple
- **Scalable**: Easy to add new models or components
- **Self-contained**: All dependencies and documentation included
- **API Ready**: Includes all Pydantic schemas for FastAPI
- **Flexible Prompt Generation**: Three different modes for different use cases
- **Import Friendly**: All models can be imported directly from `models` package

## 🔗 Dependencies

All models are loaded from Hugging Face Hub:
- PhoCLIP: `kienhoang123/ViCLIP`
- Llama3.2: `kienhoang123/Llama3.2_Poem_Analysis`
- AnimateDiff: `guoyww/animatediff-motion-adapter-v1-5-2`
- Realistic Vision: `SG161222/Realistic_Vision_V5.1_noVAE` 