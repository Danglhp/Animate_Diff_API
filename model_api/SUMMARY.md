# Model API Package - Clean Modular Structure

## What Was Reorganized

This `model_api` folder has been reorganized into a clean, modular structure with separate folders for different components and all BaseModel classes consolidated into a single file.

### Original Structure (Before Reorganization):
```
model_api/
├── __init__.py
├── main.py
├── poem_to_image_pipeline.py
├── phoclip_embedding.py
├── poem_analyzer.py
├── prompt_generator.py
├── diffusion_generator.py
├── requirements.txt
├── Dockerfile
├── test_api.py
├── README.md
└── SUMMARY.md
```

### New Clean Structure:
```
model_api/
├── __init__.py                 # Package exports
├── main.py                     # FastAPI app
├── models/                     # All model components
│   ├── __init__.py
│   ├── phoclip_embedding.py    # PhoCLIP text encoder
│   ├── poem_analyzer.py        # Poem analyzer
│   ├── prompt_generator.py     # Prompt generator
│   └── diffusion_generator.py  # Diffusion generator
├── pipeline/                   # Main pipeline orchestration
│   ├── __init__.py
│   └── poem_to_image_pipeline.py
├── schemas/                    # Pydantic models for API
│   ├── __init__.py
│   └── api_schemas.py         # All BaseModel classes
├── requirements.txt            # Dependencies
├── Dockerfile                  # Container configuration
├── test_api.py                 # Test script
├── README.md                   # Usage documentation
└── SUMMARY.md                  # This file
```

## Key Improvements Made

1. **Modular Organization**: 
   - `models/` - Contains all individual model components
   - `pipeline/` - Contains the main orchestration logic
   - `schemas/` - Contains all Pydantic BaseModel classes

2. **Clean Separation of Concerns**:
   - Models are separated from pipeline logic
   - API schemas are isolated in their own module
   - Each component has its own folder with proper `__init__.py`

3. **Consolidated Schemas**: 
   - All Pydantic BaseModel classes moved to `schemas/api_schemas.py`
   - `PoemRequest` and `PoemResponse` models in one place
   - Easy to maintain and extend API schemas

4. **Updated Imports**: 
   - All import statements updated to work with new structure
   - Proper package imports using relative paths
   - Clean dependency management

## Module Breakdown

### Models (`models/`)
- **PhoCLIPEmbedding**: Vietnamese text encoder using PhoBERT
- **PoemAnalyzer**: Llama3.2-based poem analysis
- **PromptGenerator**: Converts analysis to image prompts
- **DiffusionGenerator**: AnimateDiff animation generation

### Pipeline (`pipeline/`)
- **PoemToImagePipeline**: Orchestrates all model components

### Schemas (`schemas/`)
- **PoemRequest**: API request model for poem input
- **PoemResponse**: API response model for task status

## How to Use

### Local Development:
```bash
cd model_api
pip install -r requirements.txt
python main.py
```

### Docker Deployment:
```bash
cd model_api
docker build -t model-api .
docker run -p 8000:8000 model-api
```

### Testing:
```bash
cd model_api
python test_api.py
```

## API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /generate` - Generate animation from poem
- `GET /status/{task_id}` - Check task status
- `GET /download/{task_id}` - Download generated animation
- `GET /tasks` - List all tasks

## Benefits of New Structure

- **Modularity**: Each component is in its own folder
- **Maintainability**: Easy to find and update specific components
- **Scalability**: Simple to add new models or schemas
- **Clean Code**: Clear separation of concerns
- **API Ready**: Proper Pydantic schemas for FastAPI
- **Self-contained**: All dependencies and documentation included
- **Professional**: Follows Python package best practices

## Import Examples

```python
# Import from the package
from model_api import PoemToImagePipeline, PhoCLIPEmbedding

# Import specific modules
from model_api.models import PhoCLIPEmbedding
from model_api.pipeline import PoemToImagePipeline
from model_api.schemas import PoemRequest, PoemResponse
```

## Notes

- All models are loaded from Hugging Face Hub
- GPU acceleration is recommended for performance
- The package supports background task processing
- Fallback mechanisms are included for robustness
- Interactive API docs available at `http://localhost:8000/docs` 