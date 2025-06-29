# Model API - Clean Modular Structure Overview

## 🎯 Final Clean Structure

```
model_api/
├── __init__.py                 # Main package exports
├── main.py                     # FastAPI application entry point
├── models/                     # All model components
│   ├── __init__.py            # Models package exports
│   ├── phoclip_embedding.py   # Vietnamese text encoder
│   ├── poem_analyzer.py       # Llama3.2 poem analysis
│   ├── prompt_generator.py    # Prompt generation
│   └── diffusion_generator.py # AnimateDiff generation
├── pipeline/                   # Main orchestration
│   ├── __init__.py            # Pipeline package exports
│   └── poem_to_image_pipeline.py # Main pipeline logic
├── schemas/                    # API data models
│   ├── __init__.py            # Schemas package exports
│   └── api_schemas.py         # All Pydantic BaseModel classes
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── test_api.py                 # Package testing script
├── README.md                   # Usage documentation
├── SUMMARY.md                  # Reorganization summary
└── STRUCTURE_OVERVIEW.md       # This file
```

## 🔧 Module Responsibilities

### Models (`models/`)
**Purpose**: Individual AI model components
- **PhoCLIPEmbedding**: Vietnamese language understanding
- **PoemAnalyzer**: Poem analysis and interpretation
- **PromptGenerator**: Text-to-image prompt creation
- **DiffusionGenerator**: Animation generation

### Pipeline (`pipeline/`)
**Purpose**: Orchestration and workflow management
- **PoemToImagePipeline**: Coordinates all models and handles the complete workflow

### Schemas (`schemas/`)
**Purpose**: API data validation and serialization
- **PoemRequest**: Input validation for poem requests
- **PoemResponse**: Output formatting for API responses

## 🚀 How to Use

### 1. Start the API
```bash
cd model_api
python main.py
```

### 2. Access the API
- **API Base**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

### 3. Test the Package
```bash
python test_api.py
```

## 📦 Import Examples

```python
# Import from main package
from model_api import PoemToImagePipeline, PhoCLIPEmbedding

# Import specific modules
from model_api.models import PhoCLIPEmbedding, PoemAnalyzer
from model_api.pipeline import PoemToImagePipeline
from model_api.schemas import PoemRequest, PoemResponse

# Import individual components
from models import DiffusionGenerator
from schemas import PoemRequest
```

## 🎨 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Detailed health status |
| POST | `/generate` | Generate animation from poem |
| GET | `/status/{task_id}` | Check task status |
| GET | `/download/{task_id}` | Download generated animation |
| GET | `/tasks` | List all tasks |

## 🔄 Workflow

1. **Client** sends poem to `/generate`
2. **API** creates background task
3. **Pipeline** orchestrates models:
   - PoemAnalyzer → analyzes poem
   - PromptGenerator → creates image prompt
   - PhoCLIPEmbedding → encodes text
   - DiffusionGenerator → creates animation
4. **Client** polls `/status/{task_id}`
5. **Client** downloads from `/download/{task_id}`

## 🏗️ Architecture Benefits

- ✅ **Modular**: Each component in its own folder
- ✅ **Maintainable**: Easy to find and update code
- ✅ **Scalable**: Simple to add new models/schemas
- ✅ **Clean**: Clear separation of concerns
- ✅ **Professional**: Follows Python best practices
- ✅ **Self-contained**: All dependencies included
- ✅ **API-ready**: Proper FastAPI integration

## 🔗 Dependencies

### External Models (Hugging Face)
- PhoCLIP: `kienhoang123/ViCLIP`
- Llama3.2: `kienhoang123/Llama3.2_Poem_Analysis`
- AnimateDiff: `guoyww/animatediff-motion-adapter-v1-5-2`
- Realistic Vision: `SG161222/Realistic_Vision_V5.1_noVAE`

### Python Packages
- FastAPI, Uvicorn (API framework)
- PyTorch, Transformers (ML models)
- Diffusers (diffusion models)
- Pydantic (data validation)

## 🐳 Deployment

### Docker
```bash
docker build -t model-api .
docker run -p 8000:8000 model-api
```

### Local
```bash
pip install -r requirements.txt
python main.py
```

## 📝 Notes

- GPU acceleration recommended for performance
- Background task processing prevents API blocking
- Fallback mechanisms for robustness
- Interactive API documentation at `/docs`
- All models loaded from Hugging Face Hub 