# Model API Package

This package contains all the model-related files for the Vietnamese Poem to Image Animation API, organized in a clean, modular structure for easy deployment and management.

## 📁 Clean File Structure

```
model_api/
├── __init__.py                 # Package initialization and exports
├── main.py                     # FastAPI application entry point
├── models/                     # All model components
│   ├── __init__.py
│   ├── phoclip_embedding.py    # PhoCLIP text encoder
│   ├── poem_analyzer.py        # Poem analysis using Llama3.2
│   ├── prompt_generator.py     # Prompt generation from analysis
│   └── diffusion_generator.py  # AnimateDiff image generation
├── pipeline/                   # Main pipeline orchestration
│   ├── __init__.py
│   └── poem_to_image_pipeline.py
├── schemas/                    # Pydantic models for API
│   ├── __init__.py
│   └── api_schemas.py         # Request/Response models
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── test_api.py                 # Test script
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

# Generate animation from poem
response = requests.post("http://localhost:8000/generate", json={
    "poem": "đẩy hoa dun lá khỏi tay trời",
    "output_filename": "my_animation"
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

## 🔧 Module Components

### Models (`models/`)
- **PhoCLIPEmbedding**: Vietnamese language text encoder
- **PoemAnalyzer**: Analyzes Vietnamese poems using Llama3.2
- **PromptGenerator**: Converts poem analysis to image generation prompts
- **DiffusionGenerator**: Uses AnimateDiff for animation generation

### Pipeline (`pipeline/`)
- **PoemToImagePipeline**: Orchestrates all components and handles the complete workflow

### Schemas (`schemas/`)
- **PoemRequest**: API request model for poem input
- **PoemResponse**: API response model for task status

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

# Test the API endpoints
curl http://localhost:8000/health
```

## 📝 Architecture Benefits

- **Modular Design**: Each component is in its own folder
- **Clean Separation**: Models, pipeline, and schemas are separated
- **Easy Maintenance**: Clear structure makes updates simple
- **Scalable**: Easy to add new models or components
- **Self-contained**: All dependencies and documentation included
- **API Ready**: Includes all Pydantic schemas for FastAPI

## 🔗 Dependencies

All models are loaded from Hugging Face Hub:
- PhoCLIP: `kienhoang123/ViCLIP`
- Llama3.2: `kienhoang123/Llama3.2_Poem_Analysis`
- AnimateDiff: `guoyww/animatediff-motion-adapter-v1-5-2`
- Realistic Vision: `SG161222/Realistic_Vision_V5.1_noVAE` 