# Poem to Image Animation API - Deployment Guide

This guide explains how to deploy the modular poem-to-image animation pipeline using FastAPI and Docker.

## Project Structure

```
├── models/                     # Core model classes
│   ├── __init__.py
│   ├── phoclip_embedding.py    # PhoCLIP text encoder
│   ├── poem_analyzer.py        # Poem analysis using Llama3.2
│   ├── prompt_generator.py     # Prompt generation from analysis
│   └── diffusion_generator.py  # AnimateDiff image generation
├── pipeline/                   # Main pipeline orchestration
│   ├── __init__.py
│   └── poem_to_image_pipeline.py
├── api/                        # FastAPI application
│   └── main.py
├── outputs/                    # Generated animations (created by Docker)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose configuration
└── README_DEPLOYMENT.md        # This file
```

## Prerequisites

1. **Docker and Docker Compose** installed
2. **NVIDIA Docker** support (for GPU acceleration)
3. **CUDA-compatible GPU** (recommended for reasonable performance)
4. **PhoCLIP checkpoint** in `./phoclip_checkpoint/` directory

## Quick Start

### 1. Build and Run with Docker Compose

```bash
# Build and start the services
docker-compose up --build

# Run in background
docker-compose up -d --build
```

### 2. Check Service Status

```bash
# Check if services are running
docker-compose ps

# View logs
docker-compose logs -f poem-to-image-api
```

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Generate animation from poem
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "poem": "đẩy hoa dun lá khỏi tay trời\nnghĩ lại tình duyên luống ngậm ngùi",
       "output_filename": "my_poem"
     }'
```

## API Endpoints

### POST /generate
Generate animation from a Vietnamese poem.

**Request Body:**
```json
{
  "poem": "Your Vietnamese poem here...",
  "output_filename": "optional_filename"
}
```

**Response:**
```json
{
  "task_id": "uuid-string",
  "status": "processing",
  "message": "Animation generation started",
  "output_path": null
}
```

### GET /status/{task_id}
Check the status of a generation task.

### GET /download/{task_id}
Download the generated animation file.

### GET /tasks
List all tasks and their status.

### GET /health
Health check endpoint.

## Manual Docker Build

If you prefer to build manually:

```bash
# Build the image
docker build -t poem-to-image-api .

# Run the container
docker run -d \
  --name poem-to-image-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/phoclip_checkpoint:/app/phoclip_checkpoint \
  poem-to-image-api
```

## Configuration

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: GPU device to use (default: 0)
- `PYTHONPATH`: Python path (set to /app in container)

### Volume Mounts

- `./outputs:/app/outputs`: Generated animations
- `./phoclip_checkpoint:/app/phoclip_checkpoint`: PhoCLIP model files

## Development

### Local Development Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API locally:
```bash
python api/main.py
```

3. Test with curl or visit http://localhost:8000/docs for interactive API docs.

### Adding New Models

1. Create a new model class in `models/`
2. Update `models/__init__.py` to export the new class
3. Import and use in the pipeline

### Modifying the Pipeline

1. Edit `pipeline/poem_to_image_pipeline.py`
2. Rebuild the Docker image:
```bash
docker-compose build poem-to-image-api
docker-compose up -d
```

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues:**
   ```bash
   # Check if NVIDIA Docker is working
   docker run --rm --gpus all nvidia/cuda:11.7-base nvidia-smi
   ```

2. **Memory Issues:**
   - Reduce batch size in diffusion settings
   - Use CPU-only mode (slower but uses less memory)

3. **Model Loading Issues:**
   - Ensure PhoCLIP checkpoint is in the correct location
   - Check model file permissions

4. **API Timeout:**
   - Generation can take 2-5 minutes depending on hardware
   - Increase timeout settings in your client

### Logs and Debugging

```bash
# View real-time logs
docker-compose logs -f poem-to-image-api

# Access container shell
docker-compose exec poem-to-image-api bash

# Check GPU usage inside container
docker-compose exec poem-to-image-api nvidia-smi
```

## Performance Optimization

### GPU Memory Optimization

1. **Enable VAE slicing** (already configured)
2. **Use model CPU offload** (already configured)
3. **Reduce inference steps** in `diffusion_generator.py`

### Production Considerations

1. **Use a reverse proxy** (nginx) for load balancing
2. **Implement proper authentication**
3. **Add rate limiting**
4. **Use persistent storage** for outputs
5. **Monitor resource usage**

## Scaling

### Horizontal Scaling

```yaml
# In docker-compose.yml
services:
  poem-to-image-api:
    deploy:
      replicas: 3
```

### Load Balancing

Add nginx configuration for load balancing multiple API instances.

## Security

1. **API Authentication**: Implement JWT or API key authentication
2. **Input Validation**: Validate poem content and length
3. **Rate Limiting**: Prevent abuse
4. **File Upload Security**: Validate file types and sizes

## Monitoring

### Health Checks

The API includes health check endpoints:
- `/health`: Basic health status
- Docker health check: Automatic container health monitoring

### Metrics

Consider adding Prometheus metrics for:
- Request count and latency
- GPU utilization
- Memory usage
- Generation success rate

## Support

For issues and questions:
1. Check the logs: `docker-compose logs poem-to-image-api`
2. Verify GPU setup: `nvidia-smi`
3. Test API endpoints: `curl http://localhost:8000/health`
4. Check model files are present in `phoclip_checkpoint/` 