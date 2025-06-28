# AniDiff_Poem - Vietnamese Poem to Animation API

A FastAPI-based service that converts Vietnamese poems into animated images using PhoCLIP (Vietnamese CLIP) and AnimateDiff.

## 🌟 Features

- **Vietnamese Poem Analysis**: Uses Llama3.2 fine-tuned for Vietnamese poem analysis
- **PhoCLIP Integration**: Vietnamese language understanding for better text-to-image generation
- **AnimateDiff**: Creates smooth animations from static images
- **RESTful API**: Complete FastAPI service with background task processing
- **Docker Support**: Ready for containerized deployment
- **GPU Acceleration**: CUDA support for fast generation

## 🏗️ Architecture

```
├── models/                     # Core model classes
│   ├── phoclip_embedding.py    # PhoCLIP text encoder
│   ├── poem_analyzer.py        # Poem analysis using Llama3.2
│   ├── prompt_generator.py     # Prompt generation from analysis
│   └── diffusion_generator.py  # AnimateDiff image generation
├── pipeline/                   # Main pipeline orchestration
│   └── poem_to_image_pipeline.py
├── api/                        # FastAPI application
│   └── main.py
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
└── run_pipeline.py            # Local testing script
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Docker and Docker Compose (for containerized deployment)

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/Danglhp/AniDiff_Poem.git
   cd AniDiff_Poem
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run locally**
   ```bash
   python run_pipeline.py --poem "đẩy hoa dun lá khỏi tay trời"
   ```

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Test the API**
   ```bash
   curl -X POST "http://localhost:8000/generate" \
        -H "Content-Type: application/json" \
        -d '{
          "poem": "đẩy hoa dun lá khỏi tay trời\nnghĩ lại tình duyên luống ngậm ngùi",
          "output_filename": "my_poem"
        }'
   ```

## 📡 API Endpoints

### POST /generate
Generate animation from a Vietnamese poem.

**Request:**
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

### GET /health
Health check endpoint.

### GET /tasks
List all tasks and their status.

## 🔧 Configuration

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: GPU device to use (default: 0)
- `PYTHONPATH`: Python path (set to /app in container)

### Model Paths

- PhoCLIP checkpoint should be in `./phoclip_checkpoint/`
- Generated animations are saved to `./outputs/`

## 🛠️ Development

### Project Structure

- **models/**: Contains all AI model classes
- **pipeline/**: Main orchestration logic
- **api/**: FastAPI application and endpoints
- **tests/**: Unit tests (to be added)

### Adding New Features

1. Create new model classes in `models/`
2. Update `models/__init__.py` to export new classes
3. Integrate into the pipeline in `pipeline/poem_to_image_pipeline.py`
4. Add API endpoints in `api/main.py`

## 📊 Performance

- **Generation Time**: 2-5 minutes (depending on hardware)
- **Memory Usage**: ~8GB GPU memory recommended
- **Output Quality**: 16-frame GIF animations

## 🔍 Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**
   ```bash
   # Check if NVIDIA Docker is working
   docker run --rm --gpus all nvidia/cuda:11.7-base nvidia-smi
   ```

2. **Memory Issues**
   - Reduce batch size in diffusion settings
   - Use CPU-only mode (slower but uses less memory)

3. **Model Loading Issues**
   - Ensure PhoCLIP checkpoint is in the correct location
   - Check model file permissions

### Logs and Debugging

```bash
# View real-time logs
docker-compose logs -f poem-to-image-api

# Access container shell
docker-compose exec poem-to-image-api bash
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PhoCLIP](https://github.com/kienhoang123/ViCLIP) - Vietnamese CLIP model
- [AnimateDiff](https://github.com/guoyww/animatediff) - Animation generation
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Hugging Face](https://huggingface.co/) - Model hosting

## 📞 Support

For issues and questions:
1. Check the [Issues](https://github.com/Danglhp/AniDiff_Poem/issues) page
2. Create a new issue with detailed information
3. Include logs and error messages

---

**Made with ❤️ for Vietnamese poetry and AI art** 