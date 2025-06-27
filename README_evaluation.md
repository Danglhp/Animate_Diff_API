# Vietnamese Poem to Image Evaluation

This repository contains tools for evaluating Vietnamese poem analysis, prompt generation, and image generation using the PhoCLIP model.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. NLTK data for BLEU score calculation will be downloaded automatically when running the evaluation script.

## Scripts

### 1. Prompt Generation Evaluation (`evaluate_prompt_generation.py`)

This script evaluates prompt generation from poems by comparing generated prompts with reference prompts using BLEU and ROUGE metrics.

**Usage:**

```bash
python evaluate_prompt_generation.py
```

**Arguments:**

- `--dataset`: Hugging Face dataset name (default: "kienhoang123/Vietnamese_Poem_Analysis_VN")
- `--output-dir`: Directory to save results (default: "evaluation_results")
- `--output-file`: Output CSV file name (default: "prompt_evaluation.csv")
- `--limit`: Maximum number of samples to process (default: 50, 0 for all)
- `--log-interval`: Interval for logging progress (default: 5)
- `--use-local-model`: Use local model instead of Ollama API

**Output:**

- CSV file with evaluation results
- Summary text file with average metrics

### 2. Image Generation from Prompts (`generate_images_from_prompts.py`)

This script generates images from prompts in the dataset and optionally calculates CLIP similarity scores.

**Usage:**

```bash
python generate_images_from_prompts.py --calculate-clip-score
```

**Arguments:**

- `--dataset`: Hugging Face dataset name (default: "kienhoang123/Vietnamese_Poem_Analysis_VN")
- `--output-dir`: Directory to save generated images (default: "generated_images")
- `--model-id`: Stable Diffusion model ID (default: "runwayml/stable-diffusion-v1-5")
- `--phoclip-checkpoint`: HuggingFace repo or path to PhoCLIP checkpoint (default: "kienhoang123/ViCLIP")
- `--limit`: Maximum number of samples to process (default: 50, 0 for all)
- `--log-interval`: Interval for logging progress (default: 5)
- `--seed`: Random seed for reproducibility
- `--calculate-clip-score`: Calculate CLIP similarity score between prompt and image

**Output:**

- Generated images in the output directory
- JSON file with generation results and CLIP scores (if enabled)

## Models Used

1. **PhoCLIP**: Vietnamese CLIP model for image-text similarity 
   - HuggingFace: "kienhoang123/ViCLIP"
   - Local fallback: "e:/ViCLIP/phoclip_checkpoint"

2. **Llama3.2 with PEFT adapter**: For poem analysis and prompt generation
   - HuggingFace: "kienhoang123/Llama3.2_Poem_Analysis"

3. **Stable Diffusion**: For image generation
   - Default: "runwayml/stable-diffusion-v1-5"

## Example Workflow

1. Evaluate prompt generation quality:

```bash
python evaluate_prompt_generation.py --use-local-model
```

2. Generate images from dataset prompts:

```bash
python generate_images_from_prompts.py --calculate-clip-score
```

## Notes

- Make sure your GPU has enough memory for the models
- Both scripts are configured to process only 50 samples by default for faster evaluation
- The CLIP score calculation now uses the HuggingFace model by default with a local fallback
