import torch
import pandas as pd
import argparse
from pathlib import Path
import gc
import os
import numpy as np
from PIL import Image
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class PhoCLIPEmbedding:
    def __init__(self, model_repo="kienhoang123/ViCLIP", fallback_path="e:/ViCLIP/phoclip_checkpoint"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading PhoCLIP model...")
        
        # Always use the local model for now due to model type compatibility issues
        self._load_local_model(fallback_path)
    
    def _load_local_model(self, checkpoint_path):
        print(f"Loading local PhoCLIP model from {checkpoint_path}")
        
        # Load text encoder (PhoBERT)
        self.text_encoder = AutoModel.from_pretrained(f"{checkpoint_path}/text_encoder").to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(f"{checkpoint_path}/tokenizer", use_fast=False)
        
        # Load text projection head
        self.text_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        ).to(self.device)
        self.text_proj.load_state_dict(torch.load(f"{checkpoint_path}/text_proj.pt"))
        
        # Set to evaluation mode
        self.text_encoder.eval()
        self.text_proj.eval()
        
        print("Successfully loaded local PhoCLIP model")
    
    def encode_text(self, text):
        import torch.nn.functional as F
        
        # Tokenize the text
        inputs = self.tokenizer(
            text, 
            padding='max_length', 
            max_length=77, 
            truncation=True, 
            return_tensors='pt'
        ).to(self.device)
        
        # Get text embeddings
        with torch.no_grad():
            text_outputs = self.text_encoder(
                input_ids=inputs.input_ids, 
                attention_mask=inputs.attention_mask, 
                return_dict=True
            )
            text_cls = text_outputs.last_hidden_state[:, 0, :]
            text_proj = self.text_proj(text_cls)
            text_emb = F.normalize(text_proj, p=2, dim=-1)
            
        return text_emb.cpu().numpy()

def clean_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def load_diffusion_model(model_id="runwayml/stable-diffusion-v1-5", use_animatediff=True):
    """Load Stable Diffusion model with or without AnimateDiff"""
    print(f"Loading {'AnimateDiff' if use_animatediff else 'Stable Diffusion'} model: {model_id}")
    
    # Load AnimateDiff with motion adapter
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-2", 
        torch_dtype=torch.float16
    )
    
    pipe = AnimateDiffPipeline.from_pretrained(
        model_id, 
        motion_adapter=adapter, 
        torch_dtype=torch.float16
    )
    
    scheduler = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipe.scheduler = scheduler
    
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    
    return pipe

def generate_image(pipe, prompt, output_path, use_animatediff=True, seed=None, guidance_scale=7.5, steps=50):
    """Generate image or animation from prompt"""
    print(f"Generating {'animation' if use_animatediff else 'image'} with prompt: '{prompt}'")
    
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(seed)
    else:
        generator = None
    
    # Check prompt length and truncate if needed
    if len(prompt.split()) > 70:
        print(f"Warning: Prompt too long ({len(prompt.split())} words). Truncating...")
        prompt_words = prompt.split()[:70]
        prompt = " ".join(prompt_words)
        print(f"Truncated prompt: {prompt}")
    
    # Generate the image or animation
    if use_animatediff:
        negative_prompt = "vui vẻ, màu sắc tươi sáng, cảnh đông đúc, yếu tố hiện đại, chất lượng kém, chất lượng tệ hơn"
        
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=16,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator
        )
        frames = output.frames[0]
        export_to_gif(frames, output_path)
        
        # Also save the first frame as an image for CLIP evaluation
        image_path = output_path.replace(".gif", ".png")
        frames[0].save(image_path)
        
        print(f"Animation saved to: {output_path}")
        print(f"First frame saved to: {image_path}")
        
        return image_path
    else:
        # Generate a still image
        image = pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
        
        # Save the image
        image.save(output_path)
        print(f"Image saved to: {output_path}")
        
        return output_path

def calculate_clip_score(clip_model, image_path, prompt):
    """Calculate CLIP similarity score between image and prompt"""
    from PIL import Image
    
    # Load and encode the image
    image = Image.open(image_path)
    
    # Using the PhoCLIP model to encode text
    text_features = clip_model.encode_text(prompt)
    
    # Not implementing image encoding since PhoCLIP's image encoder would require additional adaptation
    # For a complete implementation, you would need to:
    # 1. Load the image encoder from PhoCLIP
    # 2. Process the image according to PhoCLIP's requirements
    # 3. Calculate the cosine similarity between text and image embeddings
    
    print("Note: Full CLIP score calculation requires image encoder from PhoCLIP")
    print("Text features shape:", text_features.shape)
    
    # Return a placeholder score
    return {"clip_score": 0.0, "note": "Image encoding not implemented"}

def main():
    parser = argparse.ArgumentParser(description="Generate an image from a single prompt in the dataset and evaluate with PhoCLIP")
    parser.add_argument("--dataset", type=str, default="data_vi.csv", help="Path to the dataset")
    parser.add_argument("--index", type=int, default=0, help="Index of the record to use (0-based)")
    parser.add_argument("--output-dir", type=str, default="generated_images", help="Directory to save the generated image")
    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5", help="Stable Diffusion model ID")
    parser.add_argument("--use-animatediff", action="store_true", help="Use AnimateDiff for animation generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale for diffusion model")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--calculate-clip-score", action="store_true", help="Calculate PhoCLIP similarity score")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load the dataset
        print(f"Loading dataset from {args.dataset}")
        data = pd.read_csv(args.dataset)
        
        # Check if the index is valid
        if args.index < 0 or args.index >= len(data):
            raise ValueError(f"Index {args.index} is out of range. Dataset has {len(data)} records.")
        
        # Get the prompt from the dataset
        record = data.iloc[args.index]
        prompt = record['prompt_vi']
        poem = record['content']
        title = record.get('title', f"Record_{args.index}")
        
        # Clean title for filename
        safe_title = "".join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in title)
        safe_title = safe_title[:50]  # Limit length
        
        # Set output path
        file_extension = ".gif" if args.use_animatediff else ".png"
        output_path = os.path.join(args.output_dir, f"{args.index}_{safe_title}{file_extension}")
        
        # Print record information
        print(f"\nProcessing record {args.index}:")
        print(f"Title: {title}")
        print(f"Prompt: {prompt}")
        print(f"Poem excerpt: {poem[:100]}..." if len(poem) > 100 else f"Poem: {poem}")
        
        # Load the model
        pipe = load_diffusion_model(args.model_id, use_animatediff=args.use_animatediff)
        
        # Generate the image or animation
        image_path = generate_image(
            pipe=pipe, 
            prompt=prompt, 
            output_path=output_path,
            use_animatediff=args.use_animatediff,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            steps=args.steps
        )
        
        # Calculate CLIP score if requested
        clip_score_results = {"clip_score": None}
        if args.calculate_clip_score:
            print("\nCalculating PhoCLIP similarity score...")
            # Initialize PhoCLIP
            phoclip = PhoCLIPEmbedding()
            
            # Calculate CLIP score
            clip_score_results = calculate_clip_score(phoclip, image_path, prompt)
            print(f"PhoCLIP similarity score: {clip_score_results}")
        
        # Save metadata
        metadata_path = os.path.join(args.output_dir, f"{args.index}_{safe_title}_metadata.txt")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(f"Index: {args.index}\n")
            f.write(f"Title: {title}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Poem:\n{poem}\n")
            f.write(f"\nGeneration parameters:\n")
            f.write(f"Model: {args.model_id}\n")
            f.write(f"Used AnimateDiff: {args.use_animatediff}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Guidance scale: {args.guidance_scale}\n")
            f.write(f"Steps: {args.steps}\n")
            if clip_score_results["clip_score"] is not None:
                f.write(f"\nPhoCLIP similarity score: {clip_score_results['clip_score']}\n")
                if "note" in clip_score_results:
                    f.write(f"Note: {clip_score_results['note']}\n")
        
        print(f"\nGeneration complete.")
        print(f"Output saved to {output_path}")
        print(f"Metadata saved to {metadata_path}")
        
    finally:
        # Clean up memory
        clean_memory()

if __name__ == "__main__":
    main()