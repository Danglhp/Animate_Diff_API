#!/usr/bin/env python3
"""
Example script showing how to use the updated poem-to-image pipeline 
with PhoCLIP model loaded from Hugging Face
"""

from poem_to_image_fixed import PoemToImagePipeline

def main():
    # Initialize the pipeline (will automatically load PhoCLIP from Hugging Face)
    print("Initializing Poem-to-Image Pipeline with Hugging Face PhoCLIP...")
    pipeline = PoemToImagePipeline()
    
    # Test Vietnamese prompt
    test_prompt = "cảnh hoàng hôn trên biển với những con chim bay"
    
    print(f"Generating animation for: {test_prompt}")
    
    # Generate animation using PhoCLIP text encoding
    result = pipeline.diffusion_generator.generate(
        prompt=test_prompt,
        output_path="huggingface_demo.gif"
    )
    
    print(f"Animation saved to: {result}")

if __name__ == "__main__":
    main()
