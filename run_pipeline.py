#!/usr/bin/env python3
"""
Simple script to run the poem-to-image pipeline locally for testing.
"""

import argparse
from pathlib import Path
from pipeline import PoemToImagePipeline

def main():
    parser = argparse.ArgumentParser(description="Run poem-to-image pipeline locally")
    parser.add_argument("--poem", type=str, help="Vietnamese poem text")
    parser.add_argument("--poem-file", type=str, help="Path to file containing poem")
    parser.add_argument("--output", type=str, default="animation.gif", help="Output file path")
    parser.add_argument("--use-local-model", action="store_true", help="Use local model for prompt generation")
    
    args = parser.parse_args()
    
    # Get poem from argument or file
    poem = args.poem
    if poem is None and args.poem_file:
        with open(args.poem_file, "r", encoding="utf-8") as f:
            poem = f.read()
    
    if poem is None:
        # Default poem for testing
        poem = """
        đẩy hoa dun lá khỏi tay trời , <
        nghĩ lại tình duyên luống ngậm ngùi . <
        bắc yến nam hồng , thư mấy bức , <
        đông đào tây liễu , khách đôi nơi . <
        lửa ân , dập mãi sao không tắt , <
        biển ái , khơi hoài vẫn chẳng vơi . <
        đèn nguyệt trong xanh , mây chẳng bợn , <
        xin soi xét đến tấm lòng ai ...
        """
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=== POEM TO IMAGE ANIMATION PIPELINE ===")
    print(f"Input poem: {poem.strip()}")
    print(f"Output path: {output_path}")
    print("=" * 50)
    
    try:
        # Initialize and run pipeline
        pipeline = PoemToImagePipeline(use_local_model_for_prompt=args.use_local_model)
        result_path = pipeline.process(poem, str(output_path))
        
        print(f"\n✅ Success! Animation saved to: {result_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 