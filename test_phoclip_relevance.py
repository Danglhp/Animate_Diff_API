import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import json
from pathlib import Path
import argparse
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from huggingface_hub import hf_hub_download

class PhoCLIPEmbedding:
    def __init__(self, model_repo="kienhoang123/ViCLIP", fallback_path="e:/ViCLIP/phoclip_checkpoint"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading PhoCLIP model...")
        
        # Try to load from Hugging Face first, fallback to local if needed
        try:
            self._load_huggingface_model(model_repo)
        except Exception as e:
            print(f"Failed to load from Hugging Face: {e}")
            print(f"Falling back to local model at {fallback_path}")
            self._load_local_model(fallback_path)
    
    def _load_huggingface_model(self, model_repo):
        """Load PhoCLIP model from Hugging Face Hub"""
        print(f"Loading PhoCLIP model from Hugging Face: {model_repo}")
        
        # Load tokenizer (this should work directly)
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=False)
        print("✓ Tokenizer loaded successfully")
        
        # Download the complete model file
        model_file = hf_hub_download(repo_id=model_repo, filename="model.pt")
        print("✓ Model file downloaded")
        
        # Load the complete state dict
        complete_state_dict = torch.load(model_file, map_location=self.device)
        print("✓ Model state dict loaded")
        
        # Initialize text encoder (PhoBERT base)
        self.text_encoder = AutoModel.from_pretrained("vinai/phobert-base").to(self.device)
        
        # Extract text encoder state dict
        text_encoder_state_dict = {}
        for key, value in complete_state_dict.items():
            if key.startswith('text_encoder.'):
                # Remove 'text_encoder.' prefix
                new_key = key[len('text_encoder.'):]
                text_encoder_state_dict[new_key] = value
        
        # Load text encoder weights
        self.text_encoder.load_state_dict(text_encoder_state_dict)
        print("✓ Text encoder weights loaded")
        
        # Initialize and load text projection head
        self.text_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        ).to(self.device)
        
        # Extract text projection state dict
        text_proj_state_dict = {}
        for key, value in complete_state_dict.items():
            if key.startswith('text_proj.'):
                # Remove 'text_proj.' prefix
                new_key = key[len('text_proj.'):]
                text_proj_state_dict[new_key] = value
        
        # Load projection weights
        self.text_proj.load_state_dict(text_proj_state_dict)
        print("✓ Text projection weights loaded")
        
        # Initialize and load image encoder
        self._load_image_encoder(complete_state_dict)
        
        # Set to evaluation mode
        self.text_encoder.eval()
        self.text_proj.eval()
        self.image_encoder.eval()
        self.image_proj.eval()
        
        print("Successfully loaded PhoCLIP model from Hugging Face")
    
    def _load_image_encoder(self, complete_state_dict):
        """Load image encoder components"""
        # You'll need to implement this based on your PhoCLIP architecture
        # For now, let's use a placeholder that loads from the state dict
        
        # Initialize image encoder (assuming it's a vision transformer or CNN)
        # This is a placeholder - you need to match your actual architecture
        from transformers import CLIPVisionModel
        
        try:
            # Try to load CLIP vision model as base
            self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            
            # Initialize image projection head
            self.image_proj = nn.Sequential(
                nn.Linear(512, 768),  # CLIP vision output is 512, project to 768
                nn.ReLU(),
                nn.Linear(768, 768)
            ).to(self.device)
            
            # Extract and load image encoder weights if available
            image_encoder_state_dict = {}
            image_proj_state_dict = {}
            
            for key, value in complete_state_dict.items():
                if key.startswith('image_encoder.'):
                    new_key = key[len('image_encoder.'):]
                    image_encoder_state_dict[new_key] = value
                elif key.startswith('image_proj.'):
                    new_key = key[len('image_proj.'):]
                    image_proj_state_dict[new_key] = value
            
            # Load weights if available
            if image_encoder_state_dict:
                try:
                    self.image_encoder.load_state_dict(image_encoder_state_dict, strict=False)
                    print("✓ Image encoder weights loaded")
                except:
                    print("⚠ Could not load image encoder weights, using pretrained CLIP")
            
            if image_proj_state_dict:
                try:
                    self.image_proj.load_state_dict(image_proj_state_dict)
                    print("✓ Image projection weights loaded")
                except:
                    print("⚠ Could not load image projection weights, using random initialization")
                    
        except Exception as e:
            print(f"Error loading image encoder: {e}")
            # Fallback to a simple CNN or transformer
            self.image_encoder = None
            self.image_proj = None
    
    def _load_local_model(self, checkpoint_path):
        """Load local PhoCLIP model"""
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
        
        # Try to load image components
        try:
            # Load image encoder if available
            if os.path.exists(f"{checkpoint_path}/image_encoder"):
                from transformers import CLIPVisionModel
                self.image_encoder = CLIPVisionModel.from_pretrained(f"{checkpoint_path}/image_encoder").to(self.device)
            else:
                # Use pretrained CLIP vision model as fallback
                from transformers import CLIPVisionModel
                self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            
            # Load image projection
            self.image_proj = nn.Sequential(
                nn.Linear(512, 768),
                nn.ReLU(), 
                nn.Linear(768, 768)
            ).to(self.device)
            
            if os.path.exists(f"{checkpoint_path}/image_proj.pt"):
                self.image_proj.load_state_dict(torch.load(f"{checkpoint_path}/image_proj.pt"))
            
        except Exception as e:
            print(f"Could not load image components: {e}")
            self.image_encoder = None
            self.image_proj = None
        
        # Set to evaluation mode
        self.text_encoder.eval()
        self.text_proj.eval()
        if self.image_encoder:
            self.image_encoder.eval()
            self.image_proj.eval()
        
        print("Successfully loaded local PhoCLIP model")
    
    def encode_text(self, text):
        """Encode text to embedding"""
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
    
    def encode_image(self, image):
        """Encode image to embedding"""
        if self.image_encoder is None or self.image_proj is None:
            print("Image encoder not available")
            return None
        
        # Preprocess image for CLIP vision model
        from transformers import CLIPProcessor
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Process image
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        
        # Get image embeddings
        with torch.no_grad():
            image_outputs = self.image_encoder(**inputs)
            image_features = image_outputs.pooler_output
            image_proj = self.image_proj(image_features)
            image_emb = F.normalize(image_proj, p=2, dim=-1)
        
        return image_emb.cpu().numpy()


class RelevanceChecker:
    """Check relevance between text prompts and images"""
    
    def __init__(self, images_folder="e:/ViCLIP/images"):
        self.images_folder = Path(images_folder)
        self.phoclip = PhoCLIPEmbedding()
        self.image_embeddings = {}
        self.image_files = []
        
        # Load and encode all images
        self._load_images()
    
    def _load_images(self):
        """Load and encode all images in the folder"""
        print("Loading and encoding images...")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for img_file in self.images_folder.glob("*"):
            if img_file.suffix.lower() in image_extensions:
                try:
                    # Load image
                    image = Image.open(img_file).convert('RGB')
                    
                    # Encode image
                    embedding = self.phoclip.encode_image(image)
                    
                    if embedding is not None:
                        self.image_embeddings[str(img_file)] = embedding
                        self.image_files.append(str(img_file))
                        print(f"✓ Encoded: {img_file.name}")
                    else:
                        print(f"✗ Failed to encode: {img_file.name}")
                        
                except Exception as e:
                    print(f"✗ Error processing {img_file.name}: {e}")
        
        print(f"Successfully encoded {len(self.image_embeddings)} images")
    
    def find_most_relevant_images(self, text_prompt, top_k=5):
        """Find the most relevant images for a given text prompt"""
        if not self.image_embeddings:
            print("No images available for comparison")
            return []
        
        # Encode the text prompt
        text_embedding = self.phoclip.encode_text(text_prompt)
        
        # Calculate similarities
        similarities = []
        for img_path, img_embedding in self.image_embeddings.items():
            # Calculate cosine similarity
            similarity = np.dot(text_embedding.flatten(), img_embedding.flatten()) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(img_embedding)
            )
            similarities.append((img_path, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def test_prompt_relevance(self, test_prompts):
        """Test relevance for multiple prompts"""
        results = {}
        
        for prompt in test_prompts:
            print(f"\n{'='*60}")
            print(f"Testing prompt: {prompt}")
            print(f"{'='*60}")
            
            relevant_images = self.find_most_relevant_images(prompt, top_k=5)
            results[prompt] = relevant_images
            
            for i, (img_path, similarity) in enumerate(relevant_images, 1):
                img_name = Path(img_path).name
                print(f"{i}. {img_name} (similarity: {similarity:.4f})")
        
        return results
    
    def save_results(self, results, output_file="relevance_test_results.json"):
        """Save test results to JSON file"""
        # Convert results to serializable format
        serializable_results = {}
        for prompt, image_list in results.items():
            serializable_results[prompt] = [
                {
                    "image_path": img_path,
                    "similarity": float(similarity),
                    "image_name": Path(img_path).name
                }
                for img_path, similarity in image_list
            ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Test PhoCLIP text-image relevance")
    parser.add_argument("--images-folder", type=str, default="e:/ViCLIP/images", 
                       help="Path to folder containing images")
    parser.add_argument("--test-prompts", type=str, nargs='+', 
                       help="List of text prompts to test")
    parser.add_argument("--output", type=str, default="relevance_test_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Default test prompts if none provided
    test_prompts = args.test_prompts or [
        "một người phụ nữ đẹp",
        "cảnh thiên nhiên xanh tươi", 
        "thành phố về đêm",
        "con mèo dễ thương",
        "bông hoa đẹp",
        "ô tô màu đỏ",
        "bãi biển và sóng",
        "ngôi nhà cổ",
        "trời mưa buồn",
        "ánh sáng ấm áp"
    ]
    
    # Test some Vietnamese poem-related prompts as well
    poem_prompts = [
        "tình yêu buồn và lãng mạn",
        "cảnh đẹp thiên nhiên thơ mộng", 
        "nỗi nhớ và xa cách",
        "hoa lá và mùa xuân",
        "ánh trăng và đêm tối"
    ]
    
    all_prompts = test_prompts + poem_prompts
    
    print("Starting PhoCLIP relevance test...")
    print(f"Images folder: {args.images_folder}")
    print(f"Number of test prompts: {len(all_prompts)}")
    
    # Initialize relevance checker
    checker = RelevanceChecker(args.images_folder)
    
    # Test prompts
    results = checker.test_prompt_relevance(all_prompts)
    
    # Save results
    checker.save_results(results, args.output)
    
    print("\nTest completed!")
    print(f"Check the results in: {args.output}")


if __name__ == "__main__":
    main()
