import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, CLIPVisionModel, CLIPProcessor
import torch.nn as nn
from huggingface_hub import hf_hub_download

class PhoCLIPImageSimilarityTester:
    def __init__(self, model_repo="kienhoang123/ViCLIP", fallback_path="e:/ViCLIP/phoclip_checkpoint"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_repo = model_repo
        self.fallback_path = fallback_path
        
        print(f"Loading PhoCLIP model...")
        self._load_model()
        
    def _load_model(self):
        """Load PhoCLIP model with error handling"""
        try:
            self._load_huggingface_model()
        except Exception as e:
            print(f"Failed to load from Hugging Face: {e}")
            print(f"Falling back to local model at {self.fallback_path}")
            self._load_local_model()
    
    def _load_huggingface_model(self):
        """Load PhoCLIP model from Hugging Face Hub"""
        print(f"Loading PhoCLIP model from Hugging Face: {self.model_repo}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_repo, use_fast=False)
        print("✓ Tokenizer loaded successfully")
        
        # Download the complete model file
        model_file = hf_hub_download(repo_id=self.model_repo, filename="model.pt")
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
                new_key = key[len('text_proj.'):]
                text_proj_state_dict[new_key] = value
        
        # Load projection weights
        self.text_proj.load_state_dict(text_proj_state_dict)
        print("✓ Text projection weights loaded")
        
        # Try to load vision encoder
        self._load_vision_encoder(complete_state_dict)
        
        # Set to evaluation mode
        self.text_encoder.eval()
        self.text_proj.eval()
        if hasattr(self, 'vision_encoder'):
            self.vision_encoder.eval()
            self.vision_proj.eval()
        
        print("Successfully loaded PhoCLIP model from Hugging Face")
    
    def _load_vision_encoder(self, complete_state_dict):
        """Load vision encoder components"""
        try:
            # Try to load vision encoder from state dict
            vision_encoder_state_dict = {}
            for key, value in complete_state_dict.items():
                if key.startswith('vision_encoder.'):
                    new_key = key[len('vision_encoder.'):]
                    vision_encoder_state_dict[new_key] = value
            
            if vision_encoder_state_dict:
                # Initialize vision encoder (CLIP ViT)
                self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                self.vision_encoder.load_state_dict(vision_encoder_state_dict)
                print("✓ Vision encoder weights loaded from state dict")
            else:
                # Use pretrained CLIP vision model
                self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                print("✓ Vision encoder loaded from pretrained CLIP")
            
            # Initialize and load vision projection head
            self.vision_proj = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Linear(768, 768)
            ).to(self.device)
            
            # Extract vision projection state dict
            vision_proj_state_dict = {}
            for key, value in complete_state_dict.items():
                if key.startswith('vision_proj.'):
                    new_key = key[len('vision_proj.'):]
                    vision_proj_state_dict[new_key] = value
            
            if vision_proj_state_dict:
                self.vision_proj.load_state_dict(vision_proj_state_dict)
                print("✓ Vision projection weights loaded")
            else:
                print("⚠ Vision projection weights not found, using random initialization")
            
            # Load CLIP processor for image preprocessing
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("✓ CLIP processor loaded")
            
        except Exception as e:
            print(f"Could not load vision encoder: {e}")
            self.vision_encoder = None
            self.vision_proj = None
            self.clip_processor = None
    
    def _load_local_model(self):
        """Load local PhoCLIP model"""
        print(f"Loading local PhoCLIP model from {self.fallback_path}")
        
        # Load text encoder (PhoBERT)
        self.text_encoder = AutoModel.from_pretrained(f"{self.fallback_path}/text_encoder").to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.fallback_path}/tokenizer", use_fast=False)
        
        # Load text projection head
        self.text_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        ).to(self.device)
        self.text_proj.load_state_dict(torch.load(f"{self.fallback_path}/text_proj.pt"))
        
        # Try to load vision components
        try:
            if os.path.exists(f"{self.fallback_path}/vision_encoder"):
                self.vision_encoder = CLIPVisionModel.from_pretrained(f"{self.fallback_path}/vision_encoder").to(self.device)
                self.vision_proj = nn.Sequential(
                    nn.Linear(768, 768),
                    nn.ReLU(),
                    nn.Linear(768, 768)
                ).to(self.device)
                self.vision_proj.load_state_dict(torch.load(f"{self.fallback_path}/vision_proj.pt"))
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                print("✓ Vision components loaded from local path")
            else:
                print("⚠ Vision components not found in local path")
                self.vision_encoder = None
                self.vision_proj = None
                self.clip_processor = None
        except Exception as e:
            print(f"Could not load vision components: {e}")
            self.vision_encoder = None
            self.vision_proj = None
            self.clip_processor = None
        
        # Set to evaluation mode
        self.text_encoder.eval()
        self.text_proj.eval()
        if self.vision_encoder:
            self.vision_encoder.eval()
            self.vision_proj.eval()
        
        print("Successfully loaded local PhoCLIP model")
    
    def encode_text(self, text):
        """Encode text using PhoCLIP text encoder"""
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
    
    def encode_image(self, image_path):
        """Encode image using PhoCLIP vision encoder"""
        if self.vision_encoder is None or self.vision_proj is None:
            raise ValueError("Vision encoder not available")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Process image with CLIP processor
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            # Get image embeddings
            with torch.no_grad():
                vision_outputs = self.vision_encoder(**inputs)
                image_features = vision_outputs.pooler_output
                image_proj = self.vision_proj(image_features)
                image_emb = F.normalize(image_proj, p=2, dim=-1)
            
            return image_emb.cpu().numpy()
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def load_image_dataset(self, dataset_path, max_images=100):
        """Load images from dataset path"""
        print(f"Loading images from {dataset_path}...")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        # Search for images in the dataset path
        for ext in image_extensions:
            image_paths.extend(Path(dataset_path).rglob(f"*{ext}"))
            image_paths.extend(Path(dataset_path).rglob(f"*{ext.upper()}"))
        
        # Limit number of images if specified
        if max_images and len(image_paths) > max_images:
            import random
            random.seed(42)
            image_paths = random.sample(image_paths, max_images)
            print(f"Using {max_images} randomly selected images")
        
        print(f"Found {len(image_paths)} images")
        return image_paths
    
    def encode_image_dataset(self, image_paths):
        """Encode all images in the dataset"""
        print("Encoding images...")
        
        image_embeddings = {}
        successful_encodings = 0
        
        for img_path in tqdm(image_paths, desc="Encoding images"):
            try:
                embedding = self.encode_image(str(img_path))
                if embedding is not None:
                    image_embeddings[str(img_path)] = embedding
                    successful_encodings += 1
            except Exception as e:
                print(f"Failed to encode {img_path}: {e}")
        
        print(f"Successfully encoded {successful_encodings}/{len(image_paths)} images")
        return image_embeddings
    
    def find_similar_images(self, text_prompt, image_embeddings, top_k=10):
        """Find most similar images for a text prompt"""
        if not image_embeddings:
            print("No image embeddings available")
            return []
        
        # Encode the text prompt
        text_embedding = self.encode_text(text_prompt)
        
        # Calculate similarities
        similarities = []
        for img_path, img_embedding in image_embeddings.items():
            try:
                similarity = cosine_similarity(text_embedding, img_embedding)[0][0]
                similarities.append((img_path, similarity))
            except Exception as e:
                print(f"Error calculating similarity for {img_path}: {e}")
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def test_prompt_relevance(self, test_prompts, dataset_path, max_images=100, top_k=10):
        """Test relevance of prompts against image dataset"""
        print("\n" + "="*60)
        print("TESTING PROMPT-IMAGE RELEVANCE")
        print("="*60)
        
        # Load and encode images
        image_paths = self.load_image_dataset(dataset_path, max_images)
        image_embeddings = self.encode_image_dataset(image_paths)
        
        if not image_embeddings:
            print("No images could be encoded. Cannot perform relevance testing.")
            return {}
        
        results = {}
        
        for prompt in test_prompts:
            print(f"\n{'='*50}")
            print(f"Testing prompt: {prompt}")
            print(f"{'='*50}")
            
            # Find similar images
            similar_images = self.find_similar_images(prompt, image_embeddings, top_k)
            
            results[prompt] = similar_images
            
            # Display results
            print(f"Top {len(similar_images)} most similar images:")
            for i, (img_path, similarity) in enumerate(similar_images, 1):
                img_name = Path(img_path).name
                print(f"{i:2d}. {img_name} (similarity: {similarity:.4f})")
            
            # Analyze relevance
            if similar_images:
                avg_similarity = np.mean([sim for _, sim in similar_images])
                max_similarity = max([sim for _, sim in similar_images])
                min_similarity = min([sim for _, sim in similar_images])
                
                print(f"\nRelevance Analysis:")
                print(f"  Average similarity: {avg_similarity:.4f}")
                print(f"  Maximum similarity: {max_similarity:.4f}")
                print(f"  Minimum similarity: {min_similarity:.4f}")
                
                # Assess relevance quality
                if max_similarity > 0.7:
                    print("  ✓ High relevance detected")
                elif max_similarity > 0.5:
                    print("  ⚠ Moderate relevance detected")
                else:
                    print("  ✗ Low relevance detected")
        
        return results
    
    def test_poem_prompts(self, poem_texts, dataset_path, max_images=100, top_k=10):
        """Test poem-specific prompts against image dataset"""
        print("\n" + "="*60)
        print("TESTING POEM PROMPTS AGAINST IMAGES")
        print("="*60)
        
        # Load and encode images
        image_paths = self.load_image_dataset(dataset_path, max_images)
        image_embeddings = self.encode_image_dataset(image_paths)
        
        if not image_embeddings:
            print("No images could be encoded. Cannot perform poem testing.")
            return {}
        
        results = {}
        
        for poem_line in poem_texts:
            print(f"\n{'='*50}")
            print(f"Testing poem line: {poem_line}")
            print(f"{'='*50}")
            
            # Find similar images
            similar_images = self.find_similar_images(poem_line, image_embeddings, top_k)
            
            results[poem_line] = similar_images
            
            # Display results
            print(f"Top {len(similar_images)} most similar images:")
            for i, (img_path, similarity) in enumerate(similar_images, 1):
                img_name = Path(img_path).name
                print(f"{i:2d}. {img_name} (similarity: {similarity:.4f})")
        
        return results
    
    def generate_relevance_report(self, results, output_file="phoclip_relevance_report.txt"):
        """Generate a detailed relevance report"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("PhoCLIP Text-Image Relevance Report\n")
            f.write("=" * 50 + "\n\n")
            
            for prompt, similar_images in results.items():
                f.write(f"Prompt: {prompt}\n")
                f.write("-" * 30 + "\n")
                
                if similar_images:
                    avg_sim = np.mean([sim for _, sim in similar_images])
                    max_sim = max([sim for _, sim in similar_images])
                    
                    f.write(f"Average similarity: {avg_sim:.4f}\n")
                    f.write(f"Maximum similarity: {max_sim:.4f}\n")
                    f.write(f"Top matches:\n")
                    
                    for i, (img_path, similarity) in enumerate(similar_images[:5], 1):
                        img_name = Path(img_path).name
                        f.write(f"  {i}. {img_name} ({similarity:.4f})\n")
                    
                    # Relevance assessment
                    if max_sim > 0.7:
                        f.write("  Assessment: HIGH RELEVANCE\n")
                    elif max_sim > 0.5:
                        f.write("  Assessment: MODERATE RELEVANCE\n")
                    else:
                        f.write("  Assessment: LOW RELEVANCE\n")
                else:
                    f.write("No similar images found\n")
                
                f.write("\n")
        
        print(f"Relevance report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Test PhoCLIP text-image similarity")
    parser.add_argument("--dataset-path", type=str, default="e:/ViCLIP/ktvic/ktvic_dataset",
                       help="Path to image dataset")
    parser.add_argument("--model-repo", type=str, default="kienhoang123/ViCLIP",
                       help="Hugging Face model repository")
    parser.add_argument("--fallback-path", type=str, default="e:/ViCLIP/phoclip_checkpoint",
                       help="Local model fallback path")
    parser.add_argument("--max-images", type=int, default=100,
                       help="Maximum number of images to process")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Number of top similar images to show")
    parser.add_argument("--test-poem", action="store_true",
                       help="Test poem-specific prompts")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = PhoCLIPImageSimilarityTester(args.model_repo, args.fallback_path)
    
    if args.test_poem:
        # Test poem-specific prompts
        poem_texts = [
            "đẩy hoa dun lá khỏi tay trời",
            "nghĩ lại tình duyên luống ngậm ngùi",
            "bắc yến nam hồng thư mấy bức",
            "đông đào tây liễu khách đôi nơi",
            "lửa ân dập mãi sao không tắt",
            "biển ái khơi hoài vẫn chẳng vơi",
            "đèn nguyệt trong xanh mây chẳng bợn",
            "xin soi xét đến tấm lòng ai"
        ]
        
        results = tester.test_poem_prompts(poem_texts, args.dataset_path, args.max_images, args.top_k)
    else:
        # Test general prompts
        test_prompts = [
            "một người phụ nữ",
            "con đường phố",
            "bầu trời xanh",
            "hoa lá cây cối",
            "ánh sáng mặt trời",
            "nước biển xanh",
            "mây trắng",
            "tình yêu",
            "ngôi nhà",
            "con mèo"
        ]
        
        results = tester.test_prompt_relevance(test_prompts, args.dataset_path, args.max_images, args.top_k)
    
    # Generate report
    tester.generate_relevance_report(results)
    
    # Save detailed results
    detailed_results = {}
    for prompt, similar_images in results.items():
        detailed_results[prompt] = [
            {"image_path": str(img_path), "similarity": float(sim)}
            for img_path, sim in similar_images
        ]
    
    with open("phoclip_detailed_results.json", 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: phoclip_detailed_results.json")

if __name__ == "__main__":
    main() 