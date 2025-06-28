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
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from huggingface_hub import hf_hub_download

class PhoCLIPEmbeddingTester:
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
        
        # Set to evaluation mode
        self.text_encoder.eval()
        self.text_proj.eval()
        
        print("Successfully loaded PhoCLIP model from Hugging Face")
    
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
        
        # Set to evaluation mode
        self.text_encoder.eval()
        self.text_proj.eval()
        
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
    
    def test_embedding_quality(self):
        """Test the quality of PhoCLIP embeddings"""
        print("\n" + "="*60)
        print("TESTING PHOCLIP EMBEDDING QUALITY")
        print("="*60)
        
        # Test 1: Semantic similarity between related texts
        print("\n1. Testing semantic similarity between related texts...")
        
        test_pairs = [
            ("một người đàn ông", "một người nam giới"),
            ("con mèo đen", "mèo màu đen"),
            ("hoa hồng đỏ", "bông hồng màu đỏ"),
            ("mặt trời mọc", "bình minh"),
            ("trời mưa", "mưa rơi"),
            ("ngôi nhà", "căn nhà"),
            ("con đường", "đường phố"),
            ("cây cối", "rừng cây"),
            ("biển xanh", "đại dương"),
            ("núi cao", "đỉnh núi")
        ]
        
        similarities = []
        for text1, text2 in test_pairs:
            emb1 = self.encode_text(text1)
            emb2 = self.encode_text(text2)
            sim = cosine_similarity(emb1, emb2)[0][0]
            similarities.append(sim)
            print(f"  '{text1}' vs '{text2}': {sim:.4f}")
        
        print(f"\n  Average similarity for related texts: {np.mean(similarities):.4f}")
        
        # Test 2: Semantic dissimilarity between unrelated texts
        print("\n2. Testing semantic dissimilarity between unrelated texts...")
        
        unrelated_pairs = [
            ("một người đàn ông", "con mèo đen"),
            ("hoa hồng đỏ", "mặt trời mọc"),
            ("trời mưa", "ngôi nhà"),
            ("con đường", "cây cối"),
            ("biển xanh", "núi cao"),
            ("xe hơi", "con chim"),
            ("sách vở", "bầu trời"),
            ("điện thoại", "dòng sông"),
            ("bàn ghế", "mây trắng"),
            ("áo quần", "sao trời")
        ]
        
        dissimilarities = []
        for text1, text2 in unrelated_pairs:
            emb1 = self.encode_text(text1)
            emb2 = self.encode_text(text2)
            sim = cosine_similarity(emb1, emb2)[0][0]
            dissimilarities.append(sim)
            print(f"  '{text1}' vs '{text2}': {sim:.4f}")
        
        print(f"\n  Average similarity for unrelated texts: {np.mean(dissimilarities):.4f}")
        
        # Test 3: Self-similarity (should be 1.0)
        print("\n3. Testing self-similarity...")
        
        test_texts = ["một người đàn ông", "con mèo đen", "hoa hồng đỏ", "mặt trời mọc"]
        self_similarities = []
        
        for text in test_texts:
            emb1 = self.encode_text(text)
            emb2 = self.encode_text(text)
            sim = cosine_similarity(emb1, emb2)[0][0]
            self_similarities.append(sim)
            print(f"  '{text}' vs itself: {sim:.4f}")
        
        print(f"\n  Average self-similarity: {np.mean(self_similarities):.4f}")
        
        # Test 4: Embedding statistics
        print("\n4. Testing embedding statistics...")
        
        all_texts = [text for pair in test_pairs for text in pair] + [text for pair in unrelated_pairs for text in pair]
        all_embeddings = []
        
        for text in all_texts:
            emb = self.encode_text(text)
            all_embeddings.append(emb.flatten())
        
        all_embeddings = np.array(all_embeddings)
        
        print(f"  Embedding shape: {all_embeddings.shape}")
        print(f"  Mean embedding norm: {np.mean(np.linalg.norm(all_embeddings, axis=1)):.4f}")
        print(f"  Std embedding norm: {np.std(np.linalg.norm(all_embeddings, axis=1)):.4f}")
        print(f"  Min embedding norm: {np.min(np.linalg.norm(all_embeddings, axis=1)):.4f}")
        print(f"  Max embedding norm: {np.max(np.linalg.norm(all_embeddings, axis=1)):.4f}")
        
        # Test 5: Vietnamese poem-specific embeddings
        print("\n5. Testing Vietnamese poem embeddings...")
        
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
        
        poem_embeddings = []
        for text in poem_texts:
            emb = self.encode_text(text)
            poem_embeddings.append(emb.flatten())
        
        poem_embeddings = np.array(poem_embeddings)
        
        # Calculate similarity matrix for poem lines
        poem_sim_matrix = cosine_similarity(poem_embeddings)
        
        print("  Poem line similarities:")
        for i in range(len(poem_texts)):
            for j in range(i+1, len(poem_texts)):
                sim = poem_sim_matrix[i, j]
                print(f"    '{poem_texts[i][:20]}...' vs '{poem_texts[j][:20]}...': {sim:.4f}")
        
        # Test 6: Compare with simple prompts
        print("\n6. Testing poem vs simple prompts...")
        
        simple_prompts = [
            "một người phụ nữ",
            "con đường phố",
            "bầu trời xanh",
            "hoa lá cây cối",
            "ánh sáng mặt trời",
            "nước biển xanh",
            "mây trắng",
            "tình yêu"
        ]
        
        poem_avg_emb = np.mean(poem_embeddings, axis=0).reshape(1, -1)
        
        for prompt in simple_prompts:
            prompt_emb = self.encode_text(prompt)
            sim = cosine_similarity(poem_avg_emb, prompt_emb)[0][0]
            print(f"  Poem average vs '{prompt}': {sim:.4f}")
        
        return {
            'related_similarity': np.mean(similarities),
            'unrelated_similarity': np.mean(dissimilarities),
            'self_similarity': np.mean(self_similarities),
            'embedding_stats': {
                'mean_norm': np.mean(np.linalg.norm(all_embeddings, axis=1)),
                'std_norm': np.std(np.linalg.norm(all_embeddings, axis=1))
            }
        }
    
    def test_diffusion_compatibility(self):
        """Test if PhoCLIP embeddings are compatible with diffusion models"""
        print("\n" + "="*60)
        print("TESTING DIFFUSION MODEL COMPATIBILITY")
        print("="*60)
        
        # Test prompts that should work well with diffusion models
        diffusion_prompts = [
            "một người phụ nữ đẹp",
            "con đường phố đông đúc",
            "bầu trời xanh với mây trắng",
            "hoa hồng đỏ trong vườn",
            "mặt trời mọc trên biển",
            "ngôi nhà cổ kính",
            "rừng cây xanh mát",
            "dòng sông chảy êm đềm"
        ]
        
        print("\nTesting PhoCLIP embeddings for diffusion prompts...")
        
        embeddings = []
        for prompt in diffusion_prompts:
            emb = self.encode_text(prompt)
            embeddings.append(emb.flatten())
            print(f"  '{prompt}': embedding shape {emb.shape}, norm {np.linalg.norm(emb):.4f}")
        
        embeddings = np.array(embeddings)
        
        # Check embedding diversity
        sim_matrix = cosine_similarity(embeddings)
        print(f"\nEmbedding similarity matrix shape: {sim_matrix.shape}")
        print(f"Average similarity between different prompts: {np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)]):.4f}")
        
        # Test embedding projection to diffusion model dimensions
        print("\nTesting embedding projection...")
        
        # Simulate the projection used in your pipeline
        phoclip_dim = 768
        diffusion_dim = 768  # This should match your diffusion model's expected dimension
        
        if phoclip_dim != diffusion_dim:
            projection = torch.nn.Linear(phoclip_dim, diffusion_dim).to(self.device)
            torch.nn.init.xavier_uniform_(projection.weight)
            
            projected_embeddings = []
            for emb in embeddings:
                emb_tensor = torch.from_numpy(emb).unsqueeze(0).to(self.device, dtype=torch.float16)
                proj_emb = projection(emb_tensor)
                projected_embeddings.append(proj_emb.cpu().numpy().flatten())
            
            projected_embeddings = np.array(projected_embeddings)
            print(f"  Original embedding shape: {embeddings.shape}")
            print(f"  Projected embedding shape: {projected_embeddings.shape}")
            print(f"  Projection preserves similarity: {np.corrcoef(embeddings.flatten(), projected_embeddings.flatten())[0,1]:.4f}")
        
        return {
            'embedding_diversity': np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)]),
            'embedding_norms': [np.linalg.norm(emb) for emb in embeddings]
        }
    
    def generate_test_report(self, results):
        """Generate a comprehensive test report"""
        print("\n" + "="*60)
        print("PHOCLIP EMBEDDING QUALITY REPORT")
        print("="*60)
        
        print(f"\n1. SEMANTIC SIMILARITY TESTS:")
        print(f"   Related texts similarity: {results['related_similarity']:.4f}")
        print(f"   Unrelated texts similarity: {results['unrelated_similarity']:.4f}")
        print(f"   Self-similarity: {results['self_similarity']:.4f}")
        
        print(f"\n2. EMBEDDING STATISTICS:")
        print(f"   Mean embedding norm: {results['embedding_stats']['mean_norm']:.4f}")
        print(f"   Std embedding norm: {results['embedding_stats']['std_norm']:.4f}")
        
        print(f"\n3. DIFFUSION COMPATIBILITY:")
        print(f"   Embedding diversity: {results['diffusion_compatibility']['embedding_diversity']:.4f}")
        
        # Analysis and recommendations
        print(f"\n4. ANALYSIS:")
        
        if results['related_similarity'] > 0.7:
            print("   ✓ Related texts show good semantic similarity")
        else:
            print("   ✗ Related texts show poor semantic similarity")
        
        if results['unrelated_similarity'] < 0.3:
            print("   ✓ Unrelated texts show good discrimination")
        else:
            print("   ✗ Unrelated texts show poor discrimination")
        
        if results['self_similarity'] > 0.99:
            print("   ✓ Self-similarity is correct (close to 1.0)")
        else:
            print("   ✗ Self-similarity is incorrect (should be 1.0)")
        
        if results['embedding_stats']['mean_norm'] > 0.9:
            print("   ✓ Embeddings are properly normalized")
        else:
            print("   ✗ Embeddings may not be properly normalized")
        
        print(f"\n5. RECOMMENDATIONS:")
        
        if results['related_similarity'] < 0.7:
            print("   - PhoCLIP text encoder may need retraining or fine-tuning")
            print("   - Check if the model weights are properly loaded")
        
        if results['unrelated_similarity'] > 0.3:
            print("   - PhoCLIP may not be discriminating enough between different concepts")
            print("   - Consider adjusting the temperature parameter or model architecture")
        
        if results['embedding_stats']['mean_norm'] < 0.9:
            print("   - Embeddings are not properly normalized")
            print("   - Check the normalization step in the encoding process")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Test PhoCLIP embedding quality")
    parser.add_argument("--model-repo", type=str, default="kienhoang123/ViCLIP", 
                       help="Hugging Face model repository")
    parser.add_argument("--fallback-path", type=str, default="e:/ViCLIP/phoclip_checkpoint",
                       help="Local model fallback path")
    parser.add_argument("--output-report", type=str, default="phoclip_embedding_report.json",
                       help="Output report file")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = PhoCLIPEmbeddingTester(args.model_repo, args.fallback_path)
    
    # Run tests
    quality_results = tester.test_embedding_quality()
    diffusion_results = tester.test_diffusion_compatibility()
    
    # Combine results
    all_results = {
        **quality_results,
        'diffusion_compatibility': diffusion_results
    }
    
    # Generate report
    tester.generate_test_report(all_results)
    
    # Save results
    with open(args.output_report, 'w', encoding='utf-8') as f:
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_numpy_types(all_results), f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {args.output_report}")

if __name__ == "__main__":
    main() 