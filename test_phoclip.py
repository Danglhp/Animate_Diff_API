import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    CLIPFeatureExtractor, 
    CLIPVisionModel
)
import torch.nn as nn
import glob
from tqdm import tqdm
import random

class PhoCLIPModel(nn.Module):
    """PhoCLIP: PhoBERT text encoder + CLIP ViT vision encoder + projection heads."""
    def __init__(
        self,
        text_encoder_name: str = "vinai/phobert-base",
        vision_encoder_name: str = "openai/clip-vit-base-patch32",
        embed_dim: int = 768,
        temperature_init: float = 0.07,
    ):
        super().__init__()
        # Text encoder (PhoBERT)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Vision encoder (CLIP ViT)
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_encoder_name)
        self.vision_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / temperature_init)))

    def encode_text(self, input_ids, attention_mask):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        text_cls = text_outputs.last_hidden_state[:, 0, :]
        text_proj = self.text_proj(text_cls)
        text_emb = F.normalize(text_proj, p=2, dim=-1)
        return text_emb

    def encode_image(self, pixel_values):
        vis_outputs = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
        vis_cls = vis_outputs.pooler_output
        vis_proj = self.vision_proj(vis_cls)
        vis_emb = F.normalize(vis_proj, p=2, dim=-1)
        return vis_emb

    def forward(self, input_ids, attention_mask, pixel_values):
        # Text side
        text_emb = self.encode_text(input_ids, attention_mask)

        # Vision side
        vis_emb = self.encode_image(pixel_values)

        logit_scale = self.logit_scale.exp()
        return text_emb, vis_emb, logit_scale

def load_model(checkpoint_dir):
    """Load the trained PhoCLIP model from checkpoint directory."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load tokenizer and feature extractor
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(checkpoint_dir, 'tokenizer'), use_fast=False)
    feature_extractor = CLIPFeatureExtractor.from_pretrained('openai/clip-vit-base-patch32')
    
    # Initialize model
    model = PhoCLIPModel(
        text_encoder_name=os.path.join(checkpoint_dir, 'text_encoder'),
        vision_encoder_name=os.path.join(checkpoint_dir, 'vision_encoder'),
    ).to(device)
    
    # Load projection heads
    model.text_proj.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'text_proj.pt'), map_location=device))
    model.vision_proj.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'vision_proj.pt'), map_location=device))
    
    model.eval()  # Set to evaluation mode
    return model, tokenizer, feature_extractor, device

def encode_text(text, model, tokenizer, device):
    """Encode a Vietnamese text query."""
    encoded = tokenizer(
        text,
        padding='max_length',
        max_length=77,
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    
    with torch.no_grad():
        text_emb = model.encode_text(input_ids, attention_mask)
    
    return text_emb

def encode_image(image_path, model, feature_extractor, device):
    """Encode an image."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    pixel_values = feature_extractor(images=image, return_tensors='pt')['pixel_values'].to(device)
    
    with torch.no_grad():
        image_emb = model.encode_image(pixel_values)
    
    return image_emb

def find_best_matches(text_emb, image_paths, model, feature_extractor, device, top_k=5, max_images=None):
    """Find top_k images that best match the text query."""
    similarities = []
    
    # Limit the number of images to process if specified
    if max_images and len(image_paths) > max_images:
        import random
        random.seed(42)
        image_paths = random.sample(image_paths, max_images)
        print(f"Processing {max_images} randomly selected images out of {len(image_paths)} total")
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            image_emb = encode_image(img_path, model, feature_extractor, device)
            similarity = (text_emb @ image_emb.T).item()
            similarities.append((img_path, similarity))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]

def display_results(text_query, matches):
    """Display the text query and the matching images."""
    plt.figure(figsize=(15, 10))
    plt.suptitle(f'Query: "{text_query}"', fontsize=16)
    
    for i, (img_path, similarity) in enumerate(matches):
        plt.subplot(1, len(matches), i+1)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(f"Similarity: {similarity:.3f}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results.png')
    plt.show()

def load_training_data():
    """Load the training captions to check for overfitting on training data."""
    import pandas as pd
    try:
        df = pd.read_csv('all_captions_merged.csv')
        return df
    except FileNotFoundError:
        print("Warning: all_captions_merged.csv not found. Using general queries for testing.")
        return None

def analyze_overfitting(model, tokenizer, feature_extractor, device, sample_size=100):
    """Comprehensive overfitting analysis including training data memorization check."""
    print("=" * 80)
    print("COMPREHENSIVE OVERFITTING ANALYSIS")
    print("=" * 80)
    
    # Get sample of images
    images_dir = "images"
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
    
    print(f"Total images available: {len(image_paths)}")
    
    if len(image_paths) < sample_size:
        sample_size = len(image_paths)
        print(f"Using all {sample_size} available images")
    else:
        print(f"Sampling {sample_size} images from {len(image_paths)} total images")
    
    # Random sample for diversity
    random.seed(42)  # For reproducible results
    sample_paths = random.sample(image_paths, sample_size)
    
    # Load training data for memorization check
    training_df = load_training_data()
    
    # 1. TRAINING DATA MEMORIZATION CHECK
    print("\n" + "=" * 50)
    print("1. TRAINING DATA MEMORIZATION CHECK")
    print("=" * 50)
    
    if training_df is not None:
        # Test on actual training captions
        training_sample = training_df.sample(n=min(50, len(training_df)), random_state=42)
        training_memorization_scores = []
        
        for _, row in training_sample.iterrows():
            image_name = row['image']
            caption = row['caption_vi']
            
            # Find the actual image file
            matching_images = [path for path in image_paths if image_name in os.path.basename(path)]
            
            if matching_images:
                try:
                    text_emb = encode_text(caption, model, tokenizer, device)
                    image_emb = encode_image(matching_images[0], model, feature_extractor, device)
                    similarity = (text_emb @ image_emb.T).item()
                    training_memorization_scores.append(similarity)
                except Exception as e:
                    continue
        
        if training_memorization_scores:
            avg_training_sim = sum(training_memorization_scores) / len(training_memorization_scores)
            print(f"Average similarity on training pairs: {avg_training_sim:.4f}")
            print(f"Training pairs tested: {len(training_memorization_scores)}")
            
            # Check for memorization
            if avg_training_sim > 0.9:
                print("🚨 STRONG MEMORIZATION DETECTED - Very high similarity on training data")
            elif avg_training_sim > 0.8:
                print("⚠️  MODERATE MEMORIZATION - High similarity on training data")
            else:
                print("✅ Good - Reasonable similarity on training data")
    
    # 2. DIVERSITY TEST - Test on unseen caption styles
    print("\n" + "=" * 50)
    print("2. GENERALIZATION TO UNSEEN CAPTION STYLES")
    print("=" * 50)
    
    # Test queries with different complexity levels
    test_queries = {
        "simple": [
            "một người",
            "con chó", 
            "ngôi nhà",
            "chiếc xe",
            "cây xanh"
        ],
        "medium": [
            "người đàn ông đang cười",
            "con chó đang chạy trên cỏ",
            "ngôi nhà màu xanh bên sông",
            "chiếc xe ô tô màu đỏ",
            "cây xanh to lớn trong công viên"
        ],
        "complex": [
            "một gia đình hạnh phúc đang cùng nhau ăn tối",
            "đàn chó con đang chơi đùa trong vườn hoa",
            "ngôi nhà cổ kính bên bờ biển vào lúc hoàng hôn",
            "chiếc xe đạp cũ đậu bên cửa hàng bánh mì",
            "khu rừng xanh mướt với ánh nắng xuyên qua tán lá"
        ]
    }
    
    results = {}
    
    for complexity, queries in test_queries.items():
        print(f"\nTesting {complexity.upper()} queries...")
        similarities = []
        
        for query in queries:
            print(f"  Query: '{query}'")
            text_emb = encode_text(query, model, tokenizer, device)
            
            query_similarities = []
            for img_path in sample_paths[:20]:  # Test on subset for each query
                try:
                    image_emb = encode_image(img_path, model, feature_extractor, device)
                    similarity = (text_emb @ image_emb.T).item()
                    query_similarities.append(similarity)
                except Exception as e:
                    continue
            
            if query_similarities:
                avg_sim = sum(query_similarities) / len(query_similarities)
                max_sim = max(query_similarities)
                min_sim = min(query_similarities)
                std_sim = (sum([(s - avg_sim)**2 for s in query_similarities]) / len(query_similarities))**0.5
                
                similarities.append({
                    'avg': avg_sim,
                    'max': max_sim,
                    'min': min_sim,
                    'std': std_sim,
                    'range': max_sim - min_sim
                })
        
        if similarities:
            avg_scores = [s['avg'] for s in similarities]
            std_scores = [s['std'] for s in similarities]
            range_scores = [s['range'] for s in similarities]
            
            results[complexity] = {
                'mean_similarity': sum(avg_scores) / len(avg_scores),
                'mean_std': sum(std_scores) / len(std_scores),
                'mean_range': sum(range_scores) / len(range_scores)
            }
    
    # Analyze results for overfitting indicators
    print("\n" + "=" * 60)
    print("OVERFITTING ANALYSIS RESULTS")
    print("=" * 60)
    
    # Check for signs of overfitting
    overfitting_indicators = []
    
    for complexity, stats in results.items():
        print(f"\n{complexity.upper()} Queries:")
        print(f"  Mean Similarity: {stats['mean_similarity']:.4f}")
        print(f"  Mean Std Dev:    {stats['mean_std']:.4f}")
        print(f"  Mean Range:      {stats['mean_range']:.4f}")
        
        # Overfitting indicators
        if stats['mean_similarity'] > 0.8:
            overfitting_indicators.append(f"High similarity scores for {complexity} queries")
        if stats['mean_std'] < 0.1:
            overfitting_indicators.append(f"Low variance for {complexity} queries")
        if stats['mean_range'] < 0.2:
            overfitting_indicators.append(f"Small similarity range for {complexity} queries")
    
    # Overall assessment
    print("\n" + "=" * 60)
    print("OVERFITTING ASSESSMENT")
    print("=" * 60)
    
    if len(overfitting_indicators) >= 3:
        print("🚨 HIGH RISK OF OVERFITTING:")
        for indicator in overfitting_indicators:
            print(f"  • {indicator}")
        print("\nRecommendations:")
        print("  • Add more diverse training data")
        print("  • Implement regularization techniques")
        print("  • Use data augmentation")
        print("  • Reduce model complexity")
    elif len(overfitting_indicators) >= 1:
        print("⚠️  MODERATE RISK OF OVERFITTING:")
        for indicator in overfitting_indicators:
            print(f"  • {indicator}")
        print("\nRecommendations:")
        print("  • Monitor validation performance")
        print("  • Consider early stopping")
        print("  • Add dropout or other regularization")
    else:
        print("✅ LOW RISK OF OVERFITTING:")
        print("  Model shows good generalization across different query complexities")
    
    return results

def test_animatediff_compatibility(model, tokenizer, device):
    """Test how well the model works as a text encoder for AnimateDiff-style prompts."""
    print("\n" + "=" * 80)
    print("ANIMATEDIFF COMPATIBILITY TEST")
    print("=" * 80)
    
    # AnimateDiff-style prompts (motion + visual description)
    animatediff_prompts = [
        # Simple motion descriptions
        "một người đàn ông đang đi bộ",
        "cô gái đang chạy",
        "con chó đang nhảy",
        "chiếc xe đang di chuyển",
        
        # Complex motion + scene descriptions
        "một người phụ nữ đang nhảy múa trong công viên, ánh nắng chiều",
        "đàn chim đang bay trên bầu trời xanh, có mây trắng",
        "sóng biển đang vỗ vào bờ cát, hoàng hôn",
        "cây lá đang rung rinh trong gió, mùa thu",
        
        # Detailed cinematic prompts (typical for AnimateDiff)
        "một cô gái tóc dài đang đi bộ trên phố cổ, ánh đèn vàng, buổi tối",
        "người đàn ông lái xe máy qua cầu, phong cảnh thành phố, ban ngày",
        "em bé đang chơi trong vườn hoa, bướm bay xung quanh, mùa xuân",
        
        # Style-specific prompts
        "phong cách anime, cô gái đang ngồi uống cà phê, quán café vintage",
        "phong cách realistic, người thợ làm bánh trong bếp, ánh sáng tự nhiên",
        "phong cách watercolor, phong cảnh núi non mờ ảo, sương mù"
    ]
    
    print(f"Testing {len(animatediff_prompts)} AnimateDiff-style prompts...")
    
    # Test embedding quality and consistency
    embeddings = []
    embedding_stats = []
    
    for i, prompt in enumerate(animatediff_prompts):
        print(f"  {i+1:2d}. Testing: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        
        try:
            # Get text embedding
            text_emb = encode_text(prompt, model, tokenizer, device)
            embeddings.append(text_emb.cpu().numpy())
            
            # Calculate embedding statistics
            emb_norm = torch.norm(text_emb).item()
            emb_mean = torch.mean(text_emb).item()
            emb_std = torch.std(text_emb).item()
            
            embedding_stats.append({
                'prompt': prompt,
                'norm': emb_norm,
                'mean': emb_mean,
                'std': emb_std
            })
            
        except Exception as e:
            print(f"    ❌ Error processing prompt: {e}")
    
    # Analyze embedding characteristics
    print("\n" + "=" * 50)
    print("EMBEDDING ANALYSIS")
    print("=" * 50)
    
    if embedding_stats:
        norms = [stat['norm'] for stat in embedding_stats]
        means = [stat['mean'] for stat in embedding_stats]
        stds = [stat['std'] for stat in embedding_stats]
        
        print(f"Embedding Statistics:")
        print(f"  Average Norm: {sum(norms)/len(norms):.4f} ± {(sum([(n-sum(norms)/len(norms))**2 for n in norms])/len(norms))**0.5:.4f}")
        print(f"  Average Mean: {sum(means)/len(means):.4f} ± {(sum([(m-sum(means)/len(means))**2 for m in means])/len(means))**0.5:.4f}")
        print(f"  Average Std:  {sum(stds)/len(stds):.4f} ± {(sum([(s-sum(stds)/len(stds))**2 for s in stds])/len(stds))**0.5:.4f}")
        
        # Check for good embedding properties
        norm_consistency = (sum([(n-sum(norms)/len(norms))**2 for n in norms])/len(norms))**0.5
        
        print(f"\nEmbedding Quality Assessment:")
        if norm_consistency < 0.1:
            print("  ✅ Consistent embedding norms - Good for stable generation")
        elif norm_consistency < 0.2:
            print("  ⚠️  Moderate norm variation - Should work but may need fine-tuning")
        else:
            print("  🚨 High norm variation - May cause unstable generation")
    
    # Test semantic similarity between related prompts
    print("\n" + "=" * 50)
    print("SEMANTIC SIMILARITY TEST")
    print("=" * 50)
    
    if len(embeddings) >= 4:
        import numpy as np
        embeddings_np = np.array([emb.flatten() for emb in embeddings])
        
        # Test similarity between motion-related prompts
        motion_simple = embeddings_np[0:4]  # Simple motion prompts
        motion_complex = embeddings_np[4:8]  # Complex motion prompts
        
        # Calculate average inter-group similarity
        similarities = []
        for i in range(len(motion_simple)):
            for j in range(len(motion_complex)):
                sim = np.dot(motion_simple[i], motion_complex[j]) / (
                    np.linalg.norm(motion_simple[i]) * np.linalg.norm(motion_complex[j])
                )
                similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities)
        print(f"Average similarity between simple and complex motion prompts: {avg_similarity:.4f}")
        
        if avg_similarity > 0.7:
            print("  ✅ Good semantic understanding - Captures motion concepts well")
        elif avg_similarity > 0.5:
            print("  ⚠️  Moderate semantic understanding - May need more training on motion")
        else:
            print("  🚨 Weak semantic understanding - Not ideal for motion generation")
    
    # Test prompt length handling
    print("\n" + "=" * 50)
    print("PROMPT LENGTH HANDLING")
    print("=" * 50)
    
    length_test_prompts = [
        "ngắn",  # Very short
        "một người đàn ông đang đi bộ trong công viên",  # Medium
        "một cô gái tóc dài mặc áo dài truyền thống đang đi bộ chậm rãi qua con đường phố cổ kính với những ngôi nhà màu vàng, ánh nắng chiều tà, không khí yên bình",  # Long
    ]
    
    for i, prompt in enumerate(length_test_prompts):
        try:
            text_emb = encode_text(prompt, model, tokenizer, device)
            print(f"  Length {len(prompt):3d} chars: ✅ Processed successfully")
        except Exception as e:
            print(f"  Length {len(prompt):3d} chars: ❌ Error - {e}")
    
    # Overall AnimateDiff compatibility assessment
    print("\n" + "=" * 60)
    print("ANIMATEDIFF COMPATIBILITY ASSESSMENT")
    print("=" * 60)
    
    compatibility_score = 0
    total_tests = 4
    
    if embedding_stats:
        compatibility_score += 1
        print("✅ Text encoding works for Vietnamese prompts")
    
    if norm_consistency < 0.2:
        compatibility_score += 1
        print("✅ Embedding consistency is acceptable")
    
    if avg_similarity > 0.5:
        compatibility_score += 1
        print("✅ Semantic understanding is reasonable")
    
    if len([stat for stat in embedding_stats if stat['norm'] > 0.5]) == len(embedding_stats):
        compatibility_score += 1
        print("✅ Embedding magnitudes are appropriate")
    
    print(f"\nCompatibility Score: {compatibility_score}/{total_tests}")
    
    if compatibility_score >= 3:
        print("🎉 HIGH COMPATIBILITY - Model should work well with AnimateDiff")
        print("   Recommendations:")
        print("   • Can be used as text encoder for Vietnamese AnimateDiff")
        print("   • Consider fine-tuning on motion-specific Vietnamese captions")
    elif compatibility_score >= 2:
        print("⚠️  MODERATE COMPATIBILITY - May work with some adjustments")
        print("   Recommendations:")
        print("   • Test with actual AnimateDiff pipeline")
        print("   • May need embedding normalization")
        print("   • Consider additional training on motion data")
    else:
        print("🚨 LOW COMPATIBILITY - Significant issues detected")
        print("   Recommendations:")
        print("   • More training needed on diverse Vietnamese text")
        print("   • Check tokenizer compatibility")
        print("   • Consider using a different text encoder")
    
    return embedding_stats

def main():
    parser = argparse.ArgumentParser(description='Test PhoCLIP on Vietnamese text and images')
    parser.add_argument('--checkpoint', type=str, default='phoclip_checkpoint', help='Path to model checkpoint')
    parser.add_argument('--query', type=str, help='Vietnamese text query for image search')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top matches to return')
    parser.add_argument('--analyze_overfitting', action='store_true', help='Run overfitting analysis')
    parser.add_argument('--test_animatediff', action='store_true', help='Test AnimateDiff compatibility')
    parser.add_argument('--max_images', type=int, help='Maximum number of images to process for query search (for faster testing)')
    parser.add_argument('--sample_size', type=int, default=100, help='Sample size for overfitting analysis')
    args = parser.parse_args()
    
    # Load model and supporting components
    model, tokenizer, feature_extractor, device = load_model(args.checkpoint)
    
    # Use images from the local images folder
    images_dir = "images"
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
    
    print(f"Found {len(image_paths)} images in the '{images_dir}' folder")
    
    if args.analyze_overfitting:
        # Run overfitting analysis
        analyze_overfitting(model, tokenizer, feature_extractor, device, args.sample_size)
    
    if args.test_animatediff:
        # Test AnimateDiff compatibility
        test_animatediff_compatibility(model, tokenizer, device)
    
    if args.query:
        # Encode the text query
        text_emb = encode_text(args.query, model, tokenizer, device)
        
        # Find the best matching images
        best_matches = find_best_matches(text_emb, image_paths, model, feature_extractor, device, args.top_k, args.max_images)
        
        # Display results
        display_results(args.query, best_matches)
        
        # Also print results to console
        print(f"\nTop {args.top_k} matches for query: '{args.query}'")
        for i, (img_path, similarity) in enumerate(best_matches):
            print(f"{i+1}. {os.path.basename(img_path)} - Similarity: {similarity:.4f}")
    
    if not args.query and not args.analyze_overfitting and not args.test_animatediff:
        print("Please provide one of the following options:")
        print("Example usage:")
        print("  python test_phoclip.py --query 'một người đàn ông'")
        print("  python test_phoclip.py --analyze_overfitting")
        print("  python test_phoclip.py --test_animatediff")
        print("  python test_phoclip.py --analyze_overfitting --test_animatediff")

if __name__ == '__main__':
    main()