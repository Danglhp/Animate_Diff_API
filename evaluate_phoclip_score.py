import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import csv
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import os
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class PhoCLIPEvaluator:
    def __init__(self, model_path="e:/ViCLIP/phoclip_checkpoint", device=None):
        """Initialize PhoCLIP model for evaluation"""
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        print(f"Loading PhoCLIP model from {model_path}")
        self._load_model()
        
    def _load_model(self):
        """Load the PhoCLIP model components"""
        # Load text encoder (PhoBERT)
        self.text_encoder = AutoModel.from_pretrained(f"{self.model_path}/text_encoder").to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_path}/tokenizer", use_fast=False)
        
        # Load text projection head
        self.text_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        ).to(self.device)
        self.text_proj.load_state_dict(torch.load(f"{self.model_path}/text_proj.pt", map_location=self.device))
        
        # Load vision encoder and projection if available
        try:
            from transformers import CLIPVisionModel
            self.vision_encoder = CLIPVisionModel.from_pretrained(f"{self.model_path}/vision_encoder").to(self.device)
            
            self.vision_proj = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Linear(768, 768)
            ).to(self.device)
            self.vision_proj.load_state_dict(torch.load(f"{self.model_path}/vision_proj.pt", map_location=self.device))
            
        except Exception as e:
            print(f"Vision components not found or failed to load: {e}")
            self.vision_encoder = None
            self.vision_proj = None
        
        # Set to evaluation mode
        self.text_encoder.eval()
        self.text_proj.eval()
        if self.vision_encoder:
            self.vision_encoder.eval()
            self.vision_proj.eval()
            
        print("PhoCLIP model loaded successfully")
    
    def encode_text(self, texts):
        """Encode text using PhoCLIP text encoder"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize the text
                inputs = self.tokenizer(
                    text, 
                    padding='max_length', 
                    max_length=77, 
                    truncation=True, 
                    return_tensors='pt'
                ).to(self.device)
                
                # Get text embeddings
                text_outputs = self.text_encoder(
                    input_ids=inputs.input_ids, 
                    attention_mask=inputs.attention_mask, 
                    return_dict=True
                )
                text_cls = text_outputs.last_hidden_state[:, 0, :]
                text_proj = self.text_proj(text_cls)
                text_emb = F.normalize(text_proj, p=2, dim=-1)
                
                embeddings.append(text_emb.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def encode_image(self, images):
        """Encode images using PhoCLIP vision encoder (if available)"""
        if self.vision_encoder is None:
            raise ValueError("Vision encoder not available")
        
        if not isinstance(images, list):
            images = [images]
        
        embeddings = []
        
        with torch.no_grad():
            for image in images:
                if isinstance(image, str):
                    image = Image.open(image).convert('RGB')
                
                # Preprocess image (assuming CLIP preprocessing)
                from transformers import CLIPImageProcessor
                processor = CLIPImageProcessor()
                inputs = processor(images=image, return_tensors="pt").to(self.device)
                
                # Get image embeddings
                vision_outputs = self.vision_encoder(**inputs)
                image_emb = self.vision_proj(vision_outputs.pooler_output)
                image_emb = F.normalize(image_emb, p=2, dim=-1)
                
                embeddings.append(image_emb.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def compute_similarity(self, text_embeddings, image_embeddings=None, text_embeddings_2=None):
        """Compute cosine similarity between embeddings"""
        if image_embeddings is not None:
            # Text-Image similarity
            return cosine_similarity(text_embeddings, image_embeddings)
        elif text_embeddings_2 is not None:
            # Text-Text similarity
            return cosine_similarity(text_embeddings, text_embeddings_2)
        else:
            raise ValueError("Need either image_embeddings or text_embeddings_2")


class DatasetEvaluator:
    def __init__(self, phoclip_evaluator):
        self.phoclip = phoclip_evaluator
        
    def evaluate_flickr_dataset(self, flickr_path="e:/ViCLIP/flickr"):
        """Evaluate on Flickr dataset"""
        print("Evaluating on Flickr dataset...")
        
        # Load captions
        captions_file = Path(flickr_path) / "captions_vi.txt"
        if not captions_file.exists():
            print(f"Captions file not found: {captions_file}")
            return None
        
        # Read captions
        captions_data = {}
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    img_name, caption = line.strip().split('\t', 1)
                    if img_name not in captions_data:
                        captions_data[img_name] = []
                    captions_data[img_name].append(caption)
        
        print(f"Loaded {len(captions_data)} images with captions")
        
        # Evaluate text similarity (since we don't have vision encoder working)
        scores = []
        
        for img_name, captions in tqdm(captions_data.items(), desc="Computing similarities"):
            if len(captions) >= 2:
                # Compute similarity between different captions of the same image
                emb1 = self.phoclip.encode_text(captions[0])
                emb2 = self.phoclip.encode_text(captions[1])
                sim = cosine_similarity(emb1, emb2)[0][0]
                scores.append(sim)
        
        return {
            'mean_similarity': np.mean(scores),
            'std_similarity': np.std(scores),
            'scores': scores,
            'num_samples': len(scores)
        }
    
    def evaluate_ktvic_dataset(self, ktvic_path="e:/ViCLIP/ktvic"):
        """Evaluate on KTVIC dataset"""
        print("Evaluating on KTVIC dataset...")
        
        datasets = ['train', 'test']
        all_results = {}
        
        for dataset in datasets:
            json_file = Path(ktvic_path) / "ktvic_dataset" / f"{dataset}_data.json"
            if not json_file.exists():
                print(f"Dataset file not found: {json_file}")
                continue
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            scores = []
            
            # Group captions by image
            image_captions = {}
            for item in data['annotations']:
                img_id = item['image_id']
                if img_id not in image_captions:
                    image_captions[img_id] = []
                image_captions[img_id].append(item['caption'])
            
            # Compute similarities between captions of the same image
            for img_id, captions in tqdm(image_captions.items(), desc=f"Evaluating {dataset}"):
                if len(captions) >= 2:
                    for i in range(len(captions)):
                        for j in range(i+1, len(captions)):
                            emb1 = self.phoclip.encode_text(captions[i])
                            emb2 = self.phoclip.encode_text(captions[j])
                            sim = cosine_similarity(emb1, emb2)[0][0]
                            scores.append(sim)
            
            all_results[dataset] = {
                'mean_similarity': np.mean(scores) if scores else 0,
                'std_similarity': np.std(scores) if scores else 0,
                'scores': scores,
                'num_samples': len(scores)
            }
        
        return all_results
    
    def evaluate_uitviic_dataset(self, uitviic_path="e:/ViCLIP/UIT-ViIC"):
        """Evaluate on UIT-ViIC dataset"""
        print("Evaluating on UIT-ViIC dataset...")
        
        datasets = ['train2017', 'val2017', 'test2017']
        all_results = {}
        
        for dataset in datasets:
            json_file = Path(uitviic_path) / f"uitviic_captions_{dataset}.json"
            if not json_file.exists():
                print(f"Dataset file not found: {json_file}")
                continue
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            scores = []
            
            # Group captions by image
            image_captions = {}
            for item in data['annotations']:
                img_id = item['image_id']
                if img_id not in image_captions:
                    image_captions[img_id] = []
                image_captions[img_id].append(item['caption'])
            
            # Compute similarities between captions of the same image
            for img_id, captions in tqdm(image_captions.items(), desc=f"Evaluating {dataset}"):
                if len(captions) >= 2:
                    for i in range(len(captions)):
                        for j in range(i+1, len(captions)):
                            emb1 = self.phoclip.encode_text(captions[i])
                            emb2 = self.phoclip.encode_text(captions[j])
                            sim = cosine_similarity(emb1, emb2)[0][0]
                            scores.append(sim)
            
            all_results[dataset] = {
                'mean_similarity': np.mean(scores) if scores else 0,
                'std_similarity': np.std(scores) if scores else 0,
                'scores': scores,
                'num_samples': len(scores)
            }
        
        return all_results


def plot_results(results, output_dir="evaluation_results"):
    """Plot evaluation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Mean similarities
    plt.subplot(2, 2, 1)
    datasets = []
    means = []
    stds = []
    
    for dataset_name, dataset_results in results.items():
        if isinstance(dataset_results, dict) and 'mean_similarity' in dataset_results:
            datasets.append(dataset_name)
            means.append(dataset_results['mean_similarity'])
            stds.append(dataset_results['std_similarity'])
        elif isinstance(dataset_results, dict):
            for subset_name, subset_results in dataset_results.items():
                if 'mean_similarity' in subset_results:
                    datasets.append(f"{dataset_name}_{subset_name}")
                    means.append(subset_results['mean_similarity'])
                    stds.append(subset_results['std_similarity'])
    
    plt.bar(datasets, means, yerr=stds, capsize=5)
    plt.title('Mean Similarity Scores by Dataset')
    plt.ylabel('Cosine Similarity')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of scores
    plt.subplot(2, 2, 2)
    all_scores = []
    labels = []
    
    for dataset_name, dataset_results in results.items():
        if isinstance(dataset_results, dict) and 'scores' in dataset_results:
            all_scores.extend(dataset_results['scores'])
            labels.extend([dataset_name] * len(dataset_results['scores']))
        elif isinstance(dataset_results, dict):
            for subset_name, subset_results in dataset_results.items():
                if 'scores' in subset_results:
                    all_scores.extend(subset_results['scores'])
                    labels.extend([f"{dataset_name}_{subset_name}"] * len(subset_results['scores']))
    
    plt.hist(all_scores, bins=50, alpha=0.7)
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Sample counts
    plt.subplot(2, 2, 3)
    sample_counts = []
    dataset_names = []
    
    for dataset_name, dataset_results in results.items():
        if isinstance(dataset_results, dict) and 'num_samples' in dataset_results:
            dataset_names.append(dataset_name)
            sample_counts.append(dataset_results['num_samples'])
        elif isinstance(dataset_results, dict):
            for subset_name, subset_results in dataset_results.items():
                if 'num_samples' in subset_results:
                    dataset_names.append(f"{dataset_name}_{subset_name}")
                    sample_counts.append(subset_results['num_samples'])
    
    plt.bar(dataset_names, sample_counts)
    plt.title('Number of Samples by Dataset')
    plt.ylabel('Sample Count')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/phoclip_evaluation_results.png", dpi=300, bbox_inches='tight')
    plt.show()


def save_results(results, output_file="phoclip_evaluation_results.json"):
    """Save results to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for dataset_name, dataset_results in results.items():
        if isinstance(dataset_results, dict):
            json_results[dataset_name] = {}
            for key, value in dataset_results.items():
                if key == 'scores' and isinstance(value, list):
                    json_results[dataset_name][key] = [float(score) for score in value]
                elif isinstance(value, (np.float32, np.float64)):
                    json_results[dataset_name][key] = float(value)
                elif isinstance(value, dict):
                    json_results[dataset_name][key] = {}
                    for subkey, subvalue in value.items():
                        if subkey == 'scores' and isinstance(subvalue, list):
                            json_results[dataset_name][key][subkey] = [float(score) for score in subvalue]
                        elif isinstance(subvalue, (np.float32, np.float64)):
                            json_results[dataset_name][key][subkey] = float(subvalue)
                        else:
                            json_results[dataset_name][key][subkey] = subvalue
                else:
                    json_results[dataset_name][key] = value
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")


def generate_report(results, output_file="phoclip_evaluation_report.txt"):
    """Generate a text report of the evaluation results"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("PhoCLIP Model Evaluation Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for dataset_name, dataset_results in results.items():
            f.write(f"Dataset: {dataset_name}\n")
            f.write("-" * 30 + "\n")
            
            if isinstance(dataset_results, dict) and 'mean_similarity' in dataset_results:
                f.write(f"Mean Similarity: {dataset_results['mean_similarity']:.4f}\n")
                f.write(f"Std Deviation: {dataset_results['std_similarity']:.4f}\n")
                f.write(f"Number of Samples: {dataset_results['num_samples']}\n")
                
                if dataset_results['scores']:
                    f.write(f"Min Score: {min(dataset_results['scores']):.4f}\n")
                    f.write(f"Max Score: {max(dataset_results['scores']):.4f}\n")
                    f.write(f"Median Score: {np.median(dataset_results['scores']):.4f}\n")
            
            elif isinstance(dataset_results, dict):
                for subset_name, subset_results in dataset_results.items():
                    f.write(f"  Subset: {subset_name}\n")
                    if 'mean_similarity' in subset_results:
                        f.write(f"    Mean Similarity: {subset_results['mean_similarity']:.4f}\n")
                        f.write(f"    Std Deviation: {subset_results['std_similarity']:.4f}\n")
                        f.write(f"    Number of Samples: {subset_results['num_samples']}\n")
                        
                        if subset_results['scores']:
                            f.write(f"    Min Score: {min(subset_results['scores']):.4f}\n")
                            f.write(f"    Max Score: {max(subset_results['scores']):.4f}\n")
                            f.write(f"    Median Score: {np.median(subset_results['scores']):.4f}\n")
            
            f.write("\n")
    
    print(f"Report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PhoCLIP model")
    parser.add_argument("--model-path", type=str, default="e:/ViCLIP/phoclip_checkpoint", 
                        help="Path to PhoCLIP checkpoint")
    parser.add_argument("--flickr-path", type=str, default="e:/ViCLIP/flickr", 
                        help="Path to Flickr dataset")
    parser.add_argument("--ktvic-path", type=str, default="e:/ViCLIP/ktvic", 
                        help="Path to KTVIC dataset")
    parser.add_argument("--uitviic-path", type=str, default="e:/ViCLIP/UIT-ViIC", 
                        help="Path to UIT-ViIC dataset")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", 
                        help="Output directory for results")
    parser.add_argument("--datasets", nargs='+', choices=['flickr', 'ktvic', 'uitviic'], 
                        default=['flickr', 'ktvic', 'uitviic'], help="Datasets to evaluate")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    print("Initializing PhoCLIP evaluator...")
    phoclip_evaluator = PhoCLIPEvaluator(model_path=args.model_path)
    dataset_evaluator = DatasetEvaluator(phoclip_evaluator)
    
    # Run evaluations
    all_results = {}
    
    if 'flickr' in args.datasets:
        try:
            flickr_results = dataset_evaluator.evaluate_flickr_dataset(args.flickr_path)
            if flickr_results:
                all_results['flickr'] = flickr_results
        except Exception as e:
            print(f"Error evaluating Flickr dataset: {e}")
    
    if 'ktvic' in args.datasets:
        try:
            ktvic_results = dataset_evaluator.evaluate_ktvic_dataset(args.ktvic_path)
            if ktvic_results:
                all_results['ktvic'] = ktvic_results
        except Exception as e:
            print(f"Error evaluating KTVIC dataset: {e}")
    
    if 'uitviic' in args.datasets:
        try:
            uitviic_results = dataset_evaluator.evaluate_uitviic_dataset(args.uitviic_path)
            if uitviic_results:
                all_results['uitviic'] = uitviic_results
        except Exception as e:
            print(f"Error evaluating UIT-ViIC dataset: {e}")
    
    if all_results:
        # Save results
        results_file = os.path.join(args.output_dir, "phoclip_evaluation_results.json")
        save_results(all_results, results_file)
        
        # Generate report
        report_file = os.path.join(args.output_dir, "phoclip_evaluation_report.txt")
        generate_report(all_results, report_file)
        
        # Plot results
        plot_results(all_results, args.output_dir)
        
        print("\nEvaluation completed successfully!")
        print(f"Results saved in: {args.output_dir}")
        
        # Print summary
        print("\nSummary:")
        for dataset_name, dataset_results in all_results.items():
            if isinstance(dataset_results, dict) and 'mean_similarity' in dataset_results:
                print(f"{dataset_name}: Mean similarity = {dataset_results['mean_similarity']:.4f} "
                      f"(±{dataset_results['std_similarity']:.4f}), "
                      f"Samples = {dataset_results['num_samples']}")
            elif isinstance(dataset_results, dict):
                print(f"{dataset_name}:")
                for subset_name, subset_results in dataset_results.items():
                    if 'mean_similarity' in subset_results:
                        print(f"  {subset_name}: Mean similarity = {subset_results['mean_similarity']:.4f} "
                              f"(±{subset_results['std_similarity']:.4f}), "
                              f"Samples = {subset_results['num_samples']}")
    else:
        print("No evaluation results obtained. Please check your dataset paths and model.")


if __name__ == "__main__":
    main()
