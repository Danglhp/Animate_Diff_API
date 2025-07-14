#!/usr/bin/env python3
"""
Comprehensive CLIP Testing System for Vietnamese Poem to Image Generator

This script generates 50 images for each of the 3 modes:
1. PhoCLIP Vietnamese (analysis_to_vietnamese)
2. PhoCLIP English (analysis_to_english) 
3. Base CLIP (analysis_to_english)

Then calculates CLIP scores and creates detailed metadata reports.
"""

import requests
import json
import time
import os
import sys
import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import open_clip
from pathlib import Path
import gc

# API base URL
BASE_URL = "http://localhost:8000"

# Test configuration
TEST_MODES = {
    "phoclip_vietnamese": {
        "text_encoder": "phoclip",
        "prompt_generation_mode": "analysis_to_vietnamese",
        "description": "PhoCLIP Vietnamese Mode"
    },
    "phoclip_english": {
        "text_encoder": "phoclip", 
        "prompt_generation_mode": "analysis_to_english",
        "description": "PhoCLIP English Mode"
    },
    "base_clip": {
        "text_encoder": "base",
        "prompt_generation_mode": "analysis_to_english", 
        "description": "Base CLIP Model"
    }
}

# Vietnamese poems for testing (15 diverse poems)
VIETNAMESE_POEMS = [
    "Sông gọn trường gian buồn điệp điệp",
    "Mây trắng bay về phương xa",
    "Hoa đào nở rộ mùa xuân",
    "Tiếng chim hót trong vườn",
    "Mặt trời lặn sau núi",
    "Gió thổi qua đồng lúa",
    "Mưa rơi trên mái nhà",
    "Trăng sáng soi đường đêm",
    "Cánh đồng xanh mênh mông",
    "Núi cao vời vợi mây",
    "Biển xanh sóng vỗ bờ",
    "Rừng thông xanh ngắt",
    "Thác nước chảy róc rách",
    "Cầu vồng sau cơn mưa",
    "Bông sen nở trong ao"
]

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print(f"{text.center(60)}")
    print(f"{'='*60}{Colors.ENDC}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKBLUE}ℹ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def load_clip_model():
    """Load CLIP model for scoring"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=device)
        print_success(f"CLIP model loaded on {device}")
        return model, preprocess, device
    except Exception as e:
        print_error(f"Failed to load CLIP model: {e}")
        return None, None, None

def calculate_clip_score(image_path: str, text: str, model, preprocess, device) -> float:
    """Calculate CLIP score between image and text"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Preprocess text
        text_input = open_clip.tokenize([text]).to(device)
        
        # Calculate similarity
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarity
            similarity = (image_features @ text_features.T).item()
            
        return similarity
    
    except Exception as e:
        print_error(f"Error calculating CLIP score for {image_path}: {e}")
        return 0.0

def find_generated_file(output_filename: str) -> Optional[str]:
    """Find the generated file in outputs directory"""
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        print_warning(f"Outputs directory does not exist: {outputs_dir}")
        return None
    
    print_info(f"Searching for files with filename: {output_filename}")
    
    # Look for files with the output filename (with various extensions)
    possible_extensions = ['.gif', '.png', '.jpg', '.jpeg']
    for ext in possible_extensions:
        file_path = os.path.join(outputs_dir, f"{output_filename}{ext}")
        if os.path.exists(file_path):
            print_success(f"Found exact match: {file_path}")
            return file_path
    
    # If not found with exact name, look for files containing the filename
    print_info("Searching for files containing the filename...")
    for file_path in glob.glob(os.path.join(outputs_dir, "*")):
        if output_filename.lower() in os.path.basename(file_path).lower():
            print_success(f"Found partial match: {file_path}")
            return file_path
    
    # If still not found, look for the most recent files of any image type
    print_info("Searching for most recent image files...")
    all_image_files = []
    for ext in possible_extensions:
        all_image_files.extend(glob.glob(os.path.join(outputs_dir, f"*{ext}")))
    
    if all_image_files:
        # Sort by modification time (most recent first)
        all_image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        most_recent = all_image_files[0]
        print_success(f"Found most recent image file: {most_recent}")
        print_info(f"File modification time: {datetime.fromtimestamp(os.path.getmtime(most_recent))}")
        return most_recent
    
    # List all files in outputs directory for debugging
    print_warning("No image files found. Listing all files in outputs directory:")
    if os.path.exists(outputs_dir):
        all_files = os.listdir(outputs_dir)
        if all_files:
            for file in all_files:
                file_path = os.path.join(outputs_dir, file)
                if os.path.isfile(file_path):
                    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    size = os.path.getsize(file_path)
                    print_info(f"  - {file} ({size} bytes, modified: {mtime})")
        else:
            print_info("  (directory is empty)")
    
    return None

def generate_image(poem: str, mode_config: Dict, poem_index: int) -> Optional[Dict]:
    """Generate a single image and return metadata"""
    try:
        # Create output filename
        mode_name = list(TEST_MODES.keys())[list(TEST_MODES.values()).index(mode_config)]
        output_filename = f"test_{mode_name}_{poem_index:02d}"
        
        # Prepare request data
        request_data = {
            "poem": poem,
            "output_filename": output_filename,
            "text_encoder": mode_config["text_encoder"],
            "prompt_generation_mode": mode_config["prompt_generation_mode"],
            "negative_prompt_category": "general"
        }
        
        print_info(f"Generating image {poem_index+1}/15 for {mode_config['description']}")
        
        # Send generation request
        response = requests.post(f"{BASE_URL}/generate", json=request_data)
        
        if response.status_code != 200:
            print_error(f"Failed to send request: {response.status_code}")
            return None
        
        task_id = response.json()["task_id"]
        
        # Monitor generation
        max_wait_time = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(f"{BASE_URL}/status/{task_id}")
                if response.status_code == 200:
                    status_data = response.json()
                    
                    if status_data['status'] == 'completed':
                        output_path = status_data.get('output_path', '')
                        generated_prompt = status_data.get('generated_prompt', '')
                        negative_prompt = status_data.get('negative_prompt', '')
                        
                        # Find the actual file
                        actual_file = None
                        
                        # Check if the path might be relative or absolute
                        if output_path:
                            # Try different path variations
                            possible_paths = [
                                output_path,
                                os.path.basename(output_path),  # Just filename
                                os.path.join("outputs", os.path.basename(output_path)),  # In outputs dir
                                output_path.replace("\\", "/"),  # Normalize separators
                                output_path.replace("/", "\\")   # Windows separators
                            ]
                            
                            for path in possible_paths:
                                if os.path.exists(path):
                                    print_success(f"Found file at: {path}")
                                    actual_file = path
                                    break
                        
                        # If still not found, try our discovery function
                        if not actual_file:
                            print_info("Attempting file discovery...")
                            actual_file = find_generated_file(output_filename)
                        
                        # If still not found, try to find any recently created image files
                        if not actual_file:
                            print_warning("Attempting to find any recently created image files...")
                            outputs_dir = "outputs"
                            if os.path.exists(outputs_dir):
                                # Look for any image files created in the last 5 minutes
                                current_time = time.time()
                                recent_files = []
                                
                                for file in os.listdir(outputs_dir):
                                    file_path = os.path.join(outputs_dir, file)
                                    if os.path.isfile(file_path):
                                        file_time = os.path.getmtime(file_path)
                                        if current_time - file_time < 300:  # 5 minutes
                                            if any(file.lower().endswith(ext) for ext in ['.gif', '.png', '.jpg', '.jpeg']):
                                                recent_files.append((file_path, file_time))
                                
                                if recent_files:
                                    # Sort by modification time (most recent first)
                                    recent_files.sort(key=lambda x: x[1], reverse=True)
                                    most_recent = recent_files[0][0]
                                    print_success(f"Found recently created file: {most_recent}")
                                    actual_file = most_recent
                        
                        if actual_file:
                            return {
                                'poem': poem,
                                'poem_index': poem_index,  # Fixed: use poem_index
                                'output_filename': output_filename,
                                'actual_file': actual_file,
                                'generated_prompt': generated_prompt,
                                'negative_prompt': negative_prompt,
                                'mode': mode_config['description'],
                                'mode_config': mode_config,
                                'timestamp': datetime.now().isoformat()
                            }
                        else:
                            print_error(f"Could not locate generated image file for {output_filename}")
                            return None
                    
                    elif status_data['status'] == 'failed':
                        error_msg = status_data.get('error', 'Unknown error')
                        print_error(f"Generation failed: {error_msg}")
                        return None
                
            except Exception as e:
                print_error(f"Error checking status: {e}")
            
            time.sleep(3)
        
        print_error("Generation timed out")
        return None
        
    except Exception as e:
        print_error(f"Error in generate_image: {e}")
        return None

def run_comprehensive_test():
    """Run the comprehensive CLIP testing"""
    print_header("Comprehensive CLIP Testing System")
    print_info("This will generate 20 images for each of the 3 modes and calculate CLIP scores")
    print()
    
    # Test API connection
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print_error("Cannot connect to the API server. Please make sure the server is running.")
            return
        print_success("Connected to API server")
    except Exception as e:
        print_error(f"Cannot connect to the API server: {e}")
        return
    
    # Load CLIP model
    model, preprocess, device = load_clip_model()
    if model is None:
        return
    
    # Create results directory
    results_dir = "comprehensive_clip_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize results storage
    all_results = []
    mode_results = {mode: [] for mode in TEST_MODES.keys()}
    
    # Check for existing partial results
    partial_results_file = os.path.join(results_dir, "partial_results.json")
    if os.path.exists(partial_results_file):
        try:
            with open(partial_results_file, 'r', encoding='utf-8') as f:
                partial_data = json.load(f)
                all_results = partial_data.get('all_results', [])
                mode_results = partial_data.get('mode_results', {mode: [] for mode in TEST_MODES.keys()})
            print_success(f"Loaded {len(all_results)} existing results from partial file")
        except Exception as e:
            print_warning(f"Could not load partial results: {e}")
    
    # Save partial results function
    def save_partial_results():
        try:
            with open(partial_results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'all_results': all_results,
                    'mode_results': mode_results,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print_warning(f"Could not save partial results: {e}")
    
    # Generate images for each mode
    for mode_name, mode_config in TEST_MODES.items():
        print_header(f"Testing {mode_config['description']}")
        
        mode_results[mode_name] = []
        
        for i, poem in enumerate(VIETNAMESE_POEMS):
            # Check if this generation already exists
            existing_result = None
            for result in mode_results[mode_name]:
                if result.get('poem_index') == i and result.get('poem') == poem:
                    existing_result = result
                    break
            
            if existing_result:
                print_info(f"Skipping poem {i+1}/15 (already completed): {poem[:50]}...")
                continue
            
            print_info(f"Processing poem {i+1}/15: {poem[:50]}...")
            
            # Generate image
            result = generate_image(poem, mode_config, i)
            
            if result:
                # Calculate CLIP score
                clip_score = calculate_clip_score(
                    result['actual_file'], 
                    result['generated_prompt'] or result['poem'],
                    model, preprocess, device
                )
                
                result['clip_score'] = clip_score
                mode_results[mode_name].append(result)
                all_results.append(result)
                
                print_success(f"Generated and scored: {clip_score:.4f}")
                
                # Save partial results after each successful generation
                save_partial_results()
                # Explicitly clean up memory after each image
                del clip_score
                del result
                gc.collect()
            else:
                print_error(f"Failed to generate image for poem {i+1}/15")
            
            # Small delay between requests
            time.sleep(1)
        
        print_success(f"Completed {mode_config['description']}: {len(mode_results[mode_name])}/15 images")
    
    # Calculate statistics
    print_header("Calculating Statistics")
    
    statistics = {}
    for mode_name, results in mode_results.items():
        if results:
            scores = [r['clip_score'] for r in results]
            statistics[mode_name] = {
                'count': len(results),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'median_score': np.median(scores)
            }
            print_success(f"{TEST_MODES[mode_name]['description']}: {statistics[mode_name]['mean_score']:.4f} ± {statistics[mode_name]['std_score']:.4f}")
    
    # Save detailed results
    results_file = os.path.join(results_dir, "comprehensive_clip_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'statistics': statistics,
            'all_results': all_results,
            'mode_results': mode_results,
            'test_config': TEST_MODES,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)
    
    print_success(f"Detailed results saved to: {results_file}")
    
    # Create metadata file
    metadata_file = os.path.join(results_dir, "comprehensive_clip_metadata.csv")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write("Mode,Poem Index,Poem,Generated Prompt,Negative Prompt,Image File,CLIP Score,Timestamp\n")
        for result in all_results:
            f.write(f"\"{result['mode']}\",")
            f.write(f"{VIETNAMESE_POEMS.index(result['poem']) + 1},")
            f.write(f"\"{result['poem']}\",")
            f.write(f"\"{result.get('generated_prompt', '')}\",")
            f.write(f"\"{result.get('negative_prompt', '')}\",")
            f.write(f"\"{result['actual_file']}\",")
            f.write(f"{result['clip_score']:.6f},")
            f.write(f"\"{result['timestamp']}\"\n")
    
    print_success(f"Metadata saved to: {metadata_file}")
    
    # Create visualization
    create_visualization(statistics, mode_results, results_dir)
    
    print_header("Comprehensive CLIP Testing Complete!")
    print_success(f"Results saved in: {results_dir}")

def create_visualization(statistics: Dict, mode_results: Dict, results_dir: str):
    """Create charts and visualizations"""
    print_info("Creating visualizations...")
    
    # Prepare data for plotting
    modes = list(statistics.keys())
    mean_scores = [statistics[mode]['mean_score'] for mode in modes]
    std_scores = [statistics[mode]['std_score'] for mode in modes]
    mode_labels = [TEST_MODES[mode]['description'] for mode in modes]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comprehensive CLIP Score Analysis', fontsize=16, fontweight='bold')
    
    # 1. Bar chart of mean scores
    bars = ax1.bar(mode_labels, mean_scores, yerr=std_scores, capsize=5, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax1.set_title('Mean CLIP Scores by Mode')
    ax1.set_ylabel('CLIP Score')
    ax1.set_ylim(0, max(mean_scores) * 1.2)
    
    # Add value labels on bars
    for bar, score in zip(bars, mean_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Box plot of score distributions
    all_scores = []
    labels = []
    for mode in modes:
        scores = [r['clip_score'] for r in mode_results[mode]]
        all_scores.append(scores)
        labels.append(TEST_MODES[mode]['description'])
    
    box_plot = ax2.boxplot(all_scores, labels=labels, patch_artist=True)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_title('CLIP Score Distributions')
    ax2.set_ylabel('CLIP Score')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Statistics table
    ax3.axis('tight')
    ax3.axis('off')
    
    table_data = []
    headers = ['Mode', 'Count', 'Mean', 'Std', 'Min', 'Max', 'Median']
    table_data.append(headers)
    
    for mode in modes:
        stats = statistics[mode]
        table_data.append([
            TEST_MODES[mode]['description'],
            str(stats['count']),
            f"{stats['mean_score']:.4f}",
            f"{stats['std_score']:.4f}",
            f"{stats['min_score']:.4f}",
            f"{stats['max_score']:.4f}",
            f"{stats['median_score']:.4f}"
        ])
    
    table = ax3.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax3.set_title('Detailed Statistics', fontweight='bold', pad=20)
    
    # 4. Score comparison
    ax4.plot(range(1, 21), [r['clip_score'] for r in mode_results[modes[0]]], 
             'o-', label=mode_labels[0], color='#FF6B6B', alpha=0.7)
    ax4.plot(range(1, 21), [r['clip_score'] for r in mode_results[modes[1]]], 
             's-', label=mode_labels[1], color='#4ECDC4', alpha=0.7)
    ax4.plot(range(1, 21), [r['clip_score'] for r in mode_results[modes[2]]], 
             '^-', label=mode_labels[2], color='#45B7D1', alpha=0.7)
    
    ax4.set_title('CLIP Scores by Image Index')
    ax4.set_xlabel('Image Index')
    ax4.set_ylabel('CLIP Score')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    chart_file = os.path.join(results_dir, "comprehensive_clip_analysis.png")
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_success(f"Visualization saved to: {chart_file}")

def main():
    """Main function"""
    try:
        run_comprehensive_test()
    except KeyboardInterrupt:
        print("\n")
        print_warning("Testing interrupted by user")
    except Exception as e:
        print_error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 