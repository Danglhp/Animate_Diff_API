#!/usr/bin/env python3
"""
Simple test to demonstrate metadata functionality
"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_metadata_generation():
    """Test metadata generation with a single test case"""
    print("🚀 Testing Metadata Generation")
    print("=" * 40)
    
    # Test poem
    poem = """
    Một bông hoa đẹp trong vườn
    Nở rộ dưới ánh nắng mai
    Hương thơm bay trong gió
    Làm lòng tôi thêm vui
    """
    
    # Test with PhoCLIP and Vietnamese prompt
    request_data = {
        "poem": poem,
        "output_filename": "metadata_test_phoclip",
        "text_encoder": "phoclip",
        "prompt_generation_mode": "analysis_to_vietnamese",
        "negative_prompt_category": "general"
    }
    
    print("1. Sending generation request...")
    response = requests.post(f"{BASE_URL}/generate", json=request_data)
    
    if response.status_code == 200:
        result = response.json()
        task_id = result["task_id"]
        print(f"✅ Task created: {task_id}")
        
        # Monitor the task
        print("2. Monitoring task progress...")
        max_wait_time = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_response = requests.get(f"{BASE_URL}/status/{task_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"   Status: {status_data['status']}")
                
                if status_data['status'] == 'completed':
                    print("✅ Generation completed!")
                    print(f"   Output path: {status_data.get('output_path')}")
                    
                    # Download the file
                    try:
                        download_response = requests.get(f"{BASE_URL}/download/{task_id}")
                        if download_response.status_code == 200:
                            output_dir = Path("test_outputs")
                            output_dir.mkdir(exist_ok=True)
                            output_path = output_dir / "metadata_test_phoclip.gif"
                            
                            with open(output_path, 'wb') as f:
                                f.write(download_response.content)
                            print(f"✅ File downloaded to: {output_path}")
                            
                            # Save metadata
                            save_metadata("Metadata Test PhoCLIP", request_data, output_path, status_data)
                            
                        else:
                            print(f"❌ Download failed: {download_response.status_code}")
                    except Exception as e:
                        print(f"❌ Download error: {e}")
                    
                    break
                elif status_data['status'] == 'failed':
                    print(f"❌ Generation failed: {status_data.get('error', 'Unknown error')}")
                    break
            
            time.sleep(10)  # Wait 10 seconds
    else:
        print(f"❌ Request failed: {response.status_code}")
        print(f"Response: {response.text}")

def save_metadata(test_name, request_data, output_path, status_data):
    """Save metadata about the generated image"""
    metadata_dir = Path("test_outputs/metadata")
    metadata_dir.mkdir(exist_ok=True)
    
    metadata_file = metadata_dir / f"{test_name.lower().replace(' ', '_')}_metadata.txt"
    
    # Try to get the actual generated prompt from the API response
    generated_prompt = status_data.get('generated_prompt', 'Not available in API response')
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(f"=== METADATA FOR {test_name.upper()} ===\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Task ID: {status_data.get('task_id', 'N/A')}\n")
        f.write(f"Status: {status_data.get('status', 'N/A')}\n\n")
        
        f.write("=== INPUT DATA ===\n")
        f.write(f"Poem:\n{request_data.get('poem', 'N/A')}\n\n")
        f.write(f"Text Encoder: {request_data.get('text_encoder', 'N/A')}\n")
        f.write(f"Prompt Generation Mode: {request_data.get('prompt_generation_mode', 'N/A')}\n")
        f.write(f"Negative Prompt Category: {request_data.get('negative_prompt_category', 'N/A')}\n")
        f.write(f"Output Filename: {request_data.get('output_filename', 'N/A')}\n\n")
        
        f.write("=== OUTPUT DATA ===\n")
        f.write(f"Generated Image: {output_path.name}\n")
        f.write(f"Full Path: {output_path.absolute()}\n")
        f.write(f"File Size: {output_path.stat().st_size} bytes\n")
        f.write(f"File Size (MB): {output_path.stat().st_size / (1024*1024):.2f} MB\n\n")
        
        f.write("=== PROMPT INFORMATION ===\n")
        f.write("Generated Prompt: " + generated_prompt + "\n\n")
        
        f.write("Prompt Generation Logic:\n")
        if request_data.get('text_encoder') == 'phoclip':
            f.write("- PhoCLIP text encoder used\n")
            if request_data.get('prompt_generation_mode') == 'analysis_to_vietnamese':
                f.write("- Vietnamese prompt generated from poem analysis\n")
            elif request_data.get('prompt_generation_mode') == 'analysis_to_english':
                f.write("- English prompt generated from poem analysis\n")
            else:
                f.write("- Direct prompt extraction from analysis\n")
        else:
            f.write("- Base model (CLIP) text encoder used\n")
            f.write("- English prompt generated from poem analysis\n")
        
        f.write(f"\nNegative Prompt Category: {request_data.get('negative_prompt_category', 'N/A')}\n")
        
        f.write("\n=== TECHNICAL DETAILS ===\n")
        f.write(f"Text Encoder Type: {request_data.get('text_encoder', 'N/A')}\n")
        f.write(f"Prompt Generation Mode: {request_data.get('prompt_generation_mode', 'N/A')}\n")
        f.write(f"Negative Prompt Category: {request_data.get('negative_prompt_category', 'N/A')}\n")
        f.write(f"Task Status: {status_data.get('status', 'N/A')}\n")
        f.write(f"Output Path: {status_data.get('output_path', 'N/A')}\n")
    
    print(f"📝 Metadata saved to: {metadata_file}")
    
    # Also create a summary file
    summary_file = metadata_dir / "generation_summary.txt"
    with open(summary_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Test: {test_name}\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Text Encoder: {request_data.get('text_encoder', 'N/A')}\n")
        f.write(f"Prompt Mode: {request_data.get('prompt_generation_mode', 'N/A')}\n")
        f.write(f"Image: {output_path.name}\n")
        f.write(f"Status: {status_data.get('status', 'N/A')}\n")
        f.write(f"Generated Prompt: {generated_prompt}\n")

if __name__ == "__main__":
    print("Starting Metadata Test...")
    print("Make sure the API server is running on http://localhost:8000")
    print("=" * 50)
    
    test_metadata_generation()
    
    print("\n🎉 Metadata test completed!")
    print("Check the 'test_outputs/metadata/' directory for generated metadata files.") 