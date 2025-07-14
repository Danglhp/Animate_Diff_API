#!/usr/bin/env python3
"""
Simple test script for the enhanced API
Run this after starting the API server
"""

import requests
import json
import time
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_simple_generation():
    """Test a simple generation with PhoCLIP"""
    print("=== Testing Simple Generation with PhoCLIP ===")
    
    poem = """
    Một bông hoa đẹp trong vườn
    Nở rộ dưới ánh nắng mai
    Hương thơm bay trong gió
    Làm lòng tôi thêm vui
    """
    
    request_data = {
        "poem": poem,
        "output_filename": "simple_test",
        "text_encoder": "phoclip",
        "prompt_generation_mode": "analysis_to_vietnamese",
        "negative_prompt_category": "general"
    }
    
    print("Sending request...")
    response = requests.post(f"{BASE_URL}/generate", json=request_data)
    
    if response.status_code == 200:
        result = response.json()
        task_id = result["task_id"]
        print(f"✅ Task created: {task_id}")
        
        # Monitor the task
        print("Monitoring task progress...")
        while True:
            status_response = requests.get(f"{BASE_URL}/status/{task_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"Status: {status_data['status']}")
                
                if status_data['status'] == 'completed':
                    print("✅ Generation completed!")
                    print(f"Output path: {status_data.get('output_path')}")
                    
                    # Try to download
                    try:
                        download_response = requests.get(f"{BASE_URL}/download/{task_id}")
                        if download_response.status_code == 200:
                            output_path = Path("test_output.gif")
                            with open(output_path, 'wb') as f:
                                f.write(download_response.content)
                            print(f"✅ File downloaded to: {output_path}")
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

def test_base_model():
    """Test base model generation"""
    print("\n=== Testing Base Model Generation ===")
    
    poem = """
    Hoàng hôn buông xuống trên biển
    Sóng vỗ nhẹ nhàng bờ cát
    Thuyền ai xa xa ngoài khơi
    Mang theo nỗi nhớ quê nhà
    """
    
    request_data = {
        "poem": poem,
        "output_filename": "base_test",
        "text_encoder": "base",
        "prompt_generation_mode": "analysis_to_english",
        "negative_prompt_category": "artistic"
    }
    
    print("Sending request...")
    response = requests.post(f"{BASE_URL}/generate", json=request_data)
    
    if response.status_code == 200:
        result = response.json()
        task_id = result["task_id"]
        print(f"✅ Task created: {task_id}")
        
        # Monitor the task
        print("Monitoring task progress...")
        while True:
            status_response = requests.get(f"{BASE_URL}/status/{task_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"Status: {status_data['status']}")
                
                if status_data['status'] == 'completed':
                    print("✅ Generation completed!")
                    print(f"Output path: {status_data.get('output_path')}")
                    break
                elif status_data['status'] == 'failed':
                    print(f"❌ Generation failed: {status_data.get('error', 'Unknown error')}")
                    break
            
            time.sleep(10)  # Wait 10 seconds
    else:
        print(f"❌ Request failed: {response.status_code}")
        print(f"Response: {response.text}")

def test_api_info():
    """Test API information endpoints"""
    print("=== Testing API Information ===")
    
    # Test health
    health_response = requests.get(f"{BASE_URL}/health")
    print(f"Health: {health_response.json()}")
    
    # Test text encoders
    encoders_response = requests.get(f"{BASE_URL}/text-encoders")
    print(f"Text encoders: {json.dumps(encoders_response.json(), indent=2)}")
    
    # Test negative prompt categories
    categories_response = requests.get(f"{BASE_URL}/negative-prompt-categories")
    print(f"Negative prompt categories: {json.dumps(categories_response.json(), indent=2)}")

if __name__ == "__main__":
    print("🚀 Starting Enhanced API Test")
    print("Make sure the API server is running on http://localhost:8000")
    print("=" * 50)
    
    # Test API info first
    test_api_info()
    
    # Test simple generation
    test_simple_generation()
    
    # Test base model
    test_base_model()
    
    print("\n🎉 Test completed!") 