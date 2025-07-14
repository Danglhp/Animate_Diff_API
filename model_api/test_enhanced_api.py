import requests
import json
import time
import os
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_text_encoders():
    """Test text encoders endpoint"""
    print("=== Testing Text Encoders ===")
    response = requests.get(f"{BASE_URL}/text-encoders")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_negative_prompt_categories():
    """Test negative prompt categories endpoint"""
    print("=== Testing Negative Prompt Categories ===")
    response = requests.get(f"{BASE_URL}/negative-prompt-categories")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_phoclip_vietnamese():
    """Test PhoCLIP with Vietnamese prompt generation"""
    print("=== Testing PhoCLIP with Vietnamese Prompt ===")
    
    poem = """
    Áo trắng em đi trong chiều
    Gió thổi bay tóc em bay
    Nắng vàng rơi trên vai em
    Làm tôi nhớ mãi không quên
    """
    
    request_data = {
        "poem": poem,
        "output_filename": "phoclip_vietnamese_test",
        "text_encoder": "phoclip",
        "prompt_generation_mode": "analysis_to_vietnamese",
        "negative_prompt_category": "general"
    }
    
    response = requests.post(f"{BASE_URL}/generate", json=request_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        task_id = response.json()["task_id"]
        monitor_task(task_id, "PhoCLIP Vietnamese", request_data)
    
    print()

def test_phoclip_english():
    """Test PhoCLIP with English prompt generation"""
    print("=== Testing PhoCLIP with English Prompt ===")
    
    poem = """
    Áo trắng em đi trong chiều
    Gió thổi bay tóc em bay
    Nắng vàng rơi trên vai em
    Làm tôi nhớ mãi không quên
    """
    
    request_data = {
        "poem": poem,
        "output_filename": "phoclip_english_test",
        "text_encoder": "phoclip",
        "prompt_generation_mode": "analysis_to_english",
        "negative_prompt_category": "artistic"
    }
    
    response = requests.post(f"{BASE_URL}/generate", json=request_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        task_id = response.json()["task_id"]
        monitor_task(task_id, "PhoCLIP English", request_data)
    
    print()

def test_base_model_english():
    """Test base model with English prompt generation"""
    print("=== Testing Base Model with English Prompt ===")
    
    poem = """
    Áo trắng em đi trong chiều
    Gió thổi bay tóc em bay
    Nắng vàng rơi trên vai em
    Làm tôi nhớ mãi không quên
    """
    
    request_data = {
        "poem": poem,
        "output_filename": "base_english_test",
        "text_encoder": "base",
        "prompt_generation_mode": "analysis_to_english",
        "negative_prompt_category": "technical"
    }
    
    response = requests.post(f"{BASE_URL}/generate", json=request_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        task_id = response.json()["task_id"]
        monitor_task(task_id, "Base Model English", request_data)
    
    print()

def test_custom_negative_prompt():
    """Test with custom negative prompt"""
    print("=== Testing Custom Negative Prompt ===")
    
    poem = """
    Một bông hoa đẹp trong vườn
    Nở rộ dưới ánh nắng mai
    Hương thơm bay trong gió
    Làm lòng tôi thêm vui
    """
    
    request_data = {
        "poem": poem,
        "output_filename": "custom_negative_test",
        "text_encoder": "phoclip",
        "prompt_generation_mode": "analysis_to_vietnamese",
        "negative_prompt_category": "custom",
        "custom_negative_prompt": "ugly flowers, dead plants, dark colors, scary atmosphere"
    }
    
    response = requests.post(f"{BASE_URL}/generate", json=request_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        task_id = response.json()["task_id"]
        monitor_task(task_id, "Custom Negative Prompt", request_data)
    
    print()

def test_direct_prompt_mode():
    """Test direct prompt extraction mode"""
    print("=== Testing Direct Prompt Mode ===")
    
    poem = """
    Hoàng hôn buông xuống trên biển
    Sóng vỗ nhẹ nhàng bờ cát
    Thuyền ai xa xa ngoài khơi
    Mang theo nỗi nhớ quê nhà
    """
    
    request_data = {
        "poem": poem,
        "output_filename": "direct_prompt_test",
        "text_encoder": "base",
        "prompt_generation_mode": "direct_prompt",
        "negative_prompt_category": "content"
    }
    
    response = requests.post(f"{BASE_URL}/generate", json=request_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        task_id = response.json()["task_id"]
        monitor_task(task_id, "Direct Prompt Mode", request_data)
    
    print()

def test_base_model_vietnamese():
    """Test base model with Vietnamese prompt generation mode (should still generate English)"""
    print("=== Testing Base Model with Vietnamese Prompt Mode ===")
    
    poem = """
    Áo trắng em đi trong chiều
    Gió thổi bay tóc em bay
    Nắng vàng rơi trên vai em
    Làm tôi nhớ mãi không quên
    """
    
    request_data = {
        "poem": poem,
        "output_filename": "base_vietnamese_test",
        "text_encoder": "base",
        "prompt_generation_mode": "analysis_to_vietnamese",  # This should still generate English
        "negative_prompt_category": "general"
    }
    
    response = requests.post(f"{BASE_URL}/generate", json=request_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        task_id = response.json()["task_id"]
        monitor_task(task_id, "Base Model Vietnamese Mode", request_data)
    
    print()

def test_phoclip_direct_prompt():
    """Test PhoCLIP with direct prompt extraction mode"""
    print("=== Testing PhoCLIP with Direct Prompt Mode ===")
    
    poem = """
    Một bông hoa đẹp trong vườn
    Nở rộ dưới ánh nắng mai
    Hương thơm bay trong gió
    Làm lòng tôi thêm vui
    """
    
    request_data = {
        "poem": poem,
        "output_filename": "phoclip_direct_test",
        "text_encoder": "phoclip",
        "prompt_generation_mode": "direct_prompt",
        "negative_prompt_category": "artistic"
    }
    
    response = requests.post(f"{BASE_URL}/generate", json=request_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        task_id = response.json()["task_id"]
        monitor_task(task_id, "PhoCLIP Direct Prompt", request_data)
    
    print()

def monitor_task(task_id, test_name, request_data):
    """Monitor a task until completion and save metadata"""
    print(f"Monitoring task {task_id} for {test_name}...")
    
    max_wait_time = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        response = requests.get(f"{BASE_URL}/status/{task_id}")
        if response.status_code == 200:
            status_data = response.json()
            print(f"Status: {status_data['status']}")
            
            if status_data['status'] == 'completed':
                print(f"✅ {test_name} completed successfully!")
                print(f"Output path: {status_data.get('output_path')}")
                
                # Try to download the file
                try:
                    download_response = requests.get(f"{BASE_URL}/download/{task_id}")
                    if download_response.status_code == 200:
                        output_dir = Path("test_outputs")
                        output_dir.mkdir(exist_ok=True)
                        output_path = output_dir / f"{test_name.lower().replace(' ', '_')}.gif"
                        
                        with open(output_path, 'wb') as f:
                            f.write(download_response.content)
                        print(f"✅ File downloaded to: {output_path}")
                        
                        # Save metadata
                        save_metadata(test_name, request_data, output_path, status_data)
                        
                    else:
                        print(f"❌ Failed to download file: {download_response.status_code}")
                except Exception as e:
                    print(f"❌ Download error: {e}")
                
                return True
            elif status_data['status'] == 'failed':
                print(f"❌ {test_name} failed!")
                print(f"Error: {status_data.get('error', 'Unknown error')}")
                return False
        
        time.sleep(10)  # Wait 10 seconds before checking again
    
    print(f"⏰ {test_name} timed out after 5 minutes")
    return False

def save_metadata(test_name, request_data, output_path, status_data):
    """Save metadata about the generated image"""
    metadata_dir = Path("test_outputs/metadata")
    metadata_dir.mkdir(exist_ok=True)
    
    metadata_file = metadata_dir / f"{test_name.lower().replace(' ', '_')}_metadata.txt"
    
    # Try to get the actual generated prompt from the API response
    # The API might include this in the status response
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
        if request_data.get('custom_negative_prompt'):
            f.write(f"Custom Negative Prompt: {request_data.get('custom_negative_prompt')}\n")
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
        if request_data.get('custom_negative_prompt'):
            f.write(f"Custom Negative: {request_data.get('custom_negative_prompt')}\n")
        
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

def test_list_tasks():
    """Test listing all tasks"""
    print("=== Testing List Tasks ===")
    response = requests.get(f"{BASE_URL}/tasks")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Enhanced API Tests")
    print("=" * 50)
    
    # Create test outputs directory
    test_output_dir = Path("test_outputs")
    test_output_dir.mkdir(exist_ok=True)
    
    # Test basic endpoints
    test_health_check()
    test_text_encoders()
    test_negative_prompt_categories()
    
    # Test different configurations
    test_phoclip_vietnamese()
    test_phoclip_english()
    test_base_model_english()
    test_base_model_vietnamese()  # Test base model with Vietnamese mode
    test_custom_negative_prompt()
    test_direct_prompt_mode()
    test_phoclip_direct_prompt()  # Test PhoCLIP with direct prompt
    
    # List all tasks
    test_list_tasks()
    
    print("🎉 All tests completed!")

if __name__ == "__main__":
    run_all_tests() 