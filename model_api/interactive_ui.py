import requests
import json
import time
import os
import sys
import glob
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# API base URL
BASE_URL = "http://localhost:8000"

# Storage directories
STORAGE_DIR = "generation_storage"
METADATA_FILE = os.path.join(STORAGE_DIR, "generation_metadata.json")
IMAGES_DIR = os.path.join(STORAGE_DIR, "images")

# Ensure storage directories exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

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

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def load_metadata() -> List[Dict]:
    """Load existing metadata from file"""
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print_warning(f"Could not load existing metadata: {e}")
    return []

def save_metadata(metadata: List[Dict]):
    """Save metadata to file"""
    try:
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print_success(f"Metadata saved to {METADATA_FILE}")
    except Exception as e:
        print_error(f"Failed to save metadata: {e}")

def copy_image_to_storage(image_path: str, filename: str) -> str:
    """Copy generated image to storage directory"""
    if not image_path or not os.path.exists(image_path):
        return None
    
    # Determine file extension
    _, ext = os.path.splitext(image_path)
    if not ext:
        ext = '.gif'  # Default to gif
    
    # Create storage filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    storage_filename = f"{filename}_{timestamp}{ext}"
    storage_path = os.path.join(IMAGES_DIR, storage_filename)
    
    try:
        shutil.copy2(image_path, storage_path)
        print_success(f"Image copied to storage: {storage_path}")
        return storage_path
    except Exception as e:
        print_error(f"Failed to copy image to storage: {e}")
        return None

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

def get_available_text_encoders() -> Dict[str, str]:
    """Get available text encoders from API"""
    try:
        response = requests.get(f"{BASE_URL}/text-encoders")
        if response.status_code == 200:
            data = response.json()
            # Extract ID and name from the response
            encoders = {}
            for item in data.get('text_encoders', []):
                encoders[item['id']] = f"{item['name']} - {item['description']}"
            return encoders
        else:
            return {"base": "Base CLIP Model", "phoclip": "PhoCLIP Model"}
    except:
        return {"base": "Base CLIP Model", "phoclip": "PhoCLIP Model"}

def get_negative_prompt_categories() -> Dict[str, str]:
    """Get available negative prompt categories from API"""
    try:
        response = requests.get(f"{BASE_URL}/negative-prompt-categories")
        if response.status_code == 200:
            data = response.json()
            # Extract ID and name from the response
            categories = {}
            for item in data.get('categories', []):
                categories[item['id']] = f"{item['name']} - {item['description']}"
            return categories
        else:
            return {
                "general": "General negative prompts",
                "artistic": "Artistic style negative prompts",
                "technical": "Technical quality negative prompts",
                "content": "Content-specific negative prompts",
                "custom": "Custom negative prompt"
            }
    except:
        return {
            "general": "General negative prompts",
            "artistic": "Artistic style negative prompts",
            "technical": "Technical quality negative prompts",
            "content": "Content-specific negative prompts",
            "custom": "Custom negative prompt"
        }

def display_menu(options: Dict[str, str], title: str) -> str:
    """Display a menu and get user selection"""
    print_header(title)
    
    for i, (key, description) in enumerate(options.items(), 1):
        print(f"{Colors.OKCYAN}{i}.{Colors.ENDC} {description} ({key})")
    
    while True:
        try:
            choice = input(f"\n{Colors.WARNING}Enter your choice (1-{len(options)}): {Colors.ENDC}")
            choice_num = int(choice)
            if 1 <= choice_num <= len(options):
                return list(options.keys())[choice_num - 1]
            else:
                print_error(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print_error("Please enter a valid number")

def get_custom_negative_prompt() -> str:
    """Get custom negative prompt from user"""
    print_header("Custom Negative Prompt")
    print_info("Enter your custom negative prompt to avoid certain elements in the generated image.")
    print_info("Examples: 'blurry, low quality, distorted, ugly'")
    print()
    
    while True:
        custom_prompt = input(f"{Colors.OKCYAN}Enter custom negative prompt: {Colors.ENDC}").strip()
        if custom_prompt:
            return custom_prompt
        else:
            print_error("Negative prompt cannot be empty")

def get_vietnamese_poem() -> str:
    """Get Vietnamese poem from user"""
    print_header("Vietnamese Poem Input")
    print_info("Enter your Vietnamese poem. The system will analyze it and generate an image.")
    print_info("You can enter multiple lines. Press Enter twice to finish.")
    print()
    
    lines = []
    print(f"{Colors.OKCYAN}Enter your poem (press Enter twice to finish):{Colors.ENDC}")
    
    while True:
        line = input().strip()
        if line == "" and lines:  # Empty line and we have content
            break
        elif line != "":
            lines.append(line)
        elif not lines:  # First line is empty
            print_error("Please enter at least one line of poem")
            continue
    
    poem = "\n".join(lines)
    return poem

def get_output_filename() -> str:
    """Get output filename from user"""
    print_header("Output Filename")
    print_info("Enter a name for the generated image file (without extension)")
    print_info("The file will be saved as [filename].gif")
    print()
    
    while True:
        filename = input(f"{Colors.OKCYAN}Enter filename: {Colors.ENDC}").strip()
        if filename:
            # Remove any file extension if user included it
            filename = filename.replace('.gif', '').replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
            return filename
        else:
            print_error("Filename cannot be empty")

def get_prompt_generation_mode(text_encoder: str) -> str:
    """Get prompt generation mode based on text encoder"""
    print_header("Prompt Generation Mode")
    
    if text_encoder == "phoclip":
        options = {
            "analysis_to_vietnamese": "Analyze poem and generate Vietnamese prompt",
            "analysis_to_english": "Analyze poem and generate English prompt",
            "direct_prompt": "Extract direct prompt from poem"
        }
    else:  # base model
        options = {
            "analysis_to_english": "Analyze poem and generate English prompt",
            "direct_prompt": "Extract direct prompt from poem"
        }
    
    return display_menu(options, "Choose Prompt Generation Mode")

def monitor_generation(task_id: str, poem: str, output_filename: str) -> Optional[Dict]:
    """Monitor the image generation process and return generation details"""
    print_header("Generating Image")
    print_info(f"Task ID: {task_id}")
    print_info("Please wait while the image is being generated...")
    print()
    
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
                    
                    # Debug: Print all available data from status response
                    print_info("Status response data:")
                    for key, value in status_data.items():
                        print_info(f"  {key}: {value}")
                    print()
                    
                    if output_path and os.path.exists(output_path):
                        print_success("Image generation completed!")
                        print_info(f"Output saved to: {output_path}")
                        
                        # Display generated prompt
                        if generated_prompt:
                            print_header("Generated Prompt")
                            print(f"{Colors.BOLD}Positive Prompt:{Colors.ENDC}")
                            print(f"{Colors.OKCYAN}{generated_prompt}{Colors.ENDC}")
                            print()
                        
                        if negative_prompt:
                            print(f"{Colors.BOLD}Negative Prompt:{Colors.ENDC}")
                            print(f"{Colors.WARNING}{negative_prompt}{Colors.ENDC}")
                            print()
                        
                        return {
                            'output_path': output_path,
                            'generated_prompt': generated_prompt,
                            'negative_prompt': negative_prompt
                        }
                    else:
                        print_error("Image generation completed but file not found")
                        print_info(f"Expected path: {output_path}")
                        
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
                                    return {
                                        'output_path': path,
                                        'generated_prompt': generated_prompt,
                                        'negative_prompt': negative_prompt
                                    }
                        
                        # Try to find the file using our discovery function
                        print_info("Attempting file discovery...")
                        found_file = find_generated_file(output_filename)
                        if found_file:
                            print_success(f"Found generated file: {found_file}")
                            return {
                                'output_path': found_file,
                                'generated_prompt': generated_prompt,
                                'negative_prompt': negative_prompt
                            }
                        
                        # If still not found, try to find any recently created image files
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
                                return {
                                    'output_path': most_recent,
                                    'generated_prompt': generated_prompt,
                                    'negative_prompt': negative_prompt
                                }
                        
                        print_error("Could not locate generated image file")
                        return None
                
                elif status_data['status'] == 'failed':
                    error_msg = status_data.get('error', 'Unknown error')
                    print_error(f"Image generation failed: {error_msg}")
                    return None
                
                elif status_data['status'] == 'processing':
                    print_info("Still processing... Please wait")
                
                else:
                    print_info(f"Status: {status_data['status']}")
            
        except Exception as e:
            print_error(f"Error checking status: {e}")
        
        time.sleep(3)
    
    print_error("Generation timed out")
    return None

def display_generation_summary(poem: str, settings: Dict, generation_result: Optional[Dict]):
    """Display a summary of the generation process"""
    print_header("Generation Summary")
    
    print(f"{Colors.BOLD}Poem:{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{poem}{Colors.ENDC}")
    print()
    
    print(f"{Colors.BOLD}Settings:{Colors.ENDC}")
    for key, value in settings.items():
        if key != 'poem':
            print(f"  {key}: {Colors.OKCYAN}{value}{Colors.ENDC}")
    print()
    
    if generation_result and generation_result.get('output_path'):
        print_success(f"Image successfully generated and saved to: {generation_result['output_path']}")
        
        if generation_result.get('generated_prompt'):
            print(f"{Colors.BOLD}Generated Prompt:{Colors.ENDC}")
            print(f"{Colors.OKCYAN}{generation_result['generated_prompt']}{Colors.ENDC}")
            print()
        
        if generation_result.get('negative_prompt'):
            print(f"{Colors.BOLD}Negative Prompt:{Colors.ENDC}")
            print(f"{Colors.WARNING}{generation_result['negative_prompt']}{Colors.ENDC}")
            print()
    else:
        print_error("Image generation failed")

def save_generation_metadata(poem: str, settings: Dict, generation_result: Optional[Dict], output_filename: str, task_id: str = None):
    """Save generation metadata to storage in the current folder (model_api/)"""
    if not generation_result or not generation_result.get('output_path'):
        print_warning("No generation result to save")
        return

    # Copy image to current folder (model_api/)
    image_path = generation_result['output_path']
    if image_path and os.path.exists(image_path):
        _, ext = os.path.splitext(image_path)
        if not ext:
            ext = '.gif'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{output_filename}_{timestamp}{ext}"
        dest_image_path = os.path.join(os.path.dirname(__file__), image_filename)
        try:
            shutil.copy2(image_path, dest_image_path)
            print_success(f"Image copied to: {dest_image_path}")
        except Exception as e:
            print_error(f"Failed to copy image to current folder: {e}")
            dest_image_path = None
    else:
        dest_image_path = None

    # Prepare metadata
    metadata_entry = {
        'task_id': task_id,
        'image_name': os.path.basename(dest_image_path) if dest_image_path else None,
        'prompt_used': generation_result.get('generated_prompt', ''),
        'input_poem': poem,
        'negative_prompt': generation_result.get('negative_prompt', ''),
        'output_filename': output_filename,
        'settings': settings,
        'timestamp': datetime.now().isoformat(),
        'original_image_path': image_path,
        'status': 'completed' if dest_image_path else 'failed'
    }

    # Save metadata as a JSON file in model_api/
    metadata_filename = f"{output_filename}_{timestamp}_metadata.json"
    metadata_path = os.path.join(os.path.dirname(__file__), metadata_filename)
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_entry, f, indent=2, ensure_ascii=False)
        print_success(f"Metadata saved to {metadata_path}")
    except Exception as e:
        print_error(f"Failed to save metadata in current folder: {e}")

def display_storage_info():
    """Display information about stored generations"""
    metadata = load_metadata()
    
    if not metadata:
        print_info("No generations stored yet.")
        return
    
    print_header("Storage Information")
    print_info(f"Total generations stored: {len(metadata)}")
    print_info(f"Storage directory: {STORAGE_DIR}")
    print_info(f"Images directory: {IMAGES_DIR}")
    print_info(f"Metadata file: {METADATA_FILE}")
    print()
    
    # Show recent generations
    print(f"{Colors.BOLD}Recent Generations:{Colors.ENDC}")
    recent = metadata[-5:]  # Last 5 generations
    for entry in recent:
        timestamp = entry['timestamp'][:19].replace('T', ' ')  # Format timestamp
        status = "✓" if entry['status'] == 'completed' else "✗"
        print(f"  {status} {timestamp} - {entry['output_filename']}")

def main():
    """Main interactive UI function"""
    clear_screen()
    print_header("Vietnamese Poem to Image Generator")
    print_info("Welcome to the interactive Vietnamese Poem to Image Generator!")
    print_info("This tool will help you generate images from Vietnamese poems.")
    print_info(f"All generations will be stored in: {STORAGE_DIR}")
    print()
    
    # Display storage info
    display_storage_info()
    print()
    
    try:
        # Test API connection
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print_error("Cannot connect to the API server. Please make sure the server is running.")
            print_info("Start the server with: python main.py")
            return
        print_success("Connected to API server")
        
    except Exception as e:
        print_error(f"Cannot connect to the API server: {e}")
        print_info("Please make sure the server is running on http://localhost:8000")
        return
    
    while True:
        try:
            # Step 1: Choose text encoder
            text_encoders = get_available_text_encoders()
            print_info(f"Available text encoders: {text_encoders}")
            text_encoder = display_menu(text_encoders, "Choose Text Encoder")
            print_info(f"Selected text encoder: {text_encoder}")
            
            # Step 2: Choose prompt generation mode
            prompt_mode = get_prompt_generation_mode(text_encoder)
            
            # Step 3: Choose negative prompt category
            negative_categories = get_negative_prompt_categories()
            print_info(f"Available negative categories: {negative_categories}")
            negative_category = display_menu(negative_categories, "Choose Negative Prompt Category")
            print_info(f"Selected negative category: {negative_category}")
            
            # Step 4: Get custom negative prompt if needed
            custom_negative_prompt = None
            if negative_category == "custom":
                custom_negative_prompt = get_custom_negative_prompt()
            
            # Step 5: Get Vietnamese poem
            poem = get_vietnamese_poem()
            
            # Step 6: Get output filename
            output_filename = get_output_filename()
            
            # Step 7: Prepare request data
            request_data = {
                "poem": poem,
                "output_filename": output_filename,
                "text_encoder": text_encoder,
                "prompt_generation_mode": prompt_mode,
                "negative_prompt_category": negative_category
            }
            
            if custom_negative_prompt:
                request_data["custom_negative_prompt"] = custom_negative_prompt
            
            # Step 8: Send generation request
            print_header("Sending Generation Request")
            print_info("Sending request to generate image...")
            print_info(f"Request data: {request_data}")
            
            response = requests.post(f"{BASE_URL}/generate", json=request_data)
            
            if response.status_code == 200:
                task_id = response.json()["task_id"]
                print_success(f"Request sent successfully! Task ID: {task_id}")
                
                # Step 9: Monitor generation
                generation_result = monitor_generation(task_id, poem, output_filename)
                
                # Step 10: Display summary
                display_generation_summary(poem, request_data, generation_result)
                
                # Step 11: Save metadata
                save_generation_metadata(poem, request_data, generation_result, output_filename, task_id)
                
            else:
                print_error(f"Failed to send request: {response.status_code}")
                print_error(f"Error: {response.text}")
            
            # Ask if user wants to generate another image
            print()
            print_info("Would you like to generate another image?")
            choice = input(f"{Colors.WARNING}Enter 'y' to continue or any other key to exit: {Colors.ENDC}").lower()
            
            if choice != 'y':
                print_header("Thank you for using Vietnamese Poem to Image Generator!")
                print_success("Goodbye!")
                break
            
            clear_screen()
            
        except KeyboardInterrupt:
            print("\n")
            print_warning("Operation cancelled by user")
            break
        except Exception as e:
            print_error(f"An error occurred: {e}")
            print_info("Please try again")

if __name__ == "__main__":
    main() 