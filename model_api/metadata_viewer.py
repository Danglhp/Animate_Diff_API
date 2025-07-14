import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
import subprocess
import platform

# Storage directories
STORAGE_DIR = "generation_storage"
METADATA_FILE = os.path.join(STORAGE_DIR, "generation_metadata.json")
IMAGES_DIR = os.path.join(STORAGE_DIR, "images")

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
    """Load metadata from file"""
    if not os.path.exists(METADATA_FILE):
        print_error(f"Metadata file not found: {METADATA_FILE}")
        return []
    
    try:
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print_error(f"Failed to load metadata: {e}")
        return []

def save_metadata(metadata: List[Dict]):
    """Save metadata to file"""
    try:
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print_success("Metadata saved successfully")
    except Exception as e:
        print_error(f"Failed to save metadata: {e}")

def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp

def open_image(image_path: str):
    """Open image with default system viewer"""
    if not image_path or not os.path.exists(image_path):
        print_error("Image file not found")
        return
    
    try:
        if platform.system() == "Windows":
            os.startfile(image_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", image_path])
        else:  # Linux
            subprocess.run(["xdg-open", image_path])
        print_success(f"Opened image: {image_path}")
    except Exception as e:
        print_error(f"Failed to open image: {e}")

def display_generation_details(entry: Dict):
    """Display detailed information about a generation"""
    print_header(f"Generation Details - ID: {entry['id']}")
    
    print(f"{Colors.BOLD}Timestamp:{Colors.ENDC} {format_timestamp(entry['timestamp'])}")
    print(f"{Colors.BOLD}Status:{Colors.ENDC} {entry['status']}")
    print(f"{Colors.BOLD}Output Filename:{Colors.ENDC} {entry['output_filename']}")
    print()
    
    print(f"{Colors.BOLD}Poem:{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{entry['poem']}{Colors.ENDC}")
    print()
    
    print(f"{Colors.BOLD}Settings:{Colors.ENDC}")
    for key, value in entry['settings'].items():
        if key != 'poem':
            print(f"  {key}: {Colors.OKCYAN}{value}{Colors.ENDC}")
    print()
    
    if entry.get('generated_prompt'):
        print(f"{Colors.BOLD}Generated Prompt:{Colors.ENDC}")
        print(f"{Colors.OKCYAN}{entry['generated_prompt']}{Colors.ENDC}")
        print()
    
    if entry.get('negative_prompt'):
        print(f"{Colors.BOLD}Negative Prompt:{Colors.ENDC}")
        print(f"{Colors.WARNING}{entry['negative_prompt']}{Colors.ENDC}")
        print()
    
    print(f"{Colors.BOLD}File Paths:{Colors.ENDC}")
    print(f"  Original: {entry.get('original_image_path', 'N/A')}")
    print(f"  Storage: {entry.get('storage_image_path', 'N/A')}")
    print()

def display_generations_list(metadata: List[Dict], page: int = 1, per_page: int = 10):
    """Display a paginated list of generations"""
    if not metadata:
        print_info("No generations found.")
        return
    
    total_pages = (len(metadata) + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, len(metadata))
    
    print_header(f"Generations List (Page {page}/{total_pages})")
    print_info(f"Showing {start_idx + 1}-{end_idx} of {len(metadata)} generations")
    print()
    
    for i in range(start_idx, end_idx):
        entry = metadata[i]
        timestamp = format_timestamp(entry['timestamp'])
        status = "✓" if entry['status'] == 'completed' else "✗"
        poem_preview = entry['poem'][:50] + "..." if len(entry['poem']) > 50 else entry['poem']
        
        print(f"{Colors.BOLD}{entry['id']:3d}.{Colors.ENDC} {status} {timestamp}")
        print(f"     {Colors.OKCYAN}{entry['output_filename']}{Colors.ENDC}")
        print(f"     {poem_preview}")
        print()

def search_generations(metadata: List[Dict], query: str) -> List[Dict]:
    """Search generations by poem content or filename"""
    query = query.lower()
    results = []
    
    for entry in metadata:
        if (query in entry['poem'].lower() or 
            query in entry['output_filename'].lower() or
            query in entry.get('generated_prompt', '').lower()):
            results.append(entry)
    
    return results

def delete_generation(metadata: List[Dict], generation_id: int) -> bool:
    """Delete a generation and its associated files"""
    entry = None
    for i, item in enumerate(metadata):
        if item['id'] == generation_id:
            entry = item
            del metadata[i]
            break
    
    if not entry:
        print_error(f"Generation with ID {generation_id} not found")
        return False
    
    # Delete storage image if it exists
    storage_path = entry.get('storage_image_path')
    if storage_path and os.path.exists(storage_path):
        try:
            os.remove(storage_path)
            print_success(f"Deleted storage image: {storage_path}")
        except Exception as e:
            print_warning(f"Failed to delete storage image: {e}")
    
    # Delete original image if it exists
    original_path = entry.get('original_image_path')
    if original_path and os.path.exists(original_path):
        try:
            os.remove(original_path)
            print_success(f"Deleted original image: {original_path}")
        except Exception as e:
            print_warning(f"Failed to delete original image: {e}")
    
    # Update IDs for remaining entries
    for i, item in enumerate(metadata):
        item['id'] = i + 1
    
    print_success(f"Deleted generation {generation_id}")
    return True

def export_metadata(metadata: List[Dict], filename: str):
    """Export metadata to a JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print_success(f"Metadata exported to: {filename}")
    except Exception as e:
        print_error(f"Failed to export metadata: {e}")

def main():
    """Main metadata viewer function"""
    clear_screen()
    print_header("Generation Metadata Viewer")
    print_info("Browse and manage your stored poem-to-image generations")
    print()
    
    metadata = load_metadata()
    if not metadata:
        print_info("No generations found. Run the interactive UI to generate some images first.")
        return
    
    current_page = 1
    per_page = 10
    
    while True:
        try:
            print_header("Metadata Viewer Menu")
            print("1. View generations list")
            print("2. View generation details")
            print("3. Search generations")
            print("4. Open image")
            print("5. Delete generation")
            print("6. Export metadata")
            print("7. Storage statistics")
            print("0. Exit")
            print()
            
            choice = input(f"{Colors.WARNING}Enter your choice (0-7): {Colors.ENDC}").strip()
            
            if choice == "0":
                print_header("Goodbye!")
                break
            
            elif choice == "1":
                while True:
                    display_generations_list(metadata, current_page, per_page)
                    
                    print("Navigation: n=next, p=previous, q=quit, <number>=go to page")
                    nav = input(f"{Colors.OKCYAN}Navigation: {Colors.ENDC}").strip().lower()
                    
                    if nav == 'q':
                        break
                    elif nav == 'n':
                        total_pages = (len(metadata) + per_page - 1) // per_page
                        current_page = min(current_page + 1, total_pages)
                    elif nav == 'p':
                        current_page = max(current_page - 1, 1)
                    elif nav.isdigit():
                        page_num = int(nav)
                        total_pages = (len(metadata) + per_page - 1) // per_page
                        if 1 <= page_num <= total_pages:
                            current_page = page_num
            
            elif choice == "2":
                try:
                    gen_id = int(input(f"{Colors.OKCYAN}Enter generation ID: {Colors.ENDC}"))
                    entry = None
                    for item in metadata:
                        if item['id'] == gen_id:
                            entry = item
                            break
                    
                    if entry:
                        display_generation_details(entry)
                        input(f"{Colors.WARNING}Press Enter to continue...{Colors.ENDC}")
                    else:
                        print_error(f"Generation with ID {gen_id} not found")
                except ValueError:
                    print_error("Please enter a valid number")
            
            elif choice == "3":
                query = input(f"{Colors.OKCYAN}Enter search query: {Colors.ENDC}").strip()
                if query:
                    results = search_generations(metadata, query)
                    if results:
                        print_success(f"Found {len(results)} matching generations:")
                        display_generations_list(results, 1, len(results))
                    else:
                        print_info("No matching generations found")
                    input(f"{Colors.WARNING}Press Enter to continue...{Colors.ENDC}")
            
            elif choice == "4":
                try:
                    gen_id = int(input(f"{Colors.OKCYAN}Enter generation ID: {Colors.ENDC}"))
                    entry = None
                    for item in metadata:
                        if item['id'] == gen_id:
                            entry = item
                            break
                    
                    if entry:
                        storage_path = entry.get('storage_image_path')
                        if storage_path and os.path.exists(storage_path):
                            open_image(storage_path)
                        else:
                            original_path = entry.get('original_image_path')
                            if original_path and os.path.exists(original_path):
                                open_image(original_path)
                            else:
                                print_error("No image file found for this generation")
                    else:
                        print_error(f"Generation with ID {gen_id} not found")
                except ValueError:
                    print_error("Please enter a valid number")
            
            elif choice == "5":
                try:
                    gen_id = int(input(f"{Colors.OKCYAN}Enter generation ID to delete: {Colors.ENDC}"))
                    confirm = input(f"{Colors.WARNING}Are you sure you want to delete generation {gen_id}? (y/N): {Colors.ENDC}").strip().lower()
                    
                    if confirm == 'y':
                        if delete_generation(metadata, gen_id):
                            save_metadata(metadata)
                except ValueError:
                    print_error("Please enter a valid number")
            
            elif choice == "6":
                filename = input(f"{Colors.OKCYAN}Enter export filename (default: metadata_export.json): {Colors.ENDC}").strip()
                if not filename:
                    filename = "metadata_export.json"
                export_metadata(metadata, filename)
                input(f"{Colors.WARNING}Press Enter to continue...{Colors.ENDC}")
            
            elif choice == "7":
                print_header("Storage Statistics")
                print_info(f"Total generations: {len(metadata)}")
                
                completed = sum(1 for entry in metadata if entry['status'] == 'completed')
                failed = len(metadata) - completed
                print_info(f"Completed: {completed}")
                print_info(f"Failed: {failed}")
                
                if metadata:
                    first_gen = min(metadata, key=lambda x: x['timestamp'])
                    last_gen = max(metadata, key=lambda x: x['timestamp'])
                    print_info(f"First generation: {format_timestamp(first_gen['timestamp'])}")
                    print_info(f"Last generation: {format_timestamp(last_gen['timestamp'])}")
                
                print_info(f"Storage directory: {STORAGE_DIR}")
                print_info(f"Images directory: {IMAGES_DIR}")
                print_info(f"Metadata file: {METADATA_FILE}")
                
                # Check storage size
                if os.path.exists(IMAGES_DIR):
                    total_size = 0
                    file_count = 0
                    for file in os.listdir(IMAGES_DIR):
                        file_path = os.path.join(IMAGES_DIR, file)
                        if os.path.isfile(file_path):
                            total_size += os.path.getsize(file_path)
                            file_count += 1
                    
                    size_mb = total_size / (1024 * 1024)
                    print_info(f"Images stored: {file_count}")
                    print_info(f"Total storage size: {size_mb:.2f} MB")
                
                input(f"{Colors.WARNING}Press Enter to continue...{Colors.ENDC}")
            
            else:
                print_error("Invalid choice. Please enter a number between 0 and 7.")
            
            clear_screen()
            
        except KeyboardInterrupt:
            print("\n")
            print_warning("Operation cancelled by user")
            break
        except Exception as e:
            print_error(f"An error occurred: {e}")
            input(f"{Colors.WARNING}Press Enter to continue...{Colors.ENDC}")

if __name__ == "__main__":
    main() 