#!/usr/bin/env python3
"""
Debug script for file discovery issues in the Vietnamese Poem to Image Generator.
This script helps troubleshoot problems with finding generated image files.
"""

import os
import glob
import time
from datetime import datetime
from pathlib import Path

def print_info(text: str):
    """Print info message"""
    print(f"ℹ {text}")

def print_success(text: str):
    """Print success message"""
    print(f"✓ {text}")

def print_error(text: str):
    """Print error message"""
    print(f"✗ {text}")

def print_warning(text: str):
    """Print warning message"""
    print(f"⚠ {text}")

def check_outputs_directory():
    """Check the outputs directory and its contents"""
    outputs_dir = "outputs"
    
    print("=" * 60)
    print("OUTPUTS DIRECTORY ANALYSIS")
    print("=" * 60)
    
    if not os.path.exists(outputs_dir):
        print_error(f"Outputs directory does not exist: {outputs_dir}")
        return
    
    print_success(f"Outputs directory exists: {outputs_dir}")
    
    # List all files in outputs directory
    files = os.listdir(outputs_dir)
    if not files:
        print_warning("Outputs directory is empty")
        return
    
    print_info(f"Found {len(files)} files in outputs directory:")
    
    image_files = []
    other_files = []
    
    for file in files:
        file_path = os.path.join(outputs_dir, file)
        if os.path.isfile(file_path):
            mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            size = os.path.getsize(file_path)
            
            if any(file.lower().endswith(ext) for ext in ['.gif', '.png', '.jpg', '.jpeg']):
                image_files.append((file, file_path, mtime, size))
            else:
                other_files.append((file, file_path, mtime, size))
    
    if image_files:
        print_success(f"Found {len(image_files)} image files:")
        for file, file_path, mtime, size in sorted(image_files, key=lambda x: x[2], reverse=True):
            print(f"  - {file} ({size:,} bytes, modified: {mtime})")
    else:
        print_warning("No image files found")
    
    if other_files:
        print_info(f"Found {len(other_files)} other files:")
        for file, file_path, mtime, size in other_files:
            print(f"  - {file} ({size:,} bytes, modified: {mtime})")

def test_file_discovery(filename: str):
    """Test file discovery logic for a specific filename"""
    print("\n" + "=" * 60)
    print(f"FILE DISCOVERY TEST FOR: {filename}")
    print("=" * 60)
    
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        print_error(f"Outputs directory does not exist: {outputs_dir}")
        return None
    
    print_info(f"Searching for files with filename: {filename}")
    
    # Test 1: Exact filename match with different extensions
    possible_extensions = ['.gif', '.png', '.jpg', '.jpeg']
    for ext in possible_extensions:
        file_path = os.path.join(outputs_dir, f"{filename}{ext}")
        if os.path.exists(file_path):
            print_success(f"Found exact match: {file_path}")
            return file_path
    
    # Test 2: Partial filename match (case-insensitive)
    print_info("Searching for files containing the filename...")
    for file_path in glob.glob(os.path.join(outputs_dir, "*")):
        if filename.lower() in os.path.basename(file_path).lower():
            print_success(f"Found partial match: {file_path}")
            return file_path
    
    # Test 3: Most recent image files
    print_info("Searching for most recent image files...")
    all_image_files = []
    for ext in possible_extensions:
        all_image_files.extend(glob.glob(os.path.join(outputs_dir, f"*{ext}")))
    
    if all_image_files:
        all_image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        most_recent = all_image_files[0]
        print_success(f"Found most recent image file: {most_recent}")
        print_info(f"File modification time: {datetime.fromtimestamp(os.path.getmtime(most_recent))}")
        return most_recent
    
    # Test 4: Recently created files (last 10 minutes)
    print_warning("Searching for recently created image files...")
    current_time = time.time()
    recent_files = []
    
    for file in os.listdir(outputs_dir):
        file_path = os.path.join(outputs_dir, file)
        if os.path.isfile(file_path):
            file_time = os.path.getmtime(file_path)
            if current_time - file_time < 600:  # 10 minutes
                if any(file.lower().endswith(ext) for ext in ['.gif', '.png', '.jpg', '.jpeg']):
                    recent_files.append((file_path, file_time))
    
    if recent_files:
        recent_files.sort(key=lambda x: x[1], reverse=True)
        most_recent = recent_files[0][0]
        print_success(f"Found recently created file: {most_recent}")
        return most_recent
    
    print_error("No matching files found")
    return None

def check_api_status():
    """Check if the API is running and accessible"""
    print("\n" + "=" * 60)
    print("API STATUS CHECK")
    print("=" * 60)
    
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print_success("API is running and accessible")
            return True
        else:
            print_error(f"API returned status code: {response.status_code}")
            return False
    except ImportError:
        print_error("requests module not available")
        return False
    except Exception as e:
        print_error(f"Cannot connect to API: {e}")
        return False

def main():
    """Main debug function"""
    print("VIETNAMESE POEM TO IMAGE GENERATOR - FILE DISCOVERY DEBUG")
    print("=" * 60)
    
    # Check API status
    api_running = check_api_status()
    
    # Check outputs directory
    check_outputs_directory()
    
    # Test file discovery for common filenames
    test_filenames = ["kien", "poem", "test", "output", "generated"]
    
    print("\n" + "=" * 60)
    print("FILE DISCOVERY TESTS")
    print("=" * 60)
    
    for filename in test_filenames:
        result = test_file_discovery(filename)
        if result:
            print_success(f"✓ Found file for '{filename}': {result}")
        else:
            print_error(f"✗ No file found for '{filename}'")
    
    # Additional checks
    print("\n" + "=" * 60)
    print("ADDITIONAL CHECKS")
    print("=" * 60)
    
    # Check current working directory
    print_info(f"Current working directory: {os.getcwd()}")
    
    # Check if we're in the right directory
    if os.path.exists("main.py"):
        print_success("Found main.py - likely in correct directory")
    else:
        print_warning("main.py not found - check if you're in the right directory")
    
    # Check for generation_storage directory
    if os.path.exists("generation_storage"):
        print_success("Found generation_storage directory")
        storage_files = os.listdir("generation_storage")
        print_info(f"Storage contains {len(storage_files)} items")
    else:
        print_info("generation_storage directory not found (normal for first run)")
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main() 