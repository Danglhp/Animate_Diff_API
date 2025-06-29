#!/usr/bin/env python3
"""
Test script for the clean modular Model API package
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported correctly"""
    try:
        from models import PhoCLIPEmbedding
        print("✓ PhoCLIPEmbedding imported successfully")
    except Exception as e:
        print(f"✗ Failed to import PhoCLIPEmbedding: {e}")
        return False
    
    try:
        from models import PoemAnalyzer
        print("✓ PoemAnalyzer imported successfully")
    except Exception as e:
        print(f"✗ Failed to import PoemAnalyzer: {e}")
        return False
    
    try:
        from models import PromptGenerator
        print("✓ PromptGenerator imported successfully")
    except Exception as e:
        print(f"✗ Failed to import PromptGenerator: {e}")
        return False
    
    try:
        from models import DiffusionGenerator
        print("✓ DiffusionGenerator imported successfully")
    except Exception as e:
        print(f"✗ Failed to import DiffusionGenerator: {e}")
        return False
    
    try:
        from pipeline import PoemToImagePipeline
        print("✓ PoemToImagePipeline imported successfully")
    except Exception as e:
        print(f"✗ Failed to import PoemToImagePipeline: {e}")
        return False
    
    try:
        from schemas import PoemRequest, PoemResponse
        print("✓ Schemas (PoemRequest, PoemResponse) imported successfully")
    except Exception as e:
        print(f"✗ Failed to import schemas: {e}")
        return False
    
    return True

def test_package_import():
    """Test importing from the main package"""
    try:
        from model_api import PoemToImagePipeline, PhoCLIPEmbedding, PoemRequest, PoemResponse
        print("✓ Main package imports work correctly")
        return True
    except Exception as e:
        print(f"✗ Main package imports failed: {e}")
        return False

def test_schema_validation():
    """Test Pydantic schema validation"""
    try:
        from schemas import PoemRequest, PoemResponse
        
        # Test PoemRequest
        request = PoemRequest(poem="test poem", output_filename="test")
        print("✓ PoemRequest schema validation works")
        
        # Test PoemResponse
        response = PoemResponse(
            task_id="test-id",
            status="processing",
            message="test message",
            output_path=None
        )
        print("✓ PoemResponse schema validation works")
        
        return True
    except Exception as e:
        print(f"✗ Schema validation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Clean Modular Model API Package")
    print("=" * 60)
    
    print("\n1. Testing individual module imports...")
    if not test_imports():
        print("❌ Module import tests failed")
        return
    
    print("\n2. Testing main package imports...")
    if not test_package_import():
        print("❌ Main package import tests failed")
        return
    
    print("\n3. Testing schema validation...")
    if not test_schema_validation():
        print("❌ Schema validation tests failed")
        return
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! The clean modular structure is working correctly.")
    print("\n📁 Package Structure:")
    print("  ├── models/          # All model components")
    print("  ├── pipeline/        # Main orchestration")
    print("  ├── schemas/         # Pydantic models")
    print("  └── main.py          # FastAPI application")
    print("\n🚀 To run the API:")
    print("  python main.py")
    print("\n🌐 Access the API at:")
    print("  http://localhost:8000")
    print("  http://localhost:8000/docs (interactive docs)")

if __name__ == "__main__":
    main() 