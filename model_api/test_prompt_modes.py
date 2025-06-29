#!/usr/bin/env python3
"""
Test script to demonstrate the three different prompt generation modes
Shows generated prompts without running full animation generation
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_prompt_generation_modes():
    """Test all three prompt generation modes and show generated prompts"""
    
    # Sample Vietnamese poem
    poem = """
    đẩy hoa dun lá khỏi tay trời
    nghĩ lại tình duyên luống ngậm ngùi
    bắc yến nam hồng thư mấy bức
    đông đào tây liễu khách đôi nơi
    """
    
    print("🧪 Testing Three Prompt Generation Modes")
    print("=" * 60)
    print(f"Input poem: {poem.strip()}")
    print("=" * 60)
    
    try:
        # Import the required modules
        from models import PoemAnalyzer, PromptGenerator
        
        # Initialize the poem analyzer
        print("\n📊 Initializing Poem Analyzer...")
        poem_analyzer = PoemAnalyzer()
        
        # Analyze the poem
        print("🔍 Analyzing poem...")
        full_analysis = poem_analyzer.analyze(poem)
        print(f"Full analysis:\n{full_analysis}")
        
        # Extract key elements
        print("\n📋 Extracting key elements...")
        concise_analysis = poem_analyzer.extract_elements(full_analysis)
        print(f"Concise analysis:\n{concise_analysis}")
        
        # Test all three modes
        modes = [
            "analysis_to_vietnamese",  # Mode 1: Poem analysis -> Local Llama -> Vietnamese prompt
            "direct_prompt",           # Mode 2: Extract prompt directly from analysis
            "analysis_to_english"      # Mode 3: Poem analysis -> Local Llama -> English prompt
        ]
        
        mode_descriptions = {
            "analysis_to_vietnamese": "Poem analysis → Local Llama → Vietnamese prompt",
            "direct_prompt": "Extract prompt directly from analysis",
            "analysis_to_english": "Poem analysis → Local Llama → English prompt"
        }
        
        # Initialize prompt generator
        print("\n🎨 Initializing Prompt Generator...")
        prompt_generator = PromptGenerator(use_local_model=False)  # Use Ollama API
        
        print("\n" + "=" * 60)
        print("📝 GENERATED PROMPTS FOR EACH MODE:")
        print("=" * 60)
        
        for mode in modes:
            print(f"\n🔸 Mode: {mode}")
            print(f"Description: {mode_descriptions[mode]}")
            print("-" * 50)
            
            try:
                # Generate prompt using the specific mode
                generated_prompt = prompt_generator.generate(concise_analysis, mode=mode)
                
                print(f"✅ Generated Prompt:")
                print(f"'{generated_prompt}'")
                print(f"Word count: {len(generated_prompt.split())}")
                
            except Exception as e:
                print(f"❌ Error generating prompt for mode '{mode}': {e}")
                print("Using fallback prompt...")
                fallback_prompt = prompt_generator.generate_fallback_prompt(poem)
                print(f"Fallback prompt: '{fallback_prompt}'")
        
        print("\n" + "=" * 60)
        print("🎉 Prompt generation test completed!")
        print("\n💡 Next steps:")
        print("  1. Review the generated prompts above")
        print("  2. Choose which mode you prefer")
        print("  3. Start the API server: python main.py")
        print("  4. Test with actual animation generation")
        
    except Exception as e:
        print(f"❌ Error during prompt generation test: {e}")
        print("\n💡 Make sure you have:")
        print("  - All dependencies installed: pip install -r requirements.txt")
        print("  - Ollama running locally for API calls")
        print("  - Internet connection for model downloads")

def test_api_endpoints():
    """Test API endpoints if server is running"""
    import requests
    
    print("\n🌐 API Endpoint Test")
    print("-" * 30)
    
    try:
        # Test if API is running
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running!")
            print("📋 Available endpoints:")
            print("  - GET  /health")
            print("  - POST /generate")
            print("  - GET  /status/{task_id}")
            print("  - GET  /download/{task_id}")
            print("  - GET  /tasks")
            print("\n🌐 Interactive docs: http://localhost:8000/docs")
        else:
            print(f"⚠️  API server responded with status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ API server is not running")
        print("💡 Start it with: python main.py")
    except Exception as e:
        print(f"❌ Error testing API: {e}")

def main():
    """Run all tests"""
    print("🧪 Testing Enhanced Prompt Generation (Prompt Preview Only)")
    print("=" * 70)
    
    # Test prompt generation modes
    test_prompt_generation_modes()
    
    # Test API endpoints if available
    test_api_endpoints()
    
    print("\n" + "=" * 70)
    print("✅ Testing completed!")
    print("\n🚀 To test with full animation generation:")
    print("  1. Start the API: python main.py")
    print("  2. Run: python test_prompt_modes.py")
    print("  3. Or visit: http://localhost:8000/docs")

if __name__ == "__main__":
    main() 