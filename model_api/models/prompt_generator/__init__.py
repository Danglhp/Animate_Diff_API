import requests
from collections import Counter
from typing import Dict, Any

class PromptGenerator:
    """Class to generate image generation prompts from poem analysis with multiple modes"""
    
    def __init__(self, use_local_model=False, model=None, tokenizer=None, device=None):
        self.use_local_model = use_local_model
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        # Maximum number of tokens for CLIP text encoder in Stable Diffusion
        self.max_prompt_tokens = 77
    
    def generate(self, analysis, mode="analysis_to_vietnamese"):
        """Generate a diffusion model prompt based on the specified mode"""
        if mode == "analysis_to_vietnamese":
            return self._generate_vietnamese_from_analysis(analysis)
        elif mode == "direct_prompt":
            return self._extract_direct_prompt(analysis)
        elif mode == "analysis_to_english":
            return self._generate_english_from_analysis(analysis)
        else:
            # Fallback to Vietnamese generation
            return self._generate_vietnamese_from_analysis(analysis)
    
    def _generate_vietnamese_from_analysis(self, analysis):
        """Mode 1: Generate Vietnamese prompt from poem analysis using local Llama model"""
        if self.use_local_model and self.model is not None and self.tokenizer is not None:
            prompt = self._generate_vietnamese_with_local_model(analysis)
        else:
            prompt = self._generate_vietnamese_with_ollama(analysis)
        
        return self.clean_prompt(prompt)
    
    def _generate_english_from_analysis(self, analysis):
        """Mode 3: Generate English prompt from poem analysis using local Llama model"""
        if self.use_local_model and self.model is not None and self.tokenizer is not None:
            prompt = self._generate_english_with_local_model(analysis)
        else:
            prompt = self._generate_english_with_ollama(analysis)
        
        return self.clean_prompt(prompt)
    
    def _extract_direct_prompt(self, analysis):
        """Mode 2: Extract prompt directly from poem analysis and trim excess data"""
        # Extract the prompt field from the analysis
        prompt = self._extract_prompt_field(analysis)
        
        # Trim the prompt to fit within token limits
        trimmed_prompt = self._trim_prompt(prompt)
        
        return trimmed_prompt
    
    def _extract_prompt_field(self, analysis):
        """Extract the prompt field from the poem analysis"""
        lines = analysis.split('\n')
        prompt_lines = []
        in_prompt_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for prompt-related keywords
            if any(keyword in line.lower() for keyword in [
                "prompt:", "generated prompt:", "image prompt:", "visual prompt:"
            ]):
                in_prompt_section = True
                # Extract the prompt part after the colon
                if ":" in line:
                    prompt_part = line.split(":", 1)[1].strip()
                    if prompt_part:
                        prompt_lines.append(prompt_part)
                continue
            
            # If we're in prompt section, collect lines until we hit another section
            if in_prompt_section:
                if any(section in line.lower() for section in [
                    "emotional tone:", "metaphor:", "setting:", "motion:", "analysis:", "instruction:"
                ]):
                    break
                prompt_lines.append(line)
        
        if prompt_lines:
            return " ".join(prompt_lines)
        else:
            # Fallback: use the entire analysis as prompt
            return analysis
    
    def _trim_prompt(self, prompt):
        """Trim prompt to fit within token limits"""
        words = prompt.split()
        if len(words) <= 70:  # Safe limit for diffusion models
            return prompt
        
        # Trim to 70 words
        trimmed_words = words[:70]
        return " ".join(trimmed_words)
    
    def _generate_vietnamese_with_local_model(self, analysis):
        """Generate Vietnamese prompt using local model"""
        input_text = f"""
        ### Instruction:
        Bạn là một nghệ sĩ AI chuyên tạo ra các prompt cho mô hình AI tạo hình ảnh (Midjourney, Stable Diffusion).
        Dựa trên phân tích bài thơ sau đây, hãy tạo ra một prompt ngắn gọn bằng tiếng Việt để mô hình AI có thể tạo ra hình ảnh thể hiện được không khí, cảm xúc và ẩn dụ của bài thơ.
        
        Phân tích bài thơ: {analysis}
        
        Prompt cần bao gồm:
        1. Mô tả cảnh vật/khung cảnh chính (dưới 20 từ)
        2. Cảm xúc và không khí (dưới 10 từ)
        3. Màu sắc chủ đạo (dưới 10 từ)
        4. Phong cách nghệ thuật (chỉ 1-2 từ)
        
        LƯU Ý QUAN TRỌNG: Prompt phải ngắn gọn, dưới 70 từ tổng cộng, không sử dụng từ tiếng Anh.
        Chỉ trả về prompt, không giải thích gì thêm.

        ### Response:
        """
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2
        )
        prompt = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return prompt
    
    def _generate_english_with_local_model(self, analysis):
        """Generate English prompt using local model"""
        input_text = f"""
        ### Instruction:
        You are an AI artist specializing in creating prompts for AI image generation models (Midjourney, Stable Diffusion).
        Based on the following poem analysis, create a concise English prompt that the AI model can use to generate an image that captures the atmosphere, emotions, and metaphors of the poem.
        
        Poem Analysis: {analysis}
        
        The prompt should include:
        1. Main scene/landscape description (under 20 words)
        2. Emotion and atmosphere (under 10 words)
        3. Dominant colors (under 10 words)
        4. Artistic style (only 1-2 words)
        
        IMPORTANT: The prompt must be concise, under 70 words total.
        Return only the prompt, no explanations.

        ### Response:
        """
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2
        )
        prompt = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return prompt
    
    def _generate_vietnamese_with_ollama(self, analysis):
        """Generate Vietnamese prompt using Ollama API"""
        prompt = f"""
        Bạn là một nghệ sĩ AI chuyên tạo ra các prompt cho mô hình AI tạo hình ảnh (Midjourney, Stable Diffusion).
        Dựa trên phân tích bài thơ sau đây, hãy tạo ra một prompt ngắn gọn bằng tiếng Việt để mô hình AI có thể tạo ra hình ảnh thể hiện được không khí, cảm xúc và ẩn dụ của bài thơ.
        
        Phân tích bài thơ: {analysis}
        
        Prompt cần bao gồm:
        1. Mô tả cảnh vật/khung cảnh chính (dưới 20 từ)
        2. Cảm xúc và không khí (dưới 10 từ)
        3. Màu sắc chủ đạo (dưới 10 từ)
        4. Phong cách nghệ thuật (chỉ 1-2 từ)
        
        LƯU Ý QUAN TRỌNG: Prompt phải ngắn gọn, dưới 70 từ tổng cộng, không sử dụng từ tiếng Anh.
        Chỉ trả về prompt, không giải thích gì thêm.
        """
        
        payload = {
            "model": "llama3.2:latest",
            "prompt": prompt,
            "stream": False,
        }
        
        try:
            response = requests.post(
                url='http://localhost:11434/api/generate', json=payload
            ).json()
            return response['response']
        except Exception as e:
            return f"Lỗi khi tạo prompt: {str(e)}"
    
    def _generate_english_with_ollama(self, analysis):
        """Generate English prompt using Ollama API"""
        prompt = f"""
        You are an AI artist specializing in creating prompts for AI image generation models (Midjourney, Stable Diffusion).
        Based on the following poem analysis, create a concise English prompt that the AI model can use to generate an image that captures the atmosphere, emotions, and metaphors of the poem.
        
        Poem Analysis: {analysis}
        
        The prompt should include:
        1. Main scene/landscape description (under 20 words)
        2. Emotion and atmosphere (under 10 words)
        3. Dominant colors (under 10 words)
        4. Artistic style (only 1-2 words)
        
        IMPORTANT: The prompt must be concise, under 70 words total.
        Return only the prompt, no explanations.
        """
        
        payload = {
            "model": "llama3.2:latest",
            "prompt": prompt,
            "stream": False,
        }
        
        try:
            response = requests.post(
                url='http://localhost:11434/api/generate', json=payload
            ).json()
            return response['response']
        except Exception as e:
            return f"Error generating prompt: {str(e)}"
    
    def clean_prompt(self, prompt):
        """Clean and format the generated prompt"""
        # Remove any instruction text that might have been included
        if "### Response:" in prompt:
            prompt = prompt.split("### Response:")[-1].strip()
        
        # Remove any remaining instruction markers
        prompt = prompt.replace("### Instruction:", "").strip()
        
        # Remove extra whitespace and newlines
        prompt = " ".join(prompt.split())
        
        # Ensure the prompt is not too long
        words = prompt.split()
        if len(words) > 70:
            prompt = " ".join(words[:70])
        
        return prompt
    
    def generate_fallback_prompt(self, poem):
        """Generate a simple fallback prompt if all else fails"""
        # Simple template-based prompt generation
        fallback_prompt = f"Beautiful Vietnamese landscape inspired by the poem: {poem[:50]}..."
        
        # Ensure it's not too long
        if len(fallback_prompt) > 200:
            fallback_prompt = fallback_prompt[:200] + "..."
        
        return fallback_prompt 