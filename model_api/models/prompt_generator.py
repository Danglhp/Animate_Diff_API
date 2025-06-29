import requests
from collections import Counter

class PromptGenerator:
    """Class to generate image generation prompts from poem analysis"""
    
    def __init__(self, use_local_model=False, model=None, tokenizer=None, device=None):
        self.use_local_model = use_local_model
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        # Maximum number of tokens for CLIP text encoder in Stable Diffusion
        self.max_prompt_tokens = 77
    
    def generate(self, analysis):
        """Generate a diffusion model prompt in Vietnamese based on poem analysis"""
        if self.use_local_model and self.model is not None and self.tokenizer is not None:
            prompt = self._generate_with_local_model(analysis)
        else:
            prompt = self._generate_with_ollama(analysis)
        # Clean the prompt: remove meta-instructions
        prompt = self.clean_prompt(prompt)
        return prompt
    
    def _generate_with_local_model(self, analysis):
        """Generate prompt using the local model"""
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
    
    def _generate_with_ollama(self, analysis):
        """Generate prompt using Ollama API"""
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
    
    def clean_prompt(self, prompt):
        """Clean the prompt by removing meta-instructions"""
        lines = prompt.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip lines that are meta-instructions
            if not line:
                continue
            if any(keyword in line.lower() for keyword in [
                "tạo ra một prompt", "dựa trên phân tích", "instruction", "response", 
                "phân tích bài thơ", "prompt cần bao gồm", "lưu ý quan trọng"
            ]):
                continue
            cleaned_lines.append(line)
        # Return the first non-empty, non-meta line, or join all cleaned lines
        return cleaned_lines[0] if cleaned_lines else prompt
    
    def generate_fallback_prompt(self, poem):
        """Generate a fallback prompt when poem analysis fails"""
        # Extract key words from the poem
        poem_lines = poem.strip().split('\n')
        
        # Simple keyword extraction
        keywords = []
        for line in poem_lines:
            line = line.strip()
            if line:
                # Extract Vietnamese words (basic approach)
                words = line.split()
                for word in words:
                    # Remove punctuation and special characters
                    clean_word = ''.join(c for c in word if c.isalnum() or c.isspace())
                    if len(clean_word) > 2:  # Only keep words longer than 2 characters
                        keywords.append(clean_word)
        
        # Take the most common words
        word_counts = Counter(keywords)
        top_words = [word for word, count in word_counts.most_common(5)]
        
        # Create a simple prompt
        if top_words:
            prompt = f"một cảnh đẹp với {' '.join(top_words[:3])}"
        else:
            prompt = "một cảnh đẹp thiên nhiên"
        
        return prompt 