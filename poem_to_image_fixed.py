import torch
import gc
import os
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from peft import PeftModel, PeftConfig
from compel import Compel
from huggingface_hub import hf_hub_download

class PhoCLIPEmbedding:
    def __init__(self, model_repo="kienhoang123/ViCLIP", fallback_path="e:/ViCLIP/phoclip_checkpoint"):
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoModel
        from huggingface_hub import hf_hub_download
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading PhoCLIP model...")
        
        # Try to load from Hugging Face first, fallback to local if needed
        try:
            self._load_huggingface_model(model_repo)
        except Exception as e:
            print(f"Failed to load from Hugging Face: {e}")
            print(f"Falling back to local model at {fallback_path}")
            self._load_local_model(fallback_path)
    
    def _load_huggingface_model(self, model_repo):
        """Load PhoCLIP model from Hugging Face Hub"""
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoModel
        from huggingface_hub import hf_hub_download
        
        print(f"Loading PhoCLIP model from Hugging Face: {model_repo}")
        
        # Load tokenizer (this should work directly)
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=False)
        print("✓ Tokenizer loaded successfully")
        
        # Download the complete model file
        model_file = hf_hub_download(repo_id=model_repo, filename="model.pt")
        print("✓ Model file downloaded")
        
        # Load the complete state dict
        complete_state_dict = torch.load(model_file, map_location=self.device)
        print("✓ Model state dict loaded")
        
        # Initialize text encoder (PhoBERT base)
        self.text_encoder = AutoModel.from_pretrained("vinai/phobert-base").to(self.device)
        
        # Extract text encoder state dict
        text_encoder_state_dict = {}
        for key, value in complete_state_dict.items():
            if key.startswith('text_encoder.'):
                # Remove 'text_encoder.' prefix
                new_key = key[len('text_encoder.'):]
                text_encoder_state_dict[new_key] = value
        
        # Load text encoder weights
        self.text_encoder.load_state_dict(text_encoder_state_dict)
        print("✓ Text encoder weights loaded")
        
        # Initialize and load text projection head
        self.text_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        ).to(self.device)
        
        # Extract text projection state dict
        text_proj_state_dict = {}
        for key, value in complete_state_dict.items():
            if key.startswith('text_proj.'):
                # Remove 'text_proj.' prefix
                new_key = key[len('text_proj.'):]
                text_proj_state_dict[new_key] = value
        
        # Load projection weights
        self.text_proj.load_state_dict(text_proj_state_dict)
        print("✓ Text projection weights loaded")
        
        # Set to evaluation mode
        self.text_encoder.eval()
        self.text_proj.eval()
        
        print("Successfully loaded PhoCLIP model from Hugging Face")
    
    def _load_local_model(self, checkpoint_path):
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoModel
        
        print(f"Loading local PhoCLIP model from {checkpoint_path}")
        
        # Load text encoder (PhoBERT)
        self.text_encoder = AutoModel.from_pretrained(f"{checkpoint_path}/text_encoder").to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(f"{checkpoint_path}/tokenizer", use_fast=False)
        
        # Load text projection head
        self.text_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        ).to(self.device)
        self.text_proj.load_state_dict(torch.load(f"{checkpoint_path}/text_proj.pt"))
        
        # Set to evaluation mode
        self.text_encoder.eval()
        self.text_proj.eval()
        
        print("Successfully loaded local PhoCLIP model")
    
    def encode_text(self, text):
        import torch
        import torch.nn.functional as F
        
        # Tokenize the text
        inputs = self.tokenizer(
            text, 
            padding='max_length', 
            max_length=77, 
            truncation=True, 
            return_tensors='pt'
        ).to(self.device)
        
        # Get text embeddings
        with torch.no_grad():
            text_outputs = self.text_encoder(
                input_ids=inputs.input_ids, 
                attention_mask=inputs.attention_mask, 
                return_dict=True
            )
            text_cls = text_outputs.last_hidden_state[:, 0, :]
            text_proj = self.text_proj(text_cls)
            text_emb = F.normalize(text_proj, p=2, dim=-1)
            
        return text_emb.cpu().numpy()

class PoemAnalyzer:
    """Class to analyze Vietnamese poems using the Llama3.2 model"""
    def __init__(self):
        # Define the device attribute
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # First, identify which base model this adapter was trained on
        peft_config = PeftConfig.from_pretrained("kienhoang123/Llama3.2_Poem_Analysis")
        base_model_name = peft_config.base_model_name_or_path
        
        # Load the base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load the adapter on top of the model
        self.model = PeftModel.from_pretrained(
            self.model,
            "kienhoang123/Llama3.2_Poem_Analysis",
            is_trainable=False
        )
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("kienhoang123/Llama3.2_Poem_Analysis")
    
    def analyze(self, poem, max_new_tokens=100):
        """Analyze a poem to extract key elements"""
        input_text = f"""
        ### Instruction:
        Analyze the given poem and extract its emotional tone, metaphor, setting, motion, and generate a prompt based on the poem.

        ### Input Poem:
        {poem}

        ### Poem Analysis:
        """
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.2
        )
        full_analysis = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return full_analysis
    
    def extract_elements(self, full_analysis):
        """Extract key elements from the analysis"""
        elements = {
            "emotional_tone": "",
            "metaphor": "",
            "setting": "",
            "motion": ""
        }
        
        lines = full_analysis.split('\n')
        for line in lines:
            line = line.strip()
            if "Emotional Tone:" in line:
                elements["emotional_tone"] = line.split("Emotional Tone:")[1].strip()
            elif "Metaphor:" in line:
                elements["metaphor"] = line.split("Metaphor:")[1].strip()
            elif "Setting:" in line:
                elements["setting"] = line.split("Setting:")[1].strip()
            elif "Motion:" in line:
                elements["motion"] = line.split("Motion:")[1].strip()
        
        # Create a concise analysis in Vietnamese
        concise_analysis = f"""
        Cảm xúc: {elements['emotional_tone']}
        Ẩn dụ: {elements['metaphor']}
        Bối cảnh: {elements['setting']}
        Chuyển động: {elements['motion']}
        """
        
        return concise_analysis


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
        # Remove lines that contain meta-instructions
        lines = prompt.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip lines that are meta-instructions
            if not line:
                continue
            if "tạo ra một prompt" in line or "Dựa trên phân tích" in line or "Instruction" in line or "Response" in line:
                continue
            cleaned_lines.append(line)
        # Return the first non-empty, non-meta line, or join all cleaned lines
        return cleaned_lines[0] if cleaned_lines else prompt


class DiffusionGenerator:
    """Class to generate animations from prompts using AnimateDiff"""
    
    def __init__(self):
        self._prepare_device()
        self.pipe = self._load_models()
    
    def _prepare_device(self):
        """Prepare CUDA device if available"""
        if torch.cuda.is_available():
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            torch.cuda.empty_cache()
            gc.collect()
            print("Đã chuẩn bị GPU cho diffusion model")
        else:
            print("WARNING: GPU không khả dụng, việc tạo hình ảnh có thể rất chậm")
    
    def _load_models(self):
        """Load the AnimateDiff models"""
        print("Đang tải mô hình AnimateDiff...")
        
        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-2", 
            torch_dtype=torch.float16
        )

        model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
        pipe = AnimateDiffPipeline.from_pretrained(
            model_id, 
            motion_adapter=adapter, 
            torch_dtype=torch.float16
        )
        
        scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )
        
        pipe.scheduler = scheduler

        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()
        
        return pipe
    
    def generate(self, prompt, output_path="animation.gif", negative_prompt=None):
        """Generate animation from prompt"""
        if negative_prompt is None:
            negative_prompt = "chất lượng kém, mờ, không rõ ràng"
        
        # Use the provided prompt instead of hardcoded one
        if not prompt or prompt.strip() == "":
            prompt = "Ba người phụ nữ đứng trên một con phố trong thành phố"
        
        # Check prompt length and truncate if needed
        if len(prompt.split()) > 70:
            print(f"Cảnh báo: Prompt quá dài ({len(prompt.split())} từ). Đang cắt ngắn...")
            prompt_words = prompt.split()[:70]
            prompt = " ".join(prompt_words)
            print(f"Prompt sau khi cắt ngắn: {prompt}")
        
        print(f"Đang tạo hình ảnh với prompt: {prompt}")
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=16,
            guidance_scale=7.5,
            num_inference_steps=25,
            generator=torch.Generator("cuda").manual_seed(42)
        )
        frames = output.frames[0]
        export_to_gif(frames, output_path)
        print(f"Đã lưu animation vào {output_path}")
        return output_path


class PoemToImagePipeline:
    """Main pipeline to convert poem to animation"""
    def __init__(self, use_local_model_for_prompt=False):
        # self.poem_analyzer = PoemAnalyzer()
        
        # Use your PhoCLIP model instead of CLIP
        self.phoclip_encoder = PhoCLIPEmbedding()
        
        # Initialize the prompt generator
        # self.prompt_generator = PromptGenerator(use_local_model=use_local_model_for_prompt)
        
        # Initialize the diffusion generator
        self.diffusion_generator = DiffusionGenerator()
        
        # Replace the text encoder in the diffusion pipeline with PhoCLIP
        self._integrate_phoclip_with_pipeline()
    
    def _integrate_phoclip_with_pipeline(self):
        """Replace the default text encoder with PhoCLIP"""
        # Get the diffusion pipeline
        self.pipe = self.diffusion_generator.pipe
        
        # Get the expected embedding dimension from the original text encoder
        expected_dim = self.pipe.text_encoder.config.hidden_size
        phoclip_dim = 768  # PhoCLIP output dimension
        
        # Create a projection layer if dimensions don't match
        if phoclip_dim != expected_dim:
            self.embedding_projection = torch.nn.Linear(phoclip_dim, expected_dim).to(
                self.pipe.device, dtype=torch.float16
            )
            # Initialize with identity-like mapping
            torch.nn.init.xavier_uniform_(self.embedding_projection.weight)
        else:
            self.embedding_projection = None
        
        # Store original encode_prompt method
        self.original_encode_prompt = self.pipe.encode_prompt
        
        # Replace the encode prompt method
        self.pipe.encode_prompt = self._phoclip_encode_prompt
    
    def _phoclip_encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None, lora_scale=None, **kwargs):
        """Custom encode prompt method using PhoCLIP"""
        if prompt_embeds is None:
            # Use PhoCLIP to encode the prompt
            if isinstance(prompt, str):
                prompt = [prompt]
            
            # Encode with PhoCLIP
            prompt_embeds_list = []
            for p in prompt:
                phoclip_embed = self.phoclip_encoder.encode_text(p)
                # Convert to torch tensor and move to device
                phoclip_embed = torch.from_numpy(phoclip_embed).to(device, dtype=torch.float16)
                
                # Project to expected dimensions if needed
                if self.embedding_projection is not None:
                    phoclip_embed = self.embedding_projection(phoclip_embed)
                
                # The diffusion model expects embeddings of shape (batch_size, sequence_length, hidden_size)
                # PhoCLIP gives us (batch_size, hidden_size), so we need to expand it
                # Instead of repeating, let's use a more sophisticated approach
                seq_length = 77  # Standard CLIP sequence length
                hidden_size = phoclip_embed.shape[-1]
                
                # Create a sequence by interpolating the embedding
                # This is better than just repeating the same embedding
                phoclip_embed = phoclip_embed.unsqueeze(1)  # (batch_size, 1, hidden_size)
                
                # Use the same embedding for all positions but with slight variations
                # This maintains the semantic meaning while providing sequence structure
                phoclip_embed = phoclip_embed.repeat(1, seq_length, 1)
                
                # Add small positional variations to make it more realistic
                position_embeddings = torch.randn_like(phoclip_embed) * 0.01
                phoclip_embed = phoclip_embed + position_embeddings
                
                prompt_embeds_list.append(phoclip_embed)
            
            prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
            
            # Repeat for batch size
            if num_images_per_prompt > 1:
                prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1, 1)
            
            # Handle negative prompts
            if do_classifier_free_guidance and negative_prompt_embeds is None:
                if negative_prompt is None:
                    negative_prompt = [""] * len(prompt)
                elif isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt]
                
                negative_prompt_embeds_list = []
                for np in negative_prompt:
                    if np == "":
                        # Use zero embeddings for empty negative prompt
                        neg_embed = torch.zeros_like(prompt_embeds[:1])
                    else:
                        neg_embed = self.phoclip_encoder.encode_text(np)
                        neg_embed = torch.from_numpy(neg_embed).to(device, dtype=torch.float16)
                        
                        # Project to expected dimensions if needed
                        if self.embedding_projection is not None:
                            neg_embed = self.embedding_projection(neg_embed)
                        
                        # Apply the same sequence expansion as positive prompts
                        seq_length = 77
                        neg_embed = neg_embed.unsqueeze(1).repeat(1, seq_length, 1)
                        
                        # Add small positional variations
                        position_embeddings = torch.randn_like(neg_embed) * 0.01
                        neg_embed = neg_embed + position_embeddings
                    
                    negative_prompt_embeds_list.append(neg_embed)
                
                negative_prompt_embeds = torch.cat(negative_prompt_embeds_list, dim=0)
                
                if num_images_per_prompt > 1:
                    negative_prompt_embeds = negative_prompt_embeds.repeat(num_images_per_prompt, 1, 1)
        
        return prompt_embeds, negative_prompt_embeds
    
    def process(self, poem, output_path="animation.gif"):
        """Process a poem through the full pipeline"""
        # 1. Analyze the poem
        print("=== PHÂN TÍCH BÀI THƠ ===")
        try:
            self.poem_analyzer = PoemAnalyzer()
            full_analysis = self.poem_analyzer.analyze(poem)
            print(full_analysis)
            
            # 2. Extract key elements
            print("\n=== TRÍCH XUẤT YẾU TỐ CHÍNH ===")
            concise_analysis = self.poem_analyzer.extract_elements(full_analysis)
            print(concise_analysis)
            
            # 3. Generate diffusion prompt
            print("\n=== TẠO PROMPT CHO MÔ HÌNH DIFFUSION ===")
            self.prompt_generator = PromptGenerator(use_local_model=False)
            diffusion_prompt = self.prompt_generator.generate(concise_analysis)
            print(f"Generated prompt: {diffusion_prompt}")
            
        except Exception as e:
            print(f"Error in poem analysis: {e}")
            print("Using fallback prompt generation...")
            
            # Fallback: Generate a simple prompt from the poem
            diffusion_prompt = self._generate_fallback_prompt(poem)
            print(f"Fallback prompt: {diffusion_prompt}")
        
        # 4. Generate animation
        print("\n=== TẠO HÌNH ẢNH ĐỘNG ===")
        animation_path = self.diffusion_generator.generate(diffusion_prompt, output_path)
        
        print(f"\nHoàn thành! Animation đã được tạo tại: {animation_path}")
        return animation_path
    
    def _generate_fallback_prompt(self, poem):
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
        from collections import Counter
        word_counts = Counter(keywords)
        top_words = [word for word, count in word_counts.most_common(5)]
        
        # Create a simple prompt
        if top_words:
            prompt = f"một cảnh đẹp với {' '.join(top_words[:3])}"
        else:
            prompt = "một cảnh đẹp thiên nhiên"
        
        return prompt


def main():
    parser = argparse.ArgumentParser(description="Tạo hình ảnh động từ bài thơ")
    parser.add_argument("--poem", type=str, help="Nội dung bài thơ")
    parser.add_argument("--poem-file", type=str, help="Đường dẫn đến file chứa bài thơ")
    parser.add_argument("--output", type=str, default="animation.gif", help="Đường dẫn lưu file kết quả")
    parser.add_argument("--use-local-model", action="store_true", help="Sử dụng mô hình local thay vì Ollama API")
    
    args = parser.parse_args()
    
    # Get poem from either argument or file
    poem = args.poem
    if poem is None and args.poem_file:
        with open(args.poem_file, "r", encoding="utf-8") as f:
            poem = f.read()
    
    if poem is None:
        # Use default poem if no input provided
        poem = """
        đẩy hoa dun lá khỏi tay trời , <
        nghĩ lại tình duyên luống ngậm ngùi . <
        bắc yến nam hồng , thư mấy bức , <
        đông đào tây liễu , khách đôi nơi . <
        lửa ân , dập mãi sao không tắt , <
        biển ái , khơi hoài vẫn chẳng vơi . <
        đèn nguyệt trong xanh , mây chẳng bợn , <
        xin soi xét đến tấm lòng ai ...
        """
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run the pipeline
    pipeline = PoemToImagePipeline(use_local_model_for_prompt=args.use_local_model)
    pipeline.process(poem, str(output_path))


if __name__ == "__main__":
    main()

