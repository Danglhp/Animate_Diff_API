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
        
        # Always use the local model for now due to model type compatibility issues
        self._load_local_model(fallback_path)
    
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
            return self._generate_with_local_model(analysis)
        else:
            return self._generate_with_ollama(analysis)
    
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
            negative_prompt = "bad quality, worse quality, low resolution"
        
        prompt = "an aerial view of a cyberpunk city, night time, neon lights, masterpiece, high quality"
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
        self.text_encoder = PhoCLIPEmbedding()
        
        # Initialize the prompt generator
        # self.prompt_generator = PromptGenerator(use_local_model=use_local_model_for_prompt)
        
        # Initialize the diffusion generator
        self.diffusion_generator = DiffusionGenerator()
        
        # Keep the rest of your initialization code for diffusion model
        self.pipe = AnimateDiffPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
        )
        self.pipe = self.pipe.to("cuda")
        
        # # Optional: use compel for better prompting
        # self.compel_proc = Compel(
        #     tokenizer=self.pipe.tokenizer,
        #     text_encoder=self.pipe.text_encoder,
        #     truncate_long_prompts=False,
        # )
    
    def process(self, poem, output_path="animation.gif"):
        """Process a poem through the full pipeline"""
        # 1. Analyze the poem
        # print("=== PHÂN TÍCH BÀI THƠ ===")
        # full_analysis = self.poem_analyzer.analyze(poem)
        # print(full_analysis)
        
        # # 2. Extract key elements
        # print("\n=== TRÍCH XUẤT YẾU TỐ CHÍNH ===")
        # concise_analysis = self.poem_analyzer.extract_elements(full_analysis)
        # print(concise_analysis)
        
        # # 3. Generate diffusion prompt
        # print("\n=== TẠO PROMPT CHO MÔ HÌNH DIFFUSION ===")
        # diffusion_prompt = self.prompt_generator.generate(concise_analysis)
        # print(diffusion_prompt)
        
        diffusion_prompt = ""
        # 4. Generate animation
        print("\n=== TẠO HÌNH ẢNH ĐỘNG ===")
        animation_path = self.diffusion_generator.generate(diffusion_prompt, output_path)
        
        print(f"\nHoàn thành! Animation đã được tạo tại: {animation_path}")
        return animation_path


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

