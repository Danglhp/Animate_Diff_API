import torch
import gc
import os
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

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