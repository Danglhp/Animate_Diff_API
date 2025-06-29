import torch
from pathlib import Path
from models import PhoCLIPEmbedding, PoemAnalyzer, PromptGenerator, DiffusionGenerator

class PoemToImagePipeline:
    """Main pipeline to convert poem to animation"""
    def __init__(self, use_local_model_for_prompt=False):
        # Use your PhoCLIP model instead of CLIP
        self.phoclip_encoder = PhoCLIPEmbedding()
        
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