import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
import os

class PhoCLIPEmbedding:
    def __init__(self, model_repo="kienhoang123/ViCLIP", fallback_path="e:/ViCLIP/phoclip_checkpoint"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_repo = model_repo
        self.fallback_path = fallback_path
        
        print(f"Loading PhoCLIP model...")
        self._load_model()
    
    def _load_model(self):
        """Load PhoCLIP model with error handling"""
        try:
            self._load_huggingface_model()
        except Exception as e:
            print(f"Failed to load from Hugging Face: {e}")
            print(f"Falling back to local model at {self.fallback_path}")
            self._load_local_model()
    
    def _load_huggingface_model(self):
        """Load PhoCLIP model from Hugging Face Hub"""
        print(f"Loading PhoCLIP model from Hugging Face: {self.model_repo}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_repo, use_fast=False)
        print("✓ Tokenizer loaded successfully")
        
        # Download the complete model file
        model_file = hf_hub_download(repo_id=self.model_repo, filename="model.pt")
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
                new_key = key[len('text_proj.'):]
                text_proj_state_dict[new_key] = value
        
        # Load projection weights
        self.text_proj.load_state_dict(text_proj_state_dict)
        print("✓ Text projection weights loaded")
        
        # Set to evaluation mode
        self.text_encoder.eval()
        self.text_proj.eval()
        
        print("Successfully loaded PhoCLIP model from Hugging Face")
    
    def _load_local_model(self):
        """Load local PhoCLIP model"""
        print(f"Loading local PhoCLIP model from {self.fallback_path}")
        
        # Load text encoder (PhoBERT)
        self.text_encoder = AutoModel.from_pretrained(f"{self.fallback_path}/text_encoder").to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.fallback_path}/tokenizer", use_fast=False)
        
        # Load text projection head
        self.text_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        ).to(self.device)
        self.text_proj.load_state_dict(torch.load(f"{self.fallback_path}/text_proj.pt"))
        
        # Set to evaluation mode
        self.text_encoder.eval()
        self.text_proj.eval()
        
        print("Successfully loaded local PhoCLIP model")
    
    def encode_text(self, text):
        """Encode text using PhoCLIP text encoder"""
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