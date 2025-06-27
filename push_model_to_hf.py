import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import HfApi
import shutil

# Define model paths
checkpoint_path = "e:/ViCLIP/phoclip_checkpoint"
text_encoder_path = f"{checkpoint_path}/text_encoder"
tokenizer_path = f"{checkpoint_path}/tokenizer"
text_proj_path = f"{checkpoint_path}/text_proj.pt"

# Create a temporary directory to save the model
import tempfile

with tempfile.TemporaryDirectory() as tmpdirname:
    print(f"Created temporary directory: {tmpdirname}")
    
    # Step 1: Copy the base text encoder model (PhoBERT)
    print("Copying base text encoder (PhoBERT)...")
    base_model_dir = os.path.join(tmpdirname, "text_encoder")
    os.makedirs(base_model_dir, exist_ok=True)
    
    # Copy the text encoder files
    shutil.copytree(text_encoder_path, base_model_dir, dirs_exist_ok=True)
    
    # Step 2: Copy the tokenizer
    print("Copying tokenizer...")
    tokenizer_dir = os.path.join(tmpdirname, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    # Copy the tokenizer files
    shutil.copytree(tokenizer_path, tokenizer_dir, dirs_exist_ok=True)
    
    # Step 3: Save the projection head
    print("Saving projection head...")
    proj_dir = os.path.join(tmpdirname, "projection")
    os.makedirs(proj_dir, exist_ok=True)
    
    # Copy the projection head
    shutil.copy(text_proj_path, os.path.join(proj_dir, "text_proj.pt"))    # Step 4: Create a README
    with open(os.path.join(tmpdirname, "README.md"), "w", encoding="utf-8") as f:
        f.write('''# ViCLIP - Vietnamese CLIP Text Encoder

This model is a Vietnamese adaptation of CLIP text encoder, trained on Vietnamese data.

## Model Description

- Text encoder based on PhoBERT
- Projection head to align with CLIP embedding space
- Optimized for Vietnamese text understanding

## Directory Structure

- `text_encoder/`: PhoBERT base model for Vietnamese text encoding
- `tokenizer/`: Tokenizer for the text encoder
- `projection/`: Projection head to align with CLIP embedding space

## Usage

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class PhoCLIPTextModel:
    def __init__(self, model_path="kienhoang123/ViCLIP"):
        # Load text encoder
        self.text_encoder = AutoModel.from_pretrained(f"{model_path}/text_encoder")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer", use_fast=False)
        
        # Load text projection head
        self.text_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )
        self.text_proj.load_state_dict(torch.load(f"{model_path}/projection/text_proj.pt"))
        
        # Set to evaluation mode
        self.text_encoder.eval()
        self.text_proj.eval()
    
    def encode_text(self, text, device="cuda" if torch.cuda.is_available() else "cpu"):
        # Move models to device
        self.text_encoder = self.text_encoder.to(device)
        self.text_proj = self.text_proj.to(device)
        
        # Tokenize the text
        inputs = self.tokenizer(
            text, 
            padding='max_length', 
            max_length=77, 
            truncation=True, 
            return_tensors='pt'
        ).to(device)
        
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

# Example usage
model = PhoCLIPTextModel("kienhoang123/ViCLIP")
embedding = model.encode_text("This is an example text")
```
''')
    
    # Push to Hub
    print("Pushing model to Hugging Face Hub...")
    api = HfApi()
    
    try:
        api.create_repo(
            repo_id="kienhoang123/ViCLIP",
            private=False,
            exist_ok=True
        )
        
        # Upload all files in the directory
        api.upload_folder(
            folder_path=tmpdirname,
            repo_id="kienhoang123/ViCLIP",
            commit_message="Upload PhoCLIP text model as a composite model"
        )
        print("Model successfully uploaded to Hugging Face Hub at kienhoang123/ViCLIP")
    except Exception as e:
        print(f"Error uploading model: {e}")
        print("You may need to login to Hugging Face first with: huggingface-cli login")
