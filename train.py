import os
import argparse
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    CLIPFeatureExtractor,
    CLIPVisionModel,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

class ViCaptionDataset(Dataset):
    """Dataset for Vietnamese image-caption pairs from CSV and image folder."""
    def __init__(
        self,
        csv_file: str,
        images_dir: str,
        tokenizer_name: str = "vinai/phobert-base",
        vision_model_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 77,
    ):
        print(f"Loading CSV file: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"Initial rows: {len(df)}")

        # Drop rows with missing image or caption
        df = df.dropna(subset=["image", "caption_vi"]).reset_index(drop=True)
        print(f"After dropna: {len(df)} rows")

        # Check image existence
        img_paths = df['image'].apply(lambda x: os.path.join(images_dir, x))
        exists_mask = img_paths.apply(os.path.exists)
        missing_count = (~exists_mask).sum()
        print(f"Missing images: {missing_count}/{len(df)}")

        # Keep only existing
        df = df[exists_mask].reset_index(drop=True)
        print(f"Final valid pairs: {len(df)}")

        self.df = df
        self.images_dir = images_dir
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        print(f"Loading feature extractor: {vision_model_name}")
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(vision_model_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row['image'])
        
        try:
            # Set PIL to be more permissive with corrupted files
            from PIL import ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            
            # Load and properly convert the image
            image = Image.open(img_path)
            
            # Handle palette images with transparency
            if image.mode == 'P' and 'transparency' in image.info:
                image = image.convert('RGBA')
            
            # Convert to RGB (this will properly handle RGBA images too)
            image = image.convert('RGB')
            
            pixel_values = self.feature_extractor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)

            caption = str(row['caption_vi'])
            enc = self.tokenizer(
                caption,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )
            input_ids = enc.input_ids.squeeze(0)
            attention_mask = enc.attention_mask.squeeze(0)

            return {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        except Exception as e:
            # If we encounter a corrupted image, log it and return a dummy
            print(f"Error processing image {img_path}: {str(e)}")
            
            # Create a dummy black image as a fallback
            dummy_image = Image.new('RGB', (224, 224), color='black')
            pixel_values = self.feature_extractor(images=dummy_image, return_tensors='pt')['pixel_values'].squeeze(0)
            
            caption = str(row['caption_vi'])
            enc = self.tokenizer(
                caption,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )
            input_ids = enc.input_ids.squeeze(0)
            attention_mask = enc.attention_mask.squeeze(0)
            
            return {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }


class PhoCLIPModel(nn.Module):
    """PhoCLIP: PhoBERT text encoder + CLIP ViT vision encoder + projection heads."""
    def __init__(
        self,
        text_encoder_name: str = "vinai/phobert-base",
        vision_encoder_name: str = "openai/clip-vit-base-patch32",
        embed_dim: int = 768,
        temperature_init: float = 0.07,
    ):
        super().__init__()
        # Text encoder (PhoBERT)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Vision encoder (CLIP ViT)
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_encoder_name)
        self.vision_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / temperature_init)))

    def forward(self, input_ids, attention_mask, pixel_values):
        # Text side
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        text_cls = text_outputs.last_hidden_state[:, 0, :]
        text_proj = self.text_proj(text_cls)
        text_emb = F.normalize(text_proj, p=2, dim=-1)

        # Vision side
        vis_outputs = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
        vis_cls = vis_outputs.pooler_output
        vis_proj = self.vision_proj(vis_cls)
        vis_emb = F.normalize(vis_proj, p=2, dim=-1)

        logit_scale = self.logit_scale.exp()
        return text_emb, vis_emb, logit_scale


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Dataset & DataLoader
    dataset = ViCaptionDataset(args.csv, args.images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Model
    model = PhoCLIPModel(
        text_encoder_name=args.text_encoder,
        vision_encoder_name=args.vision_encoder
    ).to(device)

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for batch in loader:
            pixel = batch['pixel_values'].to(device)
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            text_emb, vis_emb, logit_scale = model(ids, mask, pixel)
            logits = logit_scale * (vis_emb @ text_emb.T)
            labels = torch.arange(logits.size(0), device=device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)
            loss = (loss_i2t + loss_t2i) / 2
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")

    # Save checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    model.text_encoder.save_pretrained(os.path.join(args.output_dir, 'text_encoder'))
    model.vision_encoder.save_pretrained(os.path.join(args.output_dir, 'vision_encoder'))
    torch.save(model.text_proj.state_dict(), os.path.join(args.output_dir, 'text_proj.pt'))
    torch.save(model.vision_proj.state_dict(), os.path.join(args.output_dir, 'vision_proj.pt'))
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder, use_fast=False)
    tokenizer.save_pretrained(os.path.join(args.output_dir, 'tokenizer'))
    print(f"Model and tokenizer saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PhoCLIP on Vietnamese image-caption data')
    parser.add_argument('--csv',         type=str, required=True, help='Path to merged_vi_captions.csv')
    parser.add_argument('--images',      type=str, required=True, help='Directory of prefix-named images')
    parser.add_argument('--output_dir',  type=str, default='phoclip_checkpoint', help='Output checkpoint directory')
    parser.add_argument('--text_encoder',type=str, default='vinai/phobert-base', help='PhoBERT model name')
    parser.add_argument('--vision_encoder',type=str, default='openai/clip-vit-base-patch32', help='CLIPVision model name')
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--epochs',      type=int, default=10)
    parser.add_argument('--lr',          type=float, default=5e-6)
    parser.add_argument('--warmup_steps',type=int, default=1000)
    args = parser.parse_args()
    train(args)