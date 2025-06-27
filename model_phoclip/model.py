import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, AutoModel

class PhoCLIPModel(nn.Module):
    """
    CLIP model for Vietnamese:
      - Text encoder: PhoBERT-base (hidden_size=768)
      - Vision encoder: CLIP ViT-B/32
      - Two projection heads (768->768->768 + L2-normalize)
      - Learnable temperature logit_scale
    """
    def __init__(
        self,
        text_encoder_name: str = "vinai/phobert-base",
        vision_encoder_name: str = "openai/clip-vit-base-patch32",
        embed_dim: int = 768,
        temperature_init: float = 0.07,
    ):
        super().__init__()
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Vision encoder
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_encoder_name)
        self.vision_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / temperature_init)))

    def forward(self, input_ids, attention_mask, pixel_values):
        # Text side
        t_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = t_out.last_hidden_state[:, 0, :]               # [B,768]
        t_proj = self.text_proj(cls)                        # [B,768]
        text_emb = F.normalize(t_proj, p=2, dim=-1)         # [B,768]

        # Vision side
        v_out = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
        v_cls = v_out.pooler_output                         # [B,768]
        v_proj = self.vision_proj(v_cls)                    # [B,768]
        img_emb = F.normalize(v_proj, p=2, dim=-1)          # [B,768]

        # Scale
        logit_scale = self.logit_scale.exp()
        return text_emb, img_emb, logit_scale