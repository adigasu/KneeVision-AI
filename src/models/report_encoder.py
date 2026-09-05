"""
RSNA Knee Abnormality Detection - Spanish Clinical Radiology Report Text Encoder.
Encodes raw Spanish reports into dense text embeddings using pretrained Spanish BERT (BETO)
with a linear multimodal projection head.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class ClinicalReportEncoder(nn.Module):
    """
    Text encoder using pretrained Spanish BERT (BETO / Clinical-RoBERTa).
    Extracts [CLS] token embeddings and projects them into shared multimodal space.
    """

    def __init__(
        self,
        model_name: str = "dccuchile/bert-base-spanish-wwm-cased",
        embed_dim: int = 256,
        freeze_backbone: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.model_name = model_name
        self.transformer = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for p in self.transformer.parameters():
                p.requires_grad = False

        hidden_size = self.transformer.config.hidden_size
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embed_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (Batch, Seq_Len)
            attention_mask: (Batch, Seq_Len)

        Returns:
            text_embeds: L2-normalized text embeddings of shape (Batch, Embed_Dim)
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] representation (first token)
        cls_feat = outputs.last_hidden_state[:, 0, :]  # (B, Hidden_Size)
        proj_embed = self.projection(cls_feat)          # (B, Embed_Dim)
        # L2-normalize for cosine similarity / CLIP contrastive loss
        text_embeds = nn.functional.normalize(proj_embed, p=2, dim=-1)
        return text_embeds
