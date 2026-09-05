"""
RSNA Knee Abnormality Detection - Multimodal Contrastive & Multi-Task Loss Functions.
Implements InfoNCE symmetric image-text contrastive alignment and joint pseudo-label supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.losses import AsymmetricLoss


class CLIPInfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE Contrastive Loss between normalized image and text embeddings.
    """

    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_embeds: (Batch, Embed_Dim), L2-normalized
            text_embeds: (Batch, Embed_Dim), L2-normalized
            logit_scale: Scalar tensor = exp(logit_scale)

        Returns:
            Scalar contrastive loss.
        """
        # Cosine similarity matrix scaled by temperature
        logits_per_image = logit_scale * torch.matmul(image_embeds, text_embeds.t())  # (B, B)
        logits_per_text = logits_per_image.t()                                        # (B, B)

        # Ground truth diagonal targets: image i matches text i
        batch_size = image_embeds.shape[0]
        labels = torch.arange(batch_size, device=image_embeds.device)

        loss_i = self.cross_entropy(logits_per_image, labels)
        loss_t = self.cross_entropy(logits_per_text, labels)

        return (loss_i + loss_t) / 2.0


class MultimodalJointLoss(nn.Module):
    """
    Joint Loss for Multimodal Alignment:
    Total = Loss_CLIP(Images, Reports) + lambda_pseudo * Loss_ASL(Pseudo_Logits, Pseudo_Targets)
    """

    def __init__(
        self,
        lambda_pseudo: float = 1.0,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
    ):
        super().__init__()
        self.lambda_pseudo = lambda_pseudo
        self.clip_loss = CLIPInfoNCELoss()
        self.asl_loss = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)

    def forward(
        self,
        outputs: dict,
        pseudo_targets: torch.Tensor,
    ) -> dict:
        image_embeds = outputs["image_embeds"]
        text_embeds = outputs["text_embeds"]
        logit_scale = outputs["logit_scale"]
        pseudo_logits = outputs["pseudo_logits"]

        clip_loss_val = self.clip_loss(image_embeds, text_embeds, logit_scale)
        asl_loss_val = self.asl_loss(pseudo_logits, pseudo_targets)

        total_loss = clip_loss_val + self.lambda_pseudo * asl_loss_val

        return {
            "total_loss": total_loss,
            "clip_loss": clip_loss_val,
            "asl_loss": asl_loss_val,
        }
