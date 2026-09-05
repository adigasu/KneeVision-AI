import torch
import pytest
from src.models.report_encoder import ClinicalReportEncoder
from src.models.weak_multimodal import DualStreamKneeModel
from src.training.contrastive_loss import CLIPInfoNCELoss, MultimodalJointLoss


def test_clinical_report_encoder():
    model = ClinicalReportEncoder(embed_dim=128)
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    embeds = model(input_ids, attention_mask)
    assert embeds.shape == (batch_size, 128)
    # Check L2 normalization
    norms = torch.norm(embeds, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_dual_stream_knee_model_and_losses():
    model = DualStreamKneeModel(
        backbone_name="convnext_tiny",
        embed_dim=128,
        num_classes=12,
        mil_hidden_dim=64,
    )
    B, D, C, H, W = 2, 8, 3, 128, 128
    images = torch.randn(B, D, C, H, W)
    input_ids = torch.randint(0, 1000, (B, 32))
    attention_mask = torch.ones(B, 32, dtype=torch.long)
    pseudo_targets = torch.randint(0, 2, (B, 12)).float()

    outputs = model(images, input_ids, attention_mask)
    assert outputs["image_embeds"].shape == (B, 128)
    assert outputs["text_embeds"].shape == (B, 128)
    assert outputs["pseudo_logits"].shape == (B, 12)

    # Test joint loss
    criterion = MultimodalJointLoss(lambda_pseudo=1.0)
    losses = criterion(outputs, pseudo_targets)
    assert "total_loss" in losses
    assert losses["total_loss"].item() > 0

    # Test backward pass
    losses["total_loss"].backward()

    # Test backbone weights extraction
    vision_state = model.get_vision_backbone_state_dict()
    assert len(vision_state) > 0
    assert any(k.startswith("backbone.") for k in vision_state.keys())
    assert any(k.startswith("mil_pool.") for k in vision_state.keys())
