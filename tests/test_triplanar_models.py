import torch
import pytest
from src.models.cross_plane_fusion import TriPlanarKneeModel
from src.training.losses import AsymmetricLoss


def test_triplanar_model_forward_backward():
    model = TriPlanarKneeModel(
        backbone_name="convnext_tiny",
        pretrained=True,
        num_classes=12,
        mil_hidden_dim=64,
        num_heads=4,
    )
    B, D, C, H, W = 2, 8, 3, 128, 128
    sag = torch.randn(B, D, C, H, W)
    cor = torch.randn(B, D, C, H, W)
    ax = torch.randn(B, D, C, H, W)

    outputs = model(sag, cor, ax)
    assert outputs["logits"].shape == (B, 12)
    assert outputs["fused_study"].shape == (B, model.num_features)
    assert outputs["inter_plane_attn"].shape == (B, 3, 3)

    # Test Loss with Tri-State Targets (including NaNs)
    targets = torch.tensor([
        [1.0, 0.0, float("nan"), 1.0, 0.0, float("nan"), 0.0, 1.0, float("nan"), 0.0, 1.0, 0.0],
        [0.0, float("nan"), 1.0, 0.0, float("nan"), 1.0, 0.0, 0.0, 1.0, float("nan"), 0.0, 1.0],
    ])
    criterion = AsymmetricLoss()
    loss = criterion(outputs["logits"], targets)
    assert not torch.isnan(loss)
    assert loss.item() > 0

    loss.backward()
