"""
Unit tests for KneeMILModel forward/backward passes and AsymmetricLoss.
"""

import torch
from src.models.mil_backbone import KneeMILModel
from src.training.losses import AsymmetricLoss, MaskedBCEWithLogitsLoss


def test_model_forward_backward():
    # Test with dummy small backbone (e.g. resnet10t or convnext_tiny)
    model = KneeMILModel(backbone_name="resnet10t", pretrained=False, num_classes=12)
    model.train()

    batch_size = 2
    max_depth = 16
    images = torch.randn(batch_size, max_depth, 3, 128, 128)
    mask = torch.ones(batch_size, max_depth, dtype=torch.bool)
    mask[0, 12:] = False  # First item has only 12 valid slices

    targets = torch.tensor([
        [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    ])

    outputs = model(images, mask=mask)
    logits = outputs["logits"]
    attn = outputs["attention_weights"]

    assert logits.shape == (batch_size, 12)
    assert attn.shape == (batch_size, max_depth)
    assert torch.allclose(attn[0, 12:], torch.zeros_like(attn[0, 12:]), atol=1e-3)

    # Test Asymmetric Loss backward
    criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0)
    loss = criterion(logits, targets)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    loss.backward()
    # Verify gradients computed
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad

    print("\n✓ test_model_forward_backward PASSED successfully!")


if __name__ == "__main__":
    test_model_forward_backward()
