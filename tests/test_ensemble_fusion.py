import pytest
import torch
import torch.nn.functional as F

from src.models.ensemble_fusion import (
    CalibratedRankAveraging,
    BottleneckFeatureFusion,
    ResidualGatedFusion,
    DualAxisLabelSpecificABMIL,
    PathologyGroupedSoftMoE,
    PathologySlotAttention,
    SparseTopKPathologyMoE,
)

@pytest.fixture
def dummy_batch_data():
    B = 4
    num_classes = 12
    # 3 complementary backbones: ConvNeXt-Tiny (768), ConvNeXt-Small (768), DINOv2 (384)
    dims = [768, 768, 384]
    D = 16  # 16 slices per volume

    # Pre-classifier study-level embeddings: (B, 12, dim)
    model_study_feats = [torch.randn(B, num_classes, d) for d in dims]

    # Slice-level embeddings: (B, D, dim)
    slice_feats_dict = {
        'convnext_tiny': torch.randn(B, D, 768),
        'convnext_small': torch.randn(B, D, 768),
        'dinov2': torch.randn(B, D, 384),
    }

    # Model standalone logits: (B, 12)
    standalone_logits = [torch.randn(B, num_classes) for _ in dims]
    standalone_probs = [torch.sigmoid(l) for l in standalone_logits]

    return {
        'B': B,
        'num_classes': num_classes,
        'dims': dims,
        'D': D,
        'study_feats': model_study_feats,
        'slice_dict': slice_feats_dict,
        'logits': standalone_logits,
        'probs': standalone_probs,
    }


def test_calibrated_rank_averaging(dummy_batch_data):
    probs = dummy_batch_data['probs']
    num_models = len(probs)
    num_classes = dummy_batch_data['num_classes']

    module = CalibratedRankAveraging(num_models=num_models, num_classes=num_classes)
    
    # 1. Standard blended probabilities
    blended = module(probs, use_ranks=False)
    assert blended.shape == (dummy_batch_data['B'], num_classes)
    assert (blended >= 0.0).all() and (blended <= 1.0).all()

    # 2. Rank-transformed probabilities
    ranked_blended = module(probs, use_ranks=True)
    assert ranked_blended.shape == (dummy_batch_data['B'], num_classes)
    assert (ranked_blended >= 0.0).all() and (ranked_blended <= 1.0).all()

    # 3. Gradient check
    loss = blended.sum()
    loss.backward()
    assert module.weights.grad is not None


def test_bottleneck_feature_fusion(dummy_batch_data):
    study_feats = dummy_batch_data['study_feats']
    dims = dummy_batch_data['dims']
    num_classes = dummy_batch_data['num_classes']
    logits = dummy_batch_data['logits']

    fusion = BottleneckFeatureFusion(in_dims=dims, num_classes=num_classes, bottleneck_dim=128)
    
    # With standalone logits residual bypass
    out_logits = fusion(study_feats, model_standalone_logits=logits)
    assert out_logits.shape == (dummy_batch_data['B'], num_classes)

    loss = F.binary_cross_entropy_with_logits(out_logits, torch.zeros_like(out_logits))
    loss.backward()
    assert fusion.skip_weights.grad is not None


def test_residual_gated_fusion(dummy_batch_data):
    study_feats = dummy_batch_data['study_feats']
    dims = dummy_batch_data['dims']
    num_classes = dummy_batch_data['num_classes']
    B = dummy_batch_data['B']

    gated_fusion = ResidualGatedFusion(in_dims=dims, num_classes=num_classes, hidden_dim=64)
    logits, gates = gated_fusion(study_feats)

    assert logits.shape == (B, num_classes)
    assert gates.shape == (B, num_classes, len(dims))

    # Gating weights must sum to 1 across models per class
    gate_sums = gates.sum(dim=-1)
    assert torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-5)

    loss = logits.sum()
    loss.backward()
    assert gated_fusion.gate_nets[0][0].weight.grad is not None


def test_dual_axis_abmil(dummy_batch_data):
    slice_dict = dummy_batch_data['slice_dict']
    B = dummy_batch_data['B']
    D = dummy_batch_data['D']
    num_classes = dummy_batch_data['num_classes']

    abmil = DualAxisLabelSpecificABMIL(
        in_features_dict={'convnext_tiny': 768, 'convnext_small': 768, 'dinov2': 384},
        num_classes=num_classes,
        mil_hidden=64,
        cross_hidden=32,
    )

    mask = torch.ones(B, D, dtype=torch.bool)
    mask[:, -2:] = False  # Mask last 2 slices

    logits, beta, slice_attns = abmil(slice_dict, mask=mask)

    assert logits.shape == (B, num_classes)
    assert beta.shape == (B, num_classes, 3)

    # Beta across models must sum to 1
    beta_sums = beta.sum(dim=-1)
    assert torch.allclose(beta_sums, torch.ones_like(beta_sums), atol=1e-5)

    # Masked slices must have 0 attention in slice_attns
    for name, attn in slice_attns.items():
        assert attn.shape == (B, D, num_classes)
        assert torch.allclose(attn[:, -1, :], torch.zeros_like(attn[:, -1, :]), atol=1e-3)

    loss = logits.sum()
    loss.backward()
    assert abmil.cross_query.grad is not None


def test_pathology_grouped_soft_moe(dummy_batch_data):
    study_feats = dummy_batch_data['study_feats']
    dims = dummy_batch_data['dims']
    num_classes = dummy_batch_data['num_classes']
    B = dummy_batch_data['B']

    moe = PathologyGroupedSoftMoE(in_dims=dims, num_classes=num_classes, expert_dim=64)
    logits, routing_weights = moe(study_feats)

    assert logits.shape == (B, num_classes)
    assert set(routing_weights.keys()) == {'focal_tears', 'osteoarthritis', 'fluid_bone'}

    for grp_name, weights in routing_weights.items():
        assert weights.shape == (B, len(dims))
        w_sum = weights.sum(dim=-1)
        assert torch.allclose(w_sum, torch.ones_like(w_sum), atol=1e-5)

    loss = logits.sum()
    loss.backward()
    assert moe.expert_projections[0][0].weight.grad is not None


def test_pathology_slot_attention(dummy_batch_data):
    B = dummy_batch_data['B']
    num_classes = dummy_batch_data['num_classes']
    # Tokens from each model: e.g. 16 slices per model
    tokens_list = [
        torch.randn(B, 16, 768),
        torch.randn(B, 16, 768),
        torch.randn(B, 16, 384),
    ]

    slot_attn = PathologySlotAttention(
        in_features_list=[768, 768, 384],
        slot_dim=64,
        num_slots=num_classes,
        iters=2,
    )

    logits, attn = slot_attn(tokens_list)
    assert logits.shape == (B, num_classes)
    # Total tokens = 16 + 16 + 16 = 48
    assert attn.shape == (B, num_classes, 48)

    loss = logits.sum()
    loss.backward()
    assert slot_attn.slots_mu.grad is not None


def test_sparse_topk_pathology_moe(dummy_batch_data):
    study_feats = dummy_batch_data['study_feats']
    dims = dummy_batch_data['dims']
    num_classes = dummy_batch_data['num_classes']
    B = dummy_batch_data['B']

    moe = SparseTopKPathologyMoE(in_dims=dims, num_classes=num_classes, expert_dim=64, k=2)
    logits, routing_weights = moe(study_feats)

    assert logits.shape == (B, num_classes)
    for grp_name, weights in routing_weights.items():
        assert weights.shape == (B, len(dims))
        # Top-2 must have exactly 1 zero weight per sample
        num_zeros = (weights == 0.0).sum(dim=-1)
        assert (num_zeros == 1).all()
        w_sum = weights.sum(dim=-1)
        assert torch.allclose(w_sum, torch.ones_like(w_sum), atol=1e-5)

    loss = logits.sum()
    loss.backward()
    assert moe.expert_projections[0][0].weight.grad is not None
