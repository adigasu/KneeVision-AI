"""
KneeVision-AI Foundation Model Feature Extractors.
Unified interface for extracting slice embeddings from vision and medical foundation models:
- DINOv2 / DINOv3 (ViT-Small, ViT-Base, ViT-Large)
- BioMedCLIP (Microsoft PubMedBERT + ViT-B/16)
- SigLIP / MedSigLIP (Google ViT-B/16, ViT-SO400M)
- RadImageNet (Musculoskeletal / Radiology pretrained ResNet-50, 2048-dim)
- ConvNeXt / Swin Baselines
"""

from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision.models as models
import timm

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

try:
    import open_clip
except ImportError:
    open_clip = None


class BaseFoundationExtractor(nn.Module):
    def __init__(self, model_name: str, embed_dim: int, img_size: int = 224):
        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.img_size = img_size

    @torch.no_grad()
    def extract_slices(self, slices: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RadImageNetExtractor(BaseFoundationExtractor):
    def __init__(self, repo_id: str = "convergedmachine/RadImagenet", filename: str = "resnet50.pth", img_size: int = 224):
        super().__init__(model_name="radimagenet", embed_dim=2048, img_size=img_size)
        if hf_hub_download is None:
            raise ImportError("huggingface_hub is required to download RadImageNet weights.")
        weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
        model = models.resnet50()
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        clean_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("module.", "").replace("backbone.", "")
            clean_dict[new_k] = v
        model.load_state_dict(clean_dict, strict=False)
        model.fc = nn.Identity()
        self.model = model.eval()

    @torch.no_grad()
    def extract_slices(self, slices: torch.Tensor) -> torch.Tensor:
        orig_shape = slices.shape
        if len(orig_shape) == 5:
            B, D, C, H, W = orig_shape
            flat = slices.view(B * D, C, H, W)
            feats = self.model(flat)
            return feats.view(B, D, self.embed_dim)
        return self.model(slices)


class DINOExtractor(BaseFoundationExtractor):
    def __init__(self, variant: str = "vit_base_patch16_dinov3", img_size: int = 288):
        embed_dims = {
            "vit_small_patch14_dinov2.lvd142m": 384,
            "vit_base_patch14_dinov2.lvd142m": 768,
            "vit_large_patch14_dinov2.lvd142m": 1024,
            "vit_giant_patch14_dinov2.lvd142m": 1536,
            "vit_small_patch16_dinov3": 384,
            "vit_base_patch16_dinov3": 768,
            "vit_large_patch16_dinov3": 1024,
        }
        dim = embed_dims.get(variant, 768)
        super().__init__(model_name=variant, embed_dim=dim, img_size=img_size)
        self.model = timm.create_model(
            variant,
            pretrained=True,
            num_classes=0,
            img_size=img_size,
            dynamic_img_size=True,
            in_chans=3,
        ).eval()

    @torch.no_grad()
    def extract_slices(self, slices: torch.Tensor) -> torch.Tensor:
        orig_shape = slices.shape
        if len(orig_shape) == 5:
            B, D, C, H, W = orig_shape
            flat = slices.view(B * D, C, H, W)
            feats = self.model(flat)
            return feats.view(B, D, self.embed_dim)
        return self.model(slices)


class BioMedCLIPExtractor(BaseFoundationExtractor):
    def __init__(self, model_tag: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", img_size: int = 224):
        super().__init__(model_name="biomedclip", embed_dim=512, img_size=img_size)
        if open_clip is None:
            raise ImportError("open_clip_torch is required for BioMedCLIP.")
        model, _, _ = open_clip.create_model_and_transforms(model_tag)
        self.visual = model.visual.eval()

    @torch.no_grad()
    def extract_slices(self, slices: torch.Tensor) -> torch.Tensor:
        orig_shape = slices.shape
        if len(orig_shape) == 5:
            B, D, C, H, W = orig_shape
            flat = slices.view(B * D, C, H, W)
            feats = self.visual(flat)
            return feats.view(B, D, self.embed_dim)
        return self.visual(slices)


class SigLIPExtractor(BaseFoundationExtractor):
    def __init__(self, variant: str = "vit_base_patch16_siglip_224", img_size: int = 224):
        embed_dims = {
            "vit_base_patch16_siglip_224": 768,
            "vit_base_patch16_siglip_256": 768,
            "vit_large_patch16_siglip_256": 1024,
            "vit_so400m_patch14_siglip_384": 1152,
        }
        dim = embed_dims.get(variant, 768)
        super().__init__(model_name=variant, embed_dim=dim, img_size=img_size)
        self.model = timm.create_model(
            variant,
            pretrained=True,
            num_classes=0,
            img_size=img_size,
            dynamic_img_size=True,
            in_chans=3,
        ).eval()

    @torch.no_grad()
    def extract_slices(self, slices: torch.Tensor) -> torch.Tensor:
        orig_shape = slices.shape
        if len(orig_shape) == 5:
            B, D, C, H, W = orig_shape
            flat = slices.view(B * D, C, H, W)
            feats = self.model(flat)
            return feats.view(B, D, self.embed_dim)
        return self.model(slices)


class GenericTimmExtractor(BaseFoundationExtractor):
    def __init__(self, model_name: str = "convnext_small.in12k_ft_in1k", img_size: int = 288):
        model = timm.create_model(model_name, pretrained=True, num_classes=0, in_chans=3)
        dim = model.num_features
        super().__init__(model_name=model_name, embed_dim=dim, img_size=img_size)
        self.model = model.eval()

    @torch.no_grad()
    def extract_slices(self, slices: torch.Tensor) -> torch.Tensor:
        orig_shape = slices.shape
        if len(orig_shape) == 5:
            B, D, C, H, W = orig_shape
            flat = slices.view(B * D, C, H, W)
            feats = self.model(flat)
            return feats.view(B, D, self.embed_dim)
        return self.model(slices)


def get_foundation_extractor(name: str, device: str = "cuda") -> BaseFoundationExtractor:
    name_lower = name.lower()
    if "radimage" in name_lower:
        extractor = RadImageNetExtractor(img_size=224)
    elif "biomed" in name_lower:
        extractor = BioMedCLIPExtractor(img_size=224)
    elif "dinov3_small" in name_lower or "dinov3_s" in name_lower:
        extractor = DINOExtractor("vit_small_patch16_dinov3", img_size=288)
    elif "dinov3_large" in name_lower or "dinov3_l" in name_lower:
        extractor = DINOExtractor("vit_large_patch16_dinov3", img_size=288)
    elif "dinov3" in name_lower:
        extractor = DINOExtractor("vit_base_patch16_dinov3", img_size=288)
    elif "dinov2_small" in name_lower or "dinov2_s" in name_lower:
        extractor = DINOExtractor("vit_small_patch14_dinov2.lvd142m", img_size=280)
    elif "dinov2_large" in name_lower or "dinov2_l" in name_lower:
        extractor = DINOExtractor("vit_large_patch14_dinov2.lvd142m", img_size=280)
    elif "dinov2" in name_lower:
        extractor = DINOExtractor("vit_base_patch14_dinov2.lvd142m", img_size=280)
    elif "medsiglip" in name_lower or "siglip_so400m" in name_lower:
        extractor = SigLIPExtractor("vit_so400m_patch14_siglip_384", img_size=384)
    elif "siglip" in name_lower:
        extractor = SigLIPExtractor("vit_base_patch16_siglip_224", img_size=224)
    elif "swin" in name_lower:
        extractor = GenericTimmExtractor("swin_base_patch4_window7_224", img_size=224)
    elif "convnext_small" in name_lower:
        extractor = GenericTimmExtractor("convnext_small.in12k_ft_in1k", img_size=320)
    elif "convnext_tiny" in name_lower:
        extractor = GenericTimmExtractor("convnext_tiny.in12k_ft_in1k", img_size=288)
    else:
        extractor = GenericTimmExtractor(name)

    return extractor.to(device).eval()
