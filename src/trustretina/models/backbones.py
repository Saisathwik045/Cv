"""Five fixed torchvision backbones, eight independent logits and a common dropout head."""

import hashlib
import re
from pathlib import Path
import torch
from torch import nn
from torchvision import models
from trustretina.config import BACKBONES


class RetinalClassifier(nn.Module):
    def __init__(self, backbone, features, dropout, name):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, 8))
        self.backbone_name = name

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Require BCHW RGB tensors")
        return self.head(self.backbone(images))


def create_backbone(
    name, *, dropout=0.2, weights_path=None, expected_weights_sha256=None
):
    if name not in BACKBONES or not 0 <= dropout < 1:
        raise ValueError("Unknown backbone or invalid dropout")
    # Never downloads weights. A separately obtained ImageNet state must be hash verified.
    base = getattr(models, name)(weights=None)
    provenance = {
        "backbone": name,
        "initialization": "random_unfitted",
        "weights_sha256": None,
    }
    if weights_path is not None or expected_weights_sha256 is not None:
        if weights_path is None or expected_weights_sha256 is None:
            raise ValueError("Require weight path and trusted checksum together")
        digest = hashlib.sha256(Path(weights_path).read_bytes()).hexdigest()
        if digest != expected_weights_sha256:
            raise ValueError("Pretrained weight checksum mismatch")
        state=torch.load(weights_path,map_location="cpu",weights_only=True)
        if name=='densenet121':
            pattern=re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.([12]\.(?:weight|bias|running_mean|running_var))$')
            state={pattern.sub(r'\1\2',k):v for k,v in state.items()}
        base.load_state_dict(state,strict=True)
        provenance.update(
            initialization="verified_local_imagenet_state", weights_sha256=digest
        )
    if name.startswith("efficientnet"):
        features = base.classifier[-1].in_features
        base.classifier = nn.Identity()
    elif name == "resnet50":
        features = base.fc.in_features
        base.fc = nn.Identity()
    elif name == "densenet121":
        features = base.classifier.in_features
        base.classifier = nn.Identity()
    else:
        features = base.classifier[-1].in_features
        base.classifier[-1] = nn.Identity()
    model = RetinalClassifier(base, features, dropout, name)
    model.provenance = provenance
    return model
