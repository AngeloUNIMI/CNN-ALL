from __future__ import annotations

from torch import nn

from cnn_all.cnn.backbones import BACKBONES, build_backbone, get_submodule, replace_classifier


def test_all_backbone_specs_construct_without_weights() -> None:
    # Construction validates torchvision names and all feature/Grad-CAM paths.
    for name, expected in BACKBONES.items():
        model, spec = build_backbone(name, pretrained=False)
        assert get_submodule(model, spec.feature_layer) is not None
        assert get_submodule(model, spec.gradcam_layer) is not None
        head = replace_classifier(model, spec, num_classes=5)
        assert isinstance(head, nn.Linear)
        assert head.out_features == 5
