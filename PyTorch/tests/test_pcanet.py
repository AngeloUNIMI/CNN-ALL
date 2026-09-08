from __future__ import annotations

import torch

from cnn_all.config import PCANetConfig
from cnn_all.pcanet.model import TorchPCANet


def test_pcanet_fit_transform_and_roundtrip(tmp_path) -> None:
    torch.manual_seed(11)
    images = torch.rand(6, 1, 16, 16)
    cfg = PCANetConfig(
        image_size=[16, 16],
        num_stages=1,
        patch_sizes=[3],
        max_filters=[2],
        dynamic_filters=False,
        retained_variance=[0.9],
        hist_block_size=[8, 8],
        block_overlap_ratio=0.0,
        fit_batch_size=3,
        feature_batch_size=3,
        covariance_patch_chunk=1024,
        dtype="float32",
        max_training_images=100,
    )

    def factory():
        yield images[:3]
        yield images[3:]

    model = TorchPCANet(cfg, torch.device("cpu"))
    summary = model.fit(factory)
    features = model.transform(images)

    assert summary.num_filters == [2]
    assert len(summary.eigenvalues[0]) == 9
    # 2 filters -> 4 bins; 16x16 with 8x8 non-overlapping blocks -> 4 blocks.
    assert features.shape == (6, 16)
    assert torch.isfinite(features).all()

    path = tmp_path / "pcanet.pt"
    model.save(path)
    restored = TorchPCANet.load(path, torch.device("cpu"))
    torch.testing.assert_close(restored.transform(images), features)
