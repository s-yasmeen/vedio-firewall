import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch

from tapf.fer_v3 import FERV3Config, TAPFMinFERV3, multitask_loss


def test_fer_v3_forward_shapes_and_attention_normalization():
    model = TAPFMinFERV3(FERV3Config(
        n_emotions=6, n_identities=8, temporal_hidden=64,
        temporal_layers=1, dropout=0.0, pretrained=False,
    ))
    model.eval()
    x = torch.randn(2, 4, 3, 64, 64)
    with torch.no_grad():
        out = model(x, identity_grl_strength=0.5)
    assert out["emotion_logits"].shape == (2, 6)
    assert out["identity_logits"].shape == (2, 8)
    assert out["task_latent"].shape == (2, 64)
    assert out["temporal_attention"].shape == (2, 4)
    assert torch.allclose(out["temporal_attention"].sum(1), torch.ones(2), atol=1e-5)


def test_task_posterior_is_normalized():
    model = TAPFMinFERV3(FERV3Config(
        n_emotions=6, n_identities=8, temporal_hidden=32,
        temporal_layers=1, dropout=0.0, pretrained=False,
    ))
    model.eval()
    x = torch.randn(1, 3, 3, 64, 64)
    p = model.task_posterior(x)
    assert p.shape == (1, 6)
    assert torch.all(p >= 0)
    assert torch.allclose(p.sum(-1), torch.ones(1), atol=1e-5)


def test_multitask_loss_is_finite():
    model = TAPFMinFERV3(FERV3Config(
        n_emotions=6, n_identities=5, temporal_hidden=32,
        temporal_layers=1, dropout=0.0, pretrained=False,
    ))
    x = torch.randn(2, 3, 3, 64, 64)
    out = model(x, identity_grl_strength=0.7)
    losses = multitask_loss(out, torch.tensor([0, 1]), torch.tensor([2, 3]))
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["emotion_loss"])
    assert torch.isfinite(losses["identity_adversary_loss"])


def test_latent_is_not_the_release_api():
    model = TAPFMinFERV3(FERV3Config(
        n_emotions=6, n_identities=4, temporal_hidden=32,
        temporal_layers=1, dropout=0.0, pretrained=False,
    ))
    assert hasattr(model, "task_posterior")
    # The model intentionally exposes no method named release_latent.
    assert not hasattr(model, "release_latent")
