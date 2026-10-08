## python3 -m pytest tests/test_models.py
## python3 -m pytest tests/test_models.py::test_forward_shape

from pathlib import Path

import pytest
import torch
import yaml

#from raid.trainer import dict_to_namespace
from raid.models.cnn import SimpleSRCNN


def test_forward_shape():
    model = SimpleSRCNN(in_channels=3, hidden_dim=16, n_blocks=2)
    x = torch.randn(2, 3, 64, 64)
    assert model(x).shape == (2, 1, 64, 64)


def test_works_on_non_square_full_grid_size():
    # fully convolutional: any H, W must work (full grid is 961 x 1441)
    model = SimpleSRCNN(in_channels=3, hidden_dim=8, n_blocks=1)
    x = torch.randn(1, 3, 97, 145)
    assert model(x).shape == (1, 1, 97, 145)


def test_starts_as_identity_on_channel0():
    # zero-initialized output layer -> output equals the upsampled input
    model = SimpleSRCNN(in_channels=3, hidden_dim=16, n_blocks=2)
    x = torch.randn(2, 3, 32, 32)
    assert torch.allclose(model(x), x[:, :1])


def test_gradients_flow():
    model = SimpleSRCNN(in_channels=3, hidden_dim=16, n_blocks=2)
    x = torch.randn(1, 3, 32, 32)
    # output_conv starts at zero, so take one step first to make gradients reach the body
    out = model(x)
    out.sum().backward()
    assert model.output_conv.weight.grad is not None
    assert model.output_conv.weight.grad.abs().sum() > 0


"""
ROOT = Path(__file__).resolve().parents[1]


## names accepted by build_model via cfg.model.name; edit to match the implemented models
MODEL_NAMES = ["cnn", "unet"]
PATCH = 256


def make_cfg(model_name):
    raw = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    raw["model"] = {"name": model_name}
    return raw


def channels(raw):
    ## x = input variables + static fields, y = target variables (see DownscalingDataset)
    d = raw["data"]
    return len(d["input_vars"]) + len(d["static_inputs"]), len(d["target_vars"])


def test_base_config_gives_3_input_and_1_output_channel():
    assert channels(make_cfg(MODEL_NAMES[0])) == (3, 1)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_forward_maps_B_3_256_256_to_B_1_256_256(model_name, batch_size):
    ## imported here so a missing model fails only these tests instead of the whole run
    from raid.models import build_model

    raw = make_cfg(model_name)
    c_in, c_out = channels(raw)
    model = build_model(dict_to_namespace(raw)).eval()
    x = torch.randn(batch_size, c_in, PATCH, PATCH)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (batch_size, c_out, PATCH, PATCH)
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()
"""

