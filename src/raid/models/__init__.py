from .cnn import SimpleSRCNN
from .unet import UNet



def build_model(cfg):
    m = cfg.model

    ## input channels follow the data config, so adding a static field can't silently mismatch
    in_channels = len(cfg.data.input_vars) + len(cfg.data.static_inputs)
    if m.name == "cnn":
        return SimpleSRCNN(in_channels, getattr(m, "hidden_dim", 64), getattr(m, "n_blocks", 4))
    if m.name == "unet":
        return UNet(in_channels, getattr(m, "base_channels", 32), getattr(m, "depth", 3))

    raise ValueError(f"Unknown model: {m.name}")