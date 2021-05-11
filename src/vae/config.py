from dataclasses import dataclass
from typing import Tuple

@dataclass
class conv_layer:
    activation: str
    out_ch : int
    filter_size: int
    stride: int
    padding: int

@dataclass
class conv_params:
    image_channels: int
    nb_layers: int
    conv_layers: Tuple[conv_layer, conv_layer, conv_layer, conv_layer]
    fc1_in: int
    fc1_out: int
    latent_dim : int
    fc2_out: int
    deconv_layers: Tuple[conv_layer, conv_layer, conv_layer, conv_layer]
    tr_epochs : int

@dataclass
class Configuration:
    version : str
    vae_params: conv_params