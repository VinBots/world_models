from src.vae.model import VAE
from src.vae.config import conv_params, conv_layer
import dacite
import torch
import pytest
from src.vae.buffer import Buffer

@pytest.fixture
def device():
    return torch.device('cuda')
    #return torch.device('cpu')

@pytest.fixture
def test_config():
    
    conv1 = conv_layer("relu", 32, 4, 2, 0)
    conv2 = conv_layer("relu", 64, 4, 2, 0)
    conv3 = conv_layer("relu", 128, 4, 2, 0)
    conv4 = conv_layer("relu", 256, 4, 2, 0)

    deconv1 = conv_layer("relu", 128, 5, 2, 0)
    deconv2 = conv_layer("relu", 64, 5, 2, 0)
    deconv3 = conv_layer("relu", 32, 6, 2, 0)
    deconv4 = conv_layer("sigmoid", 3, 6, 2, 0)

    raw_data = {
        "image_channels": 3,
        "nb_layers" : 4,
        "conv_layers" : (conv1, conv2, conv3, conv4),
        "fc1_in": 256,
        "fc1_out": 128,
        "latent_dim" : 32,
        "fc2_out": 1024,
        "deconv_layers" : (deconv1, deconv2, deconv3, deconv4),
        "tr_epochs" : 50000,
        "batch_size": 12,
        "resize": (64,64),
        "ckp_folder": "C:/Users/vince/Documents/AI/world_models/ckp",
        "ckp_path" : "last_ckp.pth"
    }
    converters = {}
    config = dacite.from_dict(
        data_class = conv_params, 
        data = raw_data,
        config = dacite.Config(type_hooks = converters))

    return config


@pytest.fixture
def vae_net(test_config, device):
    return VAE(conv_params=test_config, device = device).to(device)


@pytest.fixture
def new_buffer():
    return Buffer ("test", 96)