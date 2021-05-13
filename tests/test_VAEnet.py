import torch
import pytest

@pytest.mark.skip()
def test_net_dim(vae_net):
    "test if the size of input equals the size of the output"
    x1 = torch.rand(2, 3, 64, 64)
    x2, mu, log_var = vae_net.forward(x1)
    assert (2, 3, 64, 64) == x2.shape

@pytest.mark.skip()
def test_output_format(vae_net):
    "test if the output is of the format torch.tensor"
    x1 = torch.rand(2, 3, 64, 64)
    x2, mu, log_var = vae_net.forward(x1)
    assert isinstance(x2, type(x1)) == True

@pytest.mark.skip()
def test_save_model (vae_net):
    vae_net.save_weights()

@pytest.mark.skip()
def test_load_weights(vae_net):
    vae_net.load_weights ("last_ckp.pth")