import torch

def test_net_dim(vae_net):

    x1 = torch.rand(2, 3, 64, 64)
    x2, mu, log_var = vae_net.forward(x1)
    assert (2, 3, 64, 64) == x2.shape

def test_output_format(vae_net):
    
    x1 = torch.rand(2, 3, 64, 64)
    x2, mu, log_var = vae_net.forward(x1)
    assert isinstance(x2, type(x1)) == True