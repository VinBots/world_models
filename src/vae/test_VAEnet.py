import unittest

from model import ConvVAE
from config import conv_params, conv_layer
import dacite
import torch

class VAENet (unittest.TestCase):

    def setUp(self) -> None:
        
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
            "tr_epochs" : 10
        }
        converters = {}
        config = dacite.from_dict(
            data_class = conv_params, 
            data = raw_data,
            config = dacite.Config(type_hooks = converters))
        self.vae = ConvVAE(conv_params=config)

        #return super().setUp()

    def test_net_dim(self):

        x1 = torch.rand(2, 3, 64, 64)
        x2, mu, log_var = self.vae.forward(x1)
        self.assertEqual((2, 3, 64, 64), x2.shape)

    def test_outpout_format(self):

        x1 = torch.rand(2, 3, 64, 64)
        x2, mu, log_var = self.vae.forward(x1)
        self.assertIsInstance(x2, type(x1))

if __name__ == "__main__":
    unittest.main()