'''
Our latent vector is sampled from a factored Gaussian distribution 
​​As the environment may give us observations as high dimensional pixel images, we first resize each image to 64x64 pixels and use this resized image as V’s observation. Each pixel is stored as three floating point values between 0 and 1 to represent each of the RGB channels. The ConvVAE takes in this 64x64x3 input tensor and passes it through 4 convolutional layers to encode it into low dimension vectors 
​​is passed through 4 of deconvolution layers used to decode and reconstruct the image.

Each convolution and deconvolution layer uses a stride of 2. The layers are indicated in the diagram in Italics as Activation-type Output Channels x Filter Size. All convolutional and deconvolutional layers use relu activations except for the output layer as we need the output to be between 0 and 1. We trained the model for 1 epoch over the data collected from a random policy, using 
​​ distance between the input image and the reconstruction to quantify the reconstruction loss we optimize for, in addition to KL loss.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# define a Conv VAE
class VAE(nn.Module):
    def __init__(self, conv_params, device = torch.device('cuda')):
        super(VAE, self).__init__()
        self.conv_params = conv_params
        #self.device = device
 
        # encoder
        self.encoder_conv = []
        #device = torch.device('cuda')
        in_c = self.conv_params.image_channels
        for i in range(self.conv_params.nb_layers):
            if i > 0:
                in_c = self.conv_params.conv_layers[i-1].out_ch

            conv_layer = nn.Conv2d(    
                in_channels = in_c, 
                out_channels = self.conv_params.conv_layers[i].out_ch, 
                kernel_size = self.conv_params.conv_layers[i].filter_size, 
                stride = self.conv_params.conv_layers[i].stride, 
                padding = self.conv_params.conv_layers[i].padding
                ).to(device)

            self.encoder_conv.append(conv_layer)

        # fully connected layers for learning representations
        self.fc1 = nn.Linear(self.conv_params.fc1_in, self.conv_params.fc1_out)
        self.fc_mu = nn.Linear(self.conv_params.fc1_out, self.conv_params.latent_dim)
        self.fc_log_var = nn.Linear(self.conv_params.fc1_out, self.conv_params.latent_dim)
        self.fc2 = nn.Linear(self.conv_params.latent_dim, self.conv_params.fc2_out)
        # decoder 
        
        self.decoder_conv = []
        in_c = self.conv_params.fc2_out
        for i in range(self.conv_params.nb_layers):
            if i > 0:
                in_c = self.conv_params.deconv_layers[i-1].out_ch

            deconv_layer = nn.ConvTranspose2d(    
                in_channels = in_c, 
                out_channels = self.conv_params.deconv_layers[i].out_ch, 
                kernel_size = self.conv_params.deconv_layers[i].filter_size, 
                stride = self.conv_params.deconv_layers[i].stride, 
                padding = self.conv_params.deconv_layers[i].padding
                ).to(device)

            self.decoder_conv.append(deconv_layer)


    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x):
        # encoding

        for i in range(self.conv_params.nb_layers):
            x = F.relu(self.encoder_conv[i](x))
        
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        z = z.view(-1, self.conv_params.fc2_out, 1, 1)
        x = z
 
        # decoding
        for i in range(self.conv_params.nb_layers):
            if self.conv_params.deconv_layers[i].activation == "relu":
                x = F.relu(self.decoder_conv[i](x))

            elif self.conv_params.deconv_layers[i].activation == "sigmoid":
                x = torch.sigmoid (self.decoder_conv[i](x))
        
        return x, mu, log_var
    
    def save_weights(self):
        """
        Saves the network checkpoints
        """
        full_path = os.path.join(self.conv_params.ckp_folder,self.conv_params.ckp_path)
        torch.save(self.state_dict(), full_path)

    def load_weights(self, path):

        """
        Loads weights of the network saved in path
        """
        full_path = self.conv_params.ckp_folder + \
            "/" + path
        self.load_state_dict(torch.load(full_path))