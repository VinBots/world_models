import numpy as np
from torch._C import dtype
from torchvision.transforms.functional import resize
import torchvision.transforms as T
import torch
import PIL

from moviepy.editor import ImageSequenceClip

class DotDict(dict):
    def __getattr__(self, name):
        return self[name]

def reshape_obs(observation, tensor_shape):
    return resize(torch.from_numpy(np.moveaxis(np.copy(observation), 2, 0)), tensor_shape)

def preprocess (observation, resize):
    preprocess = T.Compose([
        T.ToTensor(),
        T.Resize(resize)
        ])
    return preprocess(observation)

def reconstruction_loss(x_hat, x):
    eps = 1e-6
    #constant = -torch.sum(x * torch.log(x + eps) + (1 - x) * torch.log(1 - x + eps))
    #loss = (torch.nn.BCELoss(reduction='sum')(x_hat, x) - constant) / x.size()[0]
    #print ("reconstruction loss = {}".format(loss))
    loss = torch.nn.MSELoss(reduction = 'sum')(x_hat, x)
    return loss

def kld_loss(mu, logvar):
    loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size()[0]

    return loss

def tensor_to_image(tensor):
    tensor = np.array(tensor*255, dtype = np.uint8)
    return np.moveaxis(tensor, 0, 2)

def create_gif(buffer):
    dataset = [img for img in buffer.memory]
    clip = ImageSequenceClip (dataset, fps = 20)
    clip.write_gif(buffer.name + ".gif", 20)