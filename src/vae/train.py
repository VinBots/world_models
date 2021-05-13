import numpy as np
import collections
import itertools
import torch
from src.utils import reconstruction_loss, kld_loss
import torch.optim as optim


def train_vae(model, buffer, preprocess, config):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    buffer.shuffle()
    for ep in range (config.tr_epochs):
        for i in range (0, len(buffer.memory), config.batch_size):
            x_sliced_list = list(buffer.memory)[i:i+config.batch_size]
            
            x = torch.stack(
                [preprocess(np.copy(img), config.resize) for img in x_sliced_list])
            
            optimizer.zero_grad()
            x_hat, mu, logvar = model.forward(x)
            loss = reconstruction_loss (x_hat, x) + kld_loss (mu, logvar)
            loss.backward()
            optimizer.step()
        print ("Epoch {} - Loss = {}".format(ep, loss.item()))

    return True





