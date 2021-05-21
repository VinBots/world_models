"""
The MDN-RNNs were trained for 20 epochs on the data collected from a random policy agent. In the Car Racing task, the LSTM used 256 hidden units

"""

import numpy as np
import collections
import itertools
import torch
import torch.optim as optim


def train_lstm(model, buffer, preprocess, config, device):
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    buffer.shuffle()

    for ep in range (config.tr_epochs):
        
        for i in range (0, len(buffer.memory), config.batch_size):
            #calculate input x

            optimizer.zero_grad()
            x_hat = model.forward(x)

            # calculate loss
            loss.backward()
            optimizer.step()

        if ep % 100 == 0 :
            print ("Epoch {} - Loss = {}".format(ep, loss.item()))
    return True