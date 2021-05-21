"""
We use this network to model the probability distribution
of the next z in the next time step as a Mixture of Gaussian
distribution. T

we
did not model the correlation parameter between each element of z, and instead had the MDN-RNN output a diagonal
covariance matrix of a factored Gaussian distribution
"""

# https://ai.googleblog.com/2017/04/teaching-machines-to-draw.html
# https://magenta.tensorflow.org/sketch-rnn-demo


import torch


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.lstm(x)[0]
        x = self.linear(h)
        return x
    
    def get_states_across_time(self, x):
        h_c = None
        h_list, c_list = list(), list()
        with torch.no_grad():
            for t in range(x.size(1)):
                h_c = self.lstm(x[:, [t], :], h_c)[1]
                h_list.append(h_c[0])
                c_list.append(h_c[1])
            h = torch.cat(h_list)
            c = torch.cat(c_list)
        return h, c

