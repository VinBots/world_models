import torch
import torch.nn as nn
import numpy as np

"""
a = torch.cuda.is_available()
print (a)
torch.cuda.init()
torch.cuda.current_device()
torch.cuda.memory_allocated
torch.cuda.memory_cached
cuda = torch.device("cuda:0")
"""

W = torch.tensor([[2,2],[1,1]], dtype = torch.float)
#Y = X @ W.t()
U, S, V = torch.svd(W)

print (U)
print (S)
print (V)
