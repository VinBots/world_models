import torch
import torch.nn as nn
import numpy as np

a = torch.cuda.is_available()
print (a)
torch.cuda.init()
torch.cuda.current_device()
torch.cuda.memory_allocated
torch.cuda.memory_cached
cuda = torch.device("cuda:0")