import torch
import torch.nn as nn
import numpy as np

'''
x = torch.rand(1, 1024, 1, 1)
upsample = nn.ConvTranspose2d(1024, 128, 5, stride=2)
h = upsample (x)
print (h.shape)

A = [[list(range(5)), [0, 1], list(range(0))]]
'''
a = [1,2,3,4,5,6,7,8,9, 10, 11]
a = np.vstack(np.array(a))
for i in range (0, 11, 3):
    #b = np.vstack(np.array(a[i:i+3]))
    b = np.array(a[i:i+3])
    print (b)
