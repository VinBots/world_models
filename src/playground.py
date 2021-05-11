import torch
import torch.nn as nn

x = torch.rand(1, 1024, 1, 1)
upsample = nn.ConvTranspose2d(1024, 128, 5, stride=2)
h = upsample (x)
print (h.shape)