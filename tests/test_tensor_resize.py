'''
import torch
import numpy as np
from torchvision.transforms.functional import resize
import time


nb_tensors = 100

#Test 1 
start_1 = time.time()
list_np_arrays = np.random.rand(1, 3, 96, 96)
tensor_array = resize(torch.from_numpy(list_np_arrays),(64, 64))
for _ in range(nb_tensors):
    tensor_array = torch.vstack((tensor_array, resize(torch.from_numpy(np.random.rand(1, 3, 96, 96)),(64, 64))))
print (tensor_array.shape)
end_1 = time.time()

#test 2

start_2 = time.time()
list_np_arrays = np.random.rand(1, 3, 96, 96)
for _ in range(nb_tensors):
    list_np_arrays = np.vstack((list_np_arrays, np.random.rand(1, 3, 96, 96)))
tensors_stack = torch.from_numpy(list_np_arrays)
resized_tensors_stack = resize(tensors_stack, (64, 64))
print (resized_tensors_stack.shape)
end_2 = time.time()


print (f"test 1 took {end_1 - start_1} seconds")
print (f"test 2 took {end_2 - start_2} seconds")

'''