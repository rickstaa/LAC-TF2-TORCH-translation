"""Script to test if torch append is as fast as regular append."""

import numpy as np
import torch
import time

# Numpy version
print("==Numpy version==")
start_time = time.time()
test = []
test.append([2])
finish_time = time.time()
print(finish_time - start_time)

# Torch version
print("==Torch version==")
start_time = time.time()
test = torch.tensor([], dtype=torch.float32)
test = torch.cat((test, torch.tensor([2])))
finish_time = time.time()
print(finish_time - start_time)

print("==Log Torch version==")
start_time = time.time()
test = torch.tensor([2], dtype=torch.float32)
test = torch.log(test)
finish_time = time.time()
print(finish_time - start_time)

print("==numpy version==")
start_time = time.time()
np.log(3)
finish_time = time.time()
print(finish_time - start_time)
