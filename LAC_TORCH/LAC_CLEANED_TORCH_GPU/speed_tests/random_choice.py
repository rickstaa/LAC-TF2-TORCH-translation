"""Script used to asses whether numpy or torch is faster."""

import numpy as np
import torch
import time

# Numpy version
print("==Numpy version==")
start_time = time.time()
a = np.array([1, 2, 3, 4])
p = np.array([0.1, 0.1, 0.1, 0.7])
n = 2
replace = True
b = np.random.choice(a, p=p, size=n, replace=replace)
finish_time = time.time()
print(finish_time - start_time)

# Pytorch version
print("==Torch version==")
start_time = time.time()
a = torch.tensor([1, 2, 3, 4])
p = torch.tensor([0.1, 0.1, 0.1, 0.7])
n = 2
replace = True
idx = p.multinomial(num_samples=n, replacement=replace)
b = a[idx]
finish_time = time.time()
print(finish_time - start_time)
