"""Small script to see which method of calculating the Mean Squared Root error is
faster.
"""

import timeit

# Script settings
N_SAMPLE = int(5e5)

# Test manual MSE speed
setup_code = """
import torch
import torch.nn.functional as F
backup = torch.rand((256, 1))
q = torch.rand((256, 1))
"""
code = """
0.5 * F.mse_loss(q, backup)
"""
method_1_time = timeit.timeit(code, setup=setup_code, number=N_SAMPLE)

# Test torch function MSE speed
setup_code = """
import torch
import torch.nn.functional as F
backup = torch.rand((256, 1))
q = torch.rand((256, 1))
"""
code = """
loss_q1 = 0.5 * ((q - backup) ** 2).mean()
"""
method_2_time = timeit.timeit(code, setup=setup_code, number=N_SAMPLE)

# Print results
print("\nTest MSE methods:")
print(f"- Torch F MSE: {method_1_time} s")
print(f"- Manual MSE time: {method_2_time} s")
