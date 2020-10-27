"""Small script to see which method of calculating the Mean is faster.

Conclusion:
    No difference.
"""


import timeit

# Script settings
N_SAMPLE = int(1e7)

# Test torch mean function
print("\nTest torch mean methods:")
setup_code = """
import torch
log_alpha = torch.tensor(1.0, dtype=torch.float32).log()
log_alpha.requires_grad = True
alpha = log_alpha.exp()
target_entropy = -3
log_pis = torch.rand((256))
log_pis.requires_grad = True
"""
code = """
-torch.mean(alpha * (log_pis + target_entropy).detach())
"""
method_1_time = timeit.timeit(code, setup=setup_code, number=N_SAMPLE)

# Test torch mean operator
setup_code = """
import torch
log_alpha = torch.tensor(1.0, dtype=torch.float32).log()
log_alpha.requires_grad = True
alpha = log_alpha.exp()
target_entropy = -3
log_pis = torch.rand((256))
log_pis.requires_grad = True
"""
code = """
alpha_loss = -(alpha * (log_pis + target_entropy).detach()).mean()
"""
method_2_time = timeit.timeit(code, setup=setup_code, number=N_SAMPLE)

# Print results
print(f"- Torch mean function: {method_1_time} s")
print(f"- Torch mean operator: {method_2_time} s")
