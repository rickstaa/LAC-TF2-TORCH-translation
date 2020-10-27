"""Small script to see which method of calculating the Mean Squared Root error is
faster.

Conclusion:
    The manual method is two times as fast.
"""

import torch
import torch.nn.functional as F

# Script settings
N_SAMPLE = int(1e6)

# Test manual MSE speed
backup = torch.rand((256, 1))
q = torch.rand((256, 1))
for i in range(N_SAMPLE):
    0.5 * F.mse_loss(q, backup)

# # Test torch function MSE speed
# backup = torch.rand((256, 1))
# q = torch.rand((256, 1))
# for i in range(N_SAMPLE):
#     loss_q1 = 0.5 * ((q - backup) ** 2).mean()
