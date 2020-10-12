from envs.Ex3_EKF_gyro import Ex3_EKF_gyro as ENV

import random
import numpy as np

# Environment 1
random.seed(0)
np.random.seed(0)

env = ENV()
env = env.unwrapped
env.seed(0)
s = env.reset()
action = env.action_space.sample()
s_, r, done, _ = env.step(action)

# Environment 2
random.seed(0)
np.random.seed(0)
env2 = ENV()
env2 = env.unwrapped
env2.seed(0)
s2 = env2.reset()
action2 = env.action_space.sample()
s2_, r2, done2, _ = env.step(action2)

print("=ENV1=")
print(s)
print(s_)

print("=ENV2=")
print(s2)
print(s2_)
