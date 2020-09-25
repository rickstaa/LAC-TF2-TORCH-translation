"""Simple script used to test the performance of a trained model."""

import gym
import torch
import sys
import numpy as np
import machine_learning_control.simzoo.simzoo.envs
import matplotlib.pyplot as plt

from lac import LAC
from envs.oscillator import oscillator
from envs.Ex3_EKF import Ex3_EKF

ENV_NAME = "Ex3_EKF-v0"
# ENV_NAME = "Oscillator-v0"
# ENV_NAME = "Hopper-v2"
# MAX_EP_LEN = 200
MODEL_PATH = "/home/ricks/Development/tf_rewrite/Filter/log/Ex3_EKF/LAC20200922_1456/1/policy"  # Ex3 env
# MODEL_PATH = "/home/ricks/Development/tf_rewrite/Filter/log/oscillator/LAC20200921_1752/0/policy"  # Oscillator env
EP = 1000


if ENV_NAME == "Oscillator-v0":

    # Create environment
    env = oscillator()

    # Take T steps in the environment
    T = 200
    path = []
    info_dict = []
    t1 = []
    s = env.reset()
    print(f"Taking {T} steps in the oscillator environment.")

    # Get action boundaries
    a_lowerbound = env.action_space.low
    a_upperbound = env.action_space.high

    # Get trained policy
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    policy = LAC(a_dim, s_dim)
    LAC = policy.restore(MODEL_PATH)
    if not LAC:
        print("Model could not be loaded. Please try again!")
        sys.exit(0)

    # Run inference
    for i in range(int(T / env.dt)):
        a = policy.choose_action(s, True)
        action = a_lowerbound + (a + 1.0) * (a_upperbound - a_lowerbound) / 2
        s, r, done, info = env.step(action)
        # s, r, done, info = env.step(np.array([0, 0, 0]))
        path.append(s)
        info_dict.append(info)
        t1.append(i * env.dt)

    # Plot results
    # observations = (m1, m2, m3, p1, p2, p3, r1, p1 - r1)
    print("Plot results.")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    # ax.plot(t1, np.array(path)[:, 0], color="orange", label="mRNA1")
    # ax.plot(t1, np.array(path)[:, 1], color="magenta", label="mRNA2")
    # ax.plot(t1, np.array(path)[:, 2], color="sienna", label="mRNA3")
    ax.plot(t1, np.array(path)[:, 3], color="blue", label="protein1")
    # ax.plot(t1, np.array(path)[:, 4], color="cyan", label="protein2")
    # ax.plot(t1, np.array(path)[:, 5], color="green", label="protein3")
    # ax.plot(t1, np.array(path)[:, 0:3], color="blue", label="mRNA")
    # ax.plot(t1, np.array(path)[:, 3:6], color="blue", label="protein")
    ax.plot(t1, np.array(path)[:, 6], color="yellow", label="reference")
    ax.plot(t1, np.array(path)[:, 7], color="red", label="error")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    plt.show()
    print("Done")
else:

    # Create environment
    env = Ex3_EKF()

    # Get action boundaries
    a_lowerbound = env.action_space.low
    a_upperbound = env.action_space.high

    # Get trained policy
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    policy = LAC(a_dim, s_dim)
    LAC = policy.restore(MODEL_PATH)
    if not LAC:
        print("Model could not be loaded. Please try again!")
        sys.exit(0)

    # Run inference
    T = 10
    path = []
    path_2 = []
    t1 = []
    s = env.reset()
    for i in range(int(T / env.dt)):
        a = policy.choose_action(s, True)
        action = a_lowerbound + (a + 1.0) * (a_upperbound - a_lowerbound) / 2
        s, r, done, info = env.step(action)
        path_2.append(info["state_of_interest"])
        path.append(s)
        t1.append(i * env.dt)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    ax.plot(t1, path_2, color="green", label="soi")
    # ax.plot(t1, np.array(path)[:, 0], color="yellow", label="x1")
    # ax.plot(t1, np.array(path)[:, 1], color="green", label="x

    handles, labels = ax.get_legend_handles_labels()
    #
    ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    plt.show()
    print("done")
