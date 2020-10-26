from collections import OrderedDict

import numpy as np
import copy

from variant import (
    TRAIN_PARAMS,
    ENV_PARAMS,
)


def get_env_from_name(name, ENV_SEED=None):  # FIXME: Naming
    """Initializes the gym environment with the given name

    Args:
        name (str): The name of the gym environment you want to initialize.

    Returns:
        gym.Env: The gym environment.
    """
    if name == "oscillator":
        from envs.oscillator import oscillator as env

        env = env()
        env = env.unwrapped
    if name == "oscillator_double_cost":
        from envs.oscillator_double_cost import oscillator as env

        env = env()
        env = env.unwrapped
    elif name == "Ex3_EKF_gyro":
        from envs.Ex3_EKF_gyro import Ex3_EKF_gyro as env

        env = env()
        env = env.unwrapped
    elif name == "Ex3_EKF_gyro_dt":
        from envs.Ex3_EKF_gyro_dt import Ex3_EKF_gyro as env

        env = env()
        env = env.unwrapped
    elif name == "Ex3_EKF_gyro_dt_real":
        from envs.Ex3_EKF_gyro_dt_real import Ex3_EKF_gyro as env

        env = env()
        env = env.unwrapped
    if ENV_SEED is not None:
        env.seed(ENV_SEED)
    return env


def evaluate_training_rollouts(paths):
    """Evaluate the performance of the policy in the training rollouts."""
    data = copy.deepcopy(paths)
    if len(data) < 1:
        return None
    try:
        # FIXME: QD fix for getting two rewards This will break things!
        diagnostics = OrderedDict(
            (
                ("return_1", np.mean([np.sum(path["reward_1"]) for path in data])),
                ("return_2", np.mean([np.sum(path["reward_2"]) for path in data])),
                ("length", np.mean([len(p["reward_1"]) for p in data])),
            )
        )
    except KeyError:  # FIXME: Bad practice
        return
    [path.pop("reward_1") for path in data]
    [path.pop("reward_2") for path in data]
    for key in data[0].keys():
        result = [np.mean(path[key]) for path in data]
        diagnostics.update({key: np.mean(result)})

    return diagnostics


def training_evaluation(env, policy):
    """Evaluates the performance of the current policy in
    several test rollouts.

    Args:
        env (gym.Env): The gym environment you want to use.
        policy (object): The current policy.

    Returns:
        [type]: [description]
    """
    # FIXME: QD way to support two costs
    # Retrieve action space bounds from env
    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low

    # Training setting
    total_cost_1 = []
    total_cost_2 = []
    episode_length = []
    die_count = 0
    seed_average_cost_1 = []
    seed_average_cost_2 = []

    # Perform roolouts to evaluate performance
    for i in range(TRAIN_PARAMS["num_of_evaluation_paths"]):
        cost_1 = 0
        cost_2 = 0
        if env.__class__.__name__.lower() == "ex3_ekf_gyro":
            s = env.reset(eval=True)
        else:
            s = env.reset()
        for j in range(ENV_PARAMS["max_ep_steps"]):
            if ENV_PARAMS["eval_render"]:
                env.render()
            a = policy.choose_action(s, True)
            action = a_lowerbound + (a + 1.0) * (a_upperbound - a_lowerbound) / 2
            s_, r, done, _ = env.step(action)
            cost_1 += r[0]
            cost_2 += r[1]
            if j == ENV_PARAMS["max_ep_steps"] - 1:
                done = True
            s = s_
            if done:
                seed_average_cost_1.append(cost_1)
                seed_average_cost_2.append(cost_2)
                episode_length.append(j)
                if j < ENV_PARAMS["max_ep_steps"] - 1:
                    die_count += 1
                break

    # Save evaluation results
    total_cost_1.append(np.mean(seed_average_cost_1))
    total_cost_2.append(np.mean(seed_average_cost_2))
    total_cost_mean_1 = np.average(total_cost_1)
    total_cost_mean_2 = np.average(total_cost_2)
    average_length = np.average(episode_length)

    # Return evaluation results
    diagnostic = {
        "test_return_1": total_cost_mean_1,
        "test_return_2": total_cost_mean_2,
        "test_average_length": average_length,
    }
    return diagnostic
