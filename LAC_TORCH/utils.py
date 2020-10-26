"""A set of common utilities used within the algorithm code.
"""

from collections import OrderedDict
import json
import os
import os.path as osp

import numpy as np
import copy
import torch
import torch.nn as nn

from variant import (
    TRAIN_PARAMS,
    ENV_PARAMS,
)

# Script parameters
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)

# FIXME: Might be faster to replace numpy with torch!
# IMPROVE: Put ENVIRONMENTS IN CONFIGURATION FILE


def get_env_from_name(name, ENV_SEED):
    """Initializes the gym environment with the given name

    Args:
        name (str): The name of the gym environment you want to initialize.

    Returns:
        gym.Env: The gym environment.
    """
    if name.lower() == "oscillator":
        from envs.oscillator import oscillator as env

        env = env()
        env = env.unwrapped
    elif name.lower() == "ex3_ekf":
        from envs.Ex3_EKF import Ex3_EKF as env

        env = env()
        env = env.unwrapped
    elif name.lower() == "ex4_ekf":
        from envs.Ex4_EKF import Ex4_EKF as env

        env = env()
        env = env.unwrapped
    if ENV_SEED is not None:
        env.seed(ENV_SEED)
    return env


def evaluate_training_rollouts(paths):
    """Evaluates the performance of the policy in the training rollouts.

    Args:
       paths (collections.deque): The training paths.

    Returns:
        collections.OrderedDict: Dictionary with performance statistics.
    """
    data = copy.deepcopy(paths)
    if len(data) < 1:
        return None
    try:
        diagnostics = OrderedDict(
            (
                ("return", np.mean([np.sum(path["rewards"]) for path in data])),
                ("length", np.mean([len(p["rewards"]) for p in data])),
            )
        )
    except KeyError:
        return
    [path.pop("rewards") for path in data]
    for key in data[0].keys():
        result = [np.mean(path[key]) for path in data]
        diagnostics.update({key: np.mean(result)})
    return diagnostics


def training_evaluation(test_env, policy):
    """Evaluates the performance of the current policy in
    several test rollouts.

    Args:
        test_env (gym.Env): The test gym environment you want to use.
        policy (object): The current policy.

    Returns:
        collections.OrderedDict: Dictionary with performance statistics.
    """

    # Retrieve action space bounds from test_env and pass them to the policy
    a_lowerbound = test_env.action_space.low
    a_upperbound = test_env.action_space.high
    policy.act_limits = {"low": a_lowerbound, "high": a_upperbound}

    # Training setting
    total_cost = []
    episode_length = []
    die_count = 0
    seed_average_cost = []

    # Perform roolouts to evaluate performance
    for i in range(TRAIN_PARAMS["num_of_evaluation_paths"]):
        cost = 0
        s = test_env.reset()
        for j in range(ENV_PARAMS["max_ep_steps"]):
            if ENV_PARAMS["eval_render"]:
                test_env.render()

            # Retrieve (scaled) action based on the current policy
            # NOTE (rickstaa): The scaling operation is already performed inside the
            # policy based on the `act_limits` you supplied.
            a = policy.choose_action(s, True)

            # Perform action in the environment
            s_, r, done, _ = test_env.step(a)
            cost += r
            if j == ENV_PARAMS["max_ep_steps"] - 1:
                done = True
            s = s_
            if done:
                seed_average_cost.append(cost)
                episode_length.append(j)
                if j < ENV_PARAMS["max_ep_steps"] - 1:
                    die_count += 1
                break

    # Save evaluation results
    total_cost.append(np.mean(seed_average_cost))
    total_cost_mean = np.average(total_cost)
    average_length = np.average(episode_length)

    # Return evaluation results
    diagnostic = {
        "return": total_cost_mean,
        "average_length": average_length,
    }
    return diagnostic


def mlp(sizes, activation, output_activation=nn.Identity):
    """Creates a multi-layered perceptron using pytorch.

    Args:
        sizes (list): The size of each of the layers.

        activation (torch.nn.modules.activation): The activation function used for the
            hidden layers.

        output_activation (torch.nn.modules.activation, optional): The activation
            function used for the output layers. Defaults to torch.nn.Identity.

    Returns:
        torch.nn.modules.container.Sequential: The multi-layered perceptron.
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def colorize(string, color, bold=False, highlight=False):
    """Returns string surrounded by appropriate terminal color codes to
    print colorized text.

    Args:
        string (str): The string you want to print.

        color (str): The color you want the string to have. Valid colors: gray, red,
            green, yellow, blue, magenta, cyan, white, crimson.

        bold (bool): Whether you want to use bold characters for the string.

        highlight (bool): Whether you want to highlight the text.

    Returns:
        str: The colorized string.
    """

    # Import six here so that `utils` has no import-time dependencies.
    # We want this since we use `utils` during our import-time sanity checks
    # that verify that our dependencies (including six) are actually present.
    import six

    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(six.u(str(num)))
    if bold:
        attr.append(six.u("1"))
    attrs = six.u(";").join(attr)
    return six.u("\x1b[%sm%s\x1b[0m") % (attrs, string)


def convert_json(obj):
    """Converts obj to a version which can be serialized with JSON.

    Args:
        obj (object): Object which you want to convert to json.

    Returns:
        object: Serialized json object.
    """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, "__name__") and not ("lambda" in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v) for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}

        return str(obj)


def is_json_serializable(v):
    """Check if object can be serialized with JSON.

    Args:
        v (object): object you want to check.

    Returns:
        bool: Boolean specifying whether the object can be serialized by json.
    """
    try:
        json.dumps(v)
        return True
    except TypeError:
        return False


def save_config(config, output_dir):
    """Log an experiment configuration.

    Call this once at the top of your experiment, passing in all important
    config vars as a dict. This will serialize the config to JSON, while
    handling anything which can't be serialized in a graceful way (writing
    as informative a string as possible).

    Example use:

    .. code-block:: python

        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())

    Args:
        config (dict): Configuration dictionary.

        output_dir (str): Output directory.
    """
    config_json = convert_json(config)
    output = json.dumps(config_json, separators=(",", ":\t"), indent=4, sort_keys=True)
    print(colorize("INFO: Saving config.", color="cyan", bold=True))
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(osp.join(output_dir, "config.json"), "w") as out:
        out.write(output)


def clamp(data, min_bound, max_bound):
    """Clamp all the values of a input to be between the min and max boundaries.

    Args:
        data (np.ndarray/list): Input data.
        min_bound (np.ndarray/list): Array containing the desired minimum values.
        max_bound (np.ndarray/list): Array containing the desired maximum values.

    Returns:
        np.ndarray: Array which has it values clamped between the min and max
            boundaries.
    """

    # Convert arguments to numpy array is not already
    data = torch.tensor(data) if not isinstance(data, torch.Tensor) else data
    min_bound = (
        torch.tensor(min_bound)
        if not isinstance(min_bound, torch.Tensor)
        else min_bound
    )
    max_bound = (
        torch.tensor(max_bound)
        if not isinstance(max_bound, torch.Tensor)
        else max_bound
    )

    # Clamp all actions to be within the boundaries
    return (data + 1.0) * (max_bound - min_bound) / 2 + min_bound
