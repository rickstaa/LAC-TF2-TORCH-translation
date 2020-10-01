import gym
import datetime
import numpy as np
import ENV.env
import time

SEED = None
alpha = 1.0
alpha3 = 0.2
actor = [128, 64, 32]
# actor = [16, 16]
critic = [128, 64, 32]
# model = [32, 16]
episodes = int(1e5)
approx_value = True
timestr = time.strftime('%Y%m%d_%H%M')
num_of_paths_for_eval = 20
VARIANT = {
    'eval_list': [
        'LAC20200929_1810_4',
    ],
    'env_name': 'Ex3_EKF',
    'algorithm_name': 'LAC',
    'additional_description': timestr,
    # 'evaluate': False,
    'train': True,
    # 'train': False,
    'num_of_trials': 10,  # number of random seeds
    'num_of_evaluation_paths': 10,  # number of rollouts for evaluation
    'num_of_training_paths': 100,  # number of training rollouts stored for analysis
    'start_of_trial': 0,

    'evaluation_form': 'dynamic',

    'trials_for_eval': [str(i) for i in range(0, 10)],

    'evaluation_frequency': 2048,
    'trajNum': '9',
}
if VARIANT['algorithm_name'] == 'RARL':
    ITA = 0
VARIANT['log_path'] = '/'.join(
    ['./log', VARIANT['env_name'], VARIANT['algorithm_name'] + VARIANT['additional_description']])

ENV_PARAMS = {

    'Ex2_KF': {
        'max_ep_steps': 100,
        'max_global_steps': episodes,
        'max_episodes': int(1e6),
        'disturbance dim': 2,
        'eval_render': False,
        'network_structure':
            {'critic': critic,
             'actor': actor,
             },
    },
    'Ex2_EKF': {
        'max_ep_steps': 100,
        'max_global_steps': episodes,
        'max_episodes': int(1e6),
        'disturbance dim': 2,
        'eval_render': False,
        'network_structure':
            {'critic': critic,
             'actor': actor,
             },
    },
    'Ex3_EKF': {
        'max_ep_steps': 800,
        'max_global_steps': episodes,
        'max_episodes': int(1e6),
        'disturbance dim': 2,
        'eval_render': False,
        'network_structure':
            {'critic': critic,
             'actor': actor,
             },
    },
    'Ex4_EKF': {
        'max_ep_steps': 100,
        'max_global_steps': episodes,
        'max_episodes': int(1e6),
        'disturbance dim': 2,
        'eval_render': False,
        'network_structure':
            {'critic': critic,
             'actor': actor,
             },
    },
    'pose_Est': {
        'max_ep_steps': 100,
        'max_global_steps': episodes,
        'max_episodes': int(1e6),
        'disturbance dim': 2,
        'eval_render': False,
        'network_structure':
            {'critic': critic,
             'actor': actor,
             },
    },


}
ALG_PARAMS = {
    'MPC': {
        'horizon': 5,
    },

    'LQR': {
        'use_Kalman': False,
    },

    'LAC': {
        'iter_of_actor_train_per_epoch': 50,
        'iter_of_disturber_train_per_epoch': 50,
        'memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
        'alpha': alpha,
        'alpha3': alpha3,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.999,
        # 'gamma': 0.995,
        # 'gamma': 0.75,
        'steps_per_cycle': 100,
        'train_per_cycle': 80,
        'use_lyapunov': True,
        'adaptive_alpha': True,
        # 'approx_value': False,
        'approx_value': approx_value,
        'value_horizon': 2,
        'finite_horizon': True,
        # 'finite_horizon': False,
        'soft_predict_horizon': False,
        'target_entropy': None,
        'history_horizon': 0,  # 0 is using current state only
    },

    
}

EVAL_PARAMS = {
    'param_variation': {
        'param_variables': {
            'mass_of_pole': np.arange(0.05, 0.55, 0.05),  # 0.1
            'length_of_pole': np.arange(0.1, 2.1, 0.1),  # 0.5
            'mass_of_cart': np.arange(0.1, 2.1, 0.1),  # 1.0
            # 'gravity': np.arange(9, 10.1, 0.1),  # 0.1

        },
        'grid_eval': True,
        # 'grid_eval': False,
        'grid_eval_param': ['length_of_pole', 'mass_of_cart'],
        'num_of_paths': 100,  # number of path for evaluation
    },
    'impulse': {
        'magnitude_range': np.arange(80, 155, 5),
        'num_of_paths': 100,  # number of path for evaluation
        'impulse_instant': 200,
    },
    'constant_impulse': {
        'magnitude_range': np.arange(0.1, 1.0, .1),
        'num_of_paths': 20,  # number of path for evaluation
        'impulse_instant': 20,
    },
    'various_disturbance': {
        'form': ['sin', 'tri_wave'][0],
        'period_list': np.arange(2, 11, 1),
        # 'magnitude': np.array([1, 1, 1, 1, 1, 1]),
        'magnitude': np.array([80]),
        # 'grid_eval': False,
        'num_of_paths': 100,  # number of path for evaluation
    },
    'trained_disturber': {
        # 'magnitude_range': np.arange(80, 125, 5),
        # 'path': './log/cartpole_cost/RLAC-full-noise-v2/0/',
        'path': './log/HalfCheetahcost-v0/RLAC-horizon=inf-dis=.1/0/',
        'num_of_paths': 100,  # number of path for evaluation
    },
    'dynamic': {
        'additional_description': 'original',
        'num_of_paths': num_of_paths_for_eval,  # number of path for evaluation
        'plot_average': True,
        # 'plot_average': False,
        'directly_show': True,
    },
}
VARIANT['env_params'] = ENV_PARAMS[VARIANT['env_name']]
VARIANT['eval_params'] = EVAL_PARAMS[VARIANT['evaluation_form']]
VARIANT['alg_params'] = ALG_PARAMS[VARIANT['algorithm_name']]

RENDER = True


def get_env_from_name(name):
    if name == 'Ex2_EKF':
        from envs.Ex2_EKF import Ex2_EKF as env
        env = env()
        env = env.unwrapped
    elif name == 'Ex3_EKF':
        from envs.Ex3_EKF import Ex3_EKF as env
        env = env()
        env = env.unwrapped
    elif name == 'Ex4_EKF':
        from envs.Ex4_EKF import Ex4_EKF as env
        env = env()
        env = env.unwrapped
    elif name == 'Ex2_KF':
        from envs.Ex2_KF import Ex2_KF as env
        env = env()
        env = env.unwrapped
    elif name == 'pose_Est':
        from envs.Ex3_EKF import Ex3_EKF as env
        env = env()
        env = env.unwrapped
    env.seed(SEED)
    return env


def get_train(name):
    if 'LAC' in name:
        from LAC.LAC_V1 import train
    return train


def get_policy(name):

    if 'LAC' in name:
        from LAC.LAC_V1 import LAC as build_func
    elif 'LQR' in name:
        from LAC.lqr import LQR as build_func
    elif 'MPC' in name:
        from LAC.MPC import MPC as build_func

    return build_func


def get_eval(name):
    if 'LAC' in name or 'SAC_cost' in name:
        from LAC.LAC_V1 import eval

    return eval
