# NOTE: A lot of things in this code are redundant. I however choose to leave them there
# for comparison.

import tensorflow as tf
import numpy as np
import time
from .squash_bijector import SquashBijector
from .utils import evaluate_training_rollouts
import tensorflow_probability as tfp
from collections import OrderedDict, deque
import os
from copy import deepcopy
import sys
sys.path.append("..")
from robustness_eval import training_evaluation
from disturber.disturber import Disturber
from pool.pool import Pool
import logger
from variant import *

# ===============================
# BEGIN >>> Pytorch CODE ========
# ===============================
from itertools import chain
import os.path as osp
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .pytorch_a import SquashedGaussianMLPActor
from .pytorch_l import MLPLFunction
# ===============================
# END <<<<< Pytorch CODE ========
# ===============================

# Wheter you want to use Pytorch instead of tensorflow
USE_PYTORCH = True
# USE_PYTORCH = False

SCALE_DIAG_MIN_MAX = (-20, 2)
SCALE_lambda_MIN_MAX = (0, 1)

# FIXME! REMOVE LATER!
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

class LAC(object):
    def __init__(self,
                 a_dim,
                 s_dim,
                 variant,
                 action_prior = 'uniform',
                 ):



        ###############################  Model parameters  ####################################
        # self.memory_capacity = variant['memory_capacity']

        self.batch_size = variant['batch_size']
        self.network_structure = variant['network_structure']
        gamma = variant['gamma']

        tau = variant['tau']
        self.approx_value = True if 'approx_value' not in variant.keys() else variant['approx_value']
        # self.memory = np.zeros((self.memory_capacity, s_dim * 2 + a_dim+ d_dim + 3), dtype=np.float32)
        # self.pointer = 0
        if not USE_PYTORCH: # NOTE: ==== TENSORFLOW  CODE =====#
            self.sess = tf.Session()
        self._action_prior = action_prior
        s_dim = s_dim * (variant['history_horizon']+1)
        self.a_dim, self.s_dim, = a_dim, s_dim
        self.history_horizon = variant['history_horizon']
        self.working_memory = deque(maxlen=variant['history_horizon']+1)
        target_entropy = variant['target_entropy']
        if target_entropy is None:
            self.target_entropy = -self.a_dim   #lower bound of the policy entropy
        else:
            self.target_entropy = target_entropy
        self.finite_horizon = variant['finite_horizon']
        self.soft_predict_horizon = variant['soft_predict_horizon']

        if USE_PYTORCH:
            print("")
            print("=========================================================")
            print("===============Training with pytorch!====================")
            print("=========================================================")
            print("")
            # ===============================
            # BEGIN >>> Pytorch CODE ========
            # ===============================
            # FIXME: Initialize tensors Actually not needed for all variables in pytorch
            # TODO: check dimensions

            # Setup observation, actions, rewards placeholders
            # NOTE: Not needed in pytorch but used for consistency with tf
            self.S = torch.zeros(s_dim)
            self.S_ = torch.zeros(s_dim)
            self.a_input = torch.zeros(a_dim)
            self.a_input_ = torch.zeros(a_dim)
            self.R = torch.zeros(1)
            self.R_N_ = torch.zeros(1) # NOTE: Not needed for now
            self.V = torch.zeros(1) # NOTE: Not used
            self.terminal = torch.zeros(1)

            # Setup learning rate placeholders
            self.LR_A = torch.tensor(variant["lr_a"])
            self.LR_lag = torch.tensor(variant["lr_a"])
            self.LR_C = torch.tensor(variant["lr_c"])
            self.LR_L = torch.tensor(variant["lr_l"])

            # Initialize hyperparameters
            labda = variant['labda']
            alpha = variant['alpha']
            self.gamma = gamma
            self.alpha3 = variant['alpha3']
            self.polyak = (1 - tau)

            # Create trainable variables (Lagrance multipliers)
            self.log_labda = torch.tensor(labda).log()
            self.log_alpha = torch.tensor(alpha).log()
            self.log_labda.requires_grad = True  # Enable gradient computation
            self.log_alpha.requires_grad = True  # Enable gradient computation
            # self.labda = tf.clip_by_value(tf.exp(log_labda), *SCALE_lambda_MIN_MAX) # Created as property # NOTE: Created as property
            # self.alpha = tf.exp(log_alpha) #NOTE: Created as property

            # Create Main networks
            # NOTE: The self.S and self.a_input arguments are ignored in the pytorch
            # case. The action and observation space comes from self.s_dim, self.a_dim
            torch.manual_seed(50) # FIXME: REMOVE REMOVE REMOVE REMOVE
            self.ga = self._build_a(self.S) # 这个网络用于及时更新参数 # TODO: CHECK Network creation
            torch.manual_seed(40) # FIXME: REMOVE REMOVE REMOVE REMOVE
            self.lc = self._build_l(self.S, self.a_input)   # lyapunov 网络 # TODO: CHECK Network creation

            # Get other script variables
            self.use_lyapunov = variant['use_lyapunov']
            self.adaptive_alpha = variant['adaptive_alpha']

            # Create target networks
            # NOTE: The self.S and self.a_input arguments are ignored in the pytorch
            # case. The action and observation space comes from self.s_dim, self.a_dim
            torch.manual_seed(30) # FIXME: REMOVE REMOVE REMOVE REMOVE
            self.ga_ = self._build_a(self.S_) # TODO: CHECK Network creation
            torch.manual_seed(20) # FIXME: REMOVE REMOVE REMOVE REMOVE
            self.lc_ = self._build_l(self.S_, self.a_input_) # TODO: CHECK Network creation

            # Freeze target networks
            for p in self.ga_.parameters():
                p.requires_grad = False
            for p in self.lc_.parameters():
                p.requires_grad = False

            # Create untrainable lyapunov actor and l_target
            torch.manual_seed(10) # FIXME: REMOVE REMOVE REMOVE REMOVE
            self.lya_ga_ = self._build_a(self.S_)
            torch.manual_seed(5) # FIXME: REMOVE REMOVE REMOVE REMOVE
            self.lya_lc_ = self._build_l(self.S_, self.a_input)

            # Make the lyapunov actor un-trainable
            for p in self.lya_ga_.parameters():
                p.requires_grad = False
            for p in self.lya_lc_.parameters():
                p.requires_grad = False

            # Create optimizers
            # lc_params = [self.lc.w1_s, self.lc_.w1_a, self.lc_.b1, self.lc.parameters()]
            # Set up optimizers for policy, q-function and alpha temperature regularization
            self.pi_optimizer = Adam(self.ga.parameters(), lr=self.LR_A)
            self.l_optimizer = Adam(self.lc.parameters(), lr=self.LR_L) # FIXME: Weight and bias of first layer was not updated
            # self.l_optimizer = Adam(chain(*lc_params), lr=self.LR_L)
            self.log_alpha_optimizer = Adam([self.log_alpha], lr=self.LR_A)
            self.log_labda_optimizer = Adam([self.log_labda], lr=self.LR_lag) # Question: Why isn't the learnign rate of lyapunov decreased?

            # ===============================
            # END <<<<< Pytorch CODE ========
            # ===============================
        else:
            print("")
            print("=========================================================")
            print("===============Training with tensorflow!=================")
            print("=========================================================")
            print("")
            with tf.variable_scope('Actor'):
                self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
                self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
                self.a_input = tf.placeholder(tf.float32, [None, a_dim], 'a_input')
                self.a_input_ = tf.placeholder(tf.float32, [None, a_dim], 'a_input_')
                self.R = tf.placeholder(tf.float32, [None, 1], 'r')
                self.R_N_ = tf.placeholder(tf.float32, [None, 1], 'r_N_')
                self.V = tf.placeholder(tf.float32, [None, 1], 'v')
                self.terminal = tf.placeholder(tf.float32, [None, 1], 'terminal')
                self.LR_A = tf.placeholder(tf.float32, None, 'LR_A')
                self.LR_lag = tf.placeholder(tf.float32, None, 'LR_lag')
                self.LR_C = tf.placeholder(tf.float32, None, 'LR_C')
                self.LR_L = tf.placeholder(tf.float32, None, 'LR_L')
                # self.labda = tf.placeholder(tf.float32, None, 'Lambda')
                labda = variant['labda']
                alpha = variant['alpha']
                alpha3 = variant['alpha3']
                log_labda = tf.get_variable('lambda', None, tf.float32, initializer=tf.log(labda))
                log_alpha = tf.get_variable('alpha', None, tf.float32, initializer=tf.log(alpha))  # Entropy Temperature
                self.labda = tf.clip_by_value(tf.exp(log_labda), *SCALE_lambda_MIN_MAX)
                self.alpha = tf.exp(log_alpha)

                self.a, self.deterministic_a, self.a_dist = self._build_a(self.S, )  # 这个网络用于及时更新参数

                self.l = self._build_l(self.S, self.a_input)   # lyapunov 网络


                self.use_lyapunov = variant['use_lyapunov']
                self.adaptive_alpha = variant['adaptive_alpha']

                a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/actor')
                l_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/Lyapunov')

                ###############################  Model Learning Setting  ####################################
                ema = tf.train.ExponentialMovingAverage(decay=1 - tau)  # soft replacement

                def ema_getter(getter, name, *args, **kwargs):
                    return ema.average(getter(name, *args, **kwargs))
                target_update = [ema.apply(a_params),  ema.apply(l_params)]  # soft update operation

                # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
                a_, _, a_dist_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
                lya_a_, _, lya_a_dist_ = self._build_a(self.S_, reuse=True)
                # self.cons_a_input_ = tf.placeholder(tf.float32, [None, a_dim, 'cons_a_input_'])
                # self.log_pis = log_pis = self.a_dist.log_prob(self.a)
                self.log_pis = log_pis = self.a_dist.log_prob(self.a)
                self.prob = tf.reduce_mean(self.a_dist.prob(self.a))

                # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度

                l_ = self._build_l(self.S_, a_, reuse=True, custom_getter=ema_getter)
                self.l_ = self._build_l(self.S_, lya_a_, reuse=True)

                # lyapunov constraint
                self.l_derta = tf.reduce_mean(self.l_ - self.l + (alpha3) * self.R)

                labda_loss = -tf.reduce_mean(log_labda * self.l_derta)
                alpha_loss = -tf.reduce_mean(log_alpha * tf.stop_gradient(log_pis + self.target_entropy))
                self.alpha_train = tf.train.AdamOptimizer(self.LR_A).minimize(alpha_loss, var_list=log_alpha)
                self.lambda_train = tf.train.AdamOptimizer(self.LR_lag).minimize(labda_loss, var_list=log_labda)

                if self._action_prior == 'normal':
                    policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                        loc=tf.zeros(self.a_dim),
                        scale_diag=tf.ones(self.a_dim))
                    policy_prior_log_probs = policy_prior.log_prob(self.a)
                elif self._action_prior == 'uniform':
                    policy_prior_log_probs = 0.0

                if self.use_lyapunov is True:
                    a_loss = self.labda * self.l_derta + self.alpha * tf.reduce_mean(log_pis) - policy_prior_log_probs
                else:
                    a_loss = a_preloss

                self.a_loss = a_loss
                self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(a_loss, var_list=a_params)

                next_log_pis = a_dist_.log_prob(a_)
                with tf.control_dependencies(target_update):  # soft replacement happened at here
                    if self.approx_value:
                        if self.finite_horizon:
                            if self.soft_predict_horizon:
                                l_target = self.R - self.R_N_ + tf.stop_gradient(l_)
                            else:
                                l_target = self.V
                        else:
                            l_target = self.R + gamma * (1 - self.terminal) * tf.stop_gradient(l_)  # Lyapunov critic - self.alpha * next_log_pis
                            # l_target = self.R + gamma * (1 - self.terminal) * tf.stop_gradient(l_- self.alpha * next_log_pis)  # Lyapunov critic
                    else:
                        l_target = self.R

                    self.l_error = tf.losses.mean_squared_error(labels=l_target, predictions=self.l)
                    self.ltrain = tf.train.AdamOptimizer(self.LR_L).minimize(self.l_error, var_list=l_params)

                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver()
                self.diagnotics = [self.labda, self.alpha, self.l_error, tf.reduce_mean(-self.log_pis), self.a_loss]

                if self.use_lyapunov is True:
                    self.opt = [self.ltrain, self.lambda_train]
                self.opt.append(self.atrain)
                if self.adaptive_alpha is True:
                    self.opt.append(self.alpha_train)

    # ===============================
    # BEGIN >>> Pytorch CODE ========
    # ===============================
    if USE_PYTORCH:
        @property
        def alpha(self):
            return self.log_alpha.exp()

        @property
        def labda(self):
            labda_scaled = torch.clamp(
                self.log_labda.exp(),
                SCALE_lambda_MIN_MAX[0],
                SCALE_lambda_MIN_MAX[1],
            )
            return labda_scaled
    # ===============================
    # END <<<<< Pytorch CODE ========
    # ===============================

    def choose_action(self, s, evaluation = False):
        if len(self.working_memory) < self.history_horizon:
            [self.working_memory.appendleft(s) for _ in range(self.history_horizon)]

        self.working_memory.appendleft(s)
        try:
            s = np.concatenate(self.working_memory)
        except ValueError:
            print(s)

        if evaluation is True:
            if USE_PYTORCH:
                # ===============================
                # BEGIN >>> Pytorch CODE ========
                # ===============================
                try:
                    with torch.no_grad():
                        return self.ga(torch.Tensor(s).unsqueeze(0))[1].detach().squeeze().numpy()
                except ValueError:
                    return
                # ===============================
                # END <<<<< Pytorch CODE ========
                # ===============================
            else:
                try:
                    return self.sess.run(self.deterministic_a, {self.S: s[np.newaxis, :]})[0]
                except ValueError:
                    return
        else:
            if USE_PYTORCH:
                # ===============================
                # BEGIN >>> Pytorch CODE ========
                # ===============================
                # FIXME: Make sure S is updated
                with torch.no_grad():
                    return self.ga(torch.Tensor(s).unsqueeze(0))[0].detach().squeeze().numpy()
                # ===============================
                # END <<<<< Pytorch CODE ========
                # ===============================
            else:
                return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self, LR_A, LR_C, LR_L, LR_lag, batch):
        if USE_PYTORCH:
            # Question: LR_C IS redundant?

            # ===============================
            # BEGIN >>> Pytorch CODE ========
            # ===============================
            # Adjust optimizer learning rates (decay)
            for param_group in self.pi_optimizer.param_groups:
                param_group['lr'] = LR_A
            for param_group in self.l_optimizer.param_groups:
                param_group['lr'] = LR_L
            for param_group in self.log_alpha_optimizer .param_groups:
                param_group['lr'] = LR_A
            for param_group in self.log_labda_optimizer .param_groups:
                param_group['lr'] = LR_lag

            # Unpack experiences from the data dictionary
            self.S, self.a_input, self.R, self.S_, self.terminal = (
                torch.Tensor(batch["s"]),
                torch.Tensor(batch["a"]),
                torch.Tensor(batch["r"]),
                torch.Tensor(batch["s_"]),
                torch.Tensor(batch["terminal"]),
            )

            # Run optimizations (Calculate losses and perform SGD)
            # Question: Does the order mather?

            # Calculate log probability of a_input based on current policy
            # FIXME: Possible cause of deviation - Do we need to put i there or can we also put it in a function and compute multiple times?
            _, _, self.a_dist = self.ga(self.S)
            log_pis = self.a_dist # DEBUG: Tf version returns distribution and then calculates the log probability should be similar right?
            self.log_pis = log_pis.detach()

            # Calculate current and target lyapunov value
            self.l = self.lc(self.S, self.a_input)
            with torch.no_grad():
                lya_a_, _, _ = self.lya_ga_(self.S_)
                self.l_ = self.lya_lc_(self.S_, lya_a_)

            # Lyapunov constraint
            self.l_derta = torch.mean(self.l_ - self.l + (self.alpha3) * self.R)

            #####################################
            # Optimize lyapunov multiplier ######
            #####################################

            # Optimize lambda
            self.log_labda_optimizer.zero_grad()

            # Calculate lyapunov multiplier loss
            labda_loss = -torch.mean(self.log_labda * self.l_derta.detach()) # Question: The mean is redundenat here right
            # DEBUG:
            # FIXME: I changed this to possitive now
            labda_loss = torch.mean(self.log_labda * self.l_derta.detach())

            # Perform SGD
            labda_loss.backward()
            self.log_labda_optimizer.step()

            #####################################
            # Optimize lyapunov multiplier ######
            #####################################

            # Optimize alpha
            self.log_alpha_optimizer.zero_grad()

            # Calculate alpha multiplier loss
            # NOTE: This is very small!
            alpha_loss = -torch.mean(self.log_alpha * (log_pis + self.target_entropy).detach())

            # Perform SGD
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            #####################################
            # Optimize lyapunov network #########
            #####################################
            # optimize poliy net
            self.pi_optimizer.zero_grad()

            # Calculate actor los
            self.a_loss = self.labda.detach() * self.l_derta.detach() + self.alpha.detach() * torch.mean(log_pis) # TODO: Check if mean is needed

            # Perform SGD
            self.a_loss.backward()
            self.pi_optimizer.step()
            # FIXME: Action prior!
            #####################################
            # Optimize lyapunov network #########
            #####################################

            # update target networks according to exponential moving average
            # Question: In tensorflow version tf.control_dependencies specified that
            # this has to be done before the l_error calculation.
            with torch.no_grad():
                for p, p_targ in zip(self.ga.parameters(), self.ga_.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)
                for p, p_targ in zip(self.lc.parameters(), self.lc_.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

            # Calculate lyapunov loss
            self.l_optimizer.zero_grad()

            # Bellman backup for L functions
            # Question is the torch.no_Grad still needed?
            # FIXME CHECK gradient
            # with torch.no_grad():  # Make sure the gradients are not tracked

            # Get target lyapunov value out of lyapunov actor
            with torch.no_grad():
                a_, _, _ = self.ga_(self.S_)
                l_ = self.lc_(self.S_, a_)

            # Used when agent has to minimize reward is positive deviation (Minghoas version)
            l_target = self.R + self.gamma * (1 - self.terminal) * l_.detach() # FIXME: Detach not needed since already torch.no_grad

            # Calculate lyapunov loss
            self.l_error = F.mse_loss(l_target, self.l)

            # Perform SGD
            self.l_error.backward()
            self.l_optimizer.step()

            # Calculate entropy and return diagnostics
            entropy = torch.mean(-self.log_pis.detach()) # FIXME: Not needed since already done before
            return self.labda.detach(), self.alpha.detach(), self.l_error.detach(), entropy, self.a_loss.detach()
            # ===============================
            # END <<<<< Pytorch CODE ========
            # ===============================
        else:
            bs = batch['s']  # state
            ba = batch['a']  # action

            br = batch['r']  # reward

            bterminal = batch['terminal']
            bs_ = batch['s_']  # next state
            feed_dict = {self.a_input: ba,  self.S: bs, self.S_: bs_, self.R: br, self.terminal: bterminal,
                        self.LR_C: LR_C, self.LR_A: LR_A, self.LR_L: LR_L, self.LR_lag:LR_lag}
            if self.finite_horizon:
                bv = batch['value']
                b_r_ = batch['r_N_']
                feed_dict.update({self.V:bv, self.R_N_:b_r_})

            self.sess.run(self.opt, feed_dict)
            labda, alpha, l_error, entropy, a_loss = self.sess.run(self.diagnotics, feed_dict)

            return labda, alpha, l_error, entropy, a_loss

    def store_transition(self, s, a,d, r, l_r, terminal, s_):
        transition = np.hstack((s, a, d, [r], [l_r], [terminal], s_))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def normalize_input(self,input_value):
        mean_X_train = tf.math.reduce_mean(tf.reshape(input_value, [-1]))
        std_X_train = tf.math.reduce_std(tf.reshape(input_value, [-1]))
        X_train = (input_value-mean_X_train)/std_X_train
        return X_train

    def evaluate_value(self, s, a):

        if len(self.working_memory) < self.history_horizon:
            [self.working_memory.appendleft(s) for _ in range(self.history_horizon)]

        self.working_memory.appendleft(s)
        try:
            s = np.concatenate(self.working_memory)
        except ValueError:
            print(s)

        return self.sess.run(self.l, {self.S: s[np.newaxis, :], self.a_input: a[np.newaxis, :]})[0]


    def _build_a(self, s, name='actor', reuse=None, custom_getter=None):
        if USE_PYTORCH:

            # ===============================
            # BEGIN >>> Pytorch CODE ========
            # ===============================
            # Get action and observation space size
            s_dim = self.s_dim
            a_dim = self.a_dim

            # Create and return Squashed Gaussian actor
            SGA = SquashedGaussianMLPActor(s_dim, a_dim, self.network_structure, log_std_min=SCALE_DIAG_MIN_MAX[0],log_std_max=SCALE_DIAG_MIN_MAX[1])
            return SGA

            # NOTE: I tried using a sequentional (Function structure) but this is not possible
            # # Check if this is a target network
            # # FIXME: Check reuse and custom getter!
            # if reuse is None:
            #     trainable = True # Main network
            # else:
            #     trainable = False # Target network

            # # Get action and observation space size
            # s_dim = self.s_dim
            # a_dim = self.a_dim

            # # Initialize weights
            # w_a = torch.zeros([s_dim, a_dim])
            # print(w_a)

            # # TODO: Add squashed bijector

            # # Get hidden layer structure
            # n1 = self.network_structure['actor'][0] # Size of hidden layer one
            # n2 = self.network_structure['actor'][1] # Size of hidden layer two

            # # TODO: Make sure the networks are trainable
            # # Construct actor networks
            # net_0 = nn.Sequential(nn.Linear(s_dim, n1), nn.ReLU())
            # net_1 = nn.Sequential(nn.Linear(net_0[0].out_features, n2), nn.ReLU())
            # mu = nn.Sequential(net_1, nn.Linear(net_1[0].out_features, self.a_dim))  # TODOCheck: Activation = None
            # log_sigma = nn.Sequential(net_1, nn.Linear(net_1[0].out_features, self.a_dim))  #TODOCheck: Activation = None

            # # Freeze networks if trainable == False
            # # Return network
            # # FIXME: This is getting way to difficult!
            # nn.Sequential(net_0, net_1, mu, log_sigma)
            # return mu, log_sigma

            # ===============================
            # END <<<<< Pytorch CODE ========
            # ===============================
        else:
            if reuse is None:
                trainable = True
            else:
                trainable = False

            s_dim = self.s_dim
            a_dim = self.a_dim
            w_a = tf.zeros([s_dim, a_dim])
            print(w_a)

            with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):

                # s = self.normalize_input(s)
                batch_size = tf.shape(s)[0]
                squash_bijector = (SquashBijector())
                base_distribution = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.a_dim), scale_diag=tf.ones(self.a_dim))
                epsilon = base_distribution.sample(batch_size)
                ## Construct the feedforward action
                n1 = self.network_structure['actor'][0]
                n2 = self.network_structure['actor'][1]

                net_0 = tf.layers.dense(s, n1, activation=tf.nn.relu, name='l1', trainable=trainable)#原始是30
                net_1 = tf.layers.dense(net_0, n2, activation=tf.nn.relu, name='l4', trainable=trainable)  # 原始是30
                mu = tf.layers.dense(net_1, self.a_dim, activation= None, name='a', trainable=trainable)
                log_sigma = tf.layers.dense(net_1, self.a_dim, None, trainable=trainable)


                # log_sigma = tf.layers.dense(s, self.a_dim, None, trainable=trainable)


                log_sigma = tf.clip_by_value(log_sigma, *SCALE_DIAG_MIN_MAX)
                sigma = tf.exp(log_sigma)

                bijector = tfp.bijectors.Affine(shift=mu, scale_diag=sigma)
                raw_action = bijector.forward(epsilon)
                clipped_a = squash_bijector.forward(raw_action)

                ## Construct the distribution
                bijector = tfp.bijectors.Chain((
                    squash_bijector,
                    tfp.bijectors.Affine(
                        shift=mu,
                        scale_diag=sigma),
                ))
                distribution = tfp.distributions.ConditionalTransformedDistribution(
                        distribution=base_distribution,
                        bijector=bijector)

                clipped_mu = squash_bijector.forward(mu)

            return clipped_a, clipped_mu, distribution

    def _build_l(self, s, a, reuse=None, custom_getter=None):
        if USE_PYTORCH:

            # ===============================
            # BEGIN >>> Pytorch CODE ========
            # ===============================

            # Get action and observation space size
            s_dim = self.s_dim
            a_dim = self.a_dim

            # Create and return Squashed Gaussian actor
            LC = MLPLFunction(s_dim, a_dim, self.network_structure)
            # l = LC(torch.cat([s.unsqueeze(0),s.unsqueeze(0)]), a) # test network
            return LC

            # # Get hidden layer size
            # n1 = self.network_structure['critic'][0]

            # # Create hidden layers of the Lyapunov network
            # # FIXME: Why so low level. This can be done using linear layers?
            # # DEBUG: This should be similar right weighted sum as in a linear layer
            # layers = []
            # w1_s = torch.randn((self.s_dim, n1), requires_grad=True)
            # w1_a = torch.randn((self.a_dim, n1), requires_grad=True)
            # b1 = torch.randn((1, n1), requires_grad=True)
            # net_0 = F.relu(torch.matmul(s, w1_s) + torch.matmul(a, w1_a) + b1)
            # layers.append(net_0)
            # for i in range(1, len(self.network_structure['critic'])):
            #     n = self.network_structure['critic'][i]
            #     layers += [nn.Linear(self.network_structure['critic'][i], n), nn.ReLU()]

            # # Create Output layer
            # torch.sum(torch.square(layers[-1]), dim=1)

            # # Return network
            # return nn.Sequential(*layers)

            # ===============================
            # END <<<<< Pytorch CODE ========
            # ===============================
        else:
            trainable = True if reuse is None else False

            with tf.variable_scope('Lyapunov', reuse=reuse, custom_getter=custom_getter):
                n1 = self.network_structure['critic'][0]

                layers = []
                w1_s = tf.get_variable('w1_s', [self.s_dim, n1], trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n1], trainable=trainable)
                b1 = tf.get_variable('b1', [1, n1], trainable=trainable)
                net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
                layers.append(net_0)
                for i in range(1, len(self.network_structure['critic'])):
                    n = self.network_structure['critic'][i]
                    layers.append(tf.layers.dense(layers[i-1], n, activation=tf.nn.relu, name='l'+str(i+1), trainable=trainable))

                return tf.expand_dims(tf.reduce_sum(tf.square(layers[-1]), axis=1),axis=1)  # Q(s,a)

    def save_result(self, path):
        if USE_PYTORCH:
            # ===============================
            # BEGIN >>> Pytorch CODE ========
            # ===============================
            save_path = os.path.abspath(path + "/policy/model.pth")

            # Create folder if not exist
            if osp.exists(os.path.dirname(save_path)):
                print(
                    "Warning: Log dir %s already exists! Storing info there anyway."
                    % os.path.dirname(save_path)
                )
            else:
                os.makedirs(os.path.dirname(save_path))

            # Create models state dict and save
            models_state_save_dict = {
                'ga_state_dict': self.ga.state_dict(),
                'lc_state_dict': self.lc.state_dict(),
                'ga_targ_state_dict': self.ga_.state_dict(),
                'lc_targ_state_dict': self.lc_.state_dict(),
                'lya_ga_targ_state_dict': self.lya_ga_.state_dict(),
                'lya_lc_state_dict': self.lya_lc_.state_dict(),
                'log_alpha': self.log_alpha,
                'log_labda': self.log_labda,
            }
            torch.save(models_state_save_dict, save_path)
            # ===============================
            # END <<<<< Pytorch CODE ========
            # ===============================
        else:
            save_path = self.saver.save(self.sess, path + "/policy/model.ckpt")
        print("Save to path: ", save_path)

    def restore(self, path):
        if USE_PYTORCH:
            load_path = os.path.abspath(path + "/model.pth")
            # ===============================
            # BEGIN >>> Pytorch CODE ========
            # ===============================

            # Load the model state
            try:
                models_state_dict = torch.load(load_path)
            except NotADirectoryError:
                success_load = False
                return success_load

            # Restore network parameters
            # Question: Do I restore everything correctly?
            self.ga.load_state_dict(models_state_dict["ga_state_dict"])
            self.lc.load_state_dict(models_state_dict["lc_state_dict"])
            self.ga_.load_state_dict(models_state_dict["ga_targ_state_dict"])
            self.lc_.load_state_dict(models_state_dict["lc_targ_state_dict"])
            self.lya_ga_.load_state_dict(models_state_dict["lya_ga_targ_state_dict"])
            self.lya_lc_.load_state_dict(models_state_dict["lya_lc_state_dict"])
            self.log_alpha = models_state_dict["log_alpha"]
            self.log_labda = models_state_dict["log_labda"]
            # Return result
            success_load = True
            return success_load
            # ===============================
            # END <<<<< Pytorch CODE ========
            # ===============================
        else:
            model_file = tf.train.latest_checkpoint(path+'/')
            if model_file is None:
                success_load = False
                return success_load
            self.saver.restore(self.sess, model_file)
            success_load = True
            return success_load

def train(variant):
    env_name = variant['env_name']
    env = get_env_from_name(env_name)

    env_params = variant['env_params']

    max_episodes = env_params['max_episodes']
    max_ep_steps = env_params['max_ep_steps']
    max_global_steps = env_params['max_global_steps']
    store_last_n_paths = variant['num_of_training_paths']
    evaluation_frequency = variant['evaluation_frequency']

    policy_params = variant['alg_params']
    policy_params['network_structure'] = env_params['network_structure']



    min_memory_size = policy_params['min_memory_size']
    steps_per_cycle = policy_params['steps_per_cycle']
    train_per_cycle = policy_params['train_per_cycle']
    batch_size = policy_params['batch_size']

    lr_a, lr_c, lr_l = policy_params['lr_a'], policy_params['lr_c'], policy_params['lr_l']
    lr_a_now = lr_a  # learning rate for actor
    lr_c_now = lr_c  # learning rate for critic
    lr_l_now = lr_l  # learning rate for critic

    if 'Fetch' in env_name or 'Hand' in env_name:
        s_dim = env.observation_space.spaces['observation'].shape[0]\
                + env.observation_space.spaces['achieved_goal'].shape[0]+ \
                env.observation_space.spaces['desired_goal'].shape[0]
    else:
        s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    # if disturber_params['process_noise']:
    #     d_dim = disturber_params['noise_dim']
    # else:
    #     d_dim = env_params['disturbance dim']

    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low
    policy = LAC(a_dim,s_dim, policy_params)

    pool_params = {
        's_dim': s_dim,
        'a_dim': a_dim,
        'd_dim': 1,
        'store_last_n_paths': store_last_n_paths,
        'memory_capacity': policy_params['memory_capacity'],
        'min_memory_size': policy_params['min_memory_size'],
        'history_horizon': policy_params['history_horizon'],
        'finite_horizon':policy_params['finite_horizon']
    }
    if 'value_horizon' in policy_params.keys():
        pool_params.update({'value_horizon': policy_params['value_horizon']})
    else:
        pool_params['value_horizon'] = None
    pool = Pool(pool_params)
    # For analyse
    Render = env_params['eval_render']

    # Training setting
    t1 = time.time()
    global_step = 0
    last_training_paths = deque(maxlen=store_last_n_paths)
    training_started = False

    log_path = variant['log_path']
    logger.configure(dir=log_path, format_strs=['csv'])
    logger.logkv('tau', policy_params['tau'])

    logger.logkv('alpha3', policy_params['alpha3'])
    logger.logkv('batch_size', policy_params['batch_size'])
    logger.logkv('target_entropy', policy.target_entropy)

    for i in range(max_episodes):

        current_path = {'rewards': [],
                        'a_loss': [],

                        'alpha': [],

                        'lambda': [],
                        'lyapunov_error': [],
                        'entropy': [],

                        }

        if global_step > max_global_steps:
            break

        s = env.reset()
        if 'Fetch' in env_name or 'Hand' in env_name:
            s = np.concatenate([s[key] for key in s.keys()])

        for j in range(max_ep_steps):
            if Render:
                env.render()

            a = policy.choose_action(s)
            if USE_PYTORCH:
                # ===============================
                # BEGIN >>> Pytorch CODE ========
                # ===============================
                action = a_lowerbound + (a + 1.) * (a_upperbound - a_lowerbound) / 2
                # ===============================
                # END <<<<< Pytorch CODE ========
                # ===============================
            else:
                action = a_lowerbound + (a + 1.) * (a_upperbound - a_lowerbound) / 2
            # action = a

            # Run in simulator
            disturbance_input = np.zeros([a_dim + s_dim])

            s_, r, done, info = env.step(action)

            if 'Fetch' in env_name or 'Hand' in env_name:
                s_ = np.concatenate([s_[key] for key in s_.keys()])
                if info['done'] > 0:
                    done = True

            if training_started:
                global_step += 1

            if j == max_ep_steps - 1:
                done = True

            terminal = 1. if done else 0.
            pool.store(s, a, np.zeros([1]), np.zeros([1]), r, terminal, s_)
            # policy.store_transition(s, a, disturbance, r,0, terminal, s_)

            if pool.memory_pointer > min_memory_size and global_step % steps_per_cycle == 0:
                training_started = True

                for _ in range(train_per_cycle):
                    batch = pool.sample(batch_size)
                    labda, alpha, l_loss, entropy, a_loss = policy.learn(lr_a_now, lr_c_now, lr_l_now, lr_a, batch)

            if training_started:
                current_path['rewards'].append(r)
                current_path['lyapunov_error'].append(l_loss)
                current_path['alpha'].append(alpha)
                current_path['lambda'].append(labda)
                current_path['entropy'].append(entropy)
                current_path['a_loss'].append(a_loss)



            if training_started and global_step % evaluation_frequency == 0 and global_step > 0:

                logger.logkv("total_timesteps", global_step)

                training_diagnotic = evaluate_training_rollouts(last_training_paths)
                if training_diagnotic is not None:
                    if variant['num_of_evaluation_paths'] > 0:
                        eval_diagnotic = training_evaluation(variant, env, policy)
                        [logger.logkv(key, eval_diagnotic[key]) for key in eval_diagnotic.keys()]
                        training_diagnotic.pop('return')
                    [logger.logkv(key, training_diagnotic[key]) for key in training_diagnotic.keys()]
                    logger.logkv('lr_a', lr_a_now)
                    logger.logkv('lr_c', lr_c_now)
                    logger.logkv('lr_l', lr_l_now)

                    string_to_print = ['time_step:', str(global_step), '|']
                    if variant['num_of_evaluation_paths'] > 0:
                        [string_to_print.extend([key, ':', str(eval_diagnotic[key]), '|'])
                         for key in eval_diagnotic.keys()]
                    [string_to_print.extend([key, ':', str(round(training_diagnotic[key], 2)) , '|'])
                     for key in training_diagnotic.keys()]
                    print(''.join(string_to_print))

                logger.dumpkvs()
            # 状态更新
            s = s_

            # OUTPUT TRAINING INFORMATION AND LEARNING RATE DECAY
            if done:
                if training_started:
                    last_training_paths.appendleft(current_path)

                frac = 1.0 - (global_step - 1.0) / max_global_steps
                lr_a_now = lr_a * frac  # learning rate for actor
                lr_c_now = lr_c * frac  # learning rate for critic
                lr_l_now = lr_l * frac  # learning rate for critic

                break
    policy.save_result(log_path)

    print('Running time: ', time.time() - t1)
    return
