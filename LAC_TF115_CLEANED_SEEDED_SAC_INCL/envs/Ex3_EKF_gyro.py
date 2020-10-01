# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 21:09:14 2020

@author: tang
"""


import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from scipy.stats import truncnorm
import csv
import matplotlib.pyplot as plt


# This example is the RL based filter
def omega_t_sim(t):
    # input = 0*np.cos(t) * self.dt
    # 额外给一个准确的角速度（带偏移+噪声），然后由此仿真出来acc，gyro，mag的测量值
    # 1. simulate the true angular velocity
    a=200
    b=250
    c=250
    d=300
    if a<= t < (a+b)/2:
        omega_1 = 2*pow((t-a)/(b-a),2) # the real angular velocity, in q_pred direction
        omega_2 = 0  # the real angular velocity, in y direction
        omega_3 = 0  # the real angular velocity, in z direction
    elif (a+b)/2 <= t < b:
        omega_1 = 1 - 2*pow((t-b)/(b-a),2)
        omega_2 = 0
        omega_3 = 0 
    elif b <=t < c:
        omega_1 = 1
        omega_2 = 0
        omega_3 = 0  
    elif c<= t <(c+d)/2:
        omega_1 = 1-2*pow((t-c)/(d-c),2)
        omega_2 = 0
        omega_3 = 0   
    elif (c+d)/2 <= t < d:
        omega_1 = 2*pow((t-d)/(d-c),2)
        omega_2 = 0
        omega_3 = 0
    else :
        omega_1 = 0
        omega_2 = 0
        omega_3 = 0
    omega = np.array([[omega_1], [omega_2], [omega_3]])*0.07
    return omega

def omega_t_sim1(t):
    # input = 0*np.cos(t) * self.dt
    # 额外给一个准确的角速度（带偏移+噪声），然后由此仿真出来acc，gyro，mag的测量值
    # 1. simulate the true angular velocity
    if (6 * math.pi / 0.2) < t <= ( 7* math.pi / 0.2):
        omega_1 = 0 # the real angular velocity, in q_pred direction
        omega_2 = 0  # the real angular velocity, in y direction
        omega_3 = 0  # the real angular velocity, in z direction
    else:
        omega_1 = 0
        omega_2 = 0
        omega_3 = 0
    omega = np.array([[omega_1], [omega_2], [omega_3]])
    return omega

def omega_t_sim2(t):
    # input = 0*np.cos(t) * self.dt
    # 额外给一个准确的角速度（带偏移+噪声），然后由此仿真出来acc，gyro，mag的测量值
    # 1. simulate the true angular velocity
    if (6 * math.pi / 0.2) < t <= ( 7* math.pi / 0.2):
        omega_1 = np.sin((0.2 * t)) *0.4 # the real angular velocity, in q_pred direction
        omega_2 = 0  # the real angular velocity, in y direction
        omega_3 = 0  # the real angular velocity, in z direction
    else:
        omega_1 = 0
        omega_2 = 0
        omega_3 = 0
    omega = np.array([[omega_1], [omega_2], [omega_3]])
    return omega

def omega_t_sim4(t):
    # input = 0*np.cos(t) * self.dt
    # 额外给一个准确的角速度（带偏移+噪声），然后由此仿真出来acc，gyro，mag的测量值
    # 1. simulate the true angular velocity
    if (0.1 * math.pi / 0.2) < t <= (3 * math.pi / 0.2):
        omega_1 = np.sin((0.2 * t))  # the real angular velocity, in q_pred direction
        omega_2 = 0  # the real angular velocity, in y direction
        omega_3 = 0  # the real angular velocity, in z direction
    elif (19 * math.pi) < t <= (28 * math.pi):
        w_norm = np.sin(t)
        R = [[-0.1469, 0.3804, -0.9131], [-0.0470, 0.9194, 0.3906], [0.9880, 0.1003, -0.1172]]
        R = np.mod(R, 2)  # normalize rotation matrix
        w_vector = R * [np.sin(t), 0, 0]
        omega_1 = w_vector[0][0]
        omega_2 = w_vector[1][0]
        omega_3 = w_vector[2][0]
    elif (21 * math.pi * 1.5) < t <= (25 * math.pi * 1.5):
        omega_1 = np.sin(t / 1.5)
        omega_2 = 0
        omega_3 = 0
    elif (25 * math.pi * 1.5) <= t <= (29 * math.pi * 1.5):
        omega_1 = 0
        omega_2 = np.sin(t / 1.5)
        omega_3 = 0
    elif (29 * math.pi * 1.5) < t < (33 * math.pi * 1.5):
        omega_1 = 0
        omega_2 = 0
        omega_3 = np.sin(t / 1.5)
    else:
        omega_1 = 0
        omega_2 = 0
        omega_3 = 0
    omega = np.array([[omega_1], [omega_2], [omega_3]])
    return omega

def omega_t_sim5(t):
    # input = 0*np.cos(t) * self.dt
    # 额外给一个准确的角速度（带偏移+噪声），然后由此仿真出来acc，gyro，mag的测量值
    # 1. simulate the true angular velocity
    if (2 * math.pi /0.05) < t <= (6 * math.pi /0.05):
        omega_1 = np.sin((t*0.05))  # the real angular velocity, in q_pred direction
        omega_2 = 0  # the real angular velocity, in y direction
        omega_3 = 0  # the real angular velocity, in z direction
    else:
        omega_1 = 0
        omega_2 = 0
        omega_3 = 0
    omega = np.array([[omega_1], [omega_2], [omega_3]])
    return omega       

def q_t_dot(t, q_t):
    omega_t = omega_t_sim(t)
    result = np.dot(0.5*quatLeftMulMat(q_t), quatPure2Q(np.hstack(omega_t)))
    return result

def diata_q_dot(t, diata_q, omege_noise_t):
    omega_t = omega_t_sim(t) 
    omega_obs = omega_t + omege_noise_t
    # diata_omega = omege_noise_t
    # aa = np.array([0, -diata_omega[0][0], -diata_omega[1][0], -diata_omega[2][0]])
    # bb = np.hstack([diata_omega, -vecCross(np.hstack(2*omega_t+diata_omega))])
    M = quatRightMulMat(np.hstack(quatPure2Q(np.hstack(omega_t)))) - quatLeftMulMat(np.hstack(quatPure2Q(np.hstack(omega_obs))))
    result = np.dot( 0.5*M,np.vstack(diata_q))
    return result    

def Log(q):
    qw = q[0]
    qv = q[1:4]
    if(np.linalg.norm(qv)!=0):
        theta = np.arctan2(np.linalg.norm(qv),qw);
        u = qv/np.linalg.norm(qv);
        output = u*theta;
    else:
        output = np.array([0,0,0]);
    return output

def exp(omega):
    theta = np.linalg.norm(omega)
    w_n = np.linalg.norm(omega)
    if (w_n != 0):
        vector = omega / np.linalg.norm(omega)
    else:
        vector = np.array([[0], [0], [0]])
    xyz = vector * np.sin(theta)
    exp_w = np.array([np.cos(theta), xyz[0][0], xyz[1][0], xyz[2][0]])
    return exp_w


def quatRightMulMat(q):
    matrix = np.identity(4) * q[0]
    matrix[0, 0] = q[0]
    matrix[1, 0] = q[1]
    matrix[2, 0] = q[2]
    matrix[3, 0] = q[3]
    matrix[0, 1] = -q[1]
    matrix[0, 2] = -q[2]
    matrix[0, 3] = -q[3]
    matrix[2, 1] = -q[3]
    matrix[1, 2] = q[3]
    matrix[3, 1] = q[2]
    matrix[1, 3] = -q[2]
    matrix[3, 2] = -q[1]
    matrix[2, 3] = q[1]
    return matrix


def quatLeftMulMat(q):
    matrix = np.identity(4) * q[0]
    matrix[0, 0] = q[0]
    matrix[1, 0] = q[1]
    matrix[2, 0] = q[2]
    matrix[3, 0] = q[3]
    matrix[0, 1] = -q[1]
    matrix[0, 2] = -q[2]
    matrix[0, 3] = -q[3]
    matrix[2, 1] = q[3]
    matrix[1, 2] = -q[3]
    matrix[3, 1] = -q[2]
    matrix[1, 3] = q[2]
    matrix[3, 2] = q[1]
    matrix[2, 3] = -q[1]
    return matrix


def quatConj(q):
    p = np.array([[q[0]], [-q[1]], [-q[2]], [-q[3]]])
    return p


def quatPure2Q(v3):
    q = np.array([[0], [v3[0]], [v3[1]], [v3[2]]])
    return q


def vecCross(v3):
    matrix = np.array([[0, -v3[2], v3[1]], [v3[2], 0, -v3[0]], [-v3[1], v3[0], 0]])
    return matrix


def quat2eul(q):
    qw = q[0];
    qx = q[1];
    qy = q[2];
    qz = q[3];
    aSinInput = -2 * (qx * qz - qw * qy)
    if aSinInput > 1:
        aSinInput = 1
    if aSinInput < -1:
        aSinInput = -1
    eul_1 = np.arctan2(2 * (qx * qy + qw * qz), qw * qw + qx * qx - qy * qy - qz * qz)
    eul_2 = np.arcsin(aSinInput)
    eul_3 = np.arctan2(2 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz)
    return np.array([eul_1, eul_2, eul_3])


class Ex3_EKF(gym.Env):

    def __init__(self):

        self.choice = 'otherCase'  
  
        self.t = 0
        self.dt = 0.5

        self.noise_gyro_bias = np.array(
            [[0.0], [0.0], [0.0]])  # a small changing bias in angular velocity, the initial value is 0.0001
        self.cov_noise_gyro_bias = np.array([[0.00000, 0, 0], [0, 0.00000, 0], [0, 0, 0.00000]])
        # the true value of self.noise_gyro_bias & self.cov_noise_gyro_bias is unknown, 
        # and the measurement of acc&mag is used to compensate the uncertain drift caused by noise_gyro_bias
        self.cov_noise_i = np.array([[.00001, 0, 0], [0, .00001, 0], [0, 0, .00001]])
        self.cov_a = np.array([[0.0005, 0, 0], [0, 0.0005, 0], [0, 0, 0.0005]])
        self.cov_mag = np.array([[0.0003, 0, 0], [0, 0.0003, 0], [0, 0, 0.0003]])
        
        # the true initial pose (from the sensor frame to the world frame)
        self.q_t = np.random.uniform([-1,-1,-1,-1], [1,1,1,1])
        self.q_t = self.q_t /np.linalg.norm(self.q_t) 

        # displacement limit set to be [-high, high]
        high = np.array([10000, 10000, 10000])

        self.action_space = spaces.Box(low=np.array(
            [-10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10.])*0.01,
                                       high=np.array(
                                           [10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,
                                            10., 10., 10.])*0.01,
                                       dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.output = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step_1(self):  # here u1,u2=measurement, which is a result of the action
        train = False
        t = self.t
        omega = omega_t_sim(t) #simulate the trajectory
        
        # 1. update the true pose
        q_t = self.q_t  
        q_t = np.dot(quatLeftMulMat(q_t), exp(0.5 * self.dt * omega).T)
        q_t = q_t / np.linalg.norm(q_t)
        self.q_t  = q_t

        # 2. simulate the sensor measurements
        # wm=omega+noise_gyro_bias+noise_i; d(noise_gyro_bias)/dt=noise_gyro_bias_var
        # noise_i~N(0,cov_w); noise_gyro_bias_var~N(0,cov_noise_gyro_bias)
        noise_gyro_bias_var = np.random.multivariate_normal([0, 0, 0], self.cov_noise_gyro_bias).flatten()
        noise_gyro_bias_t = self.noise_gyro_bias + np.array(
            [[noise_gyro_bias_var[0]], [noise_gyro_bias_var[1]], [noise_gyro_bias_var[2]]])
        noise_i = np.random.multivariate_normal([0, 0, 0], self.cov_noise_i).flatten()
        omega_obs = omega + noise_gyro_bias_t + np.array([[noise_i[0]], [noise_i[1]], [noise_i[2]]])
        # omega_obs = omega
        self.omega = omega_obs

        # assume the pure acc goes up in proportion to omega
        acc_m_q = np.dot(np.dot(quatLeftMulMat(quatConj(self.q_t)), quatRightMulMat(self.q_t)), quatPure2Q([0, 0, -1]))
        acc_i = np.random.multivariate_normal([0., 0., 0.], self.cov_a).flatten() + np.dot([1, 1, 1],
                                                                                          np.linalg.norm(omega))
        acc_m = acc_m_q[1:4] + np.array([[acc_i[0]], [acc_i[1]], [acc_i[2]]])
        acc_m = acc_m / np.linalg.norm(acc_m)
        self.acc = acc_m
        # assume the dip angle of mag is diata = 30 degree =0.52 rad = (np.pi*30/180)
        mag_m_q = np.dot(np.dot(quatLeftMulMat(quatConj(self.q_t)), quatRightMulMat(self.q_t)),
                         quatPure2Q([np.cos(np.pi * 30 / 180), 0, np.sin(np.pi * 30 / 180)]))
        mag_i = np.random.multivariate_normal([0, 0, 0], self.cov_mag).flatten()
        mag_m = mag_m_q[1:4] + np.array([[mag_i[0]], [mag_i[1]], [mag_i[2]]])
        mag_m = mag_m / np.linalg.norm(mag_m)
        self.mag = mag_m
        self.t = self.t + self.dt
        return q_t
        

    def step_2(self,action,hat_q):

        # 3. update the hat_q and q_pred from the last round
        q_pred = np.dot(quatLeftMulMat(hat_q), exp(0.5 * self.dt * self.omega))
        q_pred = q_pred / np.linalg.norm(q_pred)

        # b_t = np.dot(np.dot(quatLeftMulMat((hat_q_pred)), quatRightMulMat(quatConj(hat_q_pred))),
        #                  quatPure2Q([mag_m[0][0], mag_m[1][0], mag_m[2][0]]))

        # 4. calculate hat_y
        y = np.vstack((self.acc, self.mag))
        hat_y_acc_q = np.dot(np.dot(quatLeftMulMat(quatConj(q_pred)), quatRightMulMat(q_pred)), quatPure2Q([0, 0, -1]))
        mag_m_q = np.dot(np.dot(quatLeftMulMat(quatConj(q_pred)), quatRightMulMat(q_pred)),
                         quatPure2Q([np.cos(np.pi * 30 / 180), 0, np.sin(np.pi * 30 / 180)]))
        hat_y = np.vstack((hat_y_acc_q[1:4], mag_m_q[1:4]))

        # y0-y2  重力加速度在世界坐标下的方向
        # y3-y5  磁场强度方向在世界坐标系下的方向
        # q0-q3  四元数，当前传感器相对于世界坐标系的旋转姿态，角度各种耦合

        hat_eta = np.array([0.0,0.0,0.0])
        
        u_11, u_21, u_31, u_41, u_51, u_61, \
        u_12, u_22, u_32, u_42, u_52, u_62, \
        u_13, u_23, u_33, u_43, u_53, u_63 = action

        hat_eta[0] = u_11 * (y[0][0] - hat_y[0]) + u_21 * (y[1][0] - hat_y[1]) + u_31 * (y[2][0] - hat_y[2]) \
                   + u_41 * (y[3][0] - hat_y[3]) + u_51 * (y[4][0] - hat_y[4]) + u_61 * (y[5][0] - hat_y[5])
        hat_eta[1] =  u_12 * (y[0][0] - hat_y[0]) + u_22 * (y[1][0] - hat_y[1]) + u_32 * (y[2][0] - hat_y[2]) \
                   + u_42 * (y[3][0] - hat_y[3]) + u_52 * (y[4][0] - hat_y[4]) + u_62 * (y[5][0] - hat_y[5])
        hat_eta[2] = u_13 * (y[0][0] - hat_y[0]) + u_23 * (y[1][0] - hat_y[1]) + u_33 * (y[2][0] - hat_y[2]) \
                   + u_43 * (y[3][0] - hat_y[3]) + u_53 * (y[4][0] - hat_y[4]) + u_63 * (y[5][0] - hat_y[5])
        hat_delta_q = exp(0.5*np.vstack(hat_eta))
        
        # 5. relinearize
        hat_q = np.inner(quatLeftMulMat(hat_delta_q), q_pred)
        hat_q = hat_q / np.linalg.norm(hat_q)
        
#        aaa = 2.0* Log(np.inner(quatRightMulMat(quatConj(self.q_pred_init)),self.q_t_init))
#        bbb = 2 / self.dt * Log(np.inner(quatLeftMulMat(q_t_lastStep),np.hstack(quatConj(np.inner(quatLeftMulMat(q_pred),hat_delta_q))))) - np.hstack(omega)
#        ccc = y-hat_y
#        cost = np.linalg.norm(aaa) + np.linalg.norm(bbb) + np.linalg.norm(ccc) 
        aaa = 2.0* Log(np.inner(quatRightMulMat(quatConj(hat_q)), np.array(self.q_t)))
        # cost = np.linalg.norm(aaa) 
        # gamma = 1.005
        # cost = np.linalg.norm(aaa) * np.power( gamma,t)
        # cost = np.linalg.norm(aaa) * np.log(self.t+1)
        cost = np.linalg.norm(aaa)
        
        
        # if cost > (3* np.log(self.t+1)):
        if cost > (100):
            done = True
            
        else:
            done = False

        # eul_hat_q = quat2eul(hat_q) 
        # eul_q_t = quat2eul(q_t)
        return hat_eta, cost, done, dict(reference=y[0],
                                        state_of_interest=np.array([hat_q[0], hat_q[1], hat_q[2], hat_q[3]]))
        # return hat_eta, cost, done, dict(reference=y[0],
        #                                 state_of_interest=np.array([hat_q[0], hat_q[1], hat_q[2], hat_q[3], q_t[0], q_t[1], q_t[2], q_t[3], hat_eta[0], hat_eta[1],hat_eta[2],cost]))




    def step(self, action):  # here u1,u2=measurement, which is a result of the action
        train = False

        u_11, u_21, u_31, u_41, u_51, u_61, \
        u_12, u_22, u_32, u_42, u_52, u_62, \
        u_13, u_23, u_33, u_43, u_53, u_63 = action

        t = self.t
        omega = omega_t_sim(t) #simulate the trajectory
        
        # 1. update the true pose
        q_t = self.q_t  
        q_t_lastStep = self.q_t  
        q_t = np.dot(quatLeftMulMat(q_t), exp(0.5 * self.dt * omega).T)
        q_t = q_t / np.linalg.norm(q_t)
        self.q_t  = q_t

        # 2. simulate the sensor measurements
        # wm=omega+noise_gyro_bias+noise_i; d(noise_gyro_bias)/dt=noise_gyro_bias_var
        # noise_i~N(0,cov_w); noise_gyro_bias_var~N(0,cov_noise_gyro_bias)
        noise_gyro_bias_var = np.random.multivariate_normal([0, 0, 0], self.cov_noise_gyro_bias).flatten()
        noise_gyro_bias_t = self.noise_gyro_bias + np.array(
            [[noise_gyro_bias_var[0]], [noise_gyro_bias_var[1]], [noise_gyro_bias_var[2]]])
        noise_i = np.random.multivariate_normal([0, 0, 0], self.cov_noise_i).flatten()
        omega_obs = omega + noise_gyro_bias_t + np.array([[noise_i[0]], [noise_i[1]], [noise_i[2]]])
        # omega_obs = omega

        # assume the pure acc goes up in proportion to omega
        acc_m_q = np.dot(np.dot(quatLeftMulMat(quatConj(self.q_t)), quatRightMulMat(self.q_t)), quatPure2Q([0, 0, -1]))
        acc_i = np.random.multivariate_normal([0., 0., 0.], self.cov_a).flatten()
        acc_m = acc_m_q[1:4] + np.array([[acc_i[0]], [acc_i[1]], [acc_i[2]]])
        # acc_m = acc_m_q[1:4]
        acc_m = acc_m / np.linalg.norm(acc_m)
        # assume the dip angle of mag is diata = 30 degree =0.52 rad = (np.pi*30/180)
        mag_m_q = np.dot(np.dot(quatLeftMulMat(quatConj(self.q_t)), quatRightMulMat(self.q_t)),
                         quatPure2Q([np.cos(np.pi * 30 / 180), 0, np.sin(np.pi * 30 / 180)]))
        mag_i = np.random.multivariate_normal([0, 0, 0], self.cov_mag).flatten()
        mag_m = mag_m_q[1:4] + np.array([[mag_i[0]], [mag_i[1]], [mag_i[2]]])
        # mag_m = mag_m_q[1:4]
        mag_m = mag_m / np.linalg.norm(mag_m)


        # 3. update the hat_q and q_pred from the last round
        hat_q = self.hat_q
        q_pred = np.dot(quatLeftMulMat(hat_q), exp(0.5 * self.dt * omega_obs))
        q_pred = q_pred / np.linalg.norm(q_pred)

        # b_t = np.dot(np.dot(quatLeftMulMat((hat_q_pred)), quatRightMulMat(quatConj(hat_q_pred))),
        #                  quatPure2Q([mag_m[0][0], mag_m[1][0], mag_m[2][0]]))

        # 4. calculate hat_y
        y = np.vstack((acc_m, mag_m))
        hat_y_acc_q = np.dot(np.dot(quatLeftMulMat(quatConj(q_pred)), quatRightMulMat(q_pred)), quatPure2Q([0, 0, -1]))
        mag_m_q = np.dot(np.dot(quatLeftMulMat(quatConj(q_pred)), quatRightMulMat(q_pred)),
                         quatPure2Q([np.cos(np.pi * 30 / 180), 0, np.sin(np.pi * 30 / 180)]))
        hat_y = np.vstack((hat_y_acc_q[1:4], mag_m_q[1:4]))
        # y0-y2  重力加速度在世界坐标下的方向
        # y3-y5  磁场强度方向在世界坐标系下的方向
        # q0-q3  四元数，当前传感器相对于世界坐标系的旋转姿态，角度各种耦合

        hat_eta = np.array([0.0,0.0,0.0])

        hat_eta[0] = u_11 * (y[0][0] - hat_y[0]) + u_21 * (y[1][0] - hat_y[1]) + u_31 * (y[2][0] - hat_y[2]) \
                   + u_41 * (y[3][0] - hat_y[3]) + u_51 * (y[4][0] - hat_y[4]) + u_61 * (y[5][0] - hat_y[5])
        hat_eta[1] =  u_12 * (y[0][0] - hat_y[0]) + u_22 * (y[1][0] - hat_y[1]) + u_32 * (y[2][0] - hat_y[2]) \
                   + u_42 * (y[3][0] - hat_y[3]) + u_52 * (y[4][0] - hat_y[4]) + u_62 * (y[5][0] - hat_y[5])
        hat_eta[2] = u_13 * (y[0][0] - hat_y[0]) + u_23 * (y[1][0] - hat_y[1]) + u_33 * (y[2][0] - hat_y[2]) \
                   + u_43 * (y[3][0] - hat_y[3]) + u_53 * (y[4][0] - hat_y[4]) + u_63 * (y[5][0] - hat_y[5])
        hat_delta_q = exp(0.5*np.vstack(hat_eta))
        
        # 5. relinearize
        hat_q = np.inner(quatLeftMulMat(hat_delta_q), q_pred)
        hat_q = hat_q/ np.linalg.norm(hat_q)
        
        aaa = 2.0* Log(np.inner(quatRightMulMat(quatConj(self.q_pred_init)),self.q_t_init))
        # bbb = 2 / self.dt * Log(np.inner(quatLeftMulMat(q_t_lastStep),np.hstack(quatConj(np.inner(quatLeftMulMat(q_pred),hat_delta_q))))) - np.hstack(omega)
        bbb = 2 / self.dt * Log(np.inner(quatLeftMulMat(quatConj(q_t_lastStep)),np.hstack((hat_q)))) - np.hstack(omega)
        ccc = y-hat_y
        cost = np.linalg.norm(aaa)**2 + np.linalg.norm(bbb)**2 + np.linalg.norm(ccc)**2

        # cost = np.sum(abs(aaa)) ** 0.2 + np.sum(abs(bbb)) ** 0.2 + np.sum(abs(ccc)) ** 0.2
        # cost = ( np.linalg.norm(bbb) )*( np.linalg.norm(bbb) )
        # aaa = 2.0* Log(np.inner(quatRightMulMat(quatConj(hat_q)), np.array(self.q_t)))
        # cost = np.linalg.norm(aaa) 
        # gamma = 1.005
        # cost = np.linalg.norm(aaa) * np.power( gamma,t)
        # cost = np.linalg.norm(aaa) * np.log(t+1)
        
        
        if cost > (100):
            done = True
            
        else:
            done = False

        
        # 6. update new for next round
        self.hat_q = hat_q
        self.state = hat_eta
        self.t = self.t + self.dt

        # eul_hat_q = quat2eul(hat_q) 
        # eul_q_t = quat2eul(q_t)
        if self.choice == 'saveData':
            return omega_obs,acc_m,mag_m,q_t,hat_q, cost, done, dict(reference=y[0],
                                        state_of_interest=np.array([hat_q[1], q_t[1],hat_q[2], q_t[2]]))
        else:
            # return hat_eta, cost, done, dict(reference=y[0],
            #                             state_of_interest=np.array([hat_q[0], hat_q[1],hat_q[2],hat_q[3], q_t[0], q_t[1], q_t[2], q_t[3]]))
            return hat_eta, cost, done, dict(reference=y[0],
                                        state_of_interest=np.array([hat_q[0], hat_q[1], hat_q[2], hat_q[3], q_t[0], q_t[1], q_t[2], q_t[3], hat_eta[0], hat_eta[1],hat_eta[2],cost]))


    def reset(self):
        self.t = 0
        self.q_t = np.random.uniform([-1,-1,-1,-1], [1,1,1,1])# the state q, the initial value can be set randomly
        # self.q_t = np.array([0.6,0.2,0.5,0.4])
        self.q_t = self.q_t /np.linalg.norm(self.q_t)
        self.q_t_init = self.q_t 
        self.hat_q = self.q_t + np.random.normal([ 0,0,0,0], [ 0.1,0.1,0.1,0.1])*0.1
        # self.hat_q = np.random.normal([0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1]) * 0.1
        self.hat_q = self.hat_q /np.linalg.norm(self.hat_q)
        self.q_pred_init = self.hat_q 

        hat_eta = np.random.normal([ 0,0,0], [ 0.1,0.1,0.1])*0.0001
        self.state = hat_eta
        
        if self.choice == 'saveData':
            omega_obs = np.array([[0],[0],[0]])
            acc_m = np.array([[0],[0],[0]])
            mag_m= np.array([[0],[0],[0]])
            return omega_obs,acc_m,mag_m,self.q_t,self.hat_q
        else:
            return hat_eta  # return hat_state

    def render(self, mode='human'):

        return

    
    def saveChoice(self, choiceIn):
        self.choice =choiceIn
        return self.choice



if __name__ == '__main__':
    env = Ex3_EKF()
    T = 3200
    
    # choice = 'saveData'
    choice = 'saveData'
    if env.saveChoice(choice) == 'saveData':  
        path = []
        omega_obs,acc_m,mag_m,q_t,hat_q = env.reset()
        measurement = np.vstack([omega_obs,acc_m,mag_m,np.vstack(q_t),np.vstack(hat_q)])
        measurement = np.hstack(measurement)
        path.append(measurement)
        for i in range(int(T / env.dt)):
            # s, r, info, done = env.step(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            # path.append(s)
            omega_obs,acc_m,mag_m,q_t,hat_q, r, info, done = env.step(np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            measurement = np.vstack([omega_obs,acc_m,mag_m,np.vstack(q_t),np.vstack(hat_q)])
            measurement = np.hstack(measurement)
            path.append(measurement)
        np.savetxt('5.csv', path, delimiter = ',')
    else:
        path = []
        # path2=[]
        t1 = []
        s = env.reset()
        for i in range(int(T / env.dt)):
            # s, r, info, done = env.step(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            # path.append(s)
            hat_q, r, info, done = env.step(np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            path.append(hat_q)
            t1.append(i * env.dt)
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.plot(t1, np.array(path)[:, 0], color='green', label='x0')
        ax.plot(t1, np.array(path)[:, 1], color='yellow', label='x1')
        ax.plot(t1, np.array(path)[:, 2], color='blue', label='x2')
        # ax.plot(t1, np.array(path)[:, 3], color='red', label='x3')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
        plt.show()
        # plt.savefig('1-.eps',format="eps")
        print('done')   
        # fig = plt.figure(figsize=(9, 6))
        # ax = fig.add_subplot(111)
        # ax.plot(t1, np.array(path2)[:, 1], color='green', label='x0')
        # ax.plot(t1, np.array(path2)[:, 1], color='yellow', label='x1')
        # ax.plot(t1, np.array(path2)[:, 2], color='blue', label='x2')
        # ax.plot(t1, np.array(path2)[:, 3], color='red', label='x3')
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
        # plt.show()
        # print('done')
