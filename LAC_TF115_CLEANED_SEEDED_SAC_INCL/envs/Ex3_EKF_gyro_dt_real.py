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

def measrement_real(t, dt, data):

    step= int(t/dt)
    # print(step)
    measurement = data[step]
    measurement.dtype = 'float64'
    q_t = measurement[0:4]
    omega_obs = measurement[4:7]
    omega_obs = np.vstack(omega_obs)
    acc_m = measurement[7:10]
    acc_m = np.vstack(acc_m)
    mag_m = measurement[10:13]
    mag_m = np.vstack(mag_m)
    return q_t,omega_obs,acc_m,mag_m

def omega_t_sim0(t):
    t=t*50*2
    # input = 0*np.cos(t) * self.dt
    # 额外给一个准确的角速度（带偏移+噪声），然后由此仿真出来acc，gyro，mag的测量值
    # 1. simulate the true angular velocity
    a=100
    b=500
    c=500
    d=700
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
    omega = np.array([[omega_1], [omega_2], [omega_3]])*0.35
    return omega

def omega_t_sim1(t):
    # input = 0*np.cos(t) * self.dt
    # 额外给一个准确的角速度（带偏移+噪声），然后由此仿真出来acc，gyro，mag的测量值
    # 1. simulate the true angular velocity
    t=t*50*2-200
    a=100
    b=400
    c=500
    d=700
    if a<= t < (a+b)/2:
        omega_1 = 2*pow((t-a)/(b-a),2) # the real angular velocity, in q_pred direction
        omega_2 = 0  # the real angular velocity, in y direction
        omega_3 = 0.25*(t-a)/a  # the real angular velocity, in z direction
    elif (a+b)/2 <= t < b:
        omega_1 = 1 - 2*pow((t-b)/(b-a),2)
        omega_2 = 0
        omega_3 = (-t+b)/b
    elif b <=t < c:
        omega_1 = 1
        omega_2 = np.sin(1/(b-d)*(t-b)*np.pi)
        omega_3 = 0
    elif c<= t <(c+d)/2:
        omega_1 = 1-2*pow((t-c)/(d-c),2)
        omega_2 = np.sin(1/(b-d)*(t-b)*np.pi)
        omega_3 = 0
    elif (c+d)/2 <= t < d:
        omega_1 = 2*pow((t-d)/(d-c),2)
        omega_2 = np.sin(1/(b-d)*(t-b)*np.pi)
        omega_3 = 0
    else :
        omega_1 = 0
        omega_2 = 0
        omega_3 = 0
    omega = np.array([[omega_1], [omega_2], [omega_3]])*1
    return omega

def omega_t_sim2(t):
    t=t*50*2
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

def omega_t_sim3(t):
    t=t*50*2
    # input = 0*np.cos(t) * self.dt
    # 额外给一个准确的角速度（带偏移+噪声），然后由此仿真出来acc，gyro，mag的测量值
    # 1. simulate the true angular velocity
    if (0.5 * math.pi / 0.008) < t <= ( 1.5* math.pi / 0.008):
        omega_1 = np.sin((0.008 * t) - 0.5* math.pi)  # the real angular velocity, in q_pred direction
        omega_2 = 0  # the real angular velocity, in y direction
        omega_3 = 0  # the real angular velocity, in z direction
    else:
        omega_1 = 0
        omega_2 = 0
        omega_3 = 0
    omega = np.array([[omega_1], [omega_2], [omega_3]])*0.4
    return omega

def omega_t_sim4(t):
    t=t*50*2
    if (1 * math.pi / 0.02) < t <= (2 * math.pi / 0.02):
        omega_1 = np.sin(t * 0.02)
        omega_2 = 0
        omega_3 = 0
    elif (2 * math.pi / 0.02) <= t <= (3 * math.pi / 0.02):
        omega_1 = 0
        omega_2 = np.sin(t * 0.02)
        omega_3 = 0
    elif (3 * math.pi / 0.02) < t < (4 * math.pi / 0.02):
        omega_1 = 0
        omega_2 = 0
        omega_3 = np.sin(t * 0.02)
    else:
        omega_1 = 0
        omega_2 = 0
        omega_3 = 0
    omega = np.array([[omega_1], [omega_2], [omega_3]])*0.4
    return omega

def omega_t_sim5(t):
    # t=t*100*2
    if (2 * math.pi / 0.02) < t <= (3 * math.pi / 0.02):
        omega_1 = np.sin(t * 0.02)
        omega_2 = 0
        omega_3 = 0
    elif (3 * math.pi / 0.02) <= t <= (4 * math.pi / 0.02):
        omega_1 = np.sin(t * 0.02)
        omega_2 = 0
        omega_3 = 0
    elif (4 * math.pi / 0.02) < t < (5 * math.pi / 0.02):
        omega_1 = np.sin(t * 0.02)
        omega_2 = 0
        omega_3 = 0
    elif (5 * math.pi / 0.02) < t <= (6 * math.pi / 0.02):
        omega_1 = np.sin(t * 0.02)
        omega_2 = 0
        omega_3 = 0
    elif (6 * math.pi / 0.02) <= t <= (7 * math.pi / 0.02):
        omega_1 = np.sin(t * 0.02)
        omega_2 = 0
        omega_3 = 0
    elif (7 * math.pi / 0.02) < t < (8 * math.pi / 0.02):
        omega_1 = np.sin(t * 0.02)
        omega_2 = 0
        omega_3 = 0
    else:
        omega_1 = 0
        omega_2 = 0
        omega_3 = 0
    omega = np.array([[omega_1], [omega_2], [omega_3]])*1
    return omega

def omega_t_sim(t):
    t=t/2.
    # input = 0*np.cos(t) * self.dt
    # 额外给一个准确的角速度（带偏移+噪声），然后由此仿真出来acc，gyro，mag的测量值
    # 1. simulate the true angular velocity
    start = 1
    risingStage = 3
    holding = 7
    a1=start
    b1=start+risingStage
    c1=start+risingStage+holding
    d1=start+2.*risingStage+holding
    a2=start+2.*risingStage+holding
    b2=start+3.*risingStage+holding
    c2=start+3.*risingStage+2.*holding
    d2=start+4.*risingStage+2.*holding
    a3=start+4.*risingStage+2.*holding
    b3=start+5.*risingStage+2.*holding
    c3=start+5.*risingStage+3.*holding
    d3=start+6.*risingStage+3.*holding
    if a1<= t < (a1+b1)/2:
        omega_1 = -2*pow((t-a1)/(b1-a1),2) # the real angular velocity, in q_pred direction
        omega_2 = 0  # the real angular velocity, in y direction
        omega_3 = 0  # the real angular velocity, in z direction
    elif (a1+b1)/2 <= t < b1:
        omega_1 = -(1 - 2*pow((t-b1)/(b1-a1),2))
        omega_2 = 0
        omega_3 = 0
    elif b1 <=t < c1:
        omega_1 = -1
        omega_2 = 0
        omega_3 = 0
    elif c1<= t <(c1+d1)/2:
        omega_1 = -(1-2*pow((t-c1)/(d1-c1),2))
        omega_2 = 0
        omega_3 = 0
    elif (c1+d1)/2 <= t < d1:
        omega_1 = -2*pow((t-d1)/(d1-c1),2)
        omega_2 = 0
        omega_3 = 0
    elif a2<= t < (a2+b2)/2:
        omega_1 = 0 # the real angular velocity, in q_pred direction
        omega_2 = 2*pow((t-a2)/(b2-a2),2)  # the real angular velocity, in y direction
        omega_3 = 0  # the real angular velocity, in z direction
    elif (a2+b2)/2 <= t < b2:
        omega_1 = 0
        omega_2 = (1 - 2*pow((t-b2)/(b2-a2),2))
        omega_3 = 0
    elif b2 <=t < c2:
        omega_1 = 0
        omega_2 = 1
        omega_3 = 0
    elif c2<= t <(c2+d2)/2:
        omega_1 = 0
        omega_2 = (1-2*pow((t-c2)/(d2-c2),2))
        omega_3 = 0
    elif (c2+d2)/2 <= t < d2:
        omega_1 = 0
        omega_2 = 2*pow((t-d2)/(d2-c2),2)
        omega_3 = 0
    elif a3<= t < (a3+b3)/2:
        omega_1 = 0 # the real angular velocity, in q_pred direction
        omega_2 = 0  # the real angular velocity, in y direction
        omega_3 = -2*pow((t-a3)/(b3-a3),2)  # the real angular velocity, in z direction
    elif (a3+b3)/2 <= t < b3:
        omega_1 = 0
        omega_2 = 0
        omega_3 = -(1 - 2*pow((t-b3)/(b3-a3),2))
    elif b3 <=t < c3:
        omega_1 = 0
        omega_2 = 0
        omega_3 = -1
    elif c3<= t <(c3+d3)/2:
        omega_1 = 0
        omega_2 = 0
        omega_3 = -(1-2*pow((t-c3)/(d3-c3),2))
    elif (c3+d3)/2 <= t < d3:
        omega_1 = 0
        omega_2 = 0
        omega_3 = -2*pow((t-d3)/(d3-c3),2)
    else :
        omega_1 = 0
        omega_2 = 0
        omega_3 = 0
    omega = np.array([[omega_1], [omega_2], [omega_3]])*0.2
    return omega

# def q_t_dot(t, q_t):
#     omega_t = omega_t_sim(t)
#     result = np.dot(0.5*quatLeftMulMat(q_t), quatPure2Q(np.hstack(omega_t)))
#     return result

# def diata_q_dot(t, diata_q, omege_noise_t):
#     omega_t = omega_t_sim(t)
#     omega_obs = omega_t + omege_noise_t
#     # diata_omega = omege_noise_t
#     # aa = np.array([0, -diata_omega[0][0], -diata_omega[1][0], -diata_omega[2][0]])
#     # bb = np.hstack([diata_omega, -vecCross(np.hstack(2*omega_t+diata_omega))])
#     M = quatRightMulMat(np.hstack(quatPure2Q(np.hstack(omega_t)))) - quatLeftMulMat(np.hstack(quatPure2Q(np.hstack(omega_obs))))
#     result = np.dot( 0.5*M,np.vstack(diata_q))
#     return result

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


class Ex3_EKF_gyro(gym.Env):

    def __init__(self):

        self.clean =  False
        self.choice = 'otherCase'
        self.realMeasurement = True
        
        # if self.realMeasurement:
        #     # p = r'trainingData.csv'
        #     p = r'D:\reinforcement learning\learning to SLAM\code\LAC_TF2_TORCH_REWRITE-master\LAC_TF115_CLEANED_SEEDED_SAC_INCL\trainingData.csv'
        #     self.readData = np.genfromtxt(p, delimiter=',')

        self.t = 0
        self.dt = 0.01

        self.noise_gyro_bias = np.array(
            [[0.0], [0.0], [0.0]])  # a small changing bias in angular velocity, the initial value is 0.0001
        self.cov_noise_gyro_bias = np.array([[0.00000, 0, 0], [0, 0.00000, 0], [0, 0, 0.00000]])
        # the true value of self.noise_gyro_bias & self.cov_noise_gyro_bias is unknown,
        # and the measurement of acc&mag is used to compensate the uncertain drift caused by noise_gyro_bias

        if not self.clean:
            self.cov_noise_i = np.array([[.0003, 0, 0], [0, .0003, 0], [0, 0, .0003]])
            self.cov_a = np.array([[0.0005, 0, 0], [0, 0.0005, 0], [0, 0, 0.0005]])
            self.cov_mag = np.array([[0.0003, 0, 0], [0, 0.0003, 0], [0, 0, 0.0003]])
        else:
            self.cov_noise_i = np.array([[.00000, 0, 0], [0, .0000, 0], [0, 0, .0000]])
            self.cov_a = np.array([[0.000, 0, 0], [0, 0.000, 0], [0, 0, 0.000]])
            self.cov_mag = np.array([[0.000, 0, 0], [0, 0.000, 0], [0, 0, 0.000]])


        # the true initial pose (from the sensor frame to the world frame)
        self.q_t = np.random.uniform([-1,-1,-1,-1], [1,1,1,1])
        self.q_t = self.q_t /np.linalg.norm(self.q_t)

        # displacement limit set to be [-high, high]
        high = np.array([10000, 10000, 10000])

        self.action_space = spaces.Box(low=np.array(
            [-10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10.])*0.0002,
                                       high=np.array(
                                           [10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,
                                            10., 10., 10.])*0.0002,
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

    def step(self, action):  # here u1,u2=measurement, which is a result of the action
        train = False
       

        u_11, u_21, u_31, u_41, u_51, u_61, \
        u_12, u_22, u_32, u_42, u_52, u_62, \
        u_13, u_23, u_33, u_43, u_53, u_63 = action
        
        if self.realMeasurement:
            q_t_lastStep = self.q_t
            # print(self.t)
            self.q_t,omega_obs,acc_m,mag_m=measrement_real(self.t, self.dt, self.data)
            q_t = self.q_t
            omega = 2 / self.dt * Log(np.inner(quatLeftMulMat(quatConj(q_t_lastStep)),np.hstack((q_t))))
            self.q_tt = np.dot(quatLeftMulMat(self.q_tt), exp(0.5 * self.dt *np.vstack( omega)).T)
        else:        
            omega = omega_t_sim(self.t) #simulate the trajectory
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
    
    
            if not self.clean:
                noise_i = np.random.multivariate_normal([0, 0, 0], self.cov_noise_i).flatten()
            else:
                noise_i = np.random.multivariate_normal([0, 0, 0], self.cov_noise_i).flatten()*0
    
            omega_obs = omega + noise_gyro_bias_t + np.array([[noise_i[0]], [noise_i[1]], [noise_i[2]]])
            # omega_obs = omega
    
            # assume the pure acc goes up in proportion to omega
            # change gravity direction
            # acc_m_q = np.dot(np.dot(quatLeftMulMat(quatConj(self.q_t)), quatRightMulMat(self.q_t)), quatPure2Q([0, 0, -1]))
            acc_m_q = np.dot(np.dot(quatLeftMulMat(quatConj(self.q_t)), quatRightMulMat(self.q_t)), quatPure2Q([0, 0, 1]))
            if not self.clean:
                acc_i = np.random.multivariate_normal([0., 0., 0.], self.cov_a).flatten()
            else:
                acc_i = np.random.multivariate_normal([0., 0., 0.], self.cov_a).flatten()*0
    
    
            acc_m = acc_m_q[1:4] + np.array([[acc_i[0]], [acc_i[1]], [acc_i[2]]])
            # acc_m = acc_m_q[1:4]
            acc_m = acc_m / np.linalg.norm(acc_m)
            # assume the dip angle of mag is diata = 30 degree =0.52 rad = (np.pi*30/180)
            mag_m_q = np.dot(np.dot(quatLeftMulMat(quatConj(self.q_t)), quatRightMulMat(self.q_t)),
                             quatPure2Q([np.cos(np.pi * 30 / 180), 0, np.sin(np.pi * 30 / 180)]))
            if not self.clean:
                mag_i = np.random.multivariate_normal([0, 0, 0], self.cov_mag).flatten()
            else:
                mag_i = np.random.multivariate_normal([0, 0, 0], self.cov_mag).flatten()*0
    
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
        # hat_y_acc_q = np.dot(np.dot(quatLeftMulMat(quatConj(q_pred)), quatRightMulMat(q_pred)), quatPure2Q([0, 0, -1]))
        hat_y_acc_q = np.dot(np.dot(quatLeftMulMat(quatConj(q_pred)), quatRightMulMat(q_pred)), quatPure2Q([0, 0, 1]))
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

        # aaa = 2.0* Log(np.inner(quatRightMulMat(quatConj(self.q_pred_init)),self.q_t_init))
        # # bbb = 2 / self.dt * Log(np.inner(quatLeftMulMat(q_t_lastStep),np.hstack(quatConj(np.inner(quatLeftMulMat(q_pred),hat_delta_q))))) - np.hstack(omega)
        # bbb = 2 / self.dt * Log(np.inner(quatLeftMulMat(quatConj(q_t_lastStep)),np.hstack((hat_q)))) - np.hstack(omega)
        # ccc = y-hat_y
        # cost = np.linalg.norm(aaa)**2 + np.linalg.norm(bbb)**2 + np.linalg.norm(ccc)**2

        # cost = np.sum(abs(aaa)) ** 0.2 + np.sum(abs(bbb)) ** 0.2 + np.sum(abs(ccc)) ** 0.2
        # cost = ( np.linalg.norm(bbb) )*( np.linalg.norm(bbb) )
        aaa = 2.0* Log(np.inner(quatRightMulMat(quatConj(hat_q)), np.array(self.q_t)))
        cost = np.linalg.norm(aaa)
        # gamma = 1.005
        # cost = np.linalg.norm(aaa) * np.power( gamma,t)
        # cost = np.linalg.norm(aaa) * np.log(t+1)

        if cost > (3):
            done = True
        else:
            done = False

        # print(cost)


        # 6. update new for next round
        self.hat_q = hat_q
        self.state = hat_eta
        self.t = self.t + self.dt

        # eul_hat_q = quat2eul(hat_q)
        # eul_q_t = quat2eul(q_t)
        if self.choice == 'saveData':
            return omega_obs,acc_m,mag_m,q_t,hat_q, cost, done, dict(reference=y[0],
                                        state_of_interest=np.array([hat_q[1], q_t[1],hat_q[2], q_t[2]]))
            # return omega,omega_obs,acc_m,mag_m,q_t,self.q_tt,hat_q, cost, done, dict(reference=y[0],
            #                             state_of_interest=np.array([hat_q[1], q_t[1],hat_q[2], q_t[2]]))
        else:
            return hat_eta, cost, done, dict(
                reference=np.array([q_t[0], q_t[1], q_t[2], q_t[3]]),
                state_of_interest=np.array([hat_q[0], hat_q[1], hat_q[2], hat_q[3]]))

            # DEBUG
            # return hat_eta, omega, cost, done, dict(
            #     reference=np.array([q_t[0], q_t[1], q_t[2], q_t[3]]),
            #     state_of_interest=np.array([hat_q[0], hat_q[1], hat_q[2], hat_q[3]]))

    def reset(self, eval=False):
        # print('reset')
        self.t = 0

        if self.realMeasurement:
            
            # p = r'trainingData.csv'
            p = r'/Users/weipan/weipan/MLC/LAC_TF2_TORCH_REWRITE/LAC_TF115_CLEANED_SEEDED_SAC_INCL/trainingData.csv'
            self.readData = np.genfromtxt(p, delimiter=',')
            
            s = int(np.random.uniform(1,7000))
            self.data = self.readData[s:(s+1000)]
            dataInit = self.readData[s-1]
            self.q_t = dataInit[0:4]
            
            self.q_tt = self.q_t
        else:  
            if not eval:
                self.q_t = np.random.uniform([-1,-1,-1,-1], [1,1,1,1])# the state q, the initial value can be set randomly
                # self.q_t = np.array([1,0,0,0])
            else:
                self.q_t = np.array([0.2,0.4,0.1,0.6])
            self.q_t = self.q_t /np.linalg.norm(self.q_t)
        self.q_t_init = self.q_t
        self.hat_q = self.q_t + np.random.normal([0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1]) * 0.1
        self.hat_q = self.hat_q /np.linalg.norm(self.hat_q)
        self.q_pred_init = self.hat_q

        hat_eta = np.random.normal([ 0,0,0], [ 0.1,0.1,0.1])*0.0001
        omega_obs = np.array([[0.],[0.],[0.]])
        self.state = hat_eta

        if self.choice == 'saveData':
            omega_obs = np.array([[0],[0],[0]])
            acc_m = np.array([[0],[0],[0]])
            mag_m= np.array([[0],[0],[0]])
            return omega_obs,acc_m,mag_m,self.q_t,self.hat_q
        else:
            # return np.hstack([hat_eta,np.hstack(omega_obs)])  # return hat_state
            return hat_eta

    def render(self, mode='human'):

        return


    def saveChoice(self, choiceIn):
        self.choice =choiceIn
        return self.choice



if __name__ == '__main__':
    env = Ex3_EKF_gyro()
    T = 1

    choice = 'saveData'
    # choice = []
    if env.saveChoice(choice) == 'saveData':
        s = env.reset()
        path = []
        steps = []
        omega_obs,acc_m,mag_m,q_t,hat_q = env.reset()
        measurement = np.vstack([omega_obs,acc_m,mag_m,np.vstack(q_t),np.vstack(hat_q)])
        measurement = np.hstack(measurement)
        path.append(measurement)
        steps.append(0)
        for i in range(int(T / env.dt)):
            # s, r, info, done = env.step(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            # path.append(s)
            omega_obs,acc_m,mag_m,q_t,hat_q, r, info, done = env.step(np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            measurement = np.vstack([omega_obs,acc_m,mag_m,np.vstack(q_t),np.vstack(hat_q)])
            measurement = np.hstack(measurement)
            path.append(measurement)
            steps.append(i)
        np.savetxt('5.csv', path, delimiter = ',')
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.plot(steps, np.array(path)[:, 9+4], color='green', label='x0', linestyle="dashed" )
        ax.plot(steps, np.array(path)[:, 10+4], color='yellow', label='x1', linestyle="dashed")
        ax.plot(steps, np.array(path)[:, 11+4], color='blue', label='x2', linestyle="dashed")
        ax.plot(steps, np.array(path)[:, 12+4], color='red', label='x3', linestyle="dashed")
        ax.plot(steps, np.array(path)[:, 9], color='green', label='x0')
        ax.plot(steps, np.array(path)[:, 10], color='yellow', label='x1')
        ax.plot(steps, np.array(path)[:, 11], color='blue', label='x2')
        ax.plot(steps, np.array(path)[:, 12], color='red', label='x3')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
        plt.show()
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.plot(steps, np.array(path)[:, 0], color='green', label='x0')
        ax.plot(steps, np.array(path)[:, 1], color='yellow', label='x1')
        ax.plot(steps, np.array(path)[:, 2], color='blue', label='x2')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
        plt.show()
        # plt.savefig('1-.eps',format="eps")
        print('done')
    else:
        path = []
        # path2=[]
        t1 = []
        s = env.reset()
        for i in range(int(T / env.dt)):
            # s, r, info, done = env.step(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            # path.append(s)
            hat_q, omega, r, info, done = env.step(np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            path.append(omega)
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
