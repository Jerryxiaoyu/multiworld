

import multiworld
import gym
import os
import numpy as np
import cv2
multiworld.register_all_envs()
import math

import transformations
import pybullet as p
from pygame.locals import QUIT, KEYDOWN

theta = math.pi/3.

orn_mat = transformations.euler_matrix(math.pi, 0,0).dot(transformations.euler_matrix(0,0,theta))
q = transformations.quaternion_from_euler(math.pi,0,theta)
orn = [ q[1], q[2], q[3], q[0]]
q1 = transformations.quaternion_from_matrix(orn_mat)
orn2 = [ q1[1], q1[2], q1[3], q1[0]]
print(q, orn, orn2)



print(p.getQuaternionFromEuler((math.pi,0,theta)))


env = gym.make('Jaco2ObjectsPusherOneXYSimpleEnv-v0',
               isRender= True,
               debug=False,  # debug for camera
               debug_joint = True, # debug for joint
               robot_info_debug= True,

               robotHomeAngle=(4.765313195362755, 3.7267309633866947, -0.16806972198770956, 1.4159912737517137, 0.1874491154963382, 4.02137015754993, 0.4529272987258747, 1.2138450740519722, 1.2945497717537808, 1.2945497717537808),
               #robot_ee_pos=(0.02, -0.45, 0.15),
               isImageObservation_debug = False,
               good_render=False,
               verbose = True)



obs = env.reset()

ndim_act = env.action_space.shape[0]
print("obs:", env.observation_space.shape)
print("action:",  ndim_act)

while 1:
    for i in range(2000):
        action = np.zeros(ndim_act)
        obs, reward, done, info = env.step(action)
        #print(info["end_effector"])

        #if done:
        #    env.reset()
        env.render()

    print('demo is over.')
