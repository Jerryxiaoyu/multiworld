import multiworld
import gym
import os
import numpy as np
import cv2
multiworld.register_all_envs()
import time
from multiworld.envs.mujoco.sawyer_xyz.sawyer_multiple_objects import MultiSawyerEnv
from multiworld.envs.mujoco import create_image_48_sawyer_reach_xy_env_v1
# SawyerPushAndReachArenaEnv-v0
# SawyerPushAndReachArenaResetFreeEnv-v0
# SawyerPushAndReachSmallArenaEnv-v0
# SawyerPickupEnv-v0
import multiworld.envs.gridworlds
#env = gym.make("FetchReach-v1" )
env = gym.make("Jaco2ObjectsPusherOneXYSimpleEnv-v1", isRender=True,
useDynamics = False,

            timeStep=0.04,
            actionRepeat=1,  # very importrant!!!
            maxSteps=125,
            action_scale_conts=0.1/10,  # m
               )
#env = create_image_48_sawyer_reach_xy_env_v1()
#env = MultiSawyerEnv()

obs = env.reset()

print("obs:", env.observation_space.shape)
print("action:", env.action_space.shape)

t1 = time.time()
reset_cnt = 0
for i in range(1000):

    action = env.action_space.sample() # np.zeros(n_dim_action)#
    obs, reward, done, info = env.step(action)

    if done :
        reset_cnt += 1
        env.reset()
    #env.render()
t2 = time.time()

print("time={:.5}".format(t2-t1))
print('reset count =', reset_cnt)



