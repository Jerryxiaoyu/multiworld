import multiworld
import gym
import os
import numpy as np
import cv2
multiworld.register_all_envs()

from multiworld.envs.mujoco.sawyer_xyz.sawyer_multiple_objects import MultiSawyerEnv
from multiworld.envs.mujoco import create_image_48_sawyer_reach_xy_env_v1
# SawyerPushAndReachArenaEnv-v0
# SawyerPushAndReachArenaResetFreeEnv-v0
# SawyerPushAndReachSmallArenaEnv-v0
# SawyerPickupEnv-v0

env = gym.make("SawyerPushNIPSEasy-v0" )

#env = create_image_48_sawyer_reach_xy_env_v1()
#env = MultiSawyerEnv()

obs = env.reset()

print("obs:", env.observation_space.shape)
print("action:", env.action_space.shape)


for i in range(2000):
    n_dim_action = env.action_space.shape[0]
    action = env.action_space.sample() # np.zeros(n_dim_action)#
    obs, reward, done, info = env.step(action)

    if done :
        env.reset()
    env.render()




