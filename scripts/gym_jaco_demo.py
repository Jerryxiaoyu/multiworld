import multiworld
import gym
import os
import numpy as np
import cv2
multiworld.register_pybullet_envs()


from multiworld.envs.pybullet import create_image_pybullet_jaco_push_primitive_xy_env_v1
from multiworld.envs.mujoco import create_image_48_sawyer_reach_xy_env_v1
#Jaco2ObjectsPusherOneXYEnv-v0
#Jaco2ObjectsPusherXYEnv-v0
env = gym.make("Jaco2ObjectsPusherOneXYEnv-v0" ,
               isRender = True,

               isRenderGoal=True, num_movable_bodies =1)


#env = create_image_pybullet_jaco_push_primitive_xy_env_v1()
#env = create_image_48_sawyer_reach_xy_env_v1()
obs = env.reset()

print("obs:", env.observation_space )
print("action:", env.action_space.shape)


for i in range(2000):
    n_dim_action = env.action_space.shape[0]
    action = env.action_space.sample() # np.zeros(n_dim_action)#
    obs, reward, done, info = env.step(action)

    if done :
        env.reset()
    env.render()




