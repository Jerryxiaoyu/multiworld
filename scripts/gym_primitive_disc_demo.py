import multiworld
import gym
import os
import numpy as np
import cv2
multiworld.register_all_envs()
import time
from multiworld.envs.mujoco.sawyer_xyz.sawyer_multiple_objects import MultiSawyerEnv
from multiworld.envs.mujoco import create_image_48_sawyer_reach_xy_env_v1

import multiworld.envs.gridworlds
from multiworld.core.image_raw_env import ImageRawEnv
from multiworld.envs.pybullet.cameras import jaco2_push_top_view_camera
from multiworld.envs.pybullet.jaco_push_primitive.jaco_push_primitive import Jaco2PushPrimitiveXY

wrapped_env = gym.make("Jaco2PushPrimitiveDiscOneXYEnv-v0" ,isRender= True,
isImageObservation= False,
   vis_debug=True)

env = wrapped_env
camera_params = jaco2_push_top_view_camera
env = ImageRawEnv(
        wrapped_env,
        init_camera=jaco2_push_top_view_camera,
        heatmap=True,
        normalize=False,
        reward_type='wrapped_env',

        image_achieved_key ='valid_depth_heightmap',
        goal_in_image_dict_key = 'valid_depth_heightmap',
    )

obs = env.reset()

print("obs:", env.observation_space.shape)
print("action:", env.action_space )

t1 = time.time()
for i in range(100):

    #action = np.array([np.random.randint(0,200), np.random.randint(0,200), np.random.randint(0,16)])
    action = env.action_space.sample()
    #
    #time.sleep(1.5)
    action = np.array([np.random.randint(0,16),0,49])
    env.visualize(action, obs)
    obs, reward, done, info = env.step(action)


    if done :
        env.reset()
    env.render()
t2 = time.time()
print('total time: {}s', t2-t1)




