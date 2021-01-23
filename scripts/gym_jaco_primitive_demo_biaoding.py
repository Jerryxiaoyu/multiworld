import multiworld
import gym
import os
import numpy as np
import cv2
multiworld.register_all_envs()

from multiworld.envs.pybullet import create_image_pybullet_jaco_push_primitive_xy_env_v1, create_image_pybullet_jaco_push_primitive_xyyaw_env_v1
from multiworld.envs.mujoco import create_image_48_sawyer_reach_xy_env_v1

env = gym.make("Jaco2PushPrimitiveOneXYEnv-v0" ,isRender=True,
                 good_render=True,
                isRenderGoal=True,

                num_movable_bodies=1,


            isRandomObjects=False,
            fixed_objects_init_pos=( -0.15, -0.35,  0.040171873,  # shape (3*n,)
                                   ),
            obj_name_list=['ball_visual', ],
            obj_scale_range=(0.01, 0.01),

             # goal_order =['x','y', 'theta'],
             #    obj_name_list=['b_cube_m' ],
             #    obj_scale_range=(1, 1.5),
               )

import numpy as np

cspace_high = np.array([0 + 0.25, -0.40 + 0.15, 0.154])
cspace_low  =  np.array((0 - 0.25, -0.40 - 0.15, 0.154))
PUSH_MAX = 0.1
cspace_offset = 0.5 * (cspace_high + cspace_low)
cspace_range = 0.5 * (cspace_high - cspace_low)

def _map2action( sp_w, ep_w):
    sp_w = np.array(sp_w)
    ep_w = np.array(ep_w)
    start = 1. /  cspace_range[0:2] * (sp_w -  cspace_offset[:2])

    motion = 1 /  PUSH_MAX * (ep_w - sp_w)[:2]

    return start, motion

obs = env.reset()


print("obs:", env.observation_space )
print("action:", env.action_space.shape)

sp_w = [ -0.15, -0.35]
ep_w = [ -0.15,-0.25]

for i in range(2000):
    n_dim_action = env.action_space.shape[0]
    action = _map2action( sp_w, ep_w)
    obs, reward, done, info = env.step(action)
    print('state:',obs['state_observation'])
    print('state_desired_goal:', obs['state_desired_goal'])

    print(info)
    print('--------------')

    if done :
        env.reset()
    env.render()




