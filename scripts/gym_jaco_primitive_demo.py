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

                num_movable_bodies=2,

               table_name = 'default',


                # isRandomObjects=False,
                # fixed_objects_init_pos=(0.0275144 , -0.53437352,  0.040171873,  # shape (3*n,)
                #                        ),
                # obj_name_list=['Lshape_train', ],
                #obj_scale_range=(0.01, 0.01),

                goal_order =['x','y', 'theta'],
                obj_name_list=['sugar_box' ], #sugar_box  b_cube_m
                obj_scale_range=(1, 1.5),

                isRandomObjects=False,
                isRandomGoals=False,
                fixed_objects_goals =[[0,0,0],[0,0,0]],

                isIgnoreGoalCollision=False,

                target_upper_space=(0.25, -0.25, 0.06),
                target_lower_space=(-0.25, -0.55, 0.06),

                obj_euler_lower_space =(0,0, -np.pi),
                obj_euler_upper_space=(0,0,  np.pi),
               )



obs = env.reset()


print("obs:", env.observation_space )
print("action:", env.action_space.shape)


for i in range(2000):
    n_dim_action = env.action_space.shape[0]
    action = env.action_space.sample() # np.zeros(n_dim_action)#
    obs, reward, done, info = env.step(action)
    print('state:',obs['state_observation'])
    print('state_desired_goal:', obs['state_desired_goal'])

    print(info)
    print('--------------')

    if done :
        env.reset()
    env.render()




