import multiworld
import gym
import os
import numpy as np
import cv2
multiworld.register_pybullet_envs()

from multiworld.envs.pybullet import create_image_pybullet_jaco_push_primitive_xy_env_v1, create_image_pybullet_jaco_push_primitive_xyyaw_env_v1
from multiworld.envs.mujoco import create_image_48_sawyer_reach_xy_env_v1


env = gym.make("Jaco2PushPrimitiveOneXYEnv-v0" ,isRender=True,
                good_render=True,
                isRenderGoal=True,

               num_movable_bodies=1,

               table_name = 'default',

               isRandomObjects=True,
                fixed_objects_init_pos=(0.0275144, -0.53437352, 0.040171873,  # shape (3*n,)
                                        ),
               # fixed_objects_init_euler=[0, 0, 0],

is_render_object_pose=True,


                #obj_scale_range=(0.01, 0.01),
#isPoseObservation = True,
                goal_order =['x','y', 'theta'],
                obj_name_list=[
                                'x_1',
                               # 'mustard_bottle',
                                #  'tomato_soup_can',
                                 # 'pudding_box',
                                 # 'potted_meat_can',
                                 # 'banana',
                                 #'bowl',
                                # 'mug',
                                 # 'rubiks_cube',
                    #'gelatin_box',
                    #'plate'
                                ], #sugar_box  b_cube_m  shapenet  sugar_box  'shapenet', "b_L1", 'b_L2'
                obj_scale_range=(1, 1),
               is_fixed_order_objects=True,
                use_random_rgba=False,

                num_RespawnObjects=2,

                # isRandomObjects=False,
                # isRandomGoals=False,
                # fixed_objects_goals =[[0,0,0],[0,0,0]],
                #
                # isIgnoreGoalCollision=False,
                #
                # target_upper_space=(0.25, -0.25, 0.06),
                # target_lower_space=(-0.25, -0.55, 0.06),

                obj_euler_lower_space =(0,0, -np.pi),
                obj_euler_upper_space=(0,0,  np.pi),
               )
#"b_cube_m" \
#"b_cube_w" \
#"b_L1" \
#"b_L2" \
#"b_semi_column" \
#"filled-073-c_lego_duplo" \
#"lshapeN_1" \
#"lshapeN_2" \
#"lshapeN_3" \
#"lshapeN_4" \
#"lshapeN_5" \
#"lshapeN_6" \
#"lshapeN_7" \
#"lshapeN_8" \


#"003_cracker_box" \
#"004_sugar_box" \
#"006_mustard_bottle" \
#"007_tuna_fish_can" \
#"008_pudding_box" \
#"009_gelatin_box" \
#"010_potted_meat_can" \
#"011_banana" \
#"035_power_drill" \
#"051_large_clamp" \
#"071_nine_hole_peg_test" \
#"flipped-065-a_cups" \

# "shapenet_bottle_1" \
# "shapenet_bottle_2" \
# "shapenet_mug_1" \
# "shapenet_mug_2" \
# "shapenet_sofa_1" \
# "shapenet_sofa_2" \
# "shapenet_phone_1" \
# "shapenet_phone_2" \
# "shape_bottle_3" \
# "shape_bottle_4" \
# "shape_bottle_5" \
# "shape_bottle_6" \
# "shape_bottle_7" \
# "phone_1" \
# "phone_2" \
#
obs = env.reset()


print("obs:", env.observation_space )
print("action:", env.action_space.shape)

# while 1:
# #     pass
# for i in range(2000):
#     n_dim_action = env.action_space.shape[0]
#     action = env.action_space.sample() # np.zeros(n_dim_action)#
#     obs, reward, done, info = env.step(action)
#     print('state:',obs['state_observation'])
#     print('state_desired_goal:', obs['state_desired_goal'])
#
#     print(info)
#     print('--------------')
#
#     if done :
#         env.reset()
#     env.render()
#



