import multiworld
import gym
import os
import numpy as np
import cv2
import math

multiworld.register_all_envs()
from multiworld.envs.pybullet.cameras import jaco2_push_top_view_camera
from multiworld.core.image_raw_env import ImageRawEnv
from multiworld.envs.pybullet import create_image_pybullet_jaco_push_primitive_xy_env_v1, create_image_pybullet_jaco_push_primitive_xyyaw_env_v1
from multiworld.envs.mujoco import create_image_48_sawyer_reach_xy_env_v1
## Jaco2PushPrimitiveXYEnv-v0
wrapped_env = gym.make("Jaco2PushPrimitiveOneXYEnv-v0" ,
# env arguments
            addTable=True,
            table_position=[0.0000000, -0.650000, -0.59000],
            table_name='white',

            good_render=True,
            isRender=True,
            isImageObservation=True,  ##IMPORTANCE
            camera_params= jaco2_push_top_view_camera,
            isImgMask=True,
            isImgDepth=True,
            isRenderGoal= False,

            # obj setting
            isRandomObjects=True,
            fixed_objects_init_pos=(-0.25, -0.3, 0.06,  # shape (3*n,)
                                    0, -0.3, 0.06,),
            obj_name_list=[ 'b_cube_m', 'b_cube_w','b_L1', 'b_L2'
                # 'sugar_box',
                #              'pudding_box'
                            ], #
            num_movable_bodies=4,


            obj_pos_upper_space = (0 + 0.15, -0.40  + 0.15, 0.16),
            obj_pos_lower_space = (0 - 0.15, -0.40   , 0.16),
            obj_max_upper_space=(0 + 0.3, -0.40 + 0.2, 0.4),
            obj_max_lower_space=(0 - 0.3, -0.40 - 0.2, -0.4),
            obj_euler_upper_space=( 0, 0, 0),
            obj_euler_lower_space=( 0, 0, 0),

            obj_safe_margin=0.05,
            obj_scale_range=(1, 1),
            obj_mass=None,
            obj_friction=0.8,
            use_random_rgba=True,
            num_RespawnObjects=5,

            # primitive
            push_delta_scale_x=0.1,
            push_delta_scale_y=0.1,
            steps_check=50,
            gripper_safe_height=0.204,
            offstage_pose=(-0.3, -0.1, 0.35416687,
                           -0.326071, 0.944798, 0.0238332, -0.021595),
            offstage_joint_pos=(4.66983682, 3.09011904, 1.22106735, 1.34376026, 0.09548322, 4.45514805, 1.82119646),
            max_phase_stpes=200,
            max_offstage_steps=900,
            max_motion_steps=200,
            num_goal_stpes=None,
            maxActionSteps=10,

            init_skip_timestep=100,

            # goal setting
            isRandomGoals=True,
            isIgnoreGoalCollision=False,
            fixed_objects_goals=( 0.15, -0.3 ,
                                  0.13, -0.4),  # shape (2*n,)
            fixed_hand_goal=(0, -0.25, 0.154),
            target_upper_space=(0.25, -0.25, 0.06),
            target_lower_space=(-0.25, -0.55, 0.06),

            # robot params
            random_robot_ee_pos=False,
            hand_init_upper_space=(0 + 0.25, -0.40 + 0.15, 0.154),
            hand_init_lower_space=(0 - 0.25, -0.40 - 0.15, 0.154),

            image=False,
            render_params={
                "target_pos": (0, -0.742, 0.642),
                "distance": 0.15,
                "yaw": 0,
                "pitch": -37,
                "roll": 0,
            },

            # base arguments
            kinova_type='j2s7s300_beam',
            control_mode='push_primitive_consXY',

            ## 10s traj horizon
            timeStep=0.01,
            actionRepeat=100,  # very importrant!!!
            maxSteps=24000,  # per max 1200
            action_scale_conts=0.1,  # m

            debug=False,
            multi_view=False,

            skip_first=0,

            robotHomeAngle=(
            4.66983682, 3.09011904, 1.22106735, 1.34376026, 0.09548322, 4.45514805, 1.82119646, 1.263, 1.347, 1.347),
            robot_ee_pos=(-0.3, -0.1, 0.35416687),
            fixed_orn=(-0.32618785, 0.94479543, 0.02328156, -0.02051954),

            fixed_finger=1.263,

            stepTextPosition=[-0.65, 0, 0.3],

            # kinova arguments
            isLimitedWorkSpace=True,
            hand_high=(0 + 0.3, -0.40 + 0.2, 0.4),
            hand_low=(0 - 0.3, -0.40 - 0.2, -0.4),
            useDynamics=True,

            # other
            debug_info=False,
                )


env = wrapped_env
camera_params = jaco2_push_top_view_camera
env = ImageRawEnv(
        wrapped_env,
        init_camera=jaco2_push_top_view_camera,
        heatmap=False,
        normalize=False,
        reward_type='wrapped_env',
    )
#env = create_image_pybullet_jaco_push_primitive_xyyaw_env_v1()
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




