import multiworld
import gym
import os
import numpy as np
import cv2
from multiworld.core.image_raw_env import ImageRawEnv
from multiworld.core.Seg_image_env import SegImageRawEnv
from multiworld.core.SegPointCloud_env import SegPointCloudRawEnv
from multiworld.envs.pybullet.cameras import jaco2_push_top_view_camera
import matplotlib.pyplot as plt

multiworld.register_all_envs()
wrapped_env = gym.make("Jaco2PushPrimitiveOneXYEnv-v0", isRender=True,
                    num_movable_bodies = 2,
                       isImgMask=True,
                       isImgDepth=True,

                       isImageObservation=False,

                       isRenderGoal=False,
                       opengl_Render_enable=False,
                       vis_debug=False)



env = SegPointCloudRawEnv(num_points=256,
                        num_bodies=2,
                        crop_min=None,
                        crop_max=None,
                        max_visible_distance_m=None,
                        confirm_target=True,
                        name=None,

                     wrapped_env=wrapped_env,


                     init_camera=jaco2_push_top_view_camera,
                     image_dim=None,
                     flatten=True,
                     transpose=True,
                     normalize=True,
                     heatmap=False,
                     reward_type='image_distance',  # image_distance    wrapped_env
                     )


obs = env.reset()

for i in range(20):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

print('Over!')
