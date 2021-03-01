import multiworld
import gym
import os
import numpy as np
import cv2
multiworld.register_all_envs()

from multiworld.envs.pybullet import create_image_pybullet_jaco_push_primitive_xy_env_v1, create_image_pybullet_jaco_push_primitive_xyyaw_env_v1
from multiworld.envs.mujoco import create_image_48_sawyer_reach_xy_env_v1
from multiworld.envs.pybullet.cameras import jaco2_push_top_view_camera
from multiworld.envs.pybullet.util.bullet_camera import create_camera
import pybullet as p

camera_params = jaco2_push_top_view_camera
camera = create_camera(p,
                       camera_params['image_height'],
                       camera_params['image_width'],
                       camera_params['intrinsics'],
                       camera_params['translation'],
                       camera_params['rotation'],
                       near=camera_params['near'],
                       far=camera_params['far'],
                       distance=camera_params['distance'],
                       camera_pose=camera_params['camera_pose'],

                       is_simulation=True)

pixel = [640, 480]
depth = 0.715
positions = camera.deproject_pixel( pixel, depth, is_world_frame=True)

pixel = [639, 479]
depth = 0.715
positions2 = camera.deproject_pixel( pixel, depth, is_world_frame=True)

print(positions - positions2)
# def transfer_bbx2world(bbx_list):
#     bbx_list = np.array(np.array(bbx_list)*[640,480,640,480], dtype=np.int)
#
#     left_top = bbx_list[:2]
#     right_bottom = bbx_list[2:]
#
#     left_top_pos = camera.deproject_pixel(left_top, depth, is_world_frame=True)
#
#     right_bottom = camera.deproject_pixel(right_bottom, depth, is_world_frame=True)
#
#     return left_top_pos, right_bottom
#
#
#
#
# CSPACE_LOW = np.array((0 - 0.25, -0.40 - 0.15, 0.154) )
# CSPACE_HIGH = np.array((0 + 0.25, -0.40 + 0.15, 0.154))
#
# cspace_offset = 0.5 * (CSPACE_HIGH + CSPACE_LOW)
# cspace_range = 0.5 * (CSPACE_HIGH - CSPACE_LOW)
# def point2action(point):
#     start = 1. /  cspace_range[0:2] * (point -  cspace_offset[:2])
#     return start
#
# left_top_pos, right_bottom = transfer_bbx2world([0.2,0.3,0.5,0.5])
#
# bbx_LT = point2action(np.array(left_top_pos[:2]))
# bbx_RB = point2action(np.array(right_bottom[:2]))
# print(left_top_pos, right_bottom)
# print(bbx_LT, bbx_RB)
#
#
# a = np.array([[-0.2, -0.35]])





