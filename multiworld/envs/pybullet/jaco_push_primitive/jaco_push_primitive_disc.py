import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
import time
import abc
import math
import os
import cv2
import random

import numpy as np
import pybullet as p
import multiworld.utils as Utils

# from multiworld.utils.logging import logger
# from .params import *
from ..jaco_xyz.base import Jaco2XYZEnv

from multiworld.math import Pose
from ..simulation.body import Body
import glob
from ..util.bullet_camera import create_camera
from multiworld.perception.camera import Camera

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

DEFAULT_TARGET_POS = (-0.15, -0.75, 0.25)
OBJECT_DEFALUT_POSITION = (0.05, -0.45, 0.25)

"""
box area:
x[-0.25, 0.25 0.13]
y[-0.65, -0.15 0.13]
"""
DEFAULT_RENDER = {
    "target_pos": (0, -0.842, 0.842),
    "distance": 0.25,
    "yaw": 0,
    "pitch": -43,
    "roll": 0,
}

from multiworld.core.multitask_env import MultitaskEnv
from gym.spaces import Box, Dict
from gym import spaces
from multiworld.math import Pose
from ..simulation.body import Body
from ..simulation.objects import Objects

import matplotlib.pyplot as plt

import copy
from collections import OrderedDict
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,

)
from multiworld.envs.pybullet.util.utils import plot_pose, clear_visualization
from .jaco_push_primitive import Jaco2PushPrimitiveXY, Jaco2PushPrimitiveXYyaw

__all__ = ['Jaco2PushPrimitiveXYDisc', 'Jaco2PushPrimitiveXYyaw']


class Jaco2PushPrimitiveXYDisc(Jaco2PushPrimitiveXY):
    def __init__(self,

                 heightmap_resolution=0.008,
                 push_dis_unit=0.05,
                 num_angle=16,
                 **kwargs):
        self.quick_init(locals())

        self.heightmap_resolution = heightmap_resolution

        self.PUSH_DIS = push_dis_unit
        self.NUM_ANGLE = num_angle

        Jaco2PushPrimitiveXY.__init__(self, **kwargs)

        self.heatmap_shape = self._get_heatmap_size()
        print('env heatmap shape:', self.heatmap_shape, self.heightmap_resolution)

        n_discrete_actioins = self.heatmap_shape[0] * self.heatmap_shape[1] * self.NUM_ANGLE

        self.action_dim = 3
        action_high = np.array([1] * self.action_dim)
        #self.action_space = spaces.Box(-action_high, action_high)
        self.action_space =   spaces.Box(-action_high, action_high)#spaces.Discrete(n_discrete_actioins)
        self._isDiscrete = True
        self.n_discrete_actioins = n_discrete_actioins

        if self.vis_debug:
            z = 0.06
            l_11 = [self._hand_init_low[0], self._hand_init_high[1], z]
            l_22 = [self._hand_init_high[0], self._hand_init_high[1], z]
            l_33 = [self._hand_init_high[0], self._hand_init_low[1], z]
            l_44 = [self._hand_init_low[0], self._hand_init_low[1], z]

            l_11_p = self.camera.project_point(l_11)
            l_22_p = self.camera.project_point(l_22)
            l_33_p = self.camera.project_point(l_33)
            l_44_p = self.camera.project_point(l_44)

            self.work_box_range_in_pixel = (l_11_p, l_33_p)

    def _get_heatmap_size(self):
        # Compute heightmap size
        heightmap_size = np.round(
            ((self.workspace_limits[1][1] - self.workspace_limits[1][0]) / self.heightmap_resolution,
             (self.workspace_limits[0][1] - self.workspace_limits[0][0]) / self.heightmap_resolution)).astype(
            int)
        return heightmap_size

    def _set_observation_space(self):
        if self.goal_order == ['x', 'y']:
            low_space = [self.objects_env.object_max_space_low[0],
                         self.objects_env.object_max_space_low[1]]
            high_space = [self.objects_env.object_max_space_high[0],
                          self.objects_env.object_max_space_high[1]]
            self.obs_box = Box(np.tile(np.array(low_space), self.num_objects),
                               np.tile(np.array(high_space), self.num_objects))

            self.state_obs_box = self.obs_box

            low_space = [self._target_lower_space[0],
                         self._target_lower_space[1]]
            high_space = [self._target_upper_space[0],
                          self._target_upper_space[1]]
            self.goal_box = Box(np.tile(np.array(low_space), self.num_objects),
                                np.tile(np.array(high_space), self.num_objects))

            self.state_goal_box = self.goal_box

            self.changed_obj_box = Box(np.array([0]),np.array([1]))


        self.observation_space = Dict([
            ('observation', self.obs_box),  # object position [x,y]*n
            ('state_observation', self.state_obs_box),  # object pose [x,y,z,roll picth, yaw] *n
            ('desired_goal', self.goal_box),  # goal object position [x,y]*n
            ('achieved_goal', self.obs_box),  # actual object position [x,y]*n
            ('state_desired_goal', self.state_goal_box),  # goal object pose [x,y,z,roll picth, yaw] *n
            ('state_achieved_goal', self.state_obs_box),  # actual pose [x,y,z,roll picth, yaw] *n
            ('changed_object', self.changed_obj_box)
        ])

    def reset(self):

        self.last_obj_pose = Pose([[0,0,0],[0,0,0]])
        self.workspace_limits = np.asarray([
            [self._hand_init_low[0], self._hand_init_high[0]],  # x
            [self._hand_init_low[1], self._hand_init_high[1]],  # y
            [ 0.035, 0.4]])  # z
        super().reset()

        self.last_obj_pose = self.movable_bodies[0].pose

        return self._get_obs()

    def _check_obs_dim(self):
        obs = self._get_obs()

        for k in self.observation_space.spaces.keys():
            assert obs[k].shape == self.observation_space.spaces[k].shape, 'obs {} shape error'.format(k)

    def cal_heatmap(self, image, depth):

        color_heightmap, depth_heightmap = self.get_heightmap(image,
                                                              depth,
                                                              self.camera.intrinsics,
                                                              self.camera.pose.matrix4,
                                                              self.workspace_limits,
                                                              self.heightmap_resolution)

        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

        return  color_heightmap, depth_heightmap, valid_depth_heightmap

    def _get_obs(self):
        if self._isImageObservation:
            images = self.camera.frames()
            # image
            rgb = images['rgb']
            depth = images['depth']
            segmask = images['segmask']
            color_heatmap, depth_heatmap , valid_depth_heightmap = self.cal_heatmap(rgb, depth)

        if self.goal_order == ['x', 'y']:
            bs = []
            orns = []
            for body in self.movable_bodies:
                bs.append(body.position[:2])
                orns.append(body.orientation.euler)

            current_pose = self.movable_bodies[0].pose
            pose_delta = np.linalg.norm(current_pose.position - self.last_obj_pose.position)
            yaw_delta = current_pose.yaw - self.last_obj_pose.yaw
            if pose_delta > 0.01 or yaw_delta >3/180.*math.pi:
                changed_object = 1
            else:
                changed_object = 0

            self.last_obj_pose = self.movable_bodies[0].pose

            pos = np.concatenate(bs)
            orn = np.concatenate(orns)

            # state =  np.concatenate((pos, orn))
            state = pos
            state_goal = self.state_goal

        new_obs = dict(
            observation=state,  # obj pos [x,y]*n
            state_observation=state,  # obj pos [x,y]*n
            desired_goal=state_goal,  # desired goal: obj pos [x,y]*n
            state_desired_goal=state_goal,  # desired goal: obj pos [x,y]*n
            achieved_goal=state,
            state_achieved_goal=state,
            changed_object = np.array([changed_object])

        )
        if self._isImageObservation:
            new_obs['image'] = rgb
            new_obs['depth'] = depth
            new_obs['segmask'] = segmask
            new_obs['color_heatmap'] = color_heatmap
            new_obs['depth_heatmap'] = depth_heatmap
            new_obs['valid_depth_heightmap'] = valid_depth_heightmap
        return new_obs

    def step(self, action):
        assert action.shape[0] == 3

        action = self._discAction2cons(action)

        return super().step(action)

    def _discAction2cons(self, action):
        # angle = action // (self.heatmap_shape[0] * self.heatmap_shape[1])
        # pix_u = action % (self.heatmap_shape[0] * self.heatmap_shape[1]) // self.heatmap_shape[1]
        # pix_v = action % (self.heatmap_shape[0] * self.heatmap_shape[1]) % self.heatmap_shape[1]

        angle = int(action[0])# // (self.heatmap_shape[0] * self.heatmap_shape[1])
        # % (self.heatmap_shape[0] * self.heatmap_shape[1]) // self.heatmap_shape[1]
        pix_v = int(action[1])# y   % (self.heatmap_shape[0] * self.heatmap_shape[1]) % self.heatmap_shape[1]
        pix_u = int(action[2])# x

        assert pix_u >= 0 and pix_u < self.heatmap_shape[1]
        assert pix_v >= 0 and pix_v < self.heatmap_shape[0]
        assert angle >= 0 and angle < self.NUM_ANGLE

        x = pix_u * self.heightmap_resolution + self.workspace_limits[0][0]
        y = pix_v * self.heightmap_resolution + self.workspace_limits[1][0]

        angle = (math.pi * 2) / self.NUM_ANGLE * angle
        delta_x = self.PUSH_DIS * math.cos(angle) / self.PUSH_DELTA_SCALE_X
        delta_y = self.PUSH_DIS * math.sin(angle) / self.PUSH_DELTA_SCALE_Y

        return np.array([x, y, delta_x, delta_y])

    def _compute_waypoints(self, action):
        """Convert action of a single step to waypoints.

        Args:
            action: The action of a single step.

        Returns:
            List of waypoints of a single step.
        """
        action = np.reshape(action, [2, 2])
        start = action[0, :]
        motion = action[1, :]

        # Start.
        x = start[0]
        y = start[1]
        z = self.start_offset[2]

        start = [x, y, z] + self._fixed_orn

        # End.
        delta_x = motion[0] * self.PUSH_DELTA_SCALE_X
        delta_y = motion[1] * self.PUSH_DELTA_SCALE_Y
        x = x + delta_x
        y = y + delta_y

        x = np.clip(x, self.robot.end_effector_lower_space[0], self.robot.end_effector_upper_space[0])
        y = np.clip(y, self.robot.end_effector_lower_space[1], self.robot.end_effector_upper_space[1])
        end = [x, y, z] + self._fixed_orn

        waypoints = [start, end]
        return waypoints

    def _get_info(self):

        ee_pos, ee_orn = self.robot.GetEndEffectorObersavations()

        object_distances = {}
        touch_distances = {}
        for i in range(self.num_objects):
            object_name = "object%d_distance" % i
            object_distance = np.linalg.norm(
                self.get_object_goal_pos(i) - self.get_object_pos(i)
            )
            object_distances[object_name] = object_distance

        info = dict(
            # end_effector=[ee_pos, ee_orn],

            success=float(sum(object_distances.values()) < 0.06),
            **object_distances,
            **touch_distances,
        )

        return info

    def _reward(self, obs, action, others=None):
        reward = self.compute_reward(action, obs)
        return reward

    def _termination(self):
        if self.terminated or self._envStepCounter > self._maxSteps:
            self._observation = self._get_obs()
            return True, 0
        if self.MAX_ACTION_STEPS is not None and self._envActionSteps > self.MAX_ACTION_STEPS:
            self._observation = self._get_obs()
            return True, 0

        if self.attributes['is_turnover']:
            self._observation = self._get_obs()
            return True, 0

        if self._check_outof_range():
            self._observation = self._get_obs()
            if self.debug_info:
                logger.warning('out of range!!!')
            return True, 0

        return False, 0


    def log_diagnostics(self, paths, logger=None, prefix=""):
        if logger is None:
            return

        statistics = OrderedDict()
        for stat_name in [
                             'hand_distance',
                             'success',
                         ] + [
                             "object%d_distance" % i for i in range(self.num_objects)
                         ] + [
                             "touch%d_distance" % i for i in range(self.num_objects)
                         ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s %s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final %s %s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))

        for key, value in statistics.items():
            logger.record_tabular(key, value)

    # multi task

    @property
    def goal_dim(self) -> int:
        return self.state_goal_box.shape[0]

    @property
    def movable_bodies(self):
        return self.objects_env.movable_bodies

    def get_goal(self):
        return {
            'desired_goal': self.state_goal,
            'state_desired_goal': self.state_goal,
        }

    def get_object_pos(self, i):
        return self.movable_bodies[i].position

    def get_object_pose(self, i):
        return self.movable_bodies[i].pose

    def compute_rewards(self, action, obs, info=None):

       # r = -np.linalg.norm(obs['state_observation'] - obs['state_desired_goal'], axis=1)

        if self.reward_mode == 0:  # dense
            r = -np.linalg.norm(obs['state_observation'] - obs['state_desired_goal'], axis=1)
            r += obs['changed_object'][:,0]
        elif self.reward_mode == 1:  # sparse
            r = obs['changed_object'][:,0] / 2.

            dis = np.linalg.norm(obs['state_observation'] - obs['state_desired_goal'], axis=1 )

            r[np.where(dis<0.06)[0]] += 1

        else:
            raise NotImplementedError

        return r

    def compute_reward(self, action, obs, info=None):

        if self.reward_mode == 0: # dense
            r = -np.linalg.norm(obs['state_observation'] - obs['state_desired_goal'])
            r += obs['changed_object']
        elif self.reward_mode == 1: # sparse
            r = obs['changed_object']/2.

            dis = np.linalg.norm(obs['state_observation'] - obs['state_desired_goal'])
            if dis < 0.06:
                r += 1

        else:
            raise NotImplementedError




        return r

    def set_goal(self, goal):
        self.state_goal = goal['state_desired_goal']

    def sample_goals(self, batch_size):

        goals = np.random.uniform(
            self.goal_box.low,
            self.goal_box.high,
            size=(batch_size, self.goal_dim),
        )
        state_goals = np.random.uniform(
            self.state_goal_box.low,
            self.state_goal_box.high,
            size=(batch_size, self.goal_dim),
        )
        return {
            'desired_goal': goals,
            'state_desired_goal': state_goals,
        }

    def get_hand_goal_pos(self):
        raise NotImplementedError

    def set_to_goal(self, goal):
        assert goal['state_desired_goal'].shape[0] == self.goal_dim

        state_goal = goal['state_desired_goal']

        object_poses = []
        for i in range(self.num_objects):
            pos = self.get_object_goal_pos_from_stategoal(state_goal, i)
            euler_orn = self.get_object_goal_euler_orn_from_stategoal(state_goal, i)
            object_poses.append(Pose([pos, euler_orn]))
        self.objects_env._reset_movable_obecjts(object_poses)

    @staticmethod
    def get_pointcloud(color_img, depth_img, camera_intrinsics):

        # Get depth image size
        im_h = depth_img.shape[0]
        im_w = depth_img.shape[1]

        # Project depth into 3D point cloud in camera coordinates
        pix_x, pix_y = np.meshgrid(np.linspace(0, im_w - 1, im_w), np.linspace(0, im_h - 1, im_h))
        cam_pts_x = np.multiply(pix_x - camera_intrinsics[0][2], depth_img / camera_intrinsics[0][0])
        cam_pts_y = np.multiply(pix_y - camera_intrinsics[1][2], depth_img / camera_intrinsics[1][1])
        cam_pts_z = depth_img.copy()
        cam_pts_x.shape = (im_h * im_w, 1)
        cam_pts_y.shape = (im_h * im_w, 1)
        cam_pts_z.shape = (im_h * im_w, 1)

        # Reshape image into colors for 3D point cloud
        rgb_pts_r = color_img[:, :, 0]
        rgb_pts_g = color_img[:, :, 1]
        rgb_pts_b = color_img[:, :, 2]
        rgb_pts_r.shape = (im_h * im_w, 1)
        rgb_pts_g.shape = (im_h * im_w, 1)
        rgb_pts_b.shape = (im_h * im_w, 1)

        cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
        rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)

        return cam_pts, rgb_pts

    def get_heightmap(self, color_img, depth_img, cam_intrinsics, cam_pose, workspace_limits, heightmap_resolution):

        # Compute heightmap size
        heightmap_size = np.round(((workspace_limits[1][1] - workspace_limits[1][0]) / heightmap_resolution,
                                   (workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_resolution)).astype(
            int)

        # Get 3D point cloud from RGB-D images
        surface_pts, color_pts = self.get_pointcloud(color_img, depth_img, cam_intrinsics)

        # Transform 3D point cloud from camera coordinates to robot coordinates
        surface_pts = np.transpose(np.dot(cam_pose[0:3, 0:3], np.transpose(surface_pts)) + np.tile(cam_pose[0:3, 3:], (
            1, surface_pts.shape[0])))

        # Sort surface points by z value
        sort_z_ind = np.argsort(surface_pts[:, 2])
        surface_pts = surface_pts[sort_z_ind]
        color_pts = color_pts[sort_z_ind]

        # Filter out surface points outside heightmap boundaries
        heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(
            np.logical_and(surface_pts[:, 0] >= workspace_limits[0][0], surface_pts[:, 0] < workspace_limits[0][1]),
            surface_pts[:, 1] >= workspace_limits[1][0]), surface_pts[:, 1] < workspace_limits[1][1]),
            surface_pts[:, 2] < workspace_limits[2][1])
        surface_pts = surface_pts[heightmap_valid_ind]
        color_pts = color_pts[heightmap_valid_ind]

        # Create orthographic top-down-view RGB-D heightmaps
        color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        depth_heightmap = np.zeros(heightmap_size)
        heightmap_pix_x = np.floor((surface_pts[:, 0] - workspace_limits[0][0]) / heightmap_resolution).astype(int)
        heightmap_pix_y = np.floor((surface_pts[:, 1] - workspace_limits[1][0]) / heightmap_resolution).astype(int)
        color_heightmap_r[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [0]]
        color_heightmap_g[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [1]]
        color_heightmap_b[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [2]]
        color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
        depth_heightmap[heightmap_pix_y, heightmap_pix_x] = surface_pts[:, 2]
        z_bottom = workspace_limits[2][0]
        depth_heightmap = depth_heightmap - z_bottom
        depth_heightmap[depth_heightmap < 0] = 0
        depth_heightmap[depth_heightmap == -z_bottom] = np.nan

        return color_heightmap, depth_heightmap

    def visualize(self, action, obs):

        # Reset.

        # rgb = obs['image']
        rgb = self._plot_box_in_image(obs['image_observation'])
        self.ax.cla()
        self.ax.imshow(rgb)

        action = self._discAction2cons(action)
        waypoints = self._compute_waypoints(action)
        self._plot_waypoints(self.ax,
                             waypoints,
                             linewidth=5,
                             c='red',
                             alpha=0.5)

        ##-----
        plt.draw()
        plt.pause(1e-3)

    def _plot_waypoints(self,
                        ax,
                        waypoints,
                        linewidth=1.0,
                        c='blue',
                        alpha=1.0):
        """Plot waypoints.

        Args:
            ax: An instance of Matplotlib Axes.
            waypoints: List of waypoints.
            linewidth: Width of the lines connecting waypoints.
            c: Color of the lines connecting waypoints.
            alpha: Alpha value of the lines connecting waypoints.
        """
        z = 0.05
        #
        # p1 = None
        # p2 = None
        # for i, waypoint in enumerate(waypoints):
        #     point1 = waypoint[:3]
        #     point1 = np.array([point1[0], point1[1], z])
        #     p1 = self.camera.project_point(point1)
        #     # if i == 0:
        #     #     ax.scatter(p1[0], p1[1],
        #     #                c=c, alpha=alpha, s=2.0)
        #     # else:
        #     #     ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
        #     #             c=c, alpha=alpha, linewidth=linewidth)
        #     # p2 = p1

        point1 = waypoints[0][:3]
        point1 = np.array([point1[0], point1[1], z])
        p1 = self.camera.project_point(point1)

        ax.scatter(p1[0], p1[1],
                   c='g', alpha=alpha, s=8)

        point1 = waypoints[1][:3]
        point1 = np.array([point1[0], point1[1], z])
        p2 = self.camera.project_point(point1)

        ax.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1],width=4,
                 #head_width=10, head_length=10,
                 fc=c, ec=c, alpha=alpha,
                 zorder=100)

    def _plot_box_in_image(self, image):

        image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        image = cv2.rectangle(image,
                              (self.work_box_range_in_pixel[0][0], self.work_box_range_in_pixel[0][1]),
                              (self.work_box_range_in_pixel[1][0], self.work_box_range_in_pixel[1][1]),
                              (0, 255, 0), 4)

        return image
