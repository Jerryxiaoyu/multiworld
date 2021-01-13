import random

import cv2
import numpy as np
import warnings
from PIL import Image
from gym.spaces import Box, Dict

from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.wrapper_env import ProxyEnv
from multiworld.envs.env_util import concatenate_box_spaces
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict
from multiworld.core.image_raw_env import ImageRawEnv, normalize_image
from multiworld.utils.segImgProc import convert2mask,get_clip_img
import multiworld.perception.point_cloud_utils as  pc_utils
import matplotlib.pyplot as plt
import copy

INF = 2**32 - 1

class ClickController(object):

    def __init__(self, obs):
        self.obs = obs

    def __call__(self, event):
        pixel = [event.xdata, event.ydata]
        z = self.obs.depth[int(pixel[1]), int(pixel[0])]
        position = self.obs.camera.deproject_pixel(pixel, z)
        self.obs.target_position = position

class SegPointCloudRawEnv(ImageRawEnv):
    def __init__(
            self,
            num_points,
            num_bodies=1,
            is_simulator = True,

            crop_min=None,
            crop_max=None,
            max_visible_distance_m=None,
            confirm_target=False,
            name=None,

            **kwargs
    ):
        self.quick_init(locals())
        super().__init__(**kwargs)
        self.camera = self.wrapped_env.camera

        self.is_simulator = is_simulator
        self.num_points = num_points
        self.num_bodies = num_bodies

        if crop_max is not None and crop_min is not None:
            self.crop_max = np.array(crop_max)[np.newaxis, :]
            self.crop_min = np.array(crop_min)[np.newaxis, :]
        else:
            self.crop_max = None
            self.crop_min = None

        self.max_visible_distance_m = max_visible_distance_m or INF
        self.confirm_target = confirm_target

        self.env = None

        if self.is_simulator:
            self.body_ids = [body.uid for body in self.wrapped_env.movable_bodies]

        if self.confirm_target:
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            self.ax = ax
            self.fig = fig

            self.target_position = None
            self.depth = None

            onclick = ClickController(obs=self)
            fig.canvas.mpl_connect('button_press_event', onclick)
            plt.ion()
            plt.show()


    def _get_imgs_dict(self, goal_flag= False ):
        images = self._wrapped_env.camera.frames()
        # image
        rgb = images['rgb']
        depth = images['depth']
        segmask = images['segmask']
        if self.heatmap:
            raise NotImplementedError

        new_obs = dict(
            image_observation=rgb,
            depth_observation=depth,
            mask_observation=segmask,  )

        point_cloud = self.camera.deproject_depth_image(depth)

        # Crop.
        if self.crop_max is not None and self.crop_min is not None:
            crop_mask = np.logical_and(
                np.all(point_cloud >= self.crop_min, axis=-1),
                np.all(point_cloud <= self.crop_max, axis=-1))
            point_cloud = point_cloud[crop_mask]

        # Segment.
        if self.is_simulator:
            segmask = images['segmask']
            segmask = segmask.flatten()
            segmask = pc_utils.convert_segment_ids(segmask, self.body_ids)
            point_cloud = pc_utils.group_by_labels(
                point_cloud, segmask, self.num_bodies, self.num_points)
        else:
            point_cloud = pc_utils.remove_table(point_cloud)
            segmask = pc_utils.cluster(
                point_cloud, num_clusters=self.num_bodies, method='dbscan')
            point_cloud = point_cloud[segmask != -1]
            segmask = pc_utils.cluster(
                point_cloud, num_clusters=self.num_bodies)
            point_cloud = pc_utils.group_by_labels(
                point_cloud, segmask, self.num_bodies, self.num_points)

        # Confirm target.
        if self.confirm_target:
            image = rgb
            # Click the target position.
            self.target_position = None
            self.depth = depth
            self.ax.cla()
            self.ax.imshow(image)
            print('Please click the target object...')
            while self.target_position is None:
                plt.pause(1e-3)
            print('Target Position: %r', self.target_position)

            # Exchange the target object with the first object.
            centers = np.mean(point_cloud, axis=1)
            dists = np.linalg.norm(
                centers - self.target_position[np.newaxis, :], axis=-1)
            target_id = dists.argmin()
            if target_id != 0:
                tmp = copy.deepcopy(point_cloud)
                point_cloud[0, :] = tmp[target_id, :]
                point_cloud[target_id, :] = tmp[0, :]

            # Show the segmented point cloud.
            pc_utils.show2d(point_cloud, self.camera, self.ax, image=image)

        positions = []

        for i in range(self.num_bodies):
            pos = [ _ for _ in self.wrapped_env.movable_bodies[i].pose.position]
            positions.append(np.array(pos))


        new_obs['point_cloud'] = point_cloud
        new_obs['position'] = np.array(positions)


        return new_obs

    def get_goal(self, is_depth= False, is_mask=False):
        goal = self.wrapped_env.get_goal()
        goal['desired_goal'] = self._img_goal
        goal['image_desired_goal'] = self._img_goal
       # goal['segrgb_desired_goal'] = self._img_goal_dict['segrgb_observation']
        goal['depth_desired_goal'] = self._img_goal_dict['depth_observation']
        goal['mask_desired_goal'] = self._img_goal_dict['mask_observation']
        return goal







