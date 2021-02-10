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

def segmentation_proc(rgb_img,  segmask,depth_img = None, body_ids=[0], num_object= 1,
                      transpose= False,  normalize=False, flatten=False):
    seg_rgbs = []
    bbx_states = []
    seg_depths = []
    if depth_img is not None:
        raise NotImplementedError

    for body_id in body_ids:
        mask = convert2mask(segmask, body_id, num_object=num_object)
        top_left, bottom_right, seg_rgb = get_clip_img(rgb_img, mask)

        if transpose:
            seg_rgb = seg_rgb.transpose((2, 0, 1))
        if normalize:
            seg_rgb = normalize_image(seg_rgb)
        if flatten:
            seg_rgb = seg_rgb.flatten()
        #seg_rgb = warped_rgb.transpose((2, 0, 1)).flatten()
        bbx_state = [top_left.x, top_left.y, bottom_right.x, bottom_right.y]

        seg_rgbs.append(seg_rgb)
        bbx_states.append(bbx_state)

    bbx_states = np.array(np.array(bbx_states)) /np.array([640,480,640,480])
        # _, _, warped_depth = get_clip_img(depth_img, mask, is_rgb=False, )
        # seg_depths[t] = warped_depth.flatten()

    return np.array(seg_rgbs), np.array(bbx_states)

class SegImageRawEnv(ImageRawEnv):
    def __init__(
            self,
            num_obj=1,
            **kwargs
    ):
        self.quick_init(locals())
        super().__init__(**kwargs)
        self.num_obj = num_obj
        self.segImg_fun = segmentation_proc
        #super().__init__(**kwargs)


    def _get_imgs_dict(self, goal_flag= False ):
        images = self._wrapped_env.camera.frames()
        # image
        rgb = images['rgb']
        depth = images['depth']
        segmask = images['segmask']
        if self.heatmap:
            color_heatmap, depth_heatmap, valid_depth_heightmap = self._wrapped_env.cal_heatmap(rgb, depth)

        new_obs = dict(
            image_observation=rgb,
            depth_observation=depth,
            mask_observation=segmask,
                )
        if self.heatmap:
            new_obs['color_heatmap'] = color_heatmap
            new_obs['depth_heatmap'] = depth_heatmap
            new_obs['valid_depth_heightmap'] = valid_depth_heightmap
            new_obs['heatmap']= np.concatenate((color_heatmap, depth_heatmap[:,:,np.newaxis]), axis=2)

        if self.goal_heatmap and not goal_flag:
            delta_depth = valid_depth_heightmap - self._img_goal
            depth_new = np.concatenate((valid_depth_heightmap[None], delta_depth[None], self._img_goal[None]), axis=0)
            depth_new = depth_new.transpose((1, 2, 0))
            new_obs['goal_heatmap'] = np.concatenate((color_heatmap, depth_new ), axis=2)

        seg_rgbs, bbx_states = self.segImg_fun(rgb, segmask,depth_img = None, body_ids=[_ for _ in range(self.num_obj)], num_object= self.num_obj,
                                               transpose=self.transpose, normalize=self.normalize, flatten=self.flatten
                                               )

        new_obs['segrgb_observation'] = seg_rgbs
        new_obs['bbx_observation'] = bbx_states

        return new_obs

    def get_goal(self, is_depth= False, is_mask=False):
        goal = self.wrapped_env.get_goal()
        goal['desired_goal'] = self._img_goal
        goal['image_desired_goal'] = self._img_goal
        goal['segrgb_desired_goal'] = self._img_goal_dict['segrgb_observation']

        goal['depth_desired_goal'] = self._img_goal_dict['depth_observation']
        goal['mask_desired_goal'] = self._img_goal_dict['mask_observation']
        goal['bbx_desired_goal'] = self._img_goal_dict['bbx_observation']
        return goal

    def _update_obs(self, obs):
        extra_obs = self._get_imgs_dict()
        for key in extra_obs.keys():
            obs[key] = extra_obs[key]

        obs['image_desired_goal'] = self._img_goal_dict[self.goal_dict_key]
        obs['image_achieved_goal'] = obs[self.image_achieved_key]

        obs['segrgb_desired_goal'] = self._img_goal_dict['segrgb_observation']
        obs['segrgb_achieved_goal'] = obs['segrgb_observation']

        obs['bbx_desired_goal'] = self._img_goal_dict['bbx_observation']
        obs['bbx_achieved_goal'] = obs['bbx_observation']

        return obs






