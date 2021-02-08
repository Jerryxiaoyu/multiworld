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


class ImageRawEnv(ProxyEnv, MultitaskEnv):
    def __init__(
            self,
            wrapped_env,
            init_camera=None,
            heatmap = False,
            goal_heatmap = False,
            normalize=False,
            image_dim = None,
            transpose = False,
            flatten=False,
            goal_in_image_dict_key = 'image_observation',
            image_achieved_key ='image_observation',
            reward_type='wrapped_env',
            threshold=10,
            image_length=None,
            presampled_goals=None,
            non_presampled_goal_img_is_garbage=False,
            recompute_reward=True,
            **kwargs
    ):

        """
        :param wrapped_env:
        :param imsize:
        :param init_camera:
        :param transpose:
        :param grayscale:
        :param normalize:
        :param reward_type:
        :param threshold:
        :param image_length:
        :param presampled_goals:
        :param non_presampled_goal_img_is_garbage: Set this option to True if
        you want to allow the code to work without presampled goals,
        but where the underlying env doesn't support set_to_goal. As the name,
        implies this will make it so that the goal image is garbage if you
        don't provide pre-sampled goals. The main use case is if you want to
        use an ImageEnv to pre-sample a bunch of goals.
        """
        self.quick_init(locals())
        super().__init__(wrapped_env)
        self.wrapped_env.hide_goal_markers = True

        self.init_camera = init_camera
        self.goal_dict_key = goal_in_image_dict_key
        self.image_achieved_key = image_achieved_key

        self.image_dim = image_dim
        self.transpose = transpose
        self.flatten = flatten

        self.heatmap = heatmap
        self.goal_heatmap = goal_heatmap
        self.im_height = self.init_camera['image_height']
        self.im_width = self.init_camera['image_width']

        self.normalize = normalize
        self.recompute_reward = recompute_reward
        self.non_presampled_goal_img_is_garbage = non_presampled_goal_img_is_garbage

        if image_length is not None:
            self.image_length = image_length
        else:
            if self.image_dim is None:
                self.image_length = 3 * self.im_height * self.im_width
                self.image_shape = (self.im_height, self.im_width)
            else:
                self.image_length = 3* self.image_dim * self.image_dim
                self.image_shape = (self.image_dim, self.image_dim)
        self.channels =   3

        # This is torch format rather than PIL image

        # Flattened past image queue
        # init camera
        if init_camera is not None:
            self._wrapped_env.initialize_camera(init_camera)

        img_space = Box(0, 1, (self.image_length,), dtype=np.float32)
        self._img_goal = img_space.sample() #has to be done for presampling
        spaces = self.wrapped_env.observation_space.spaces.copy()

        spaces['image_observation'] = img_space
        spaces['image_desired_goal'] = img_space
        spaces['image_achieved_goal'] = img_space

        if self.heatmap:
            self.heatmap_shape = self.wrapped_env.heatmap_shape
            spaces['color_heatmap'] = Box(0, 1, (self.heatmap_shape[0] * self.heatmap_shape[1] * 3,),
                                                          dtype=np.float32)
            spaces['depth_heatmap'] = Box(0, 1, (self.heatmap_shape[0] * self.heatmap_shape[1],),
                                                          dtype=np.float32)
            spaces['valid_depth_heightmap'] = Box(0, 1, (self.heatmap_shape[0] * self.heatmap_shape[1],),
                                                                  dtype=np.float32)
            spaces['heatmap'] = Box(0, 1, (self.heatmap_shape[0] * self.heatmap_shape[1] * 4,),
                                                    dtype=np.float32)

        if self.goal_heatmap:
            spaces['goal_heatmap'] = Box(0, 1, (self.heatmap_shape[0] * self.heatmap_shape[1] * 6,),
                                    dtype=np.float32)
        self.observation_space = Dict(spaces)
        self.action_space = self.wrapped_env.action_space
        self.reward_type = reward_type
        self.threshold = threshold

        self._presampled_goals = presampled_goals
        if self._presampled_goals is None:
            self.num_goals_presampled = 0
        else:
            self.num_goals_presampled = presampled_goals[random.choice(list(presampled_goals))].shape[0]


    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        if self.recompute_reward:
            reward = self.compute_reward(action, new_obs)
        self._update_info(info, obs)
        return new_obs, reward, done, info

    def _update_info(self, info, obs):
        achieved_goal = obs[self.image_achieved_key]
        desired_goal = self._img_goal
        image_dist = np.linalg.norm(achieved_goal-desired_goal)
        image_success = (image_dist<self.threshold).astype(float)-1
        info['image_dist'] = image_dist
        info['image_success'] = image_success

    def reset(self):
        obs = self.wrapped_env.reset()
        if self.num_goals_presampled > 0:
            raise  NotImplementedError
            goal = self.sample_goal()
            self._img_goal = goal['image_desired_goal']
            self.wrapped_env.set_goal(goal)
            for key in goal:
                obs[key] = goal[key]
        elif self.non_presampled_goal_img_is_garbage:
            # This is use mainly for debugging or pre-sampling goals.
            self._img_goal = self._get_imgs_dict()
        else:
            env_state = self.wrapped_env.get_env_state()
            self.wrapped_env.set_to_goal(self.wrapped_env.get_goal())
            self._img_goal_dict = self._get_imgs_dict(goal_flag= True)
            self._img_goal = self._img_goal_dict[self.goal_dict_key]
            self.wrapped_env.set_env_state(env_state)
        return self._update_obs(obs)

    def _get_obs(self):
        return self._update_obs(self.wrapped_env._get_obs())

    def _update_obs(self, obs):
        extra_obs = self._get_imgs_dict()
        for key in extra_obs.keys():
            obs[key] = extra_obs[key]

        obs['image_desired_goal'] = self._img_goal_dict[self.goal_dict_key]
        obs['image_achieved_goal'] = obs[self.image_achieved_key]
        obs['pointcloud_desired_goal'] = self._img_goal_dict['point_cloud']



        return obs
    def _get_rgb_img(self):
        images = self._wrapped_env.camera.frames()
        # image
        rgb = images['rgb']
        return rgb

    def _get_imgs_dict(self, goal_flag= False ):
        images = self._wrapped_env.camera.frames()
        # image
        rgb = images['rgb']
        depth = images['depth']
        segmask = images['segmask']
        if self.heatmap:
            color_heatmap, depth_heatmap, valid_depth_heightmap = self._wrapped_env.cal_heatmap(rgb, depth)

        if self.image_dim is not None:
            rgb = cv2.resize(rgb, (self.image_dim, self.image_dim), interpolation=cv2.INTER_AREA)
        if self.transpose:
            rgb = rgb.transpose((2,0,1))
        if self.normalize:
            rgb = normalize_image(rgb)
        if self.flatten:
            rgb = rgb.flatten()

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

        return new_obs

    def render(self, mode='wrapped'):
        if mode == 'wrapped':
            self.wrapped_env.render()
        pass


    """
    Multitask functions
    """
    def get_goal(self, is_depth= False, is_mask=False):
        goal = self.wrapped_env.get_goal()
        goal['desired_goal'] = self._img_goal
        goal['image_desired_goal'] = self._img_goal

        goal['depth_desired_goal'] = self._img_goal_dict['depth_observation']
        goal['mask_desired_goal'] = self._img_goal_dict['mask_observation']
        return goal

    def set_goal(self, goal):
        ''' Assume goal contains both image_desired_goal and any goals required for wrapped envs'''
        self._img_goal = goal['image_desired_goal']
        self.wrapped_env.set_goal(goal)

    def sample_goals(self, batch_size):
        if self.num_goals_presampled > 0:
            idx = np.random.randint(0, self.num_goals_presampled, batch_size)
            sampled_goals = {
                k: v[idx] for k, v in self._presampled_goals.items()
            }
            return sampled_goals
        if batch_size > 1:
            warnings.warn("Sampling goal images is slow")
        img_goals = np.zeros((batch_size, self.image_length))
        goals = self.wrapped_env.sample_goals(batch_size)
        pre_state = self.wrapped_env.get_env_state()
        for i in range(batch_size):
            goal = self.unbatchify_dict(goals, i)
            self.wrapped_env.set_to_goal(goal)
            img_goals[i, :] = self._get_flat_img()
        self.wrapped_env.set_env_state(pre_state)
        goals['desired_goal'] = img_goals
        goals['image_desired_goal'] = img_goals
        return goals

    def compute_rewards(self, actions, obs):
        if self.reward_type=='wrapped_env':
            return self.wrapped_env.compute_rewards(actions, obs)


        if self.reward_type=='image_distance':
            achieved_goals = obs['image_observation'].flatten()
            desired_goals = obs['image_desired_goal'].flatten()
            dist = np.linalg.norm(achieved_goals - desired_goals )

            return [-dist]
        elif self.reward_type=='image_sparse':
            achieved_goals = obs['image_observation'].flatten()
            desired_goals = obs['image_desired_goal'].flatten()
            dist = np.linalg.norm(achieved_goals - desired_goals )

            return [-(dist > self.threshold).astype(float)]
        else:
            raise NotImplementedError()

    def get_diagnostics(self, paths, **kwargs):
        statistics = self.wrapped_env.get_diagnostics(paths, **kwargs)
        for stat_name_in_paths in ["image_dist", "image_success"]:
            stats = get_stat_in_paths(paths, 'env_infos', stat_name_in_paths)
            statistics.update(create_stats_ordered_dict(
                stat_name_in_paths,
                stats,
                always_show_all_stats=True,
            ))
            final_stats = [s[-1] for s in stats]
            statistics.update(create_stats_ordered_dict(
                "Final " + stat_name_in_paths,
                final_stats,
                always_show_all_stats=True,
            ))
        return statistics



class SegImageRawEnv(ImageRawEnv):
    def __init__(
            self,
            num_obj=1,
            **kwargs
    ):
        self.num_obj = num_obj
        super().__init__(**kwargs)



def normalize_image(image, dtype=np.float64):
    assert image.dtype == np.uint8
    return dtype(image) / 255.0

def unormalize_image(image):
    assert image.dtype != np.uint8
    return np.uint8(image * 255.0)
