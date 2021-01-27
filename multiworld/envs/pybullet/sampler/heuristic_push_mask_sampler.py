import os


import numpy as np
from datetime import datetime
import os
import cv2
import math



import matplotlib.pyplot as plt
from multiworld.perception.depth_utils import scale_mask

#Z_HEIGHT = 0.0714#0.0315#0.064
Z_HEIGHT =  0.0315#0.064
class HeuristicPushMaskSampler(object):
    """Heuristics push sampler from mask image."""

    def __init__(self,
                 camera,
                 cspace_low,
                 cspace_high,


                 mask_scale = 1.05,
                 mask_bias =2,
                 num_object=2,
                 start_margin=0.05,
                 motion_margin=0.01,
                 max_attemps=20000):
        """Initialize.

        Args:
            cspace_low:
            cspace_high:
            translation_x:
            translation_y:
            start_margin:
            motion_margin:
            max_attemps:
        """
        self.cspace_low = np.array(cspace_low)
        self.cspace_high = np.array(cspace_high)
        self.cspace_offset = 0.5 * (self.cspace_high + self.cspace_low)
        self.cspace_range = 0.5 * (self.cspace_high - self.cspace_low)

        ## hard code!!!   for ottermodel , mask bias =2 ; for multiworld , mask bias =3
        self.mask_body_index_list = [_ + mask_bias for _ in range(num_object)]
        self.camera = camera

        self.start_margin = start_margin
        self.motion_margin = motion_margin
        self.max_attemps = max_attemps

        self.last_end = None

        self.PUSH_DELTA_SCALE_X = self.PUSH_DELTA_SCALE_Y = 0.1
        self.PUSH_MIN =  0.04
        self.PUSH_MAX = 0.1
        self.PUSH_SCALE = 2.
        self.MASK_MARGIN_SCALE = mask_scale


        self.num_object = num_object

    def sample(self,

               segmask,
               depth,
               body_mask_index=0,
               num_samples=1):
        assert body_mask_index < self.num_object
        list_action = []

        for i in range(num_samples):
            action = self._sample(segmask, depth, body_mask_index)
            list_action.append(action)

        return np.stack(list_action, axis=0)

    def _convert_mask(self, segmask, body_mask_index, scale = 1):
        mask = segmask.copy()
        mask[mask != self.mask_body_index_list[body_mask_index]] = 0
        mask[mask == self.mask_body_index_list[body_mask_index]] = 255

        scaled_mask = scale_mask(mask, scale, value=self.mask_body_index_list[body_mask_index])

        return scaled_mask

    def _sample(self, segmask, depth, body_mask_index):

        scale_mask = self._convert_mask(segmask, body_mask_index, 0.85)

        idx = np.argwhere(scale_mask == self.mask_body_index_list[body_mask_index])

        sampled_index = np.random.randint(0, idx.shape[0], size=1)

        sampled_points = idx[sampled_index]
        sampled_points = sampled_points[:, ::-1]

        base_point = sampled_points[0]

        if depth.dtype == np.uint16:
            depth = depth / 1000.
        bp_w = self.camera.deproject_pixel(base_point, depth[base_point[1], base_point[0]], is_world_frame=True)[:2]

        for i in range(self.max_attemps):

            angle = np.random.uniform(-np.pi, np.pi)

            dis = np.random.uniform(self.PUSH_MIN / self.PUSH_SCALE, self.PUSH_MAX / self.PUSH_SCALE)
            assert self.PUSH_SCALE >= 1
            push_scale = np.random.uniform(1, self.PUSH_SCALE)

            start_x = dis * np.cos(angle) + bp_w[0]
            #start_y = dis * np.sin(angle) + bp_w[1]
            if angle >= 0:
                start_y = bp_w[1] + np.sqrt(dis ** 2 - (start_x - bp_w[0]) ** 2)
            else:
                start_y = bp_w[1] - np.sqrt(dis ** 2 - (start_x - bp_w[0]) ** 2)

            sp_w = np.array([start_x, start_y], dtype=np.float32)

            margin_mask =  self._convert_mask(segmask, body_mask_index, 1.3)
            if self._check_not_inside_mask(margin_mask, sp_w):
                ep_w = push_scale * (bp_w - sp_w) + sp_w
                ep_w = np.array(ep_w, dtype=np.float32)
                break

            if i == self.max_attemps - 1:
                raise Exception('HeuristicSampler did not find a good sample.')

        start, motion = self._map2action(sp_w, ep_w)

        action = np.concatenate([start, motion], axis=-1)
        action = np.clip(action, -1, 1)


        return action

    def push_control(self, segmask, depth, body_mask_index, state_goal):

        scale_mask = self._convert_mask(segmask, body_mask_index, self.MASK_MARGIN_SCALE)

        idx = np.argwhere(scale_mask == self.mask_body_index_list[body_mask_index])

        sampled_index = np.random.randint(0, idx.shape[0], size=1)

        sampled_points = idx[sampled_index]
        sampled_points = sampled_points[:, ::-1]

        base_point = sampled_points[0]

        if depth.dtype == np.uint16:
            depth = depth / 1000.
        bp_w = self.camera.deproject_pixel(base_point, depth[base_point[1], base_point[0]], is_world_frame=True)[:2]

        for i in range(self.max_attemps):

            angle = np.random.uniform(-np.pi, np.pi)

            dis = np.random.uniform(self.PUSH_MIN / self.PUSH_SCALE, self.PUSH_MAX / self.PUSH_SCALE)
            assert self.PUSH_SCALE >= 1
            push_scale = np.random.uniform(1, self.PUSH_SCALE)

            start_x = dis * np.cos(angle) + bp_w[0]

            if angle >= 0:
                start_y = bp_w[1] + np.sqrt(dis ** 2 - (start_x - bp_w[0]) ** 2)
            else:
                start_y = bp_w[1] - np.sqrt(dis ** 2 - (start_x - bp_w[0]) ** 2)

            sp_w = np.array([start_x, start_y], dtype=np.float32)

            if self._check_not_inside_mask(scale_mask, sp_w, scale = 1.3):
                ep_w = push_scale * (bp_w - sp_w) + sp_w
                ep_w = np.array(ep_w, dtype=np.float32)
                break

            if i == self.max_attemps - 1:
                raise Exception('HeuristicSampler did not find a good sample.')

        start, motion = self._map2action(sp_w, ep_w)
        action = np.concatenate([start, motion], axis=-1)
        action = np.clip(action, -1 ,1)
        return action

    def _map2action(self, sp_w, ep_w):
        start = 1. / self.cspace_range[0:2] * (sp_w - self.cspace_offset[:2])
        motion = 1 / self.PUSH_MAX * (ep_w - sp_w)[:2]

        return start, motion

    def _check_not_inside_mask(self, segmask, point_world):
        point_world = np.r_[point_world, Z_HEIGHT]
        pixel_uv = self.camera.project_point(point_world, is_world_frame=True)

        if segmask[pixel_uv[1], pixel_uv[0]] in self.mask_body_index_list:
            return False
        else:
            return True

    def plot_push(self, action, rgb=None, copy=True, show=True):
        waypoints = self._compute_waypoints(action)

        print('push length :', np.linalg.norm(np.array(waypoints[0]) - np.array(waypoints[1])))
        sp_uv = self.camera.project_point(np.r_[waypoints[0], Z_HEIGHT], is_world_frame=True)
        ep_uv = self.camera.project_point(np.r_[waypoints[1], Z_HEIGHT], is_world_frame=True)

        if rgb is not None:
            push_img = rgb.copy() if copy else rgb
            push_img = cv2.line(push_img, (sp_uv[0], sp_uv[1]), (ep_uv[0], ep_uv[1]), (255, 0, 0), 5, 1)
            cv2.circle(push_img, (sp_uv[0], sp_uv[1]), 5, [0, 255, 0], 4)
            cv2.circle(push_img, (ep_uv[0], ep_uv[1]), 5, [255, 0, 0], 4)
            # cv2.circle(push_img, (base_point[0], base_point[1]), 5, [0,0,0], 4 )
            if show:
                plt.imshow(push_img)
                plt.show()
        return push_img

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
        x = start[0] * self.cspace_range[0] + self.cspace_offset[0]
        y = start[1] * self.cspace_range[1] + self.cspace_offset[1]

        start = [x, y]

        # End.
        delta_x = motion[0] * self.PUSH_DELTA_SCALE_X
        delta_y = motion[1] * self.PUSH_DELTA_SCALE_Y
        x = x + delta_x
        y = y + delta_y

        end = [x, y]

        waypoints = [start, end]
        return waypoints



def main():
    import pybullet as p
    from multiworld.envs.pybullet.util.bullet_camera import create_camera

    camera_params = {"target_pos": (0.0, -0.461, -0.004),
                     "distance": 1,
                     "yaw": 0,
                     "pitch": -63.684,
                     "roll": 0,
                     "fov": 60,
                     "near": 0.1,
                     "far": 100.0,
                     "image_width": 640,
                     "image_height": 480,
                     'intrinsics': [610.911, 0., 321.936, 0., 611.021, 236.665, 0., 0., 1.],
                     'translation': None,
                     'rotation': None,

                     # 'camera_pose' :{'translation': [0.0 ,  -0.767,  0.60],
                     #                  'rotation':[-2.588,  0,  0],},  # tilt
                     'camera_pose': {'translation': [0.0, -0.4, 0.75],
                                     'rotation': [-math.pi, 0, 0], },  # upright
                     }

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
    sampler = HeuristicPushMaskSampler(camera,
                                       cspace_high=(0 + 0.25, -0.40 + 0.15, 0.154),
                                       cspace_low=(0 - 0.25, -0.40 - 0.15, 0.154),
                                       num_object=4)

    fig_path = '/home/drl/PycharmProjects/JerryRepos/ottermodels/scripts/camera_test'
    rgb = cv2.imread(os.path.join(fig_path, 'rgb.png'))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth = cv2.imread(os.path.join(fig_path, 'depth.png'), cv2.IMREAD_UNCHANGED)
    segmask = cv2.imread(os.path.join(fig_path, 'segmask.png'), cv2.IMREAD_UNCHANGED)


    num_samples = 10
    action = sampler.sample(segmask, depth, body_mask_index=2, num_samples=num_samples)

    for i in range(num_samples):
        push_img = sampler.plot_push(action[i], rgb=rgb, copy=False, show=False)

    plt.imshow(push_img)
    plt.show()


if __name__ == '__main__':
    main()
