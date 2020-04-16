import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)
import time
import abc
import math
import numpy as np
import pybullet as p
import os
import cv2


from .base import Jaco2XYZEnv

from multiworld.core.serializable import Serializable
import copy

from multiworld.core.multitask_env import MultitaskEnv
from gym.spaces import Box, Dict

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

OBJECT_URDF_LISTS=[
 "objects/blocks/block_column.urdf",
# "objects/blocks/block_cube_m.urdf",
# "objects/blocks/block_cube_w.urdf",
# "objects/blocks/block_cuboid.urdf",
# "objects/blocks/block_cuboid2.urdf",
# "objects/blocks/block_L1.urdf",
# "objects/blocks/block_L2.urdf",
# "objects/blocks/block_semi_column.urdf",
# "objects/blocks/block_triangle.urdf",

# "objects/blocks/block_semi_hole.urdf",
# "objects/blocks/block_semi_hole_ext.urdf",
    
# "YCB/002_master_chef_can/master_chef.urdf",
# "YCB/004_sugar_box/sugar_box.urdf",
# "YCB/005_tomato_soup_can/tomato_soup_can.urdf",
# "YCB/008_pudding_box/pudding_box.urdf",
# "YCB/010_potted_meat_can/potted_meat_can.urdf",
# "YCB/011_banana/banana.urdf",
# "YCB/013_apple/apple.urdf",
#
# "YCB/006_mustard_bottle/mustard_bottle.urdf",

#"YCB/007_tuna_fish_can

]
from multiworld.math import Pose
from ..simulation.body import Body
from ..simulation.objects import Objects

__all__ = [ 'Jaco2BlockPusherXYZ']

class Jaco2BlockPusherXYZ(Jaco2XYZEnv,   MultitaskEnv):
    def __init__(self,
                 random_init_object_position= True,
                 random_target_position = False,
                 default_target_position = DEFAULT_TARGET_POS,


                 obj_name_list=[],
                 num_movable_bodies=3,


                 target_upper_space=(0.2, -0.75, 0.25),
                 target_lower_space=(-0.2, -0.85, 0.25),
                 initPos_upper_space=(0.2, -0.45, 0.25),
                 initPos_lower_space=(-0.2, -0.55, 0.25),
                 default_init_position = OBJECT_DEFALUT_POSITION,
                 object_EulerOrn = (0, -math.pi/2,  math.pi/2),


                 reward_mode=0,

                # render_params=DEFAULT_RENDER,
                 **kwargs):
        # objects
        self.num_objects = num_movable_bodies
        self.objects_env = Objects(obj_name_list, num_movable_bodies, **kwargs)

        isRandomGoals = True
        isIgnoreGoalCollision = False
        # goal
        self._isRandomGoals  = isRandomGoals

        self._isIgnoreGoalCollision = isIgnoreGoalCollision
        self.fixed_objects_goals = []



        self._default_target_positon = list(default_target_position)


        self._target_upper_space = target_upper_space
        self._target_lower_space = target_lower_space

        self._initPos_lower_space = initPos_lower_space
        self._initPos_upper_space = initPos_upper_space

        self.goal_point = list(default_target_position)



        self._random_init_cup_position = random_init_object_position
        self._default_init_position = default_init_position
        self._object_EulerOrn = object_EulerOrn


        if os.getenv('REWARD_CHOICE') is not None:
            self.reward_mode = int(os.getenv('REWARD_CHOICE'))
            print('Reward Mode : ', self.reward_mode)
        else:
            self.reward_mode = reward_mode
            print ('Default Reward mode: ', self.reward_mode)

        Jaco2XYZEnv.__init__(self,  **kwargs)

    def _set_observation_space(self):

        self.obs_box = Box(
            np.array(self._hand_low),
            np.array(self._hand_high),
            )
        self.goal_box = Box(
            np.array(self._target_lower_space),
            np.array(self._target_lower_space),
            )

        self.observation_space = Dict([
            ('observation', self.obs_box),
            ('state_observation', self.obs_box),
            ('desired_goal', self.goal_box),
            ('state_desired_goal', self.goal_box),

        ])

    def reset(self):
        super().reset()
        self.objects_env.reset()

        self.state_goal = self.sample_goal_for_rollout()
        self._isRenderGoal = True
        if self._isRenderGoal:

        return  self._get_obs()

    def _get_obs(self):
        e = self.robot.GetEndEffectorObersavations()[0][:3]
        bs = []

        for body in  self.movable_bodies :
            b = body.position
            bs.append(b)
        b = np.concatenate(bs)
        x =  np.concatenate((e, b))
        g = self.state_goal

        new_obs = dict(
            observation=x,          # end-effector
            state_observation=x,    # end_effector & obj positions
            desired_goal=g,         # desired goal
            state_desired_goal=g,   # desired goal

        )

        return new_obs

    def _get_info(self):
        end_effector_pose = self.robot.GetEndEffectorObersavations()
        return {"end_effector":end_effector_pose}

    def _reward(self, obs, action, others=None):
        torque = self.robot.GetMotorTorques()

        dist = 0

        if self.reward_mode == 0:
            reward_dist = -np.square(dist).sum()
            reward_ctrl = -np.linalg.norm(torque) * 0.0001
            reward = reward_dist + reward_ctrl
        else:
            raise NotImplementedError

        return reward


    # multi task
    @property
    def movable_bodies(self):
        return self.objects_env.movable_bodies


    def sample_goals(self, batch_size):
        goals = np.random.uniform(
            self.goal_box.low,
            self.goal_box.high,
            size=(batch_size, self.goal_box.low.size),
        )
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }
    def get_goal(self):
        return {
            'desired_goal': self.state_goal,
            'state_desired_goal': self.state_goal,
        }

    def compute_rewards(self, action, obs, info=None):
        r = -np.linalg.norm(obs['state_achieved_goal'] - obs['state_desired_goal'], axis=1)
        return r

    def set_goal(self, goal):
        self.state_goal = goal['state_desired_goal']


    def sample_goal_for_rollout(self):
        if self._isRandomGoals:
            if not self._isIgnoreGoalCollision:
                poses_list = self.objects_env._sample_body_poses( self.num_objects)
                pos = []
                for pose in poses_list:
                    pos.append(pose.position)

                goals = np.concatenate(pos)
            else:

                goals = np.concatenate([np.random.uniform(self._target_lower_space, self._target_upper_space) for i in range(self.num_objects)])
        else:

            goals = self.fixed_objects_goals.copy()


        return goals

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        self.set_hand_xy(state_goal[:2])
        for i in range(self.num_objects):
            x = 2 + 2 * i
            y = 4 + 2 * i
            self.set_object_xy(i, state_goal[x:y])