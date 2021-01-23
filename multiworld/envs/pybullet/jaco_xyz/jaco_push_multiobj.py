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
from collections import OrderedDict
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,

)
from multiworld.core.multitask_env import MultitaskEnv
from gym.spaces import Box, Dict

DEFAULT_TARGET_POS = (-0.15, -0.75, 0.25)
OBJECT_DEFALUT_POSITION = (0.05, -0.45, 0.25)



from multiworld.math import Pose
from ..simulation.body import Body
from ..simulation.objects import Objects

__all__ = [ 'Jaco2BlockPusherXYZ', 'Jaco2BlockPusherXYSimpleEnv']

class Jaco2BlockPusherXYZ(Jaco2XYZEnv,  MultitaskEnv):
    def __init__(self,
                 # target
                 isRandomGoals=True,
                 isIgnoreGoalCollision = False,
                 fixed_objects_goals =[],
                 fixed_hand_goal =[],
                 target_upper_space=(0.2, -0.75, 0.25),
                 target_lower_space=(-0.2, -0.85, 0.25),

                 #obj
                 obj_name_list=['b_cube_m'],
                 num_movable_bodies=3,
                 isRandomObjects = True,
                 fixed_objects_init_pos=[],

                 # other
                 initPos_upper_space=(0.2, -0.45, 0.25),
                 initPos_lower_space=(-0.2, -0.55, 0.25),

                 reward_mode=0,
                 isRenderGoal = True,

                # render_params=DEFAULT_RENDER,
                 **kwargs):
        self.quick_init(locals())
        # import objects & init pos
        self.num_objects = num_movable_bodies

        self._isRandomObjects = isRandomObjects
        if self._isRandomObjects:
            obj_fixed_poses = None
        else:
            obj_fixed_poses=[]
            assert len(fixed_objects_init_pos) == self.num_objects * 3
            for i in range(self.num_objects):
                obj_fixed_poses.append(Pose([fixed_objects_init_pos[i*3:i*3+3],[0,0,0]]))
        self.objects_env = Objects(obj_name_list, num_movable_bodies,is_fixed = (not isRandomObjects),
                                   obj_fixed_poses = obj_fixed_poses, **kwargs)


        # goal setting
        self._isRandomGoals = isRandomGoals
        self._isIgnoreGoalCollision = isIgnoreGoalCollision
        self.fixed_objects_goals = fixed_objects_goals
        self.fixed_hand_goal = fixed_hand_goal
        self._target_upper_space = target_upper_space
        self._target_lower_space = target_lower_space

        self._isRenderGoal = isRenderGoal
        if self._isRenderGoal:
            self.goal_render_env = Objects(['ball_visual'], num_movable_bodies +1 ,
                                           obj_scale_range=(0.02, 0.02),
                                           use_random_rgba=True,
                                           num_RespawnObjects=None,
                                           is_fixed=True,
                                          )

        self._initPos_lower_space = initPos_lower_space
        self._initPos_upper_space = initPos_upper_space

        if os.getenv('REWARD_CHOICE') is not None:
            self.reward_mode = int(os.getenv('REWARD_CHOICE'))
            print('Reward Mode : ', self.reward_mode)
        else:
            self.reward_mode = reward_mode
            print ('Default Reward mode: ', self.reward_mode)

        Jaco2XYZEnv.__init__(self,  **kwargs)

    def _set_observation_space(self):

        obs_box = Box(np.array(self._hand_low),
                      np.array(self._hand_high), )

        state_obs_box = Box(np.tile(np.array(self._hand_low),self.num_objects+1),
                            np.tile(np.array(self._hand_high), self.num_objects + 1) )

        self.goal_box = Box(np.tile(np.array(self._target_lower_space),self.num_objects +1 ),
                            np.tile(np.array(self._target_upper_space), self.num_objects +1 ) )


        self.observation_space = Dict([
            ('observation',  obs_box),
            ('state_observation',  state_obs_box),
            ('desired_goal', self.goal_box),
            ('achieved_goal', state_obs_box),
            ('state_desired_goal', self.goal_box),
            ('state_achieved_goal', state_obs_box),

        ])

    def reset(self):
        super().reset()
        self.objects_env.reset()

        self.state_goal = self.sample_goal_for_rollout()

        if self._isRenderGoal:
            movable_poses = []
            # add hand pose
            pose = Pose([self.get_hand_goal_pos(), (0, 0, 0)])
            movable_poses.append(pose)
            # add obj pose
            for i in range(self.num_objects):
                pose = Pose([self.get_object_goal_pos(i),  (0,0,0)])
                movable_poses.append(pose)

            self.goal_render_env.reset(movable_poses)

        self._check_obs_dim()
        return  self._get_obs()

    def _check_obs_dim(self):
        obs = self._get_obs()

        for k in self.observation_space.spaces.keys():
            assert obs[k].shape == self.observation_space.spaces[k].shape , 'obs ({}) shape {} does not match the shape obs space {} '.format(k, obs[k].shape, self.observation_space.spaces[k].shape)


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
            observation=e,          # end-effector
            state_observation=x,    # end_effector & obj positions
            desired_goal=g,         # desired goal
            state_desired_goal=g,   # desired ee and obj goal
            achieved_goal=b,
            state_achieved_goal =x,
        )

        return new_obs

    def _get_info(self):
        ee_pos, ee_orn = self.robot.GetEndEffectorObersavations()

        hand_distance = np.linalg.norm(
            self.get_hand_goal_pos() - ee_pos
        )
        object_distances = {}
        touch_distances = {}
        for i in range(self.num_objects):
            object_name = "object%d_distance" % i
            object_distance = np.linalg.norm(
                self.get_object_goal_pos(i) - self.get_object_pos(i)
            )
            object_distances[object_name] = object_distance
            touch_name = "touch%d_distance" % i
            touch_distance = np.linalg.norm(
                ee_pos - self.get_object_pos(i)
            )
            touch_distances[touch_name] = touch_distance
        info = dict(
            #end_effector=[ee_pos, ee_orn],
            hand_distance=hand_distance,
            success=float(hand_distance + sum(object_distances.values()) < 0.06),
            **object_distances,
            **touch_distances,
        )

        return info

    def _reward(self, obs, action, others=None):
        reward = self.compute_reward(action, obs)

        return reward


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
        return 3 + self.num_objects*3

    @property
    def movable_bodies(self):
        return self.objects_env.movable_bodies


    def sample_goals(self, batch_size):
        goals = np.random.uniform(
            self.goal_box.low,
            self.goal_box.high,
            size=(batch_size, self.goal_dim),
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

    def get_object_goal_pos(self, i):
        x = 3 + 3 * i
        y = 6 + 3 * i
        return self.state_goal[x:y]
    def get_hand_goal_pos(self):
        return self.state_goal[:3]

    def get_object_pos(self, i):
        return self.movable_bodies[i].position

    def compute_rewards(self, action, obs, info=None):
        r = -np.linalg.norm(obs['state_observation'] - obs['state_desired_goal'], axis=1)
        return r

    def compute_reward(self, action, obs, info=None):
        r = -np.linalg.norm(obs['state_observation'] - obs['state_desired_goal'])
        return r


    def set_goal(self, goal):
        self.state_goal = goal['state_desired_goal']


    def sample_goal_for_rollout(self):
        if self._isRandomGoals:
            if not self._isIgnoreGoalCollision:
                poses_list = self.objects_env._sample_body_poses(self.num_objects)
                pos = []
                for pose in poses_list:
                    pos.append(pose.position)

                object_goals = np.concatenate(pos)

                GOAL_HAND_OBJ_MARGIN = 0.03
                while True:
                    hand_goal = np.random.uniform(self._hand_init_low, self._hand_init_high)
                    touching = []
                    for i in range(self.num_objects):
                        t = np.linalg.norm(hand_goal - object_goals[i*3:i*3+3]) < GOAL_HAND_OBJ_MARGIN
                        touching.append(t)

                    if not any(touching):
                        break

            else:
                hand_goal = np.random.uniform(self._hand_init_low, self._hand_init_high)
                object_goals = np.concatenate([np.random.uniform(self._target_lower_space, self._target_upper_space) for i in range(self.num_objects)])
        else:
            assert len(self.fixed_objects_goals) == self.num_objects * 3, "shape (3n, )"
            object_goals = np.array(self.fixed_objects_goals).copy()
            hand_goal = np.array(self.fixed_hand_goal).copy()
        g = np.hstack((hand_goal, object_goals))
        return g


    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']

        self.set_hand_xyz(state_goal[:3])

        object_poses =[]
        for i in range(self.num_objects):
            object_poses.append(Pose([state_goal[3+i*3:6+i*3],[0,0,0]]))
        self.objects_env._reset_movable_obecjts(object_poses)


    def get_env_state(self):
        # get robot states
        joint_state = self.robot.GetTrueMotorAngles()
        # get object states
        object_state = [x.pose for x in self.movable_bodies]

        state = joint_state, object_state
        return copy.deepcopy(state)

    def set_env_state(self, state):
        joint_state, object_state = state
        self.robot._ResetJointState(joint_state)
        self.objects_env._reset_movable_obecjts(object_state)

    def set_hand_xyz(self,pos):
        ee_pose = [pos[0], pos[1], pos[2],
                   self._robot_init_ee_orn[0], self._robot_init_ee_orn[1], self._robot_init_ee_orn[2],
                   self._robot_init_ee_orn[3]]
        self._reset_robot_end_efector(target_pose=ee_pose)


class Jaco2BlockPusherXYSimpleEnv(Jaco2BlockPusherXYZ):
    def __init__(self, **kwargs):
        Jaco2BlockPusherXYZ.__init__(self, **kwargs)

        if self._isRenderGoal:
            self.goal_render_env.remove_all_body()
            self.goal_render_env = Objects(['ball_visual'], self.num_objects,
                                           obj_scale_range=(0.02, 0.02),
                                           use_random_rgba=True,
                                           num_RespawnObjects=None,
                                           is_fixed=True,
                                           )
    def _set_observation_space(self):

        obs_box = Box(np.array(self._hand_low),
                      np.array(self._hand_high), )

        state_obs_box = Box(np.tile(np.array(self._hand_low),self.num_objects+1 ),
                            np.tile(np.array(self._hand_high), self.num_objects+1  ) )

        self.goal_box = Box(np.tile(np.array(self._target_lower_space),self.num_objects  ),
                            np.tile(np.array(self._target_upper_space), self.num_objects  ) )


        self.observation_space = Dict([
            ('observation',  state_obs_box),
            ('state_observation',  state_obs_box),
            ('desired_goal', self.goal_box),
            ('achieved_goal', self.goal_box),
            ('state_desired_goal', self.goal_box),
            ('state_achieved_goal', state_obs_box),

        ])

    def _get_obs(self):
        ee_pos = self.robot.GetEndEffectorObersavations()[0][:3]

        bs = []
        for body in self.movable_bodies:
            b = body.position
            bs.append(b)
        obj_pos = np.concatenate(bs)


        x = np.concatenate((ee_pos, obj_pos))
        g = self.state_goal

        new_obs = dict(
            observation=x,  # end-effector
            state_observation=x,  # end_effector & obj positions
            desired_goal=g,  # desired goal
            state_desired_goal=g,  # desired ee and obj goal
            achieved_goal=obj_pos,
            state_achieved_goal=x,
        )

        return new_obs

    def reset(self):
        obs = super().reset()

        if self._isRenderGoal:

            movable_poses = []
            # add hand pose
            if not self._reset_finished:  # hard code, just to avoid goal render error
                pose = Pose([self.get_hand_goal_pos(), (0, 0, 0)])
                movable_poses.append(pose)
            # add obj pose
            for i in range(self.num_objects):
                pose = Pose([self.get_object_goal_pos(i), (0, 0, 0)])
                movable_poses.append(pose)

            self.goal_render_env.reset(movable_poses)

        return obs

    def _reward(self, obs, action, others=None):
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])

        return reward

    @property
    def goal_dim(self) -> int:
        return  self.num_objects * 3

    def sample_goal_for_rollout(self):
        if self._isRandomGoals:
            if not self._isIgnoreGoalCollision:
                poses_list = self.objects_env._sample_body_poses(self.num_objects)
                pos = []
                for pose in poses_list:
                    pos.append(pose.position)

                object_goals = np.concatenate(pos)

                # GOAL_HAND_OBJ_MARGIN = 0.03
                # while True:
                #     hand_goal = np.random.uniform(self._hand_init_low, self._hand_init_high)
                #     touching = []
                #     for i in range(self.num_objects):
                #         t = np.linalg.norm(hand_goal - object_goals[i*3:i*3+3]) < GOAL_HAND_OBJ_MARGIN
                #         touching.append(t)
                #
                #     if not any(touching):
                #         break

            else:
                #hand_goal = np.random.uniform(self._hand_init_low, self._hand_init_high)
                object_goals = np.concatenate([np.random.uniform(self._target_lower_space, self._target_upper_space) for i in range(self.num_objects)])
        else:
            assert len(self.fixed_objects_goals) == self.num_objects * 3, "shape (3n, )"
            object_goals = np.array(self.fixed_objects_goals).copy()
            #hand_goal = np.array(self.fixed_hand_goal).copy()

        return object_goals


    def get_object_goal_pos(self, i):
        x =   3 * i
        y = 3 + 3 * i
        return self.state_goal[x:y]



    def compute_rewards(self, action, obs, info=None):
        r = -np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'], axis=1)
        return r

    # def compute_reward(self, action, obs, info=None):
    #     r = -np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
    #     return r

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        if len(achieved_goal.shape) == 1:
             r = -np.linalg.norm(achieved_goal - desired_goal)
        elif len(achieved_goal.shape) == 2:
            r = -np.linalg.norm(achieved_goal - desired_goal, axis=1)
        else:
            raise NotImplementedError
        return r

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
            touch_name = "touch%d_distance" % i
            touch_distance = np.linalg.norm(
                ee_pos - self.get_object_pos(i)
            )
            touch_distances[touch_name] = touch_distance
        info = dict(
            end_effector=[ee_pos, ee_orn],

            is_success=float( sum(object_distances.values()) < 0.05),
            **object_distances,
            **touch_distances,
        )

        return info

    def set_goal(self, goal):
        self.state_goal = goal['state_desired_goal']


    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']

        object_poses = []
        for i in range(self.num_objects):
            object_poses.append(Pose([state_goal[0 + i * 3:3 + i * 3], [0, 0, 0]]))
        self.objects_env._reset_movable_obecjts(object_poses)

