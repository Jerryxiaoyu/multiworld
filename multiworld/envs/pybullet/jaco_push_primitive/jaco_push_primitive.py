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

#from multiworld.utils.logging import logger
#from .params import *
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

__all__ = ['Jaco2PushPrimitiveXY' , 'Jaco2PushPrimitiveXYyaw']
SUCCESS_THESHOLD = 0.02


DELTA_COM_X = 0.01
DELTA_COM_Y = -0.005
# DELTA_COM_X = 0.0
# DELTA_COM_Y = 0


class Jaco2PushPrimitiveXY(Jaco2XYZEnv,   MultitaskEnv):
    def __init__(self,

                 # goal setting
                 # target
                 isRandomGoals=True,
                 isIgnoreGoalCollision=False,
                 fixed_objects_goals=[],
                 fixed_hand_goal=[],
                 target_upper_space=(0.2, -0.75, 0.25),
                 target_lower_space=(-0.2, -0.85, 0.25),

                 goal_order =['x','y'],
                 isGoalImg = False,

                 isImageObservation = False,
                 # obj
                 obj_name_list=[],
                 num_movable_bodies=3,
                 isRandomObjects=True,
                 fixed_objects_init_pos=[],

                 # primitive
                 push_delta_scale_x=0.1,
                 push_delta_scale_y=0.1,
                 steps_check=50,
                 gripper_safe_height=0.35,
                 offstage_pose=(-0.27536532, -0.14509082, 0.33775887 ,
                                 0.29393724, 0.94632232, -0.13347848, -0.01607945 ),
                 offstage_joint_pos=(),
                 max_phase_stpes=200,
                 max_offstage_steps=800,
                 max_motion_steps=200,
                 num_goal_stpes=None,
                 maxActionSteps=None,

                 # other params
                 skip_first= 0,
                 reward_mode = 0,
                 init_skip_timestep = 100,
                 is_debug_camera= False,
                 debug_info = False,

                 isRenderGoal=True,
                 vis_debug = False,
                 is_render_object_pose = False,

                 is_enable_file_pose = False,
                 file_pose_path = None,

                 **kwargs):
        self.quick_init(locals())

        self.debug_info = debug_info
        self.vis_debug = vis_debug
        self.is_render_object_pose = is_render_object_pose
        self.is_enable_file_pose = False
        if is_enable_file_pose:
            if  goal_order == ['x', 'y']:
                fixed_objects_goals = np.zeros((num_movable_bodies, 2))
            elif  goal_order == ['x', 'y', 'theta']:
                fixed_objects_goals = np.zeros((num_movable_bodies, 3))
            else:
                raise NotImplementedError

        # push primitive params
        self.PUSH_DELTA_SCALE_X = push_delta_scale_x
        self.PUSH_DELTA_SCALE_Y = push_delta_scale_y
        self.STEPS_CHECK = steps_check
        self.GRIPPER_SAFE_HEIGHT = gripper_safe_height
        self.OFFSTAGE_POSE = offstage_pose
        self.OFFSTAGE_JOINT_POS = offstage_joint_pos
        self.MAX_PHASE_STEPS = max_phase_stpes
        self.MAX_OFFSTAGE_STEPS = max_offstage_steps
        self.MAX_MOTION_STEPS= max_motion_steps
        self.NUM_GOAL_STEPS = num_goal_stpes
        self.MAX_ACTION_STEPS = maxActionSteps

        # Execution phases.
        self.phase_list = ['initial', 'pre', 'start', 'motion',
                           'post', 'offstage',  'done']

        # import objects & init pos
        self.num_objects = num_movable_bodies

        self._isRandomObjects = isRandomObjects
        if is_enable_file_pose:
            obj_fixed_poses =[]
            for i in range(self.num_objects):
                obj_fixed_poses.append(Pose([[0,0,0], [0, 0, 0]]))

        else:
            if self._isRandomObjects:
                obj_fixed_poses = None
            else:
                obj_fixed_poses = []
                assert len(fixed_objects_init_pos) == self.num_objects * 3
                for i in range(self.num_objects):
                    obj_fixed_poses.append(Pose([fixed_objects_init_pos[i * 3:i * 3 + 3], [0, 0, 0]]))
        self.objects_env = Objects(obj_name_list, num_movable_bodies, is_fixed=(not isRandomObjects),
                                   obj_fixed_poses=obj_fixed_poses, **kwargs)

        # goal setting
        self._isRandomGoals = isRandomGoals
        self._isIgnoreGoalCollision = isIgnoreGoalCollision
        self.fixed_objects_goals =  fixed_objects_goals
        self.fixed_hand_goal = fixed_hand_goal
        self._target_upper_space = target_upper_space
        self._target_lower_space = target_lower_space

        self._isRenderGoal = isRenderGoal
        if self._isRenderGoal:
            self.goal_render_env = Objects(['ball_visual'], num_movable_bodies,
                                           obj_scale_range=(0.02, 0.02),
                                           use_random_rgba=True,
                                           num_RespawnObjects=None,
                                           is_fixed=True,
                                           )



        self.goal_order= goal_order
        self.isGoalImg = isGoalImg

        self._isImageObservation =  isImageObservation
        # others
        self._skip_first = skip_first
        self.INIT_SKIP_TIMESTEP = init_skip_timestep

        if os.getenv('REWARD_CHOICE') is not None:
            self.reward_mode = int(os.getenv('REWARD_CHOICE'))
            #logger.info('Setting reward mode : %s'% self.reward_mode)
        else:
            self.reward_mode = reward_mode
            print('reward mode:', reward_mode)
            #logger.info('Setting default Reward mode: %s'%self.reward_mode)

        # Visualization.
        if self.vis_debug:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            plt.ion()
            plt.show()
            self.ax = ax
            self.vis_plot_count = 0
            time_now = time.strftime('%Y-%m-%d_%H%M%S', time.localtime(time.time()))
            plot_name = "debug_plot_{}".format(time_now)
            self.plot_res_folder = os.path.join('/home/drl/res_plot', plot_name)
            os.makedirs(self.plot_res_folder)

        self._debug_goal_ids = []
        Jaco2XYZEnv.__init__(self, **kwargs)

        # (H, n, 7)
        self.is_enable_file_pose = is_enable_file_pose
        if self.is_enable_file_pose:
            from multiworld.utils.io import IO

            self.file_pose_path = file_pose_path
            pose_dict = IO(self.file_pose_path).read_pickle()
            self.object_init_pose_list = pose_dict['init_pose']
            self.object_goal_pose_list = pose_dict['goal_pose']
            # self.object_init_pose_list = np.array([[[0, -0.35, 0.05, 0,0,np.pi/3],
            #                                         [-0.2, -0.35, 0.05, 0, 0, np.pi / 3],
            #                                         ],
            #                                        [[0.1, -0.35, 0.05, 0, 0, np.pi / 3],
            #                                         [-0.2, -0.35, 0.05, 0, 0, np.pi / 3],
            #                                         ]
            #                                        ])
            # self.object_goal_pose_list = np.array([[[0.2, -0.35, 0.05, 0, 0, 1 ],
            #                                         [0.25, -0.35, 0.05, 0, 0, 1],
            #                                         ],
            #                                        [[0.25, -0.35, 0.05, 0, 0, 1],
            #                                         [0.25, -0.35, 0.05, 0, 0, 1],
            #                                         ]
            #                                        ])
        self.current_file_index = 0

    def _set_observation_space(self):
        if self.goal_order == ['x','y']:
            low_space = [self.objects_env.object_max_space_low[0] ,
                         self.objects_env.object_max_space_low[1]]
            high_space = [self.objects_env.object_max_space_high[0] ,
                         self.objects_env.object_max_space_high[1]]
            self.obs_box = Box(np.tile(np.array(low_space), self.num_objects).reshape((-1,2)),
                               np.tile(np.array(high_space), self.num_objects).reshape((-1,2)))

            self.state_obs_box = self.obs_box

            low_space = [self._target_lower_space[0],
                         self._target_lower_space[1]]
            high_space = [self._target_upper_space[0],
                          self._target_upper_space[1]]
            self.goal_box = Box(np.tile(np.array(low_space),   self.num_objects ).reshape((-1,2)),
                                 np.tile(np.array(high_space), self.num_objects ).reshape((-1,2)))

            self.state_goal_box = self.goal_box

        elif self.goal_order == ['x', 'y', 'theta']:
            low_space = [self.objects_env.object_max_space_low[0],
                         self.objects_env.object_max_space_low[1],
                         -np.pi]
            high_space = [self.objects_env.object_max_space_high[0],
                          self.objects_env.object_max_space_high[1],
                          np.pi]
            self.obs_box = Box(np.tile(np.array(low_space), self.num_objects ).reshape((-1,3)),
                          np.tile(np.array(high_space), self.num_objects ).reshape((-1,3)))

            self.state_obs_box = Box(np.tile(np.array(low_space), self.num_objects).reshape((-1,3)),
                                     np.tile(np.array(high_space), self.num_objects).reshape((-1,3)))

            low_space = [self._target_lower_space[0],
                         self._target_lower_space[1],
                         -np.pi]
            high_space = [self._target_upper_space[0],
                          self._target_upper_space[1],
                          np.pi]
            self.goal_box = Box(np.tile(np.array(low_space), self.num_objects).reshape((-1,3)),
                                np.tile(np.array(high_space), self.num_objects).reshape((-1,3)))

            self.state_goal_box = self.goal_box

        self.observation_space = Dict([
            ('observation',  self.obs_box),                  # object position [x,y]*n
            ('state_observation',  self.state_obs_box),      # object pose [x,y,z,roll picth, yaw] *n
            ('desired_goal', self.goal_box),                 # goal object position [x,y]*n
            ('achieved_goal', self.obs_box),                 # actual object position [x,y]*n
            ('state_desired_goal', self.state_goal_box),     # goal object pose [x,y,z,roll picth, yaw] *n
            ('state_achieved_goal', self.state_obs_box),     # actual pose [x,y,z,roll picth, yaw] *n

        ])

    def reset(self):
        super().reset()

        if self.is_enable_file_pose:
            cur_pose_list = self.object_init_pose_list[self.current_file_index].flatten()
            movable_poses =[]
            for i in range(self.num_objects):
                movable_poses.append(Pose([cur_pose_list[i * 6:i * 6 + 3], cur_pose_list[i * 6+3:i * 6 + 6]]))
            # self.current_file_index += 1
            # self.current_file_index % self.object_init_pose_list.shape[0]
        else:
            movable_poses = None
        self.objects_env.reset(movable_poses=movable_poses)
        self.state_goal = self.sample_goal_for_rollout()

        if self.is_enable_file_pose:
            self.current_file_index += 1
            self.current_file_index %= self.object_init_pose_list.shape[0]
            print("Current_file_index: ",  self.current_file_index)

        if self._isRenderGoal:
            # movable_poses = []
            #
            # # add obj pose
            # for i in range(self.num_objects):
            #     pose = Pose([self.get_object_goal_pos(i),  self.get_object_goal_orn(i)])
            #     movable_poses.append(pose)
            #     #plot_pose(pose, axis_length=0.1)
            # self.goal_render_env.reset(movable_poses)

            if len(self._debug_goal_ids) != 0:
                for i in range(len(self._debug_goal_ids)):
                    self._p.removeUserDebugItem(self._debug_goal_ids[i])
                self._debug_goal_ids = []
                clear_visualization()
            for i in range(self.num_objects):
                goal_pos = self.get_object_goal_pos(i)
                debug_goal_id = self._p.addUserDebugLine(goal_pos, [goal_pos[0], goal_pos[1], goal_pos[2]+0.03], [255, 0, 0], 5)
                self._debug_goal_ids.append(debug_goal_id)
                pose = Pose([self.get_object_goal_pos(i), self.get_object_goal_orn(i)])
                plot_pose(pose, axis_length=0.1)


        for i in range(self.INIT_SKIP_TIMESTEP):
            self._p.stepSimulation()
        self._envStepCounter = 0
        self._envActionSteps = 0

        self.plot_boundary()

        if self.vis_debug:
            self.vis_plot_count += 1
            self.plot_res  = os.path.join(self.plot_res_folder, str(self.vis_plot_count))
            os.makedirs(self.plot_res)

        #self._check_obs_dim()
        return  self._get_obs()

    def plot_goalAndObject_pose(self):
        clear_visualization()
        for i in range(self.num_objects):
            pose = Pose([self.get_object_goal_pos(i), self.get_object_goal_orn(i)])
            plot_pose(pose, axis_length=0.1)

            pose = self.get_object_pose(i)
            plot_pose(pose, axis_length=0.1)

    def _check_obs_dim(self):
        obs = self._get_obs()

        for k in self.observation_space.spaces.keys():
            assert obs[k].shape == self.observation_space.spaces[k].shape , 'obs {} shape {} does not match the shape obs space {} '.format(k, obs[k].shape, self.observation_space.spaces[k].shape)

    def _get_obs(self):
        if self._isImageObservation:
            images = self.camera.frames()
            # image
            rgb = images['rgb']
            depth = images['depth']
            segmask = images['segmask']

        if self.goal_order == ['x', 'y']:
            bs = []
            for body in self.movable_bodies:
                bs.append(body.position[:2])
            pos = np.array(bs)

            state = np.array(pos)
            state_goal = self.state_goal
        elif self.goal_order == ['x', 'y', 'theta']:

            pos_orn = []
            for body in self.movable_bodies:
                pos_orn.append(np.concatenate((body.position[:2], np.array([body.orientation.euler[2]]))))

            state =  np.array(pos_orn)#.flatten()
            state_goal = self.state_goal

        new_obs = dict(
            observation= state,              # obj pos [x,y]*n
            state_observation=state,         # obj pos [x,y]*n
            desired_goal=state_goal,         # desired goal: obj pos [x,y]*n
            state_desired_goal=state_goal,   # desired goal: obj pos [x,y]*n
            achieved_goal=state,
            state_achieved_goal =state,

        )
        if self._isImageObservation:
            new_obs['image'] = rgb
            new_obs['depth'] = depth
            new_obs['segmask'] = segmask
        return new_obs


    def step(self, action):
        self._excute_action( action)
        self._envActionSteps += 1

        self._observation = self._get_obs()

        self.check_dis_objects()

        info = self._get_info()

        reward = self._reward(self._observation, action)
        done, reward_terminal = self._termination()
        reward += reward_terminal

        return self._observation, reward, done, info

    def _termination(self ):
        if self.is_success:
            return True, 0
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
                print('out of range!!!')
            return True, 0

        return False, 0

    def simulation_step(self):
        if self._envStepCounter % 5  ==0:
            self.robot.send_joint_command()

        self._p.stepSimulation()
        self._envStepCounter += 1

        if self._renders is True and self._envStepCounter % 100 == 0:
            time.sleep(0.001)
            time_info = 'step: %d' % self._envStepCounter + '  ' + 'time: %.2fs' \
                        % (self._envStepCounter * self._timeStep) + '  ' + 'actStep: %d' % self._envActionSteps
            self.debug_ino_step = self._p.addUserDebugText(time_info, self._stepTextPosition,
                                                           textColorRGB=[1, 0, 0], textSize=1.5,
                                                           replaceItemUniqueId=self.debug_ino_step)
            if self.is_render_object_pose:
                self.plot_goalAndObject_pose()
    def _excute_action(self, action):
        """
        :param action:
        :return:
        """

        self.attributes = {
           # 'num_episodes': self.num_episodes,
           # 'num_steps': self.num_steps,
            #'layout_id': self.layout_id,
           # 'movable_body_mask':  movable_body_mask,
            'is_safe': True,
            'is_effective': True,
            'is_turnover':False
        }

        waypoints = self._compute_all_waypoints(action)

        self.phase = 'initial'
        self.num_waypoints = 0
        self.interrupt = False
        self.max_phase_steps=None

        if self.phase == 'initial':
            self.robot._ResetJointState()
            self.robot.move_to_gripper_pose(self.OFFSTAGE_POSE)
            #self.robot.move_to_joint_positions(self.OFFSTAGE_JOINT_POS)

        self.start_status = self._get_movable_status()

        while (self.phase != 'done'):

            self.simulation_step()
            if not (self._envStepCounter %
                    self.STEPS_CHECK == 0):
                continue

            # Phase transition.
            if self._is_phase_ready():
                self.phase = self._get_next_phase()

                self.max_phase_steps = self._envStepCounter
                if self.phase == 'motion':
                    self.max_phase_steps += (self.MAX_MOTION_STEPS )
                elif self.phase == 'offstage' or self.phase == 'pre' :
                    self.max_phase_steps += (self.MAX_OFFSTAGE_STEPS )
                else:
                    self.max_phase_steps += (self.MAX_PHASE_STEPS )

                if self.phase == 'pre':
                    pose = waypoints[self.num_waypoints][0].copy()
                    #pose.z =  GRIPPER_SAFE_HEIGHT
                    pose[2] = self.GRIPPER_SAFE_HEIGHT
                    self.robot.move_to_gripper_pose(pose)

                elif self.phase == 'start':
                    pose = waypoints[self.num_waypoints][0]
                    self.robot.move_to_gripper_pose(pose)

                elif self.phase == 'motion':
                    pose = waypoints[self.num_waypoints][1]
                    self.robot.move_to_gripper_pose(pose)

                elif self.phase == 'post':
                    self.num_waypoints += 1
                    pose = self.robot.end_effector_pose
                    #pose.z =  GRIPPER_SAFE_HEIGHT
                    pose[2] = self.GRIPPER_SAFE_HEIGHT
                    self.robot.move_to_gripper_pose(pose)

                elif self.phase == 'offstage':
                    self.robot._ResetJointState()
                    self.robot.move_to_gripper_pose(self.OFFSTAGE_POSE)
                    #self.robot.move_to_joint_positions(self.OFFSTAGE_JOINT_POS)

            self.interrupt = False

            #if not self._check_safety():
            #    self.interrupt = True
            #    self.attributes['is_safe'] = False

            if self.interrupt:
                if self.phase == 'done':
                    self._done = True
                    break

        # wait for stable
        for _ in range(100):
            self.simulation_step()

        self.end_status = self._get_movable_status()

        if self._check_turnover():
            self.attributes['is_effective'] = False
            self.attributes['is_turnover'] = True

        else:
            self.attributes['is_effective'] = self._check_effectiveness()

        return True
    def _is_phase_ready(self):

        if self.interrupt:
            return True

        if self.robot.is_limb_reached() and self.robot.is_gripper_reached():
            #self.robot.arm.reset_targets()
            return True

        if self.max_phase_steps is None:
            return False
        if self._envStepCounter >= self.max_phase_steps:
            if self.debug_info:
                logger.warning('Phase %s timeout.', self.phase)
            #self.robot.arm.reset_targets()
            return True

        return False
    def _get_next_phase(self):
        """Get the next phase of the current phase.

        Returns:
            The next phase as a string variable.
        """
        if self.phase in self.phase_list:
            if self.interrupt:
                if self.phase not in ['post', 'offstage', 'done']:
                    return 'post'

            i = self.phase_list.index(self.phase)
            if i >= len(self.phase_list):
                raise ValueError('phase %r does not have a next phase.')
            else:
                if self.NUM_GOAL_STEPS is not None:
                    if (self.phase == 'post' and
                            self.num_waypoints < self.NUM_GOAL_STEPS):
                        return 'pre'

                return self.phase_list[i + 1]
        else:
            raise ValueError('Unrecognized phase: %r' % self.phase)
    def _compute_all_waypoints(self, action):
        """Convert action of a single step or multiple steps to waypoints.

        Args:
            action: Action of a single step or actions of multiple steps.

        Returns:
            List of waypoints of a single step or multiple steps.
        """
        if self.NUM_GOAL_STEPS is None:
            waypoints = [self._compute_waypoints(action)]
        else:
            waypoints = [
                self._compute_waypoints(action[i])
                for i in range(self.NUM_GOAL_STEPS)]
        return waypoints
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
        x = start[0] * self.start_range[0]  + self.start_offset[0]  + DELTA_COM_X
        y = start[1] * self.start_range[1]  + self.start_offset[1] + DELTA_COM_Y
        z = self.start_offset[2]
        angle = 0.0

        start = [x, y, z] + self._fixed_orn

        # End.
        delta_x = motion[0] * self.PUSH_DELTA_SCALE_X
        delta_y = motion[1] * self.PUSH_DELTA_SCALE_Y
        x = x + delta_x + DELTA_COM_X
        y = y + delta_y + DELTA_COM_Y

        x = np.clip(x, self.robot.end_effector_lower_space[0], self.robot.end_effector_upper_space[0])
        y = np.clip(y, self.robot.end_effector_lower_space[1], self.robot.end_effector_upper_space[1])
        end =  [x, y, z] + self._fixed_orn

        waypoints = [start, end]
        return waypoints
    def _check_effectiveness(self):
        """Check if the action is effective.

        Returns:
            True if at least one of the object has a translation or orientation
                larger than the threshold, False otherwise.
        """
        MIN_DELTA_POSITION = 0.01
        MIN_DELTA_ANGLE = 0.01

        is_simulation = True
        if  is_simulation:
            delta_position = np.linalg.norm(
                    self.end_status[0] - self.start_status[0], axis=-1)
            delta_position = np.sum(delta_position)

            delta_angle = self.end_status[1][:,2] - self.start_status[1][:,2]  ## only check yaw for each obj
            delta_angle = (delta_angle + np.pi) % (2 * np.pi) - np.pi
            delta_angle = np.abs(delta_angle)
            delta_angle = np.sum(delta_angle)

            if (delta_position <=  MIN_DELTA_POSITION and
                    delta_angle <=   MIN_DELTA_ANGLE):
                #if self.config.DEBUG:
                #    logger.warning('Ineffective action.')
                return False

        return True
    def _check_turnover(self):
        TURNOVER_THRESHOLD = math.pi/8.
        is_simulation = True
        if is_simulation:

            delta_angle = abs(self.end_status[1][:,:2]  - self.start_status[1][:,:2])  ## only check yaw for each obj

            if (delta_angle > TURNOVER_THRESHOLD).any():
                # if self.config.DEBUG:
                #    logger.warning('Ineffective action.')
                return True
        return False
    def _check_outof_range(self):
        is_simulation = True
        if is_simulation:

            pos_xy = self.end_status[0][:,:2]

            for i in range(pos_xy.shape[0]):
                point = Point(pos_xy[i][0], pos_xy[i][1])
                if not point.within(self.objects_env.object_space_polygon):
                    return True

        return False
    def _get_movable_status(self):
        """Get the status of the movable objects.

        Returns:
            Concatenation of the positions and Euler angles of all objects in
                the simulation, None in the real world.
        """
        is_simulation = True
        if is_simulation:
            positions = [body.position for body in self.movable_bodies]
            angles = [body.orientation.euler for body in self.movable_bodies]
            return [np.stack(positions, axis=0), np.stack(angles, axis=0)]

        return None

    def check_dis_objects(self):
        object_distances = {}
        object_eluler_error = {}
        for i in range(self.num_objects):
            object_name = "object%d_distance" % i
            object_distance = np.linalg.norm(
                self.get_object_goal_pos(i) - self.get_object_pos(i)
            )
            eluer_angle = self.get_object_goal_orn(i)[2] - self.get_object_eluer(i)[2]

            object_distances[object_name] = object_distance

            if self.goal_order == ['x', 'y', 'theta']:
                euler_error = abs(eluer_angle) * 180/np.pi
                object_name = "object%d_euler_error" % i
                object_eluler_error[object_name] = euler_error

        if self.goal_order == ['x', 'y']:
            self._is_success = float(sum(object_distances.values()) / self.num_objects < SUCCESS_THESHOLD)
        elif self.goal_order == ['x', 'y', 'theta']:
            mean_dis = sum(object_distances.values())/self.num_objects
            mean_angle = sum(object_eluler_error.values())/self.num_objects

            self._is_success = mean_dis < SUCCESS_THESHOLD and mean_angle < 5
        else:
            raise NotImplementedError

        self._object_distances = object_distances
        self._object_eluler_error = object_eluler_error
        return  [object_distances, object_eluler_error]

    @property
    def is_success(self):
        return self._is_success
    @property
    def object_distances(self):
        return self._object_distances

    @property
    def object_eluler_error(self):
        return self._object_eluler_error

    def _get_info(self):
        ee_pos, ee_orn = self.robot.GetEndEffectorObersavations()
        info = dict(
            end_effector=[ee_pos, ee_orn],
            success=self._is_success,
            **self.object_distances,
            **self.object_eluler_error,
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
        return self.state_goal_box.shape

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

    def get_object_eluer(self, i):
        return self.movable_bodies[i].orientation.euler

    def get_object_pose(self, i):
        return self.movable_bodies[i].pose

    def compute_rewards(self, action, obs, info=None):
        r = -np.linalg.norm(obs['state_observation'] - obs['state_desired_goal'], axis=1)
        return r

    def compute_reward(self, action, obs, info=None):
        r = -np.linalg.norm(obs['state_observation'] - obs['state_desired_goal'])
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
        assert goal['state_desired_goal'].shape  == self.goal_dim , "goal['state_desired_goal'].shape: ".format(goal['state_desired_goal'].shape)

        state_goal = goal['state_desired_goal']
       # euler_orns = [[-math.pi,-math.pi/2,0], [0,0,0]]
       #  object_poses =[]
       #  for i in range(self.num_objects):
       #      if i ==1:
       #          object_poses.append( self.movable_bodies[1].pose)
       #      else:
       #          pos = [0, -0.45, 0.06]#self.get_object_goal_pos_from_stategoal(state_goal, i)
       #          euler_orn = euler_orns[i]#self.get_object_goal_euler_orn_from_stategoal(state_goal, i)
       #          object_poses.append(Pose([pos,euler_orn]))
        object_poses = []
        for i in range(self.num_objects):
            pos = self.get_object_goal_pos_from_stategoal(state_goal, i)
            euler_orn =  self.get_object_goal_euler_orn_from_stategoal(state_goal, i)
            object_poses.append(Pose([pos, euler_orn]))

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


    def get_object_goal_pos(self, i):
        if self.goal_order == ['x', 'y']:
            # x = self.state_goal[2*i]
            # y = self.state_goal[2*i+1]
            x = self.state_goal[i, 0]
            y = self.state_goal[i, 1]
            z = 0.05

        elif self.goal_order == ['x', 'y', 'theta']:
            x = self.state_goal[i, 0]
            y = self.state_goal[i, 1]
            # x = self.state_goal[3 * i]
            # y = self.state_goal[3 * i + 1]
            z = 0.05

        return  [x,y,z]

    def get_object_goal_orn(self, i):
        roll = 0
        pitch = 0
        yaw = 0

        if self.goal_order == ['x', 'y', 'theta']:
            #yaw = self.state_goal[3 * i + 2]
            yaw = self.state_goal[i, 2]

        return [roll, pitch, yaw]

    def get_object_goal_pos_from_stategoal(self, state_goal, i):
        if self.goal_order == ['x', 'y']:
            x =  state_goal[i, 0]
            y =  state_goal[i, 1]
            # x =  state_goal[2*i]
            # y =  state_goal[2*i+1]
            z = 0.05

        elif self.goal_order == ['x', 'y', 'theta']:
            x =  state_goal[i, 0]
            y =  state_goal[i, 1]
            # x =  state_goal[3 * i]
            # y =  state_goal[3 * i + 1]
            z = 0.05

        return  [x, y, z]

    def get_object_goal_euler_orn_from_stategoal(self, state_goal, i):

        roll = 0
        pitch = 0
        yaw = 0

        if self.goal_order == ['x', 'y', 'theta']:
            #yaw = state_goal[3 * i + 2]
            yaw =  state_goal[i, 2]

        return [0,0,yaw]

    def sample_goal_for_rollout(self):
        if self._isRandomGoals:
            if not self._isIgnoreGoalCollision:
                poses_list = self.objects_env._sample_body_poses(self.num_objects)

                if self.goal_order ==['x', 'y']:
                    pos = []

                    for pose in poses_list:
                        pos.append(pose.position[:2])
                        #orn.append(pose.orientation.euler[2])
                    pos = np.array(pos)
                    #orn = np.array(orn)

                    object_goals = pos
                elif self.goal_order == ['x', 'y', 'theta']:
                    pos_orn = []
                    for pose in poses_list:
                        pos_orn.append(np.concatenate((pose.position[:2],  np.array([pose.orientation.euler[2]]))))

                    object_goals = np.array(pos_orn)#.flatten()
                else:
                    raise NotImplementedError

            else:
                if self.goal_order ==['x', 'y']:
                    object_goals = np.concatenate([np.random.uniform(self._target_lower_space[:2],
                                                                     self._target_upper_space[:2]) for _ in
                                                   range(self.num_objects)]).reshape((-1, 2))
                elif self.goal_order == ['x', 'y', 'theta']:
                    pos_goals = np.concatenate([np.random.uniform(self._target_lower_space[:2],
                                                                     self._target_upper_space[:2]) for _ in
                                                   range(self.num_objects)]).reshape((-1, 2))
                    yaw_goals = np.random.uniform(-np.pi,np.pi, size=(self.num_objects, 1))
                    object_goals = np.concatenate([pos_goals, yaw_goals], axis=-1)
                else:
                    raise NotImplementedError

        else:
            if self.is_enable_file_pose:
                if self.goal_order ==['x', 'y']:
                    object_goals = self.object_goal_pose_list[self.current_file_index][:, :2]

                elif self.goal_order == ['x', 'y', 'theta']:
                    pos_goals = self.object_goal_pose_list[self.current_file_index][:, :2]
                    yaw_goals = self.object_goal_pose_list[self.current_file_index][:, -1:]
                    object_goals = np.concatenate([pos_goals, yaw_goals], axis=-1)
                else:
                    raise NotImplementedError

            else:

                # assert len(self.fixed_objects_goals) == self.goal_dim, "require the shape of {} is {}, but got {}".format(
                #     'fixed_objects_goals', self.goal_dim, len(self.fixed_objects_goals))
                assert  np.array(self.fixed_objects_goals).shape == self.goal_dim
                object_goals = np.array(self.fixed_objects_goals).copy()

       ## object_goals = np.array([np.random.uniform( -0.06, -0.15) , np.random.uniform(-0.37,-0.45)])  hard code
        return object_goals

    def visualize(self, action, info):
        MAX_STATE_PLOTS = 10

        # Reset.
        images = self.camera.frames()
        rgb = images['rgb']
        self.ax.cla()
        self.ax.imshow(rgb)

        obj_index = 0

        ##-----your plot
        states = self.get_object_pos(obj_index)[:2]

        # if 'best_pred_states' in info:
        #     pred_states = info['best_pred_states']
        #
        #     num_info_samples = info['num_samples']
        #     max_plots = min(num_info_samples,  MAX_STATE_PLOTS)
        #     for i in range(max_plots):
        #         if i == 0:
        #             c = 'orange'
        #             alpha = 0.8
        #         else:
        #             c = 'gold'
        #             alpha = 0.4
        #         self._plot_single_state_trajectory(self.ax,
        #                                states,
        #                                pred_states[i],
        #                                #terminations[i],
        #                                c=c,
        #                                alpha=alpha)

        n_hor = info['best_actions'][0].shape[0]

        n_hor = 1

        # plot action

        c = 'red'  # 'royalblue'
        linewidth = 3.0
        alpha = 0.8

        waypoints = self._compute_waypoints(action)
        self._plot_waypoints(self.ax,
                             waypoints,
                             linewidth=linewidth,
                             c=c,
                             alpha=0.5)

        # n_hor = 1
        # for t in range(n_hor):
        #     action = info['best_actions'][0][t]
        #     # plot action
        #     if t ==0:
        #         c = 'red'#'royalblue'
        #         linewidth = 3.0
        #         alpha = 0.8
        #     else:
        #         alpha -= 0.1
        #         alpha = max(0.2, alpha)
        #     waypoints = self._compute_waypoints(action)
        #     self._plot_waypoints(self.ax,
        #                          waypoints,
        #                          linewidth=linewidth,
        #                          c=c,
        #                          alpha=0.5)

        num_traj_itr = info['best_pred_states'].shape[0]
        c_list =['forestgreen', 'olive', 'lawngreen', 'red', 'yellow']
        # for itr in range(info['itr']-1):
        #     num_traj_itr = min(num_traj_itr, MAX_STATE_PLOTS)
        #     for i in range(num_traj_itr):
        #         if i ==0:
        #             alpha =0.8
        #         else:
        #             alpha =0.4
        #         pred_states = info['stat_info']['traj_state_{}'.format(itr)]
        #
        #         self._plot_single_state_trajectory(self.ax,
        #                                            states,
        #                                            pred_states[i],
        #                                            # terminations[i],
        #                                            c=c_list[itr],
        #                                            alpha=alpha)

        # fiel_name = os.path.join(self.plot_res, 'debug_{}.png'.format(self._envActionSteps) )
        # plt.savefig(fiel_name)

        ##-----
        plt.draw()
        plt.pause(1e-3)
    def visualize_random_action(self, action, info):
        MAX_STATE_PLOTS = 10

        # Reset.
        images = self.camera.frames()
        rgb = images['rgb']
        self.ax.cla()
        self.ax.imshow(rgb)

        obj_index = 0

        ##-----your plot
        states = self.get_object_pos(obj_index)[:2]

        t=0
        # plot action
        if t ==0:
            c = 'red'#'royalblue'
            linewidth = 3.0
            alpha = 0.8

        waypoints = self._compute_waypoints(action)
        self._plot_waypoints(self.ax,
                             waypoints,
                             linewidth=linewidth,
                             c=c,
                             alpha=0.5)


        c_list =['forestgreen', 'olive', 'lawngreen', 'red', 'yellow']


        ##-----
        plt.draw()
        plt.pause(1e-3)

    def visualize_sna(self, action, info):
        MAX_STATE_PLOTS = 10


        # Reset.
        images = self.camera.frames()
        rgb = images['rgb']
        self.ax.cla()
        self.ax.imshow(rgb)

        body_idx = 0
        goal_pixel = self.camera.project_point([self.state_goal[body_idx, 0],self.state_goal[body_idx, 1], 0.035], is_world_frame=True)
        self.ax.scatter(goal_pixel[0], goal_pixel[1], c='r', alpha=1, s=8.0, marker='x')

        obj_index = 0

        ##-----your plot
        states = self.get_object_pos(obj_index)
        pixel_state = self.camera.project_point(states, is_world_frame=True )

        if 'best_pred_states' in info:
            pred_states = info['best_pred_states'] *np.array([640/64., 480/64.])


            num_info_samples = info['num_samples']
            max_plots = min(num_info_samples,  MAX_STATE_PLOTS)
            for i in range(max_plots):
                if i == 0:
                    c = 'orange'
                    alpha = 0.8
                else:
                    c = 'gold'
                    alpha = 0.4
                self._plot_single_pixel_state_trajectory(self.ax,
                                     pixel_state,
                                       pred_states[i],
                                       #terminations[i],
                                       c=c,
                                       alpha=alpha)

        n_hor = info['best_actions'][0].shape[0]

        n_hor = 1
        for t in range(n_hor):
            action = info['best_actions'][0][t]
            # plot action
            if t ==0:
                c = 'red'#'royalblue'
                linewidth = 3.0
                alpha = 0.8
            else:
                alpha -= 0.1
                alpha = max(0.2, alpha)
            waypoints = self._compute_waypoints(action)
            self._plot_waypoints(self.ax,
                                 waypoints,
                                 linewidth=linewidth,
                                 c=c,
                                 alpha=0.5)

        num_traj_itr = info['best_pred_states'].shape[0]
        c_list =['forestgreen', 'olive', 'lawngreen', 'red', 'yellow']
        # for itr in range(info['itr']-1):
        #     num_traj_itr = min(num_traj_itr, MAX_STATE_PLOTS)
        #     for i in range(num_traj_itr):
        #         if i ==0:
        #             alpha =0.8
        #         else:
        #             alpha =0.4
        #         pred_states = info['stat_info']['traj_state_{}'.format(itr)]
        #
        #         self._plot_single_state_trajectory(self.ax,
        #                                            states,
        #                                            pred_states[i],
        #                                            # terminations[i],
        #                                            c=c_list[itr],
        #                                            alpha=alpha)

        fiel_name = os.path.join(self.plot_res, 'debug_{}.png'.format(self._envActionSteps) )
        plt.savefig(fiel_name)

        ##-----
        plt.draw()
        plt.pause(1e-3)

    def _plot_single_pixel_state_trajectory(self,
                                      ax,
                                      pixel_state, #
                                      pred_pixel,#n nv
                                      c='lawngreen',
                                      alpha=1.0):
        """
        state : (2,)
        pred_states :[num_steps, 3]
        """
        num_steps = pred_pixel.shape[0]
        z = 0.05


        p1 = pixel_state

        for t in range(num_steps):

            p2 = pred_pixel[t]


            ax.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1],
                     head_width=10, head_length=10,
                     fc=c, ec=c, alpha=alpha,
                     zorder=100)

            p1 = p2

    def _plot_single_state_trajectory(self,
                                      ax,
                                      states,
                                      pred_states,
                                      c='lawngreen',
                                      alpha=1.0):
        """
        state : (2,)
        pred_states :[num_steps, 2]
        """
        num_steps = pred_states.shape[0]
        z = 0.05

        points1 = np.array(list(states) + [z])
        p1 = self.camera.project_point(points1)

        for t in range(num_steps):
            points2 = np.array(list(pred_states[t]) + [z])
            p2 = self.camera.project_point(points2)

            # if np.linalg.norm(points2 - points1) < 0.1:
            #     continue

            ax.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1],
                     head_width=10, head_length=10,
                     fc=c, ec=c, alpha=alpha,
                     zorder=100)
            points1 = points2
            p1 = p2
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
        z = 0.0315

        p1 = None
        p2 = None
        for i, waypoint in enumerate(waypoints):
            point1 = waypoint[:3]
            point1 = np.array([point1[0], point1[1], z])
            p1 = self.camera.project_point(point1)
            if i == 0:
                ax.scatter(p1[0], p1[1],
                           c=c, alpha=alpha, s=10.0)
            else:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        c=c, alpha=alpha, linewidth=linewidth)
            p2 = p1





class Jaco2PushPrimitiveXYyaw(Jaco2PushPrimitiveXY):
    def __init__(self, **kwargs):
        Jaco2PushPrimitiveXY.__init__(self, **kwargs)

    def _set_observation_space(self):
        #  x y yaw
        low_space = [self.objects_env.object_max_space_low[0] , self.objects_env.object_max_space_low[1] ,
                     self.objects_env.object_euler_space_low[2]]
        high_space = [self.objects_env.object_max_space_high[0] , self.objects_env.object_max_space_high[1],
                      self.objects_env.object_euler_space_high[2]]
        self.obs_box = Box(np.tile(np.array(low_space), self.num_objects),
                           np.tile(np.array(high_space), self.num_objects))

        self.state_obs_box = self.obs_box

        assert len(self._target_lower_space) == 6
        low_space = [self._target_lower_space[0],
                     self._target_lower_space[1], self._target_lower_space[5]]
        high_space = [self._target_upper_space[0],
                      self._target_upper_space[1], self._target_upper_space[5]]
        self.goal_box = Box(np.tile(np.array(low_space),   self.num_objects ),
                             np.tile(np.array(high_space), self.num_objects ))

        self.state_goal_box = self.goal_box

        self.observation_space = Dict([
            ('observation',  self.obs_box),                  # object position [x,y,z]*n
            ('state_observation',  self.state_obs_box),      # object pose [x,y,z,roll picth, yaw] *n
            ('desired_goal', self.goal_box),                 # goal object position [x,y,z]*n
            ('achieved_goal', self.obs_box),                 # actual object position [x,y,z]*n
            ('state_desired_goal', self.state_goal_box),     # goal object pose [x,y,z,roll picth, yaw] *n
            ('state_achieved_goal', self.state_obs_box),     # actual pose [x,y,z,roll picth, yaw] *n

        ])
    def _get_obs(self):
        if self._isImageObservation:
            images = self.camera.frames()
            # image
            rgb = images['rgb']
            depth = images['depth']
            segmask = images['segmask']

        bs = []
        orn = []
        for body in  self.movable_bodies :
            bs.append(body.position[:2])
            orn.append(body.orientation.euler[2])

        pos = np.concatenate(bs)
        state =  np.concatenate((pos, orn)) # x y yaw
        state_goal = self.state_goal

        new_obs = dict(
            observation= state,              #  obj [x,y,yaw]*n
            state_observation=state,         # obj [x,y,yaw]*n
            desired_goal=state_goal,         # desired goal: obj [x,y,yaw]*n
            state_desired_goal=state_goal,   # desired ee and obj goal :obj [x,y,yaw]*n
            achieved_goal=state,
            state_achieved_goal =state,

            # object_positions = ,
            # object_orientations = ,
        )

        if self._isImageObservation:
            new_obs['image'] = rgb
            new_obs['depth'] = depth
            new_obs['segmask'] = segmask
        return new_obs

    def get_object_goal_pos(self, i):

        x = self.state_goal[3 * i]
        y = self.state_goal[3 * i + 1]
        z = self._target_upper_space[2]
        return [x, y, z]
    def get_object_goal_orn(self, i):
        roll = 0
        pitch = 0
        yaw = self.state_goal[3 * i + 2]
        return [roll, pitch, yaw]

    def get_object_goal_pos_from_stategoal(self, state_goal, i):
        x = self.state_goal[3 * i]
        y = self.state_goal[3 * i + 1]
        z = self._target_upper_space[2]
        return [x, y, z]

    def get_object_goal_euler_orn_from_stategoal(self, state_goal, i):
        roll = 0
        pitch = 0
        yaw =  state_goal[3 * i + 2]
        return [roll, pitch, yaw]


    def sample_goal_for_rollout(self):
        if self._isRandomGoals:
            if not self._isIgnoreGoalCollision:
                poses_list = self.objects_env._sample_body_poses(self.num_objects)


                object_goals = []
                for pose in poses_list:
                    pos = pose.position[:2]
                    orn = np.array([pose.orientation.euler[2]])
                    object_goals.append(np.concatenate((pos, orn)))

                object_goals = np.concatenate(object_goals)

            else:
                object_goals = self.observation_space.spaces['state_desired_goal'].sample()
        else:
            assert len(self.fixed_objects_goals) == self.goal_dim, "require the shape of {} is {}, but got {}".format('fixed_objects_goals', self.goal_dim, len(self.fixed_objects_goals))
            object_goals = np.array(self.fixed_objects_goals).copy()

        return object_goals

    def compute_rewards(self, actionJaco2PushPrimitiveXY, obs, info=None):
        r = -np.linalg.norm(obs['state_observation'] - obs['state_desired_goal'], axis=1)
        return r

    def compute_reward(self, action, obs, info=None):
        r = -np.linalg.norm(obs['state_observation'] - obs['state_desired_goal'])
        return r


class Jaco2PushPrimitiveXYGoalEnv(Jaco2PushPrimitiveXY):
    def __init__(self, **kwargs):
        Jaco2PushPrimitiveXY.__init__(self, **kwargs)


    def _reward(self, obs, action, others=None):
        reward = self.compute_reward(obs['state_observation'], obs['state_desired_goal'])
        return reward

    def compute_reward(self, achieved, goal, info=None):
        r = -np.linalg.norm(achieved - goal, axis=-1)
        return r
    
    def _get_info(self):
       # ee_pos, ee_orn = self.robot.GetEndEffectorObersavations()

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

            is_success=float(  sum(object_distances.values()) < 0.06),
            **object_distances,
           # **touch_distances,
        )

        return info


class Jaco2PushPrimitiveXYGoalAddDistEnv(Jaco2PushPrimitiveXY):
    def __init__(self, **kwargs):
        Jaco2PushPrimitiveXY.__init__(self, **kwargs)

    def _reward(self, obs, action, others=None):
        reward = self.compute_reward(obs['state_observation'], obs['state_desired_goal'])
        return reward

    def compute_reward(self, achieved, goal, info=None):
        r = -np.linalg.norm(achieved - goal, axis=-1)
        return r

    def _set_observation_space(self):
        if self.goal_order == ['x','y']:
            low_space = [self.objects_env.object_max_space_low[0] ,
                         self.objects_env.object_max_space_low[1]]
            high_space = [self.objects_env.object_max_space_high[0] ,
                         self.objects_env.object_max_space_high[1]]
            obs_box = Box(np.tile(np.array(low_space), self.num_objects + 1),
                               np.tile(np.array(high_space), self.num_objects + 1))

            self.state_obs_box = Box(np.tile(np.array(low_space), self.num_objects ),
                               np.tile(np.array(high_space), self.num_objects ))



            low_space = [self._target_lower_space[0],
                         self._target_lower_space[1]]
            high_space = [self._target_upper_space[0],
                          self._target_upper_space[1]]
            self.goal_box = Box(np.tile(np.array(low_space),   self.num_objects ),
                                 np.tile(np.array(high_space), self.num_objects ))

            self.state_goal_box = self.goal_box
        elif self.goal_order == ['x', 'y', 'theta']:
            low_space = [self.objects_env.object_max_space_low[0],
                         self.objects_env.object_max_space_low[1],
                         -np.pi]
            high_space = [self.objects_env.object_max_space_high[0],
                          self.objects_env.object_max_space_high[1],
                          np.pi]
            obs_box = Box(np.tile(np.array(low_space), self.num_objects + 1),
                          np.tile(np.array(high_space), self.num_objects + 1))

            self.state_obs_box = Box(np.tile(np.array(low_space), self.num_objects),
                                     np.tile(np.array(high_space), self.num_objects))

            low_space = [self._target_lower_space[0],
                         self._target_lower_space[1],
                         -np.pi]
            high_space = [self._target_upper_space[0],
                          self._target_upper_space[1],
                          np.pi]
            self.goal_box = Box(np.tile(np.array(low_space), self.num_objects),
                                np.tile(np.array(high_space), self.num_objects))

            self.state_goal_box = self.goal_box


        self.observation_space = Dict([
            ('observation',  obs_box),                  # object position [x,y]*n
            ('state_observation',  self.state_obs_box),      # object pose [x,y,z,roll picth, yaw] *n
            ('desired_goal', self.goal_box),                 # goal object position [x,y]*n
            ('achieved_goal', self.state_obs_box),                 # actual object position [x,y]*n
            ('state_desired_goal', self.state_goal_box),     # goal object pose [x,y,z,roll picth, yaw] *n
            ('state_achieved_goal', self.state_obs_box),     # actual pose [x,y,z,roll picth, yaw] *n

        ])

    def _get_obs(self):
        if self._isImageObservation:
            images = self.camera.frames()
            # image
            rgb = images['rgb']
            depth = images['depth']
            segmask = images['segmask']

        bs = []
        orns = []
        for body in self.movable_bodies:
            bs.append(body.position[:2])
            orns.append(body.orientation.euler)

        if self.goal_order == ['x', 'y']:
            pos = np.concatenate(bs)

            state =  np.concatenate((pos,  pos - self.state_goal))

            state_goal = self.state_goal

        new_obs = dict(
            observation= state,              # obj pos [x,y]*n
            state_observation=pos,         # obj pos [x,y]*n
            desired_goal=state_goal,         # desired goal: obj pos [x,y]*n
            state_desired_goal=state_goal,   # desired goal: obj pos [x,y]*n
            achieved_goal=pos,
            state_achieved_goal =pos,
        )
        if self._isImageObservation:
            new_obs['image'] = rgb
            new_obs['depth'] = depth
            new_obs['segmask'] = segmask
        return new_obs





