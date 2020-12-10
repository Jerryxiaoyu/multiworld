import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)

import abc
import math
import gym
import time
import random
import cv2
import copy

import transformations

from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet_data
import pybullet as p


from .manipulator import Manipulator
from .pybullet_env import BasePybulletEnv
from .util.camera import pybullet_camera


from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
    get_asset_full_path,
)

import copy

from multiworld.core.multitask_env import MultitaskEnv
from .util.bullet_camera import create_camera
from .cameras import jaco2_push_top_view_camera, jaco2_push_lateral_view_camera


DEFAULT_RENDER = {
"target_pos": (0.32, 0.1, -0.00),
"distance": 1.5,
"yaw": 180,
"pitch": -41,
"roll": 0,
}

DEFAULT_CAMERA =  jaco2_push_top_view_camera


UNIT_ACTION_DESCRETE = 0.002
ACTION_SCALE_CONTINUOUS = 0.002

NOISE_STD_DISCRETE = 0.000      # Add noise to actions, so the env is not fully deterministic  0.01
NOISE_STD_CONTINUOUS = 0.000    # noise standard deviation (for continuous relative control)


ROBOT_BASEPOS= (0,0,0)
ROBOT_BASE_ORN_EULER= (0,0,0)

ROBOT_HOME_EE_POS = (0.092, -0.443, 0.369)
ROBOT_HOME_EE_ORN = (0.727, -0.01, 0.029, 0.686)
ROBOT_HOME_ANGLE = (4.543, 3.370, -0.264, 0.580, 2.705, 4.350, 6.425, 0, 0,0 )
FIXED_FINGER_ANGLE = 0.0

INIT_JOINT_STD = 0.1


__all__ = ['ManipulatorXYZEnv']

CONTROL_MODES_LISTS = {
"pos_consXY":{
              "n_disc_actions": None,
              "n_dim_action": 2,
               "OnlyXY" :True
             },

"pos_consXYZ":{
                "n_disc_actions": None,
                "n_dim_action": 3,
                "OnlyXY" :False
            },
# TODO add xyz rot grasp env control mode
"pos_consXYZ_Rot_grasp":{
                "n_disc_actions": None,
                "n_dim_action": 5,
                "OnlyXY" :False
            },
"pos_discXY":{
                "n_disc_actions": 5,
                "n_dim_action": None,
                "OnlyXY": True
             },
"pos_discXYZ":{
                "n_disc_actions": 7,
                "n_dim_action": None,
                "OnlyXY" :False
                },

"push_primitive_consXY":{
              "n_disc_actions": None,
              "n_dim_action": 4,
               "OnlyXY" :True
             },
"push_primitive_discXY":{
              "n_disc_actions": 10,
              "n_dim_action": None,
               "OnlyXY" :True
             },

}

TABLE_UDRF_LIST = {
'default':"objects/table/table.urdf",
'white':"objects/table/table_white.urdf"

}

class ManipulatorXYZEnv(BasePybulletEnv, Serializable,  metaclass=abc.ABCMeta):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 # control mode
                 control_mode = 'pos_consXY',
                 action_scale_conts=ACTION_SCALE_CONTINUOUS,
                 action_scale_disc=UNIT_ACTION_DESCRETE,
                 isAddActionNoise=False,

                 # sim settings
                 timeStep = 0.01,
                 actionRepeat=10,
                 maxSteps=1000,

                 isRender=False,
                 urdfRoot = pybullet_data.getDataPath(),
                 isEnableSelfCollision = True,
                 hard_reset=False,

                 # camera & image info
                 isImageObservation_debug = False, # just for debug camera params
                 camera_params      =   DEFAULT_CAMERA,
                 shadow_enable      =   False,              # Bool, enable shadow in an image
                 isImgMask          =   False,              # Bool, if to enable mask mode
                 isImgDepth         =   False,              # Bool, if to enable depth mode

                 # render setting
                 opengl_Render_enable  =   True,              # Bool, default(True): use hardware opengl
                 render_params      =   DEFAULT_RENDER,

                 # robot setting
                 random_init_arm_angle = False,             # Bool, if to randomize joint angle
                 init_joint_std =   INIT_JOINT_STD,
                 robot_basePos      =   ROBOT_BASEPOS,         # robot base position
                 robot_baseOrnEuler =   ROBOT_BASE_ORN_EULER,  # robot base orientation
                 robot_ee_pos   =   ROBOT_HOME_EE_POS,      # robot initial end-efector position (x,y,z) wrt. Cartesian
                 fixed_orn      =   ROBOT_HOME_EE_ORN,      # robot initial end-efector orientation [x,y,z,w]
                 robotHomeAngle =   ROBOT_HOME_ANGLE,       # robot initial joint angle (n,), n is Dof of the robot
                 fixed_finger   =   FIXED_FINGER_ANGLE,     # robot initial fingle angel, scalor
                 random_robot_ee_pos = False,               # Bool, if to randomize end-effector pose in a specified space
                 hand_init_upper_space  =   (0, 0, 0),      # random sample space for robot initialization
                 hand_init_lower_space  =   (0, 0, 0),      # random sample space for robot initialization

                 hand_high              =   (0.22, -0.2,  0.4), # robot work space
                 hand_low               =   (-0.22, -0.6, 0.0), # robot work space

                 # env setting
                 default_plane_pos      = (0,0,-1),
                 addTable               =   True,
                 table_name             =   'default',
                 table_position         =   [0.0000000, -0.650000, -0.40000],
                 stepTextPosition       =   [-0.8, 0, 0.0],


                 # debug info
                 debug=False,
                 debug_joint=False,
                 state_vis=False,
                 robot_info_debug=False,
                 good_render = False,

                 # others
                 num_check_initPos = 20,
                 ignore_initError = True,


                 *args, **kwargs):
        """

        :param control_mode:
        :param timeStep:     simulation step (s)
        :param actionRepeat: control step (period) = timeStep * actionRepeat
        :param maxSteps: max sim steps per epiosode. Horizon (s) = maxSteps * timeSteps

        """
        self._reset_finished = False
        BasePybulletEnv.__init__(self)
        #Utils.get_fun_arguments(inspect.currentframe())
        self.__dict__.update(kwargs)
        self._p = p
        self.kwargs = kwargs

        self._control_mode = CONTROL_MODES_LISTS[control_mode]
        self._action_scale_conts = action_scale_conts
        self._action_scale_disc = action_scale_disc
        self._isAddActionNoise = isAddActionNoise

        self._timeStep = timeStep
        self._actionRepeat =  actionRepeat
        self._isEnableSelfCollision =  isEnableSelfCollision
        self._renders = isRender
        self._maxSteps = maxSteps
        self._urdfRoot = urdfRoot

        self._addTable = addTable
        self._table_position = table_position
        self._table_name = table_name
        self._default_plane_pos = default_plane_pos
        self._robot_urdfRoot = os.path.join(currentdir, 'assets')

        self._debug = debug
        self._debug_joint = debug_joint
        self._state_vis = state_vis
        self._robot_info_debug = robot_info_debug

        self._custom_light = False

        self._isImageObservation_debug = isImageObservation_debug
        self._isImgMask = isImgMask
        self._isImgDepth = isImgDepth
        self._mask_flags = self._p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if self._isImgMask else self._p.ER_NO_SEGMENTATION_MASK

        self._stepTextPosition = stepTextPosition

        self.camera_params= camera_params
        self._opengl_Render_enable = opengl_Render_enable
        self.renderer = self._p.ER_BULLET_HARDWARE_OPENGL if opengl_Render_enable else  self._p.ER_TINY_RENDERER   #p.ER_TINY_RENDERER  #p.ER_BULLET_HARDWARE_OPENGL  #
        self._render_params = render_params
        self._shadow_flag = 1 if shadow_enable else 0

        self._robot_basePos = robot_basePos
        self._robot_baseOrnEuler = robot_baseOrnEuler

        self._random_init_arm_angle = random_init_arm_angle
        self._init_joint_std = init_joint_std

        self._robot_init_ee_pos = list(robot_ee_pos)
        self._fixed_orn = self._robot_init_ee_orn = list(fixed_orn)
        self._robotHomeAngle = robotHomeAngle
        self._fixed_finger = fixed_finger

        self._random_robot_ee_pos = random_robot_ee_pos
        self._hand_init_low = np.array(hand_init_lower_space)
        self._hand_init_high = np.array(hand_init_upper_space)
        self._hand_low = np.array(hand_low)
        self._hand_high = np.array(hand_high)

        self._num_check_initPos = num_check_initPos
        self._ignore_initError = ignore_initError

        self.terminated = 0
        self._observation = []
        self._envStepCounter = 0
        self.observations = None

        self.good_render = good_render
        ##-------------init process-------------
        self._set_observation_space()
        self._seed()
        self._init_render_setting()     # connect pybullet, and set render & camera
        self.reset()                    # load env, robot
        self._init_control_mode()       # set control mode

        self.viewer = None
        self._hard_reset =  hard_reset  # This assignment need to be after reset()
        self._reset_finished = True

    def _init_render_setting(self):
        # 1. connet pybullet server, and set render
        GUI_FLAG = False
        if self._renders:
            if not GUI_FLAG:
                cid = self._p.connect(self._p.SHARED_MEMORY)
                if (cid < 0):
                    cid = self._p.connect(self._p.GUI)
                else:
                    self._p.connect(self._p.DIRECT)

                if self.good_render:
                    self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)
                    self._p.configureDebugVisualizer(self._p.COV_ENABLE_SHADOWS, 0)

                self._p.resetDebugVisualizerCamera(self._render_params["distance"], self._render_params["yaw"],
                                                   self._render_params["pitch"],  self._render_params["target_pos"] )
            else:
                self._p.connect(self._p.DIRECT)
            if self._debug:
                # Debug sliders for moving the camera
                self.cam_x_slider = self._p.addUserDebugParameter("cam_x_slider", -0.5 , 0.5 , self.camera_params['camera_pose']['translation'][0])
                self.cam_y_slider = self._p.addUserDebugParameter("cam_y_slider", -0.9, 0.2, self.camera_params['camera_pose']['translation'][1])
                self.cam_z_slider = self._p.addUserDebugParameter("cam_z_slider", -0.2, 1, self.camera_params['camera_pose']['translation'][2])
                self.cam_rotx_slider = self._p.addUserDebugParameter("cam_rotx_slider", -math.pi, math.pi, self.camera_params['camera_pose']['rotation'][0] )
                self.cam_roty_slider = self._p.addUserDebugParameter("cam_roty_slider", -math.pi, math.pi, self.camera_params['camera_pose']['rotation'][1] )
                self.cam_rotz_slider = self._p.addUserDebugParameter("cam_rotz_slider", -math.pi, math.pi, self.camera_params['camera_pose']['rotation'][2] )

            self.debug_ino_step = self._p.addUserDebugText('step: %d' % self._envStepCounter,
                                                         [-0.8, 0, 0.0], textColorRGB=[0, 0, 0], textSize=1.5 )
        else:
            self._p.connect(self._p.DIRECT)

        self._hard_reset = True

        ## 2. set camera
        self.initialize_camera(self.camera_params)

        ## 3. set light
        if self._custom_light:
            light_x = np.random.uniform(-3, 3)
            light_y = np.random.uniform(1, 3)
            self._p.configureDebugVisualizer(lightPosition=[light_x, light_y, 3])

    def initialize_camera(self, camera_params):
        self.camera = create_camera(self._p,
                                     camera_params['image_height'],
                                     camera_params['image_width'],
                                     camera_params['intrinsics'],
                                     camera_params['translation'],
                                     camera_params['rotation'],
                                     near= camera_params['near'],
                                     far= camera_params['far'],
                                     distance= camera_params['distance'],
                                     camera_pose= camera_params['camera_pose'],

                                     isImgDepth=self._isImgDepth,
                                     isImgMask=self._isImgMask,
                                     shadow_enable=self._shadow_flag,
                                     opengl_Render_enable=self._opengl_Render_enable,

                                    is_simulation=True)
        self._image_width = camera_params['image_width']
        self._image_height = camera_params['image_height']

    def debug_camera(self):
        camera_params =  copy.copy(self.camera_params)
        camera_params['camera_pose']['translation'] = (self._p.readUserDebugParameter(self.cam_x_slider),
                                       self._p.readUserDebugParameter(self.cam_y_slider),
                                       self._p.readUserDebugParameter(self.cam_z_slider))
        camera_params['camera_pose']['rotation'] = (self._p.readUserDebugParameter(self.cam_rotx_slider),
                                                      self._p.readUserDebugParameter(self.cam_roty_slider),
                                                      self._p.readUserDebugParameter(self.cam_rotz_slider))

        self.initialize_camera(camera_params)



    def plot_boundary(self):

        for z in [self._hand_init_low[2]-0.1, self._hand_init_high[2]]:

            l_11 = [self._hand_init_low[0], self._hand_init_high[1], z]
            l_22 = [self._hand_init_high[0], self._hand_init_high[1], z]
            l_33 = [self._hand_init_high[0], self._hand_init_low[1], z]
            l_44 = [self._hand_init_low[0], self._hand_init_low[1], z]

            self._p.addUserDebugLine(l_11, l_22, [255, 0, 0], 2)
            self._p.addUserDebugLine(l_22, l_33, [255, 0, 0], 2)
            self._p.addUserDebugLine(l_33, l_44, [255, 0, 0], 2)
            self._p.addUserDebugLine(l_44, l_11, [255, 0, 0], 2)

        # self._p.loadURDF(os.path.join(self._robot_urdfRoot, 'stickers/square_visual_ws.urdf'),
        #                  self.start_offset,
        #                  [0.000000, 0.000000, 0.0, 1.0],
        #                  useFixedBase=True,
        #                  )


    def _init_control_mode(self):
        n_discrete_actioins = self._control_mode["n_disc_actions"]
        self.action_dim = self._control_mode["n_dim_action"]

        if n_discrete_actioins is not None:
            self.action_space = spaces.Discrete(n_discrete_actioins)
            self._isDiscrete = True
            self.n_discrete_actioins= n_discrete_actioins
            assert self.action_dim is None

        if self.action_dim is not None:
            self._action_bound = 1
            action_high = np.array([self._action_bound] * self.action_dim)
            self.action_space = spaces.Box(-action_high, action_high)
            self._isDiscrete = False

            assert  n_discrete_actioins is None

        self._OnlyXYControl= self._control_mode["OnlyXY"]
        # define Obsevation Space

    def _set_observation_space(self):
        return NotImplementedError

    def build_env(self):
        pass
    def load_robot(self):
        self.robot = Manipulator(p,
                                 robot_name='jaco2',
                                 urdfRootPath=os.path.join(self._robot_urdfRoot, "jaco2/urdf/j2s7s300.urdf"),
                                 arm_dof=7,
                                 gripper_dof=3,
                                 endeffector_linkname="j2s7s300_end_effector",

                                 timeStep=self._timeStep,
                                 building_env=False,  ## use gym env

                                 useInverseKinematics=True,  # IMPORTANCE! It determines the mode of the motion.
                                 torque_control_enabled=False,




                                 is_fixed=True,

                                 state_vis=False,
                                 robot_info_debug=False,
                                 debug_joint=True,

                                 basePosition=self._robot_basePos,
                                 baseOrientationEuler=self._robot_baseOrnEuler,

                                 init_configuration=self.robot_init_pos,

                                 hand_low=self._hand_low,
                                 hand_high=self._hand_high,

                                 **self.kwargs
                                 )

    def reset(self):
        if self._hard_reset:
            self._p.resetSimulation()
            self._p.setPhysicsEngineParameter(numSolverIterations=150)
            self._p.setTimeStep(self._timeStep)

            # import plane and table
           # self.Uid_plane = self._p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), self._default_plane_pos)
            if self.good_render:
                self.Uid_plane = self._p.loadURDF(os.path.join(self._robot_urdfRoot, "planes/plane_ceramic.urdf"), self._default_plane_pos)
            else:
                self.Uid_plane = self._p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), self._default_plane_pos)
            self._p.changeDynamics(self.Uid_plane, -1, contactStiffness = 1. , contactDamping = 1)

            if self._addTable:
                table_urdf = TABLE_UDRF_LIST[self._table_name]
                self.Uid_table = self._p.loadURDF(os.path.join(self._robot_urdfRoot, table_urdf),
                                           self._table_position,
                                           [0.000000, 0.000000, 0.0, 1.0],
                                          useFixedBase=True)
                self._p.changeDynamics(self.Uid_table, 0, lateralFriction = 1.0, spinningFriction = 0.2, rollingFriction= 0)


            self.build_env()
            self._p.setGravity(0, 0, -9.81)

            self.robot_init_pos = self.get_init_joint_angle()

            # load the ROBOT
            self.load_robot()

        else:
            self.robot.reset(reload_urdf=False)

        self.terminated = 0
        self._envStepCounter = 0

        self._reset_robot_end_efector()  # very important!!

        self._observation = []#self._get_obs()
        return  self._observation

    def _reset_robot_end_efector(self, target_pose = None):

        def ExecuteOnePosition(xyzq):  # runToInitConfiguration():
            num_check = self._num_check_initPos
            for i in range(num_check):

                jointStates = self.robot.CalInverseKinetics(xyzq[:3], xyzq[3:])

                self.robot._ResetJointState(jointStates)

                if self._check_reach_ee(xyzq, verbose=False):
                    break
                if i == (num_check - 1):
                    if self._ignore_initError:
                       # print('Warning: The initial position was not reached. And re-select a new configuration.')
                        return False
                    raise Exception("The init position was not reached.")
            return True

        self.robot._ResetJointState()

        if self._random_robot_ee_pos and target_pose is None:
            num_attempt = 5
            for num in range(num_attempt):
                xyzq = self.get_init_end_effector_position()
                if ExecuteOnePosition(xyzq):
                    break
                if num == num_attempt - 1:
                    raise Exception("Several attempts to robot initilization failed.")

            self.robot.Update_End_effector_pos()
        elif target_pose is not None:
            assert np.array(target_pose).shape[0] == 7
            num_attempt = 5
            for num in range(num_attempt):
                xyzq = target_pose
                if ExecuteOnePosition(xyzq):
                    break
                if num == num_attempt - 1:
                    raise Exception("Several attempts to robot initilization failed.")


            self.robot.Update_End_effector_pos()
        else:

            num_attempt = 5
            for num in range(num_attempt):

                xyzq = self._robot_init_ee_pos +  self._robot_init_ee_orn
                if ExecuteOnePosition(xyzq):
                    break
                if num == num_attempt - 1:
                    raise Exception("Several attempts to robot initilization failed.")

            self.robot.Update_End_effector_pos()

    def _check_reach_ee(self, target_xyzq, verbose=False):
        ee_pos, ee_orn = self.robot.GetEndEffectorObersavations()
        ee_pos = np.array(ee_pos)
        target_xyz = np.array(target_xyzq[:3])
        dis = np.linalg.norm(ee_pos-target_xyz)

        if dis<0.01:
            return True
        else:
            return False

    def get_init_end_effector_position(self):
        if self._random_robot_ee_pos:

            # pos_x = np.random.normal(self._robot_init_ee_pos[0], 0.1)
            # pos_y = np.random.normal(self._robot_init_ee_pos[1], 0.1)
            # pos_z = np.random.normal(self._robot_init_ee_pos[2], 0.05)

            pos_x = np.random.uniform(self._hand_init_low[0], self._hand_init_high[0])
            pos_y = np.random.uniform(self._hand_init_low[1], self._hand_init_high[1])
            pos_z = np.random.uniform(self._hand_init_low[2], self._hand_init_high[2])

            start_XYZQ =  [pos_x, pos_y,pos_z,
                          self._robot_init_ee_orn[0],self._robot_init_ee_orn[1],self._robot_init_ee_orn[2],self._robot_init_ee_orn[3],
                          ]
        else:
            start_XYZQ = [self._robot_init_ee_pos[0], self._robot_init_ee_pos[1],self._robot_init_ee_pos[2],
                         self._robot_init_ee_orn[0], self._robot_init_ee_orn[1], self._robot_init_ee_orn[2], self._robot_init_ee_orn[3],
                          ]
        return start_XYZQ

    def get_init_joint_angle(self):
        if self._random_init_arm_angle:
            #TODO  jaco dependent
            start_pos = self._robotHomeAngle + np.append(np.random.normal(0, self._init_joint_std, size = 7), [0,0,0])
        else:
            start_pos = self._robotHomeAngle
        return start_pos

    def __del__(self):
        self._p.disconnect()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def _get_obs(self):
        return NotImplementedError
    def _get_info(self):
        return {}
    def _termination(self):
        if self.terminated or self._envStepCounter >= self._maxSteps:
            self._observation = self._get_obs()
            return True, 0
        return False, 0
    def _reward(self, obs, action ):
        return NotImplementedError

    def _get_orn_from_theta(self, theta):
        return NotImplementedError
        #theta = math.pi / 3.

        init_euler = self._p.getEulerFromQuaternion(self._fixed_orn)
        ## jaco dependent :if griper forward axis is Z
        orn_mat = transformations.euler_matrix(init_euler[0], init_euler[1], init_euler[2]).dot(transformations.euler_matrix(0, 0, theta))
        q  = transformations.quaternion_from_matrix(orn_mat)

        orn = [q[1], q[2], q[3], q[0]]
        return orn

    def step(self, action):
        finger_angle = [self._fixed_finger]  # Close the gripper
        orn = self._fixed_orn

        if (self._isDiscrete):
            dv_x = dv_y = dv_z = copy.copy(self._action_scale_disc)   # velocity per physics step.

            # Add noise to action
            if self._isAddActionNoise:
                dv_x += self.np_random.normal(0.0, scale=NOISE_STD_DISCRETE)
                dv_y += self.np_random.normal(0.0, scale=NOISE_STD_DISCRETE)
                dv_z += self.np_random.normal(0.0, scale=NOISE_STD_DISCRETE)

            dx = [0, -dv_x, dv_x,        0,     0,      0,      0][action]
            dy = [0,  0,    0,      -dv_y,      dv_y,   0,      0][action]
            dz = [0,  0,    0,          0,       0,     -dv_z, dv_z][action]

            if self._OnlyXYControl:
                dz = 0

            realAction = np.concatenate(([dx, dy, dz], orn, finger_angle))
        else:
            assert action.shape[0] == self.action_space.shape[0]
            action = np.clip(action, -self._action_bound, self._action_bound)
            dv = copy.copy(self._action_scale_conts)

            # Add noise to action
            if self._isAddActionNoise:
                dv += self.np_random.normal(0.0, scale=NOISE_STD_CONTINUOUS)

            if self._OnlyXYControl:
                dx = action[0] * dv
                dy = action[1] * dv
                dz = 0
            else:
                dx = action[0] * dv
                dy = action[1] * dv
                dz = action[2] * dv

            realAction = np.concatenate(([dx, dy, dz], orn, finger_angle))
        #t2 = time.time()
        #print('step time :', t2-t1)

        if self._debug:
            if self._isImageObservation_debug:
                self.debug_camera()
                img = self.get_image(mode='rgb')
        return self.step2(realAction)

    def step2(self, action):
       # t1 = time.time()
        if self._debug_joint:
            self.robot.ApplyDebugJoint()

            for i in range(self._actionRepeat):
                self._p.stepSimulation()
                self._envStepCounter += 1
        else:
            new_action = action.copy()
            new_action[0] = action[0] /float(self._actionRepeat)
            new_action[1] = action[1] /float(self._actionRepeat)
            new_action[2] = action[2] /float(self._actionRepeat)

            for st in range(self._actionRepeat):
                self.robot.ApplyAction(new_action)
                self._p.stepSimulation()
                self._envStepCounter += 1

            # if self._termination():
            #     break
        if self._renders:
            time.sleep(0.001)
            if self._renders is True and self._envStepCounter % 10 == 0:
                time_info = 'step: %d' % self._envStepCounter +'  '+'time: %.2fs' \
                            % (self._envStepCounter* self._timeStep)
                self.debug_ino_step = self._p.addUserDebugText(time_info, self._stepTextPosition,
                                    textColorRGB=[1, 0, 0], textSize=1.5, replaceItemUniqueId=self.debug_ino_step)

        self._observation = self._get_obs()

        #t2 = time.time()
        #print('step time :', t2 - t1)

        info = self._get_info()

        reward = self._reward(self._observation, action)
        done, reward_terminal = self._termination()
        reward += reward_terminal

        return  self._observation , reward, done, info


    def render(self, mode="human",  close=False, result_full=False):
        """
        :param mode:
        :param close:
        :return: rgb image (height, width, 3)
        """
        pass

    def get_image(self, width=None, height=None, mode='rgb'):

        images = self.camera.frames()

        if mode =='rgb':
            if width is not None and height is not None and (width != self.image_width or height != self.image_height):
                return cv2.resize(images['rgb'], (height, width), interpolation=cv2.INTER_NEAREST)
            else:
                return images['rgb']
        elif mode =='rgbd':
            if width is not None and height is not None and (width != self.image_width or height != self.image_height):
                rgb_img = cv2.resize(images['rgb'], (height, width), interpolation=cv2.INTER_NEAREST)
            else:
                rgb_img = images['rgb']

            return {
                'rgb': rgb_img,
                'depth': images['depth'],

            }
        elif mode == 'rgbd-seg':
            if width is not None and height is not None and (width != self.image_width or height != self.image_height):
                rgb_img = cv2.resize(images['rgb'], (height, width), interpolation=cv2.INTER_NEAREST)
            else:
                rgb_img = images['rgb']
            return {
                'rgb': rgb_img,
                'depth':images['depth'],
                'segmask':images['segmask']
            }

        else:
            raise NotImplementedError

    @property
    def start_range(self):
        """
        end effector range/2 wrt Catisian, [x,y,z]
        """
        return  (self._hand_init_high  -  self._hand_init_low)/2

    @property
    def start_offset(self):
        """
        end effector offset wrt Catisian, [x,y,z]
        """
        return (self._hand_init_high + self._hand_init_low)/2.

    @property
    def image_width(self):
        return self._image_width
    @property
    def image_height(self):
        return self._image_height


