import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import numpy as np
import time
import pybullet as p


from ..base_manipulator import ManipulatorXYZEnv
from ..manipulator import Manipulator


KINOVA_TYPES = {
'j2s7s300': {
            'urdf': 'jaco2/urdf/j2s7s300.urdf',
            'arm_dof': 7,
            'gripper_dof': 3,
            'ee_link_name': "j2s7s300_end_effector",
            },
'j2s7s300_beam': {
            'urdf': 'jaco2/urdf/j2s7s300_beam.urdf',
            'arm_dof': 7,
            'gripper_dof': 3,
            'ee_link_name': "j2s7s300_end_effector",
            },

}


class Jaco2XYZEnv(ManipulatorXYZEnv ):
    def __init__(self,
                 kinova_type = 'j2s7s300',
                 *args, **kwargs):

        self._kinova_type = kinova_type
        assert self._kinova_type in KINOVA_TYPES.keys()
        ManipulatorXYZEnv.__init__(self, **kwargs)

    def load_robot(self):
        self.robot = Manipulator(p,
                                 robot_name='jaco2',
                                 urdfRootPath=os.path.join(self._robot_urdfRoot,
                                                           KINOVA_TYPES[self._kinova_type]['urdf']),
                                 arm_dof=KINOVA_TYPES[self._kinova_type]['arm_dof'],
                                 gripper_dof=KINOVA_TYPES[self._kinova_type]['gripper_dof'],
                                 endeffector_linkname=KINOVA_TYPES[self._kinova_type]['ee_link_name'],

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


