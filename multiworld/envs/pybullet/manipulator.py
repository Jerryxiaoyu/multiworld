import os

import copy
import math
import numpy as np

import pybullet_data

#import multiworld.utils as Utils
import inspect
INIT_CONFIGURATION =  [math.pi/2, math.pi, math.pi, math.pi/6, 0, math.pi/2, 0, 1, 1, 1]
from multiworld.math import Pose
import queue



X_HIGH = 0.3
X_LOW = -0.3
Y_HIGH = -0.3
Y_LOW = -0.85
Z_HIGH = 0.6
Z_LOW = 0.2

MAX_VELOCITY = 1.0
SENSOR_NOISE_STDDEV = (0.0, 0.0, 0.0)
class Manipulator:
  def __init__(self,
               pybullet_client,
               robot_name = 'j2s7s300',
               urdfRootPath=os.path.abspath('assets/jaco2/urdf/j2s7s300.urdf'),
               timeStep=0.01,
               
               useInverseKinematics=True,
               torque_control_enabled=False,
               
               isLimitedWorkSpace = True,
               building_env = True,
               is_fixed = True,
               
               verbose = False,
               state_vis= False,
               robot_info_debug= False,
               debug_joint=False,

               hand_low= (X_LOW, Y_LOW, Z_LOW),
               hand_high = (X_HIGH, Y_HIGH, Z_HIGH),
               basePosition = ( -0.000000, 0.000000, 0.000000 ),
               baseOrientationEuler = (0.000000, 0.000000, 0),
    
               arm_dof = 7,
               gripper_dof = 3,
               endeffector_linkname = "j2s7s300_joint_end_effector", 
               init_configuration = INIT_CONFIGURATION,
               useDynamics = True,
               **kwargs
               ):

    #Utils.get_fun_arguments(inspect.currentframe())
    self.robot_name = robot_name
    self._pybullet_client = pybullet_client
    self._urdfRootPath = urdfRootPath
    self._timeStep = timeStep
    self._isBuildEnv = building_env

    self._is_fixed = is_fixed
    self._basePosition = basePosition
    self._baseOrientation = self._pybullet_client.getQuaternionFromEuler(baseOrientationEuler)
    self._init_jointPositions = init_configuration
    self._isLimitedWorkSpace = isLimitedWorkSpace
    
    
    self._torque_control_enabled = torque_control_enabled
    self._useInverseKinematics = useInverseKinematics  # 0
    self._useSimulation = useDynamics
    self._useNullSpace = 1  # set 1, otherwise cause robot unstable
    self._useOrientation = 1  # set 1, otherwise cause robot unstable
    

    self._endEffectorLinkName = endeffector_linkname

    self._numFingers = gripper_dof   
    self._numArmJoints = arm_dof
 
    # debug flag
    self._verbose  = verbose
    self._robot_info_debug = robot_info_debug
    self._debug_joint =  debug_joint

    # robot model parameters
    self._maxForce = 30.
    self._max_joint_velocity = MAX_VELOCITY
    self._max_velocity = MAX_VELOCITY
    self._fingerAForce = 2
    self._fingerBForce = 2.5
    self._fingerTipForce = 20

    self._positionGain = 4
    self._velocityGain = 0.6
    
    # observation parameters
    self._observation_noise_stdev = SENSOR_NOISE_STDDEV
    self.OnlyEndEffectorObervations = useInverseKinematics
 
    if self._isBuildEnv:
      self.build_env()

    # import the robot into env
    self.reset()


    # ik paramters
    # lower limits for null space
    self.ll = self.jointLowerLimit[:self.numMotors ]
    # upper limits for null space
    self.ul = self.jointUpperLimit[:self.numMotors ]
    # joint ranges for null space
    self.jr = [3.5, 4,  6.8,   4.8, 5.8, 4.5, 7, 0.01, 0.01, 0.01]
    # restposes for null space
    self.rp = [1.1, 2.8, -3.2, 0.6, -0.505, 1.9, 0.12, 0.01, 0.01, 0.01]
    #joint damping coefficents
    self.jd = [5, 5, 5, 5, 5, 5, 5, 0.01, 0.01,   0.01]

    # setting the workspace wrt ee
    self.ee_X_upperLimit = hand_high[0]
    self.ee_X_lowerLimit = hand_low[0]
    self.ee_Y_upperLimit = hand_high[1]
    self.ee_Y_lowerLimit = hand_low[1]
    self.ee_Z_upperLimit = hand_high[2]
    self.ee_Z_lowerLimit = hand_low[2]



    self.joint_waypoints = queue.Queue()
    self.target_pose_waypoints = queue.Queue()
    self._joint_cmd = None
    self.target_pose = None
    self.target_joint = None
    self.last_jointPose = self.rp

    if self._useInverseKinematics:
       ee_pose_res = self.GetEndEffectorObersavations()

       self.endEffectorPos = ee_pose_res[0] 
       self.endEffectorOrn = ee_pose_res[1]

       self._joint_target_pos = np.array(init_configuration)[:self._numArmJoints]
       #self.endEffectorAngle  = self.GetTrueMotorAngles()[6]
       # state = self._pybullet_client.getLinkState(self.robotUid,
       #                                            self.EndEffectorIndex, computeForwardKinematics=1)
       #print("reset state: ", state)

       #print("pos :", self.endEffectorPos, self.endEffectorOrn)
       #self.endEffectorOrn_euler = list(p.getEulerFromQuaternion(self.endEffectorOrn ))
       #print('init ee angle :', self.GetTrueMotorAngles(), '  ,', self.endEffectorAngle )

    # if use visdom to visualize states
    self.state_vis = state_vis
    if self.state_vis:
      self.t = 0
      self.vis = visdom.Visdom(env='kinova')


  def Update_End_effector_pos(self):
    ee_pose_res = self.GetEndEffectorObersavations()

    self.endEffectorPos = ee_pose_res[0]
    self.endEffectorOrn = ee_pose_res[1]

    if self._verbose:
      print("reset after ee pose: ", ee_pose_res)

  def print_robot_info(self):
      num_joints = self._pybullet_client.getNumJoints(self.robotUid)
      for i in range(num_joints):
          joint_info = self._pybullet_client.getJointInfo(self.robotUid, i)
          print(joint_info)

      print('joint id :', self._joint_name_to_id)

      print("Print the robot info: ")
      print("motorNames", self.motorNames)
      print("jointLowerLimit", self.jointLowerLimit)
      print("jointUpperLimit", self.jointUpperLimit)
      print("jointMaxForce", self.jointMaxForce)
      print("jointMaxVelocity", self.jointMaxVelocity)

  def build_env(self):
    # build env
    self.plane = self._pybullet_client.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath())
    self._pybullet_client.changeVisualShape(self.plane, -1, rgbaColor=[1, 1, 1, 0.9])
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
    self._pybullet_client.setGravity(0, 0, -9.81)

  def reset(self, reload_urdf=True):
    if reload_urdf:
      self.robotUid = self._pybullet_client.loadURDF(
                          os.path.join(self._urdfRootPath),
                          self._basePosition,
                          self._baseOrientation,
                          useFixedBase=self._is_fixed,
                          flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
      
      ## display the link info
      # for i in range(self._pybullet_client.getNumJoints(self.robotUid)):
      #   print(self._pybullet_client.getJointInfo(self.robotUid, i))
 
      self._BuildJointNameToIdDict()
      self._GetJointInfo()
      self._ResetJointState()

      # set fingers friction:
      # for linkIndex in [10, 11, 12, 13, 14, 15]:
      #   self._pybullet_client.changeDynamics(self.robotUid, linkIndex,
      #                                        lateralFriction=10,
      #                                        spinningFriction=0.2,
      #                                        rollingFriction=0,
      #                                        contactStiffness=-1,
      #                                        contactDamping=-1,
      #                                        )

      # # reset joint angle
      # for i in range(self.numMotors):
      #   self._SetDesiredMotorAngleById(self.motorIndices[i], self._init_jointPositions[i], max_velocity= 10)
    else:
      self._pybullet_client.resetBasePositionAndOrientation(self.robotUid, self._basePosition, self._baseOrientation)
      self._pybullet_client.resetBaseVelocity(self.robotUid, [0, 0, 0], [0, 0, 0])

      self._ResetJointState()

    ee_res = self.GetEndEffectorObersavations()
    self.endEffectorPos = ee_res[0]#list(KINOVA_HOME_EE_POS)  # [0.09, -0.4, 0.4]#ee_res[0]
    self.endEffectorOrn = ee_res[1]#list(KINOVA_HOME_EE_ORN)
      # # reset joint ange
      # for i in range(self.numMotors):
      #   self._SetDesiredMotorAngleById(self.motorIndices[i], self._init_jointPositions[i], max_velocity=10)

    if self._verbose:
      print('reset joint angle: ', self.GetTrueMotorAngles())
      print('reset end-effortor: ', self.GetEndEffectorObersavations())

      self.print_robot_info()

    if self._robot_info_debug:
      self.Xpos_info = self._pybullet_client.addUserDebugText('x:', [-0.8, 0, 0.6], textColorRGB=[1, 0, 0], textSize=1.5 )
      self.Ypos_info = self._pybullet_client.addUserDebugText('y:', [-0.8, 0, 0.5], textColorRGB=[1, 0, 0], textSize=1.5)
      self.Zpos_info = self._pybullet_client.addUserDebugText('z:', [-0.8, 0, 0.4], textColorRGB=[1, 0, 0], textSize=1.5)

  def _BuildJointNameToIdDict(self):
    """
    Build Joint Dict  
    :return: 
    """
    num_joints = self._pybullet_client.getNumJoints(self.robotUid)
    self._joint_name_to_id = {}
    self._link_name_to_id ={}
    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.robotUid, i)
      self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]
      self._link_name_to_id[joint_info[12].decode("UTF-8")] = joint_info[0]

      #print(joint_info)

    #print('joint id :', self._joint_name_to_id)
  def _GetJointInfo(self):
    """
    :return: 
    """
    self.actuator_joint = []
    self.motorNames = []
    self.motorIndices = []
    self.joint_q_index = []
    self.joint_u_index = []
    self.jointLowerLimit = []
    self.jointUpperLimit = []
    self.jointMaxForce = []
    self.jointMaxVelocity = []
    self.paramIds = []
    self.paramNames = []
    for i in range(self._pybullet_client.getNumJoints(self.robotUid)):
      joint_info = self._pybullet_client.getJointInfo(self.robotUid, i)
      qIndex = joint_info[3]
      if qIndex > - 1:  # JOINT_FIXED
        self.motorNames.append(joint_info[1].decode("UTF-8"))
        self.motorIndices.append(i)
        self.joint_q_index.append(joint_info[3])
        self.joint_u_index.append(joint_info[4])
        self.jointLowerLimit.append(joint_info[8])
        self.jointUpperLimit.append(joint_info[9])
        self.jointMaxForce.append(joint_info[10])
        self.jointMaxVelocity.append(joint_info[11])

        # jointName = joint_info[1]
        # jointType = joint_info[2]
        # if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
        #   jointState = self._pybullet_client.getJointState(self.robotUid, i)
        #   self.paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, 0))
        #   self.paramNames.append(jointName.decode("utf-8"))

    self.numMotors = len(self.motorNames)
    
    # print("Print the robot info: ")
    # print("motorNames", self.motorNames)
    # print("jointLowerLimit",self.jointLowerLimit)
    # print("jointUpperLimit",self.jointUpperLimit)
    # print("jointMaxForce",self.jointMaxForce)
    # print("jointMaxVelocity",self.jointMaxVelocity)


    if  self._debug_joint:
      for i in range(self.numMotors):
        jointName = self.motorNames[i]
        self.paramIds.append(self._pybullet_client.addUserDebugParameter(jointName, -8, 8, self._init_jointPositions[i]))
        self.paramNames.append(jointName)

    self.EndEffectorIndex = self._link_name_to_id[self._endEffectorLinkName]
  def _ResetJointState(self, jointPositions = None):

    if jointPositions is None:
      state = self._init_jointPositions
    else:
      state = jointPositions
    for i in range(self._numArmJoints+self._numFingers):
      self._pybullet_client.resetJointState(
        self.robotUid,
        self._joint_name_to_id[self.motorNames[i]],
        state[i],
        targetVelocity=0)


  def _SetMotorTorqueById(self, motor_id, torque):
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.robotUid,
        jointIndex=motor_id,
        controlMode=self._pybullet_client.TORQUE_CONTROL,
        force=torque)
  def _SetDesiredMotorAngleById(self, motor_id, desired_angle, desired_vel=None, max_velocity = None):
    #if max_velocity is None:
    #    max_velocity = self.jointMaxVelocity[motor_id]
    if desired_vel is None:
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.robotUid,
          jointIndex=motor_id,
          controlMode=self._pybullet_client.POSITION_CONTROL,
          targetPosition=desired_angle,
          positionGain= self._positionGain,
          velocityGain= self._velocityGain,
          maxVelocity = self._max_velocity,#self.max_velocity,
          force=self._maxForce)
    else:
      self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.robotUid,
        jointIndex=motor_id,
        controlMode=self._pybullet_client.POSITION_CONTROL,
        targetPosition=desired_angle,
        targetVelocity = desired_vel,
        positionGain=0.5,
        velocityGain=0.1,
        maxVelocity=self._max_velocity,  # self.max_velocity,
        force=self._maxForce)
  def _SetDesiredMotorVelocityById(self, motor_id, desired_vel, max_velocity = None):
    #if max_velocity is None:
    #    max_velocity = self.jointMaxVelocity[motor_id]
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.robotUid,
        jointIndex=motor_id,
        controlMode=self._pybullet_client.VELOCITY_CONTROL,
        targetVelocity = desired_vel,

        maxVelocity = self._max_velocity, #self.max_velocity,
        force=self._maxForce)
  def _SetDesiredMotorTorqueById(self, motor_id, desired_torque, max_velocity=None):
    # if max_velocity is None:
    #    max_velocity = self.jointMaxVelocity[motor_id]
    self._pybullet_client.setJointMotorControl2(
      bodyIndex=self.robotUid,
      jointIndex=motor_id,
      controlMode=self._pybullet_client.TORQUE_CONTROL,
      force=desired_torque)
  def _AddSensorNoise(self, sensor_values, noise_stdev):
    if noise_stdev <= 0:
      return sensor_values
    observation = sensor_values + np.random.normal( scale=noise_stdev, size=sensor_values.shape)
    return observation

  def GetActionDimension(self):
    if (self._useInverseKinematics):
      return 5  #position x,y,z angle and finger angle
    return len(self.motorIndices)
  def GetObservationDimension(self):
    return len(self.GetObservation())
  def GetTrueMotorAngles(self):
    """Gets the joints angles at the current moment.
    """
    motor_angles = [
        self._pybullet_client.getJointState(self.robotUid, motor_id)[0]
        for motor_id in self.motorIndices]
    return motor_angles
  def GetMotorAngles(self):
    """Gets the actual joint angles with noise.
    This function mimicks the noisy sensor reading and adds latency. The motor
    angles that are delayed, noise polluted, and mapped to [-pi, pi].
    Returns:
      Motor angles polluted by noise and latency, mapped to [-pi, pi].
    """
    motor_angles = self._AddSensorNoise(np.array(self.GetTrueMotorAngles()[0:self.numMotors]),
        self._observation_noise_stdev[0])
    return motor_angles  # delete maping
  def GetTrueMotorVelocities(self):
    """Get the velocity of all joints.
    Returns:
      Velocities of all joints.
    """
    motor_velocities = [
        self._pybullet_client.getJointState(self.robotUid, motor_id)[1]
        for motor_id in self.motorIndices]

    return motor_velocities
  def GetMotorVelocities(self):
    """Get the velocity of all eight motors.

    This function mimicks the noisy sensor reading and adds latency.
    Returns:
      Velocities of all eight motors polluted by noise and latency.
    """
    return self._AddSensorNoise(
        np.array( self.GetTrueMotorVelocities()[0:self.numMotors]),
        self._observation_noise_stdev[1])
  def GetTrueMotorTorques(self):
    """Get the amount of torque the motors are exerting.

    Returns:
      Motor torques of all eight motors.
    """
    #TODO if considering motor dynamics, need to add a coversion function of motor torques.
    motor_torques = [
          self._pybullet_client.getJointState(self.robotUid, motor_id)[3]
          for motor_id in self.motorIndices ]

    return motor_torques
  def GetMotorTorques(self):
    """Get the amount of torque the motors are exerting.
    This function mimicks the noisy sensor reading and adds latency.
    Returns:
      Motor torques of all eight motors polluted by noise and latency.
    """
    return self._AddSensorNoise( np.array( self.GetTrueMotorTorques()[0: self.numMotors]),
                                 self._observation_noise_stdev[2])
  def GetEndEffectorObersavations(self):
    """Get the end effector of kinova wrt world Cartesian space

    Returns:
      Position of the end effecotr:[x, y, z] wrt world Cartesian space
      Oreintation of the end effector, in quaternion form. [x,y,z,w]
    """
    state = self._pybullet_client.getLinkState(self.robotUid,
                                               self.EndEffectorIndex, computeForwardKinematics=1)
    # ee_pos = state[4]
    # ee_orn = state[5]
    ee_pos = state[4]
    ee_orn = state[5]

    return np.array(ee_pos), np.array(ee_orn)
  def GetObservation(self):
    """Get the observations of kinova.

    It includes the angles, velocities, torques and the orientation of the base.

    Returns:
      The observation list. observation[0:8] are motor angles. observation[8:16]
      are motor velocities, observation[16:24] are motor torques.
      observation[24:28] is the orientation of the base, in quaternion form.
    """
    observation = []

    if self.OnlyEndEffectorObervations:
      ee_pos, ee_orn = self.GetEndEffectorObersavations()
      ee_euler = self._pybullet_client.getEulerFromQuaternion(ee_orn)

      observation.extend(ee_pos.tolist())
      observation.extend(ee_orn.tolist())
    else:
      observation.extend(self.GetMotorAngles().tolist())
      observation.extend(self.GetMotorVelocities().tolist())
      observation.extend(self.GetMotorTorques().tolist())

    return observation

  def GetObservationUpperBound(self):
    """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation. See GetObservation() for the details
        of each element of an observation.
    """
    if self.OnlyEndEffectorObervations:
      raise NotImplementedError
    else:
      upper_bound = np.array([0.0] * self.GetObservationDimension())
      upper_bound[0:self.numMotors] = self.jointUpperLimit  # Joint angle.
      upper_bound[self.numMotors:2 * self.numMotors] = self.jointMaxVelocity  #    Joint velocity.
      upper_bound[2 * self.numMotors:3 * self.numMotors] = self.jointMaxForce # Joint torque.

    return upper_bound
  def GetObservationLowerBound(self):
    """Get the lower bound of the observation."""
    if self.OnlyEndEffectorObervations:
      raise NotImplementedError
    else:
      lower_bound = np.array([0.0] * self.GetObservationDimension())
      lower_bound[0:self.numMotors] = self.jointLowerLimit  # Joint angle.
      lower_bound[self.numMotors:2 * self.numMotors] = self.jointMaxVelocity*(-1)  # Joint velocity.
      lower_bound[2 * self.numMotors:3 * self.numMotors] = self.jointMaxForce*(-1)  # Joint torque.

    return lower_bound
  def CalInverseKinetics(self, pos, orn):

    jointPoses = self._pybullet_client.calculateInverseKinematics(self.robotUid, self.EndEffectorIndex, pos, orn,
                                                                  lowerLimits=self.ll, upperLimits=self.ul,
                                                                  jointRanges=self.jr, restPoses=self.rp,
                                                                  residualThreshold=0.001, jointDamping=self.jd)

    return jointPoses

  def CheckEEpos(self):
    ee_pos, ee_orn = self.GetEndEffectorObersavations()

    if ee_pos[0]  >  self.ee_X_upperLimit or ee_pos[0] < self.ee_X_lowerLimit:
      return True
    if ee_pos[1]  >  self.ee_Y_upperLimit or ee_pos[1] < self.ee_Y_lowerLimit:
      return True
    if ee_pos[2]  >  self.ee_Z_upperLimit or ee_pos[2] < self.ee_Z_lowerLimit:
      return True

    return False
  def Enable_Torque_mode(self):
    for i in range(self._numArmJoints):
      motor_id = self.motorIndices[i]
      maxForce = 0
      mode = p.VELOCITY_CONTROL
      p.setJointMotorControl2(self.robotUid, motor_id,
                              controlMode=mode, force=maxForce)

  def DebugJointControl(self):
    for i in range(self.numMotors):
      c = self.paramIds[i]
      targetPos = self._pybullet_client.readUserDebugParameter(c)

      motor_id = self.motorIndices[i]
      self._SetDesiredMotorAngleById(motor_id, targetPos)
  def ApplyAction(self, commands):
    """
    end-effector position difference mode

    :param commands:

    1. control end-effector in Cartesian Space. useInverseKinematics is True.
    np.array o list with size 8
    [dx, dy, dz, orn, fingerAngle]
    dx,dy,dz : the relative value of end-effector in Cartesian Space.R^{3}, Unit: meter
    orn : the quaterion of the end-effector, R^{4},  [x,y,z,w]
    fingerAngle : angle of the fingers, R^{1}

    2. control joint angle derectly in joint configuration space. useInverseKinematics is False.
    np.array o list with size 10
    [j1,j2,...,j7, f1, f2, f3] Unit: rad

    """
    if (self._useInverseKinematics):
      if np.array(commands).size != 8:
        raise Exception("Command size is not matched, require a command with the size of 8 but got ",
                        np.array(commands).size)

      # ee_pose_res = self.GetEndEffectorObersavations()
      # endEffectorPos = ee_pose_res[0]
      # endEffectorOrn = ee_pose_res[1]

      self.endEffectorPos[0] += commands[0]
      self.endEffectorPos[1] += commands[1]
      self.endEffectorPos[2] += commands[2]

      if self._verbose:
        print('real action: ', commands[:3])

      new_commands = commands.copy()  # very important!!
      new_commands[0:3] = self.endEffectorPos

      self.ApplyAction_EndEffectorPose(new_commands)

    else:
      if np.array(commands).size != self.numMotors:  # 10 for j2n7s300
        raise Exception("Command size is not matched, require a command with the size of ", self.numMotors, " but got ", np.array(commands).size)
      for i in range (self.numMotors):
        motor_id = self.motorIndices[i]
        self._SetDesiredMotorAngleById(motor_id, commands[i])
  def ApplyDebugJoint(self):

      for i in range(self.numMotors):
        joint_val = self._pybullet_client.readUserDebugParameter(self.paramIds[i])

        motor_id = self.motorIndices[i]
        self._SetDesiredMotorAngleById(motor_id, joint_val)

      # ee_pos, ee_orn = self.GetEndEffectorObersavations()
      # if self._robot_info_debug:
      #   x_pos = "x: %.3f" % ee_pos[0]
      #   y_pos = "y: %.3f" % ee_pos[1]
      #   z_pos = "z: %.3f" % ee_pos[2]
      #
      #   self._pybullet_client.addUserDebugText(x_pos, [-0.8, 0, 0.6], textColorRGB=[1, 0, 0], textSize=1.5,
      #                                          replaceItemUniqueId=self.Xpos_info)
      #   self._pybullet_client.addUserDebugText(y_pos, [-0.8, 0, 0.5], textColorRGB=[1, 0, 0], textSize=1.5,
      #                                          replaceItemUniqueId=self.Ypos_info)
      #   self._pybullet_client.addUserDebugText(z_pos, [-0.8, 0, 0.4], textColorRGB=[1, 0, 0], textSize=1.5,
      #                                          replaceItemUniqueId=self.Zpos_info)
  def ApplyAction_Velocity(self, commands):
    """
    :param commands:

    """
    if np.array(commands).size != 10:
      raise Exception("Command size is not matched, require a command with the size of 8 but got ",
                      np.array(commands).size)

    for i in range (self._numArmJoints):
      motor_id = self.motorIndices[i]
      self._SetDesiredMotorVelocityById(motor_id, commands[i])

    #fingers
    for i in range(self._numFingers):
      fingerAngle = commands[-self._numFingers +i]
      finger_id = self.motorIndices[self._numArmJoints + i]
      self._pybullet_client.setJointMotorControl2(self.robotUid,finger_id,
                                                  self._pybullet_client.POSITION_CONTROL,
                                                  targetPosition= fingerAngle,
                                                  force=self._fingerTipForce)
  def ApplyAction_Torque(self, commands):
    """
    :param commands:


    """
    if np.array(commands).size != 10:
      raise Exception("Command size is not matched, require a command with the size of 10 but got ",
                      np.array(commands).size)

    for i in range (self._numArmJoints):
      motor_id = self.motorIndices[i]
      self._SetDesiredMotorTorqueById(motor_id, commands[i])

    #fingers
    for i in range(self._numFingers):
      fingerAngle = commands[-self._numFingers +i]
      finger_id = self.motorIndices[self._numArmJoints + i]
      self._pybullet_client.setJointMotorControl2(self.robotUid,finger_id,
                                                  self._pybullet_client.POSITION_CONTROL,
                                                  targetPosition= fingerAngle,
                                                  force=self._fingerTipForce)
  def ApplyAction_EndEffectorPose(self, commands):
    """
    end-effector position control mode
    given absolute position wrt. Cartesian world
    :param commands:

     1. control end-effector in Cartesian Space. useInverseKinematics is True.
    np.array o list with size 9
    [x, y, z, orn, fingerAngle]
    x,y,z : the value of end-effector in Cartesian Space. R^{3},Unit: meter
    orn : the quaterion of the end-effector, R^{4},[x,y,z,w]
    fingerAngle : angle of the fingers, R^{1},

    2. control joint angle derectly in joint configuration space. useInverseKinematics is False.
    np.array o list with size 10
    [j1,j2,...,j7, f1, f2, f3] Unit: rad

    :return:
    """
    if (self._useInverseKinematics):
      if np.array(commands).size != 8:
        raise Exception("Command size is not matched, require a command with the size of 8 but got ",
                        np.array(commands).size)
      x = commands[0]
      y = commands[1]
      z = commands[2]
      orn = commands[3:7]
      fingerAngle = commands[7]

      if self._isLimitedWorkSpace:
        x = np.clip(x, self.ee_X_lowerLimit, self.ee_X_upperLimit)
        y = np.clip(y, self.ee_Y_lowerLimit, self.ee_Y_upperLimit)
        z = np.clip(z, self.ee_Z_lowerLimit, self.ee_Z_upperLimit)

      pos = np.array([x,y,z])

      if self._verbose:
        print('command: ', commands[0:3])
        ee_pos, ee_orn = self.GetEndEffectorObersavations()
        print('end-effecter position: ', ee_pos)
        print('end-effecter orentation ', ee_orn)
        print('joint angle', self.GetMotorAngles())
      if self.state_vis:
        ee_pos, ee_orn = self.GetEndEffectorObersavations()
        self.vis.line(X=np.array([self.t]),
                      Y=np.column_stack((np.array([pos[0]]), np.array([ee_pos[0]]))),
                      opts=dict(showlegend=True, title='X position'), win='X position', update='append', )
        self.vis.line(X=np.array([self.t]),
                      Y=np.column_stack((np.array([pos[1]]), np.array([ee_pos[1]]))),
                      opts=dict(showlegend=True, title='Y position'), win='Y position', update='append', )
        self.vis.line(X=np.array([self.t]),
                      Y=np.column_stack((np.array([pos[2]]), np.array([ee_pos[2]]))),
                      opts=dict(showlegend=True, title='Z position'), win='Z position', update='append', )
        self.t += self._timeStep

      # ik
      if (self._useNullSpace == 1):
        if (self._useOrientation == 1):
          jointPoses = self._pybullet_client.calculateInverseKinematics(self.robotUid, self.EndEffectorIndex, pos, orn,
                                                                        lowerLimits=self.ll, upperLimits=self.ul,
                                                                        jointRanges=self.jr, restPoses=self.rp,
                                                                        residualThreshold=0.001, jointDamping=self.jd)
        else:
          jointPoses = self._pybullet_client.calculateInverseKinematics(self.robotUid, self.EndEffectorIndex, pos,
                                                                        lowerLimits=self.ll, upperLimits=self.ul,
                                                                        jointRanges=self.jr, restPoses=self.rp,
                                                                       residualThreshold=0.001, jointDamping=self.jd)
        #print('ik angle:', jointPoses)
        self._joint_target_pos = jointPoses[:self._numArmJoints]
        if (self._useSimulation):
          for i in range(self._numArmJoints):
            motor_id = self.motorIndices[i]
            self._SetDesiredMotorAngleById(motor_id, jointPoses[i])
        else:
          # reset the joint state (ignoring all dynamics, not recommended to use during simulation)
          for i in range(self._numArmJoints):
            self._pybullet_client.resetJointState(self.robotUid, self._joint_name_to_id[self.motorNames[i]], jointPoses[i])
      # fingers
      for i in range(self._numFingers):
        finger_id = self.motorIndices[self._numArmJoints + i]
        self._pybullet_client.setJointMotorControl2(self.robotUid, finger_id, self._pybullet_client.POSITION_CONTROL,
                                                    targetPosition=fingerAngle, force=self._fingerTipForce)

    else:
      raise Exception("You must set \'useInverseKinematics==True\' in the ApplyAction_EndEffectorPose function.")

  def ApplyFinger_Action(self, commands):
    """
    :param commands: [f1,f2,f3] angle, R^{3}, radian
    :return:
    """
    assert np.array(commands).shape[0]== self._numFingers

    fingerAngles = np.array(commands)

    # fingers
    for i in range(self._numFingers):
      finger_id = self.motorIndices[self._numArmJoints + i]
      self._pybullet_client.setJointMotorControl2(self.robotUid, finger_id, self._pybullet_client.POSITION_CONTROL,
                                                  targetPosition=fingerAngles[i], force=self._fingerTipForce)


  def move_to_gripper_pose2(self, commands):
    """
    end-effector position control mode
    given absolute position wrt. Cartesian world
    :param commands:

     1. control end-effector in Cartesian Space. useInverseKinematics is True.
    np.array o list with size 7
    [x, y, z, orn ]
    x,y,z : the value of end-effector in Cartesian Space. R^{3},Unit: meter
    orn : the quaterion of the end-effector, R^{4},[x,y,z,w]
    fingerAngle : angle of the fingers, R^{1},

    2. control joint angle derectly in joint configuration space. useInverseKinematics is False.
    np.array o list with size 10
    [j1,j2,...,j7, f1, f2, f3] Unit: rad

    :return:
    """
    if (self._useInverseKinematics):
      if np.array(commands).size != 7:
        raise Exception("Command size is not matched, require a command with the size of 8 but got ",
                        np.array(commands).size)
      x = commands[0]
      y = commands[1]
      z = commands[2]
      orn = commands[3:7]

      if self._isLimitedWorkSpace:
        x = np.clip(x, self.ee_X_lowerLimit, self.ee_X_upperLimit)
        y = np.clip(y, self.ee_Y_lowerLimit, self.ee_Y_upperLimit)
        z = np.clip(z, self.ee_Z_lowerLimit, self.ee_Z_upperLimit)

      self.endEffectorPos[0] = x
      self.endEffectorPos[1] = y
      self.endEffectorPos[2] = z
      self.endEffectorOrn = orn

      pos = np.array([x,y,z])

      if self._verbose:
        print('command: ', commands[0:3])
        ee_pos, ee_orn = self.GetEndEffectorObersavations()
        print('end-effecter position: ', ee_pos)
        print('end-effecter orentation ', ee_orn)
        print('joint angle', self.GetMotorAngles())

      # ik
      if (self._useNullSpace == 1):
        if (self._useOrientation == 1):
          jointPoses = self._pybullet_client.calculateInverseKinematics(self.robotUid, self.EndEffectorIndex, pos, orn,
                                                                        lowerLimits=self.ll, upperLimits=self.ul,
                                                                        jointRanges=self.jr, restPoses=self.rp,
                                                                        residualThreshold=0.001, jointDamping=self.jd)
        else:
          jointPoses = self._pybullet_client.calculateInverseKinematics(self.robotUid, self.EndEffectorIndex, pos,
                                                                        lowerLimits=self.ll, upperLimits=self.ul,
                                                                        jointRanges=self.jr, restPoses=self.rp,
                                                                       residualThreshold=0.001, jointDamping=self.jd)

        self._joint_target_pos = jointPoses[:self._numArmJoints]
        if (self._useSimulation):
          for i in range(self._numArmJoints):
            motor_id = self.motorIndices[i]
            self._SetDesiredMotorAngleById(motor_id, jointPoses[i])

        else:
          # reset the joint state (ignoring all dynamics, not recommended to use during simulation)
          for i in range(self._numArmJoints):
            self._pybullet_client.resetJointState(self.robotUid, self._joint_name_to_id[self.motorNames[i]], jointPoses[i])


    else:
      raise Exception("You must set \'useInverseKinematics==True\' in the ApplyAction_EndEffectorPose function.")


  #def move_to_gripper_pose_queue(self, pose):
  def move_to_gripper_pose(self, pose):
    """
    end-effector position control mode
    given absolute position wrt. Cartesian world
    :param commands:

     1. control end-effector in Cartesian Space. useInverseKinematics is True.
    np.array o list with size 7
    [x, y, z, orn ]
    x,y,z : the value of end-effector in Cartesian Space. R^{3},Unit: meter
    orn : the quaterion of the end-effector, R^{4},[x,y,z,w]
    fingerAngle : angle of the fingers, R^{1},

    2. control joint angle derectly in joint configuration space. useInverseKinematics is False.
    np.array o list with size 10
    [j1,j2,...,j7, f1, f2, f3] Unit: rad

    :return:
    """

    END_EFFECTOR_STEP = 0.005

    start_pose = self.end_effector_Pose
    end_pose = Pose([pose[0:3], pose[3:7]])


    delta_position = end_pose.position - start_pose.position
    num_waypoints = int(np.linalg.norm(delta_position)
                        / END_EFFECTOR_STEP)

    waypoints = []

    for i in range(num_waypoints):
      scale = float(i) / float(num_waypoints)
      position = start_pose.position + delta_position * scale
      euler = end_pose.euler
      waypoint = Pose([position, euler])
      waypoints.append(waypoint)
      self.target_pose_waypoints.put(waypoint)

    waypoints.append(end_pose)
    self.target_pose_waypoints.put(end_pose)

    self.target_pose = end_pose


  def move_to_joint_positions(self, jointPoses):
    assert np.array(jointPoses).shape[0] >= self._numArmJoints

    for i in range(self._numArmJoints):
      motor_id = self.motorIndices[i]
      self._SetDesiredMotorAngleById(motor_id, jointPoses[i])

    self.target_joint = jointPoses

  def send_joint_command(self):

    if not self.target_pose_waypoints.empty():
      self._joint_cmd = self.target_pose_waypoints.get()

      if self._joint_cmd is not None:

        pos = self._joint_cmd.position
        orn = self._joint_cmd.quaternion


        jointPose = self._pybullet_client.calculateInverseKinematics(self.robotUid, self.EndEffectorIndex, pos, orn,
                                                                      lowerLimits=self.ll, upperLimits=self.ul,
                                                                      jointRanges=self.jr, restPoses=self.last_jointPose,
                                                                      residualThreshold=0.001, jointDamping=self.jd)

        self.last_jointPose = jointPose
        for i in range(self._numArmJoints):
          motor_id = self.motorIndices[i]
          self._SetDesiredMotorAngleById(motor_id, jointPose[i])

  def check_ee_reached(self):
    END_EFFECTOR_POS_THRESHOLD = 0.01

    current_pose = self.end_effector_Pose

    dis = np.linalg.norm(current_pose.position - self.target_pose.position)

    if dis <= END_EFFECTOR_POS_THRESHOLD:
      return True
    else:
      return False

  def check_joint_reached(self, velocity=False):
    position_threshold = 0.0087
    velocity_threshold = 0.05

    current_position = self.GetMotorAngles()[:self._numArmJoints]
    delta_position = self.target_joint - current_position
    position_reached = (abs(delta_position) < position_threshold)

    if velocity:
      current_velocity = self.GetMotorVelocities()[:self._numArmJoints]
      delta_velocity = np.zeros(self._numArmJoints) - current_velocity

      velocity_reached = (abs(delta_velocity) < velocity_threshold)

      if not (position_reached.all() and velocity_reached.all()):
        return False
      return True

    if not (position_reached.all()):
      return False
    return True

  def is_limb_reached(self, velocity=False ):

    if self.target_pose is not None:
      if self.check_ee_reached():
        self.target_pose = None
        return True
      else:
        return False

    if self.target_joint is not None:
      if self.check_joint_reached(velocity=velocity):
        self.target_joint = None
        return True
      else:
        return False

    return True


    position_threshold = 0.0087
    velocity_threshold = 0.05

    current_position = self.GetMotorAngles()[:self._numArmJoints]
    delta_position = self._joint_target_pos - current_position
    position_reached = (abs(delta_position) <  position_threshold)

    if velocity:
      current_velocity = self.GetMotorVelocities()[:self._numArmJoints]
      delta_velocity = np.zeros(self._numArmJoints) - current_velocity

      velocity_reached = ( abs(delta_velocity) < velocity_threshold )

      if not (position_reached.all() and velocity_reached.all()):
        return False
      return True

    if not (position_reached.all()  ):
      return False
    return True



  def is_gripper_reached(self, verbose=False):

    return True


  @property
  def end_effector_pose(self):
    ee_pos, ee_orn = self.GetEndEffectorObersavations()
    return np.concatenate((ee_pos,ee_orn))

  @property
  def end_effector_Pose(self):
    ee_pos, ee_orn = self.GetEndEffectorObersavations()
    return Pose([ee_pos,ee_orn])


  @property
  def end_effector_lower_space(self):
    return [self.ee_X_lowerLimit, self.ee_Y_lowerLimit, self.ee_Z_lowerLimit]

  @property
  def end_effector_upper_space(self):
    return [self.ee_X_upperLimit, self.ee_Y_upperLimit, self.ee_Z_upperLimit]





