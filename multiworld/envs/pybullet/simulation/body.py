"""The body class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from multiworld.math import Pose, Orientation
import numpy as np

class Body:
    """Body."""

    def __init__(self,
                 pybullet_client,
                 filename,
                 pose,
                 scale=1.0,
                 is_static=False,
                 name=None,
                 ):
        """Initialize.

        Args:
            simulator: The simulator of the entity.
            filename: Path to the URDF file.
            pose: The initial pose of the body.
            scale: The scaling factor of the body.
            is_static: If the body is static.
            name: The name of the entity.
        """

        self._pybullet_client = pybullet_client

        self._uid = None
        self._name = name
        self._scoped_name = None

        suffix = filename.split('.')[-1]
        if suffix == 'urdf':
            self._uid =  self._pybullet_client.loadURDF( filename, pose.position, pose.quaternion,
                                                         useFixedBase=is_static,  useMaximalCoordinates=False,
                                                         globalScaling=scale)
        elif suffix =='sdf':
            uids =  self._pybullet_client.loadSDF( filename,  useMaximalCoordinates=False,
                                                         globalScaling=scale)
            assert len(uids)  == 1, "SDF only contrains ONE body."
            self._uid = uids[0]
            self._pybullet_client.resetBasePositionAndOrientation(self._uid, pose.position, pose.quaternion)
        else:
            raise NotImplementedError


        self._initial_relative_pose = pose
        self._is_static = is_static


        self._constraints = []

        if name is None:
            model_name, _ = os.path.splitext(os.path.basename(filename))
            self._name = '%s_%s' % (model_name, self.uid)

    @property
    def uid(self):
        return self._uid

    # @property
    # def links(self):
    #     return self._links
    #
    # @property
    # def joints(self):
    #     return self._joints

    @property
    def pose(self):
        position, quaternion = self._pybullet_client.getBasePositionAndOrientation(
            bodyUniqueId=self.uid, )

        return Pose([position, quaternion])

    @property
    def position(self):
        return self.pose.position

    @property
    def orientation(self):
        return self.pose.orientation

    # @property
    # def joint_positions(self):
    #     return [joint.position for joint in self.joints]
    #
    # @property
    # def joint_velocities(self):
    #     return [joint.velocity for joint in self.joints]
    #
    # @property
    # def joint_lower_limits(self):
    #     return [joint.lower_limit for joint in self.joints]
    #
    # @property
    # def joint_upper_limits(self):
    #     return [joint.upper_limit for joint in self.joints]
    #
    # @property
    # def joint_max_efforts(self):
    #     return [joint.max_effort for joint in self.joints]
    #
    # @property
    # def joint_max_velocities(self):
    #     return [joint.max_velocity for joint in self.joints]
    #
    # @property
    # def joint_dampings(self):
    #     return [joint.damping for joint in self.joints]
    #
    # @property
    # def joint_frictions(self):
    #     return [joint.friction for joint in self.joints]
    #
    # @property
    # def joint_ranges(self):
    #     return [joint.range for joint in self.joints]

    @property
    def linear_velocity(self):
        linear_velocity, _ = self._pybullet_client.getBaseVelocity(
            bodyUniqueId=self.uid, )
        return np.array(linear_velocity, dtype=np.float32)

    @property
    def angular_velocity(self):
        _, angular_velocity = self._pybullet_client.getBaseVelocity(
            bodyUniqueId=self.uid, )
        return np.array(angular_velocity, dtype=np.float32)

    @property
    def mass(self):
        if self._mass is None:
            mass, _, _, _, _, _, _, _, _, _ = self._pybullet_client.getDynamicsInfo(
            bodyUniqueId=self.uid, linkIndex=-1, )
            self._mass = mass
        return self._mass

    @property
    def dynamics(self):
        (mass, lateral_friction, _, _, _, _,
         rolling_friction, spinning_friction, _, _) = self._pybullet_client.getDynamicsInfo(
            bodyUniqueId=self.uid, linkIndex=-1, )

        return {
            'mass': mass,
            'lateral_friction': lateral_friction,
            'rolling_friction': rolling_friction,
            'spinning_friction': spinning_friction,
        }

    @property
    def is_static(self):
        return self._is_static

    @property
    def contacts(self):
        return self.physics.get_body_contacts(self.uid)

    @pose.setter
    def pose(self, value):
        pose = Pose(value)
        position = list(pose.position)
        quaternion = list(pose.quaternion)
        self._pybullet_client.resetBasePositionAndOrientation(
            bodyUniqueId=self.uid, posObj=position, ornObj=quaternion,
            )



    @position.setter
    def position(self, value):
        position = list(value)
        _, quaternion = self._pybullet_client.getBasePositionAndOrientation(self.uid)
        self._pybullet_client.resetBasePositionAndOrientation(
            bodyUniqueId=self.uid, posObj=position, ornObj=quaternion,
            )


    @orientation.setter
    def orientation(self, value):
        orientation = value
        assert isinstance(orientation, Orientation)
        position, _ = self._pybullet_client.getBasePositionAndOrientation(self.uid)
        quaternion = list(orientation.quaternion)
        self._pybullet_client.resetBasePositionAndOrientation(
            bodyUniqueId=self.uid, posObj=position, ornObj=quaternion,
            )

    # @joint_positions.setter
    # def joint_positions(self, value):
    #     for joint, joint_position in zip(self.joints, value):
    #         joint.position = joint_position

    @linear_velocity.setter
    def linear_velocity(self, value):
        linear_velocity = list(value)
        self._pybullet_client.resetBaseVelocity(
            bodyUniqueId=self, linearVelocity=linear_velocity,
            )



    @angular_velocity.setter
    def angular_velocity(self, value):
        angular_velocity = list(value)
        self._pybullet_client.resetBaseVelocity(
            bodyUniqueId=self.uid, angularVelocity=angular_velocity,
            )


    @mass.setter
    def mass(self, value):
        self._pybullet_client.changeDynamics(
            bodyUniqueId=self.uid, linkIndex=-1, mass=value,
            )

    # def get_joint_by_name(self, name):
    #     """Get the joint by the joint name.,
    #
    #     Args:
    #         name: The joint name.
    #
    #     Returns:
    #         An instance of Joint. Return None if the joint is not found.
    #     """
    #     for joint in self.joints:
    #         if joint.name == name:
    #             return joint
    #
    #     raise ValueError('The joint %s is not found in body %s.'
    #                      % (name, self.name))
    #
    # def get_link_by_name(self, name):
    #     """Get the link by the link name.,
    #
    #     Args:
    #         name: The link name.
    #
    #     Returns:
    #         An instance of Link. Return None if the link is not found.
    #     """
    #     for link in self.links:
    #         if link.name == name:
    #             return link
    #
    #     raise ValueError('The link %s is not found in body %s.'
    #                      % (name, self.name))

    def update(self):
        """Update disturbances."""
        pass

    def set_dynamics(self,
                     mass=None,
                     lateral_friction=None,
                     rolling_friction=None,
                     spinning_friction=None,
                     ):
        """Set dynmamics.

        Args:
            mass: The mass of the body.
            lateral_friction: The lateral friction coefficient.
            rolling_friction: The rolling friction coefficient.
            spinning_friction: The spinning friction coefficient.
        """
        kwargs = dict()
        #kwargs['physicsClientId'] = self.uid
        kwargs['bodyUniqueId'] = self.uid
        kwargs['linkIndex'] = -1

        if mass is not None:
            kwargs['mass'] = mass

        if lateral_friction is not None:
            kwargs['lateralFriction'] = lateral_friction

        if rolling_friction is not None:
            kwargs['rollingFriction'] = rolling_friction

        if spinning_friction is not None:
            kwargs['spinningFriction'] = spinning_friction

        self._pybullet_client.changeDynamics(**kwargs)



    def set_color(self, rgba=None, specular=None):
        """Set color.

        Args:
            rgba: The color in RGBA.
            specular: The specular of the object.
        """
        kwargs = dict()
       # kwargs['physicsClientId'] = self.uid
        kwargs['objectUniqueId'] = self.uid
        kwargs['linkIndex'] = -1

        if rgba is not None:
            kwargs['rgbaColor'] = rgba

        if specular is not None:
            kwargs['specularColor'] = specular

        self._pybullet_client.changeVisualShape(**kwargs)


    def remove_body(self):
        self._pybullet_client.removeBody(self.uid)


