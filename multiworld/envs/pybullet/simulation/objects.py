import numpy as np
from multiworld.math import Pose
from ..simulation.body import Body
import glob
import os
import math

import random
import pybullet as p

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import transformations
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import json

ROBOT_URDF_PATH = os.path.join(os.path.dirname(currentdir), 'assets')
OBJECTS_DICT = {

'master_chef':["YCB/002_master_chef_can/master_chef.urdf",(0,0,0), []],
'sugar_box':["YCB/004_sugar_box/sugar_box.urdf",(math.pi,-math.pi/2,0), []],
'tomato_soup_can':["YCB/005_tomato_soup_can/tomato_soup_can.urdf",(0,0,0), []],
'mustard_bottle':["YCB/006_mustard_bottle/mustard_bottle.urdf",  (-math.pi/2, 0,  -math.pi/6), []],
'pudding_box':["YCB/008_pudding_box/pudding_box.urdf",(0,0,0), []],
'potted_meat_can':["YCB/010_potted_meat_can/potted_meat_can.urdf", (0,0,0), []],
'banana':["YCB/011_banana/banana.urdf",(0,0,0), []],
'apple':["YCB/013_apple/apple.urdf",(0,0,0), []],
'cracker_box':["YCB/003_cracker_box/cracker_box.urdf", (0,-math.pi/2,0), []],
'gelatin_box':["YCB/009_gelatin_box/gelatin_box.urdf", (0,-math.pi/2,0), []],
'bowl':["YCB/024_bowl/bowl.urdf", (0,0,0), []],
'mug':["YCB/025_mug/mug.urdf", (0,0,0), []],
'plate':["YCB/029_plate/plate.urdf", (0,0,0), []],
'rubiks_cube':["YCB/077_rubiks_cube/rubikes_cube.urdf", (0,0,0), []],
'cups_a':["YCB/065-a_cups/cups_a.urdf", (0,0,0), []],


'b_column': ["objects/blocks/block_column.urdf",(0,0,0), []],
'b_semi_column': ["objects/blocks/block_semi_column.urdf",(0,0,0), []],
'b_cube_m': ["objects/blocks/block_cube_m.urdf",(0,0,0), []],
'b_cube_w': ["objects/blocks/block_cube_w.urdf",(0,0,0), []],
'b_cuboid': ["objects/blocks/block_cuboid.urdf",(0,0,0), []],
'b_cuboid2': ["objects/blocks/block_cuboid2.urdf",(math.pi/2,0,0), []],#big
'b_L1':     ["objects/blocks/block_L1.urdf",(0,0,0), []],
'b_L2':     ["objects/blocks/block_L2.urdf",(0,0,0), []],


'ball_visual': ["objects/balls/ball_visual.urdf",(0,0,0), []],

'box_b':  ['objects/box/box_blue.urdf',(0,0,0), []],

'lshape_1':['Lshapes/auto_gen_objects_14420_5139.sdf',(0,0,0), []],
'Lshape_train' :['Lshapes/train',(0,0,0), []],


'shapenet':['shapenet/02876657',(0,0,0), []],
}


def get_config_value(config):
    """Get the value of an configuration item.

    If the config is None, return None. If the config is a value, return the
    value. Otherwise, the config must by a tuple or list represents low and
    high values to sample the property from.
    """
    if config is None:
        return None
    elif isinstance(config, (int, float)):
        return config
    elif isinstance(config, (list, tuple)):
        return np.random.uniform(low=config[0], high=config[1])
    else:
        raise ValueError('config %r of type %r is not supported.'
                         % (config, type(config)))

class Objects(object):
    def __init__(self,

                 obj_name_list,
                 num_movable_bodies,
                 is_fixed=False,
                 obj_fixed_poses =[],
                 obj_pos_upper_space=(0.2, -0.45, 0.20),
                 obj_pos_lower_space=(-0.2, -0.55, 0.20),
                 obj_max_upper_space=(0 + 0.3, -0.40 + 0.2, 0.4),
                 obj_max_lower_space=(0 - 0.3, -0.40 - 0.2, -0.4),
                 obj_euler_upper_space=(np.pi, np.pi, np.pi),
                 obj_euler_lower_space=(-np.pi, -np.pi, -np.pi),
                 obj_safe_margin=0.01,
                 obj_scale_range=(1, 1),
                 obj_mass=None,
                 obj_friction=None,
                 use_random_rgba=False,
                 num_RespawnObjects=None,

                  **kwargs
                    ):
        self._p = p
        self.OBJ_SCALE_RANGE = obj_scale_range
        self.NUM_MOVABLE_BODIES = self.num_objects = num_movable_bodies

        self.OBJ_NAME_LIST = obj_name_list
        self.OBJ_POS_UPPER_SPACE = obj_pos_upper_space
        self.OBJ_POS_LOWER_SPACE = obj_pos_lower_space
        self.OBJ_MAX_UPPER_SPACE = obj_max_upper_space
        self.OBJ_MAX_LOWER_SPACE = obj_max_lower_space
        self.OBJ_EULER_UPPER_SPACE = obj_euler_upper_space
        self.OBJ_EULER_LOWER_SPACE = obj_euler_lower_space
        self.OBJ_SAFE_MARGIN = obj_safe_margin

        self.USE_RANDOM_RGBA = use_random_rgba
        self.OBJ_MASS = obj_mass
        self.OBJ_FRICTION = obj_friction

        self._reset_counter = 0
        self.movable_bodies = []
        self.base_eulers_list = []
        self._num_RespawnObjects = num_RespawnObjects

        self.is_fixed=is_fixed
        self.OBJ_FIXED_POSES = obj_fixed_poses

        if 'Lshape_train' in self.OBJ_NAME_LIST:
            self.OBJ_NAME_LIST.remove('Lshape_train')
            sdf_dir = OBJECTS_DICT['Lshape_train'][0]
            sdf_files = os.listdir(os.path.join(ROBOT_URDF_PATH,sdf_dir))
            for i in range(len(sdf_files)):
                obj_name = 'lshapeTrain_{}'.format(i)
                OBJECTS_DICT[obj_name] =  [sdf_dir+'/{}'.format(sdf_files[i]), (0, 0, 0), []]
                self.OBJ_NAME_LIST.append(obj_name)

            print('update OBJECT DICT')

        self.load_shapenet_urdf2list()


    def load_shapenet_urdf2list(self):
        if 'shapenet' in self.OBJ_NAME_LIST:
            self.OBJ_NAME_LIST.remove('shapenet')

            self.shapenet_info = json.load(open(os.path.join(ROBOT_URDF_PATH, 'shapenet/shapenet_id.json'), 'r'))
            self.shapenet_path = os.path.join(ROBOT_URDF_PATH, 'shapenet')


            for cat in self.shapenet_info.keys():
                for i, obj_id in enumerate(self.shapenet_info[cat]['object_id']):
                    obj_name = 'shapenet_{}_{}'.format(cat, i)
                    OBJECTS_DICT[obj_name] =  [self.load_shapenet(obj_name), (0, 0, 0), []]
                    self.OBJ_NAME_LIST.append(obj_name)

            print('update OBJECT DICT')

    def _load_movable_objects(self):
        """Load movable bodies."""
        if self._num_RespawnObjects is not None and self._reset_counter >= self._num_RespawnObjects:
            # delete all the blocks
            for body in self.movable_bodies:
                body.remove_body()
            self.movable_bodies = []
            self.base_eulers_list = []

        if len(self.movable_bodies) ==0:
            self.target_movable_paths = []
            self.base_eulers = []
            for obj_name in self.OBJ_NAME_LIST:
                if not os.path.isabs(obj_name):
                    file_path = os.path.join(ROBOT_URDF_PATH, OBJECTS_DICT[obj_name][0])
                    self.base_eulers.append(OBJECTS_DICT[obj_name][1])
                else:
                    file_path = obj_name
                self.target_movable_paths += glob.glob(file_path)
            assert len(self.target_movable_paths) > 0


            is_valid = False
            while not is_valid:
                is_valid = True

                # Sample placements of the movable objects.
                is_target = False
                movable_poses = self._sample_body_poses( self.NUM_MOVABLE_BODIES )

                for i in range(self.NUM_MOVABLE_BODIES):
                    index = random.randint(0, len(self.target_movable_paths)-1)

                    urdf_path = self.target_movable_paths[index]
                    base_pose = self.base_eulers[index]

                    pose = movable_poses[i]
                    scale = np.random.uniform(*self.OBJ_SCALE_RANGE)
                    name = 'movable_%d' % i

                    #TODO load base pose from CONFIG, or from sample function
                    #pose.euler = base_pose  ## TODO load base pose from CONFIG, or from sample function
                    base_rot_mat = Pose(value=[[0, 0, 0], self.base_eulers[index]]).matrix4
                    rot = base_rot_mat.dot(pose.matrix4)
                    euler = transformations.euler_from_matrix(rot)
                    pose.euler = euler  ## TODO  use base pose or Not

                    # Add object.
                    obj_body = Body(self._p, urdf_path, pose, scale=scale, name=name)

                    if self.USE_RANDOM_RGBA:
                        r = np.random.uniform(0., 1.)
                        g = np.random.uniform(0., 1.)
                        b = np.random.uniform(0., 1.)
                        obj_body.set_color(rgba=[r, g, b, 1.0], specular=[0, 0, 0])

                    mass =  get_config_value(self.OBJ_MASS)
                    lateral_friction =  get_config_value(self.OBJ_FRICTION)
                    obj_body.set_dynamics(
                        mass=mass,
                        lateral_friction=lateral_friction,
                        rolling_friction=None,
                        spinning_friction=None)

                    self.base_eulers_list.append(self.base_eulers[index])
                    self.movable_bodies.append(obj_body)
            self._reset_counter = 0
        else:
            self._reset_movable_obecjts()
        self._reset_counter += 1


    def _sample_body_poses(self,  num_samples, max_attemps=32):
        """Sample body poses.

        Args:
            num_samples: Number of samples.
            body_config: Configuration of the body.
            max_attemps: Maximum number of attemps to find a feasible
                placement.

        Returns:
            List of poses.
        """

        while True:
            movable_poses = []

            for i in range(num_samples):
                num_attemps = 0
                is_valid = False
                while not is_valid and num_attemps <= max_attemps:
                    pose = Pose.uniform(x=[self.OBJ_POS_LOWER_SPACE[0], self.OBJ_POS_UPPER_SPACE[0]],
                                        y=[self.OBJ_POS_LOWER_SPACE[1], self.OBJ_POS_UPPER_SPACE[1]],
                                        z=[self.OBJ_POS_LOWER_SPACE[2], self.OBJ_POS_UPPER_SPACE[2]],
                                        roll=[self.OBJ_EULER_LOWER_SPACE[0], self.OBJ_EULER_UPPER_SPACE[0]],
                                        pitch=[self.OBJ_EULER_LOWER_SPACE[1], self.OBJ_EULER_UPPER_SPACE[1]],
                                        yaw=[self.OBJ_EULER_LOWER_SPACE[2], self.OBJ_EULER_UPPER_SPACE[2]],)

                    # Check if the new pose is distant from other bodies.
                    is_valid = True
                    for other_pose in movable_poses:
                        dist = np.linalg.norm(
                            pose.position[:2] - other_pose.position[:2])

                        if dist <  self.OBJ_SAFE_MARGIN:
                            is_valid = False
                            num_attemps += 1
                            break

                if not is_valid:
                    logger.warning('Cannot find the placement of body %d. '
                                'Start re-sampling.', i)
                    break
                else:
                    movable_poses.append(pose)

            if i >= num_attemps -1 :
                break

        return movable_poses

    def _reset_movable_obecjts(self, movable_poses=None):
        if movable_poses is None:
            movable_poses = self._sample_body_poses(self.NUM_MOVABLE_BODIES)
        for i, body in enumerate(self.movable_bodies):


            # update pose.
            body.pose= movable_poses[i]

            #-------------------

    def _load_movable_fixed_objects(self, movable_poses):
        """Load movable bodies."""
        if self._num_RespawnObjects is not None and self._reset_counter >= self._num_RespawnObjects:
            # delete all the blocks
            for body in self.movable_bodies:
                body.remove_body()
            self.movable_bodies = []
            self.base_eulers_list = []

        assert movable_poses is not None


        if len(self.movable_bodies) == 0:
            self.target_movable_paths = []
            self.base_eulers = []
            for obj_name in self.OBJ_NAME_LIST:
                if not os.path.isabs(obj_name):
                    file_path = os.path.join(ROBOT_URDF_PATH, OBJECTS_DICT[obj_name][0])
                    self.base_eulers.append(OBJECTS_DICT[obj_name][1])
                else:
                    file_path = obj_name
                self.target_movable_paths += glob.glob(file_path)
            assert len(self.target_movable_paths) > 0

            for i in range(self.NUM_MOVABLE_BODIES):
                #index = i#random.randint(0, len(self.target_movable_paths) - 1)  # for fixed object
                index = random.randint(0, len(self.target_movable_paths) - 1)
                urdf_path = self.target_movable_paths[index]
                base_pose = self.base_eulers[index]

                pose = movable_poses[i]
                scale = np.random.uniform(*self.OBJ_SCALE_RANGE)
                name = 'movable_%d' % i

                base_rot_mat = Pose(value=[[0, 0, 0], self.base_eulers[index]]).matrix4
                rot = base_rot_mat.dot(pose.matrix4)
                euler = transformations.euler_from_matrix(rot)
                pose.euler = euler  ## TODO  use base pose or Not


                # Add object.
                obj_body = Body(self._p, urdf_path, pose, scale=scale, name=name)

                if self.USE_RANDOM_RGBA:
                    r = np.random.uniform(0., 1.)
                    g = np.random.uniform(0., 1.)
                    b = np.random.uniform(0., 1.)
                    obj_body.set_color(rgba=[r, g, b, 1.0], specular=[0, 0, 0])

                mass = get_config_value(self.OBJ_MASS)
                lateral_friction = get_config_value(self.OBJ_FRICTION)
                obj_body.set_dynamics(
                    mass=mass,
                    lateral_friction=lateral_friction,
                    rolling_friction=None,
                    spinning_friction=None)

                self.movable_bodies.append(obj_body)
                self.base_eulers_list.append(self.base_eulers[index])
            self._reset_counter = 0
        self._reset_counter += 1

        self._reset_movable_obecjts(movable_poses)

    def reset(self, movable_poses=None ):
        if self.is_fixed:
            if movable_poses is None:
                assert len(self.OBJ_FIXED_POSES) == self.num_objects
                movable_poses = self.OBJ_FIXED_POSES
            self._load_movable_fixed_objects(movable_poses)
        else:
            self._load_movable_objects()


    @property
    def object_sample_space_high(self):
        return self.OBJ_POS_UPPER_SPACE
    @property
    def object_sample_space_low(self):
        return self.OBJ_POS_LOWER_SPACE

    @property
    def object_max_space_high(self):
        return self.OBJ_MAX_UPPER_SPACE

    @property
    def object_max_space_low(self):
        return self.OBJ_MAX_LOWER_SPACE

    @property
    def object_euler_space_low(self):
        return self.OBJ_EULER_LOWER_SPACE

    @property
    def object_euler_space_high(self):
        return self.OBJ_EULER_UPPER_SPACE

    @property
    def object_space_polygon(self):
        obj_space_polygon = Polygon([(self.OBJ_MAX_LOWER_SPACE[0], self.OBJ_MAX_UPPER_SPACE[1]),
                                          (self.OBJ_MAX_UPPER_SPACE[0], self.OBJ_MAX_UPPER_SPACE[1]),
                                          (self.OBJ_MAX_UPPER_SPACE[0], self.OBJ_MAX_LOWER_SPACE[1]),
                                          (self.OBJ_MAX_LOWER_SPACE[0], self.OBJ_MAX_LOWER_SPACE[1])])

        return obj_space_polygon

    def remove_all_body(self):
        for body in self.movable_bodies:
            body.remove_body()
        self.movable_bodies =[]
        self.base_eulers_list =[]

    def load_random_shapenet(self):
        object_cat_cur = np.random.choice(list(self.shapenet_info.keys()))
        category_id = self.shapenet_info[object_cat_cur]['category_id']
        tmp = np.random.choice(len(self.shapenet_info[object_cat_cur]['object_id']))
        object_id = self.shapenet_info[object_cat_cur]['object_id'][tmp]
        urdf_path = os.path.join(self.shapenet_path, '%s/%s/obj.urdf' % (category_id, object_id))

        return urdf_path

    def load_shapenet(self, object_name=None):
        name_list = object_name.split('_')
        if name_list[0] == 'shapenet':
            if len(name_list) == 2:
                category_name = name_list[1]
                object_id = None
            elif len(name_list) == 3:
                category_name = name_list[1]
                object_id = name_list[2]
            else:
                raise NotImplementedError

            category_id = self.shapenet_info[category_name]['category_id']

            if object_id is None:
                object_id = np.random.choice(len(self.shapenet_info[category_id]['object_id']))

            object_id = self.shapenet_info[category_name]['object_id'][int(object_id)]
            urdf_path = os.path.join(self.shapenet_path, '%s/%s/obj.urdf' % (category_id, object_id))
        else:
            urdf_path = None

        return urdf_path











