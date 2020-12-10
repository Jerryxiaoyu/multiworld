import numpy as np
import math

jaco2_push_top_view_camera = {"target_pos": (0.0, -0.461, -0.004),
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

                             # camera in world pose Twc
                             'camera_pose': {'translation': [0.0, -0.4, 0.75],
                                             'rotation': [-math.pi, 0, 0], },  # upright
                             }


# jaco2_push_lateral_view_camera = {"target_pos": (0.0, -0.461, -0.004),
#                              "distance": 1,
#                              "yaw": 0,
#                              "pitch": -63.684,
#                              "roll": 0,
#                              "fov": 60,
#                              "near": 0.1,
#                              "far": 100.0,
#                              "image_width": 640,
#                              "image_height": 480,
#                              'intrinsics': [610.911, 0., 321.936, 0., 611.021, 236.665, 0., 0., 1.],
#                              'translation': None,
#                              'rotation': None,
#
#                              # camera in world pose Twc
#                              'camera_pose': {'translation': [0,-0.767,0.571],
#                                              'rotation': [-2.579, 0, 0], },  # upright
#                              }

jaco2_push_lateral_view_camera = {"target_pos": (0.0, -0.461, -0.004),
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

                             # camera in world pose Twc
                             'camera_pose': {'translation': [0,-0.796,0.533],
                                             'rotation':    [-2.579, 0, 0], },  # upright
                             }


DEFAULT_CAMERA =  {"target_pos": (0.0, -0.461, -0.004),
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

                   # camera in world pose Twc
                   'camera_pose': {'translation': [0.0, -0.4, 0.75],
                                   'rotation': [-math.pi, 0, 0], },  # upright
                   }