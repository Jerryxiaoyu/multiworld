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
