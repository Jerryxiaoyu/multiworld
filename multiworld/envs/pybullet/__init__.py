import gym
from gym.envs.registration import register
import logging
import math

LOGGER = logging.getLogger(__name__)
REGISTERED = False


def register_pybullet_envs():
    global REGISTERED
    if REGISTERED:
        return
    REGISTERED = True
    LOGGER.info("Registering multiworld pybullet gym environments")

    """
    Reaching tasks
    """
    register(
        id='Jaco2ObjectsPusherXYEnv-v0',
        entry_point='multiworld.envs.pybullet.jaco_xyz.jaco_push_multiobj:Jaco2BlockPusherXYZ',
        max_episode_steps=200,
        reward_threshold=20000.0,
        kwargs=dict(
            # env arguments
            addTable=True,
            table_position=[0.0000000, -0.650000, -0.59000],
            table_name='white',

            # obj setting
            obj_name_list=['b_cube_m', 'b_cube_w', 'b_L1', 'b_L2',
                           'b_cube_m', 'b_cube_w', 'b_L1', 'b_L2'],
            num_movable_bodies=2,
            obj_pos_upper_space=(0 + 0.25, -0.40 + 0.15, 0.06),
            obj_pos_lower_space=(0 - 0.25, -0.40 - 0.15, 0.06),
            obj_max_upper_space=(0 + 0.3, -0.40 + 0.2, 0.4),
            obj_max_lower_space=(0 - 0.3, -0.40 - 0.2, -0.4),

            obj_euler_upper_space=(0, 0, 0),
            obj_euler_lower_space=(0, 0, 0),
            obj_safe_margin=0.05,
            obj_scale_range=(1, 1),
            obj_mass=None,
            obj_friction=0.8,
            use_random_rgba=True,
            num_RespawnObjects=2,

            # block area


            # robot params
            random_robot_ee_pos=False,
            hand_init_upper_space=(0 + 0.25, -0.40 + 0.15, 0.154),
            hand_init_lower_space=(0 - 0.25, -0.40 - 0.15, 0.154),

            random_init_object_position=True,
            default_init_position=(0.15, -0.35, 0.16),
            initPos_upper_space=(0.2, -0.2, 0.16),
            initPos_lower_space=(-0.2, -0.6, 0.16),
            object_EulerOrn=(0, -math.pi / 2, math.pi / 2),

            random_target_position=False,
            default_target_position=(0.0, -0.43, 0.115),
            # target_upper_space=(0.3, -0.40, 0.125),
            # target_lower_space=(-0.3, -0.45, 0.125),

            image=False,
            render_params={
                "target_pos": (0, -0.742, 0.642),
                "distance": 0.15,
                "yaw": 0,
                "pitch": -37,
                "roll": 0,
            },
            camera_params={"target_pos": (0.0, -0.461, -0.004),
                           "distance": 0.480,
                           "yaw": 0,
                           "pitch": -63.684,
                           "roll": 0,
                           "fov": 70,
                           "near": 0.1,
                           "far": 100.0,
                           "image_width": 64,
                           "image_height": 64, },

            objects_list=['b_cube_m', 'b_cube_w', 'b_L1', 'b_L2',
                          'b_cube_m', 'b_cube_w', 'b_L1', 'b_L2',
                           ],

            # base arguments
            kinova_type='j2s7s300_beam',
            control_mode='pos_consXY',

            ## 10s traj horizon
            timeStep=0.01,
            actionRepeat=10,  # very importrant!!!
            maxSteps=1000,
            action_scale_conts=0.01,  #m
            isRender=True,
            debug=False,
            multi_view=False,
            isImageObservation=False,



            robotHomeAngle=(4.800, 2.947, -0.253, 0.842, 0.000, 4.211, 0.505, 1.263, 1.347, 1.347),
            robot_ee_pos=(0.03741799, -0.1755366, 0.25416687),
            fixed_orn=(-0.32618785, 0.94479543, 0.02328156, -0.02051954),

            fixed_finger=1.263,

            stepTextPosition=[-0.55, 0, 0.3],

            # kinova arguments
            isLimitedWorkSpace=True,
            hand_high=(0 + 0.3, -0.40 + 0.2, 0.4),
            hand_low=(0 - 0.3, -0.40 - 0.2, -0.4),
            useDynamics=True,

        )
    )





def create_image_pybullet_jaco_push_primitive_xy_env_v1():
    from multiworld.core.image_env import ImageEnv


    wrapped_env = gym.make('Jaco2PushPrimitiveXYEnv-v0', isRender=True)
    #return wrapped_env
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_xyz_reacher_camera_v0,
        transpose=True,
        normalize=True,
    )




