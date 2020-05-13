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

        kwargs=dict(
            # env arguments
            addTable=True,
            table_position=[0.0000000, -0.650000, -0.59000],
            table_name='white',

            # obj setting
            isRandomObjects=True,
            fixed_objects_init_pos = (0, -0.3, 0.06,        # shape (3*n,)
                                      0, -0.5, 0.06,),
            obj_name_list=['b_cube_m', 'b_cube_w', 'b_L1', 'b_L2',
                           'b_cube_m', 'b_cube_w', 'b_L1', 'b_L2'],
            num_movable_bodies=2,
            obj_pos_upper_space=(0 + 0.2, -0.25, 0.06),
            obj_pos_lower_space=(0 - 0.2, -0.55, 0.06),
            obj_max_upper_space=(0 + 0.28, -0.20, 0.4),
            obj_max_lower_space=(0 - 0.28, -0.60, -0.4),

            obj_euler_upper_space=(0, 0, 0),
            obj_euler_lower_space=(0, 0, 0),
            obj_safe_margin=0.05,
            obj_scale_range=(1, 1),
            obj_mass=None,
            obj_friction=0.8,
            use_random_rgba=True,
            num_RespawnObjects=5,

            # goal
            isRandomGoals=True,
            isIgnoreGoalCollision=False,
            fixed_objects_goals= (-0.15,-0.3,0.06,
                                   0.15, -0.5, 0.06,), # shape (3*n,)
            fixed_hand_goal = (0, -0.25, 0.154),
            target_upper_space=(0.25, -0.25, 0.06),
            target_lower_space=(-0.25, -0.55, 0.06),

            # camera
            # camera_params = {},
            shadow_enable=False,
            isImgMask=False,
            isImgDepth=False,

            # robot params
            random_robot_ee_pos=False,
            hand_init_upper_space=(0 + 0.28, -0.2, 0.154),
            hand_init_lower_space=(0 - 0.28, -0.6, 0.154),


            # base arguments
            kinova_type='j2s7s300_beam',
            control_mode='pos_consXY',

            ## 30s traj horizon
            ## action control rate: 1s
            timeStep=0.01,
            actionRepeat=100,  # very importrant!!!
            maxSteps=3000,
            action_scale_conts=0.1,  #m

            robotHomeAngle= (4.800, 2.947, -0.253, 0.842, 0.000, 4.211, 0.505, 1.263, 1.347, 1.347),
            robot_ee_pos=   (0.02, -0.25,  0.15),
            fixed_orn=    (-0.32618785, 0.94479543, 0.02328156, -0.02051954),
            fixed_finger=1.263,



            # kinova arguments
            isLimitedWorkSpace=True,
            hand_high   =   (0 + 0.3, -0.40 + 0.2, 0.4),
            hand_low    =   (0 - 0.3, -0.40 - 0.2, -0.4),
            useDynamics=True,

            render_params={
                "target_pos": (0, -0.742, 0.642),
                "distance": 0.15,
                "yaw": 0,
                "pitch": -37,
                "roll": 0,
            },
            stepTextPosition=[-0.55, 0, 0.3],

            # debug setting
            isRenderGoal=False,
            isRender=True,
            debug=False,

        )
    )

    register(
        id='Jaco2ObjectsPusherOneXYEnv-v0',
        entry_point='multiworld.envs.pybullet.jaco_xyz.jaco_push_multiobj:Jaco2BlockPusherXYZ',

        kwargs=dict(
            # env arguments
            addTable=True,
            table_position=[0.0000000, -0.650000, -0.59000],
            table_name='white',

            # obj setting
            isRandomObjects=True,
            fixed_objects_init_pos=(0, -0.3, 0.06,  # shape (3*n,)
                                    0, -0.5, 0.06,),
            obj_name_list=['b_cube_m', 'b_cube_w', 'b_L1', 'b_L2',
                           'b_cube_m', 'b_cube_w', 'b_L1', 'b_L2'],
            num_movable_bodies=1,
            obj_pos_upper_space=(0 + 0.2, -0.25, 0.06),
            obj_pos_lower_space=(0 - 0.2, -0.55, 0.06),
            obj_max_upper_space=(0 + 0.28, -0.20, 0.4),
            obj_max_lower_space=(0 - 0.28, -0.60, -0.4),

            obj_euler_upper_space=(0, 0, 0),
            obj_euler_lower_space=(0, 0, 0),
            obj_safe_margin=0.05,
            obj_scale_range=(1, 1),
            obj_mass=None,
            obj_friction=0.8,
            use_random_rgba=True,
            num_RespawnObjects=5,

            # goal
            isRandomGoals=True,
            isIgnoreGoalCollision=False,
            fixed_objects_goals=(-0.15, -0.3, 0.06,
                                 0.15, -0.5, 0.06,),  # shape (3*n,)
            fixed_hand_goal=(0, -0.25, 0.154),
            target_upper_space=(0.25, -0.25, 0.06),
            target_lower_space=(-0.25, -0.55, 0.06),

            # camera
            # camera_params = {},
            shadow_enable=False,
            isImgMask=False,
            isImgDepth=False,

            # robot params
            random_robot_ee_pos=False,
            hand_init_upper_space=(0 + 0.28, -0.2, 0.154),
            hand_init_lower_space=(0 - 0.28, -0.6, 0.154),

            # base arguments
            kinova_type='j2s7s300_beam',
            control_mode='pos_consXY',

            ## 30s traj horizon
            ## action control rate: 1s
            timeStep=0.01,
            actionRepeat=100,  # very importrant!!!
            maxSteps=3000,
            action_scale_conts=0.1,  # m

            robotHomeAngle=(4.800, 2.947, -0.253, 0.842, 0.000, 4.211, 0.505, 1.263, 1.347, 1.347),
            robot_ee_pos=(0.02, -0.25, 0.15),
            fixed_orn=(-0.32618785, 0.94479543, 0.02328156, -0.02051954),
            fixed_finger=1.263,

            # kinova arguments
            isLimitedWorkSpace=True,
            hand_high=(0 + 0.3, -0.40 + 0.2, 0.4),
            hand_low=(0 - 0.3, -0.40 - 0.2, -0.4),
            useDynamics=True,

            render_params={
                "target_pos": (0, -0.742, 0.642),
                "distance": 0.15,
                "yaw": 0,
                "pitch": -37,
                "roll": 0,
            },
            stepTextPosition=[-0.55, 0, 0.3],

            # debug setting
            isRenderGoal=False,
            isRender=False,
            debug=False,

        )
    )
    register(
        id='Jaco2PushPrimitiveXYEnv-v0',
        entry_point='multiworld.envs.pybullet.jaco_push_primitive.jaco_push_primitive:Jaco2PushPrimitiveXY',
        max_episode_steps=200,
        reward_threshold=20000.0,
        kwargs=dict(
            # env arguments
            addTable=True,
            table_position=[0.0000000, -0.650000, -0.59000],
            table_name='white',

            isImageObservation=False,

            # obj setting
            isRandomObjects=True,
            fixed_objects_init_pos=(0, -0.3, 0.06,  # shape (3*n,)
                                    0, -0.5, 0.06,),
            obj_name_list=['b_cube_m', 'b_cube_w', 'b_L1', 'b_L2',
                           'b_cube_m', 'b_cube_w', 'b_L1', 'b_L2'],
            num_movable_bodies=1,
            obj_pos_upper_space=(0 + 0.2, -0.25, 0.06),
            obj_pos_lower_space=(0 - 0.2, -0.55, 0.06),
            obj_max_upper_space=(0 + 0.28, -0.20, 0.4),
            obj_max_lower_space=(0 - 0.28, -0.60, -0.4),

            obj_euler_upper_space=(0, 0, 0),
            obj_euler_lower_space=(0, 0, 0),
            obj_safe_margin=0.05,
            obj_scale_range=(1, 1),
            obj_mass=None,
            obj_friction=0.8,
            use_random_rgba=True,
            num_RespawnObjects=5,

            # primitive
            push_delta_scale_x=0.1,
            push_delta_scale_y=0.1,
            steps_check=50,
            gripper_safe_height=0.204,
            offstage_pose=(-0.3, -0.1, 0.35416687,
                           -0.326071, 0.944798, 0.0238332, -0.021595),
            offstage_joint_pos=(4.66983682, 3.09011904, 1.22106735, 1.34376026, 0.09548322, 4.45514805, 1.82119646),
            max_phase_stpes=200,
            max_offstage_steps=900,
            max_motion_steps=200,
            num_goal_stpes=None,
            maxActionSteps=10,

            init_skip_timestep=100,

            # goal setting
            isRandomGoals=True,
            isIgnoreGoalCollision=False,
            fixed_objects_goals=(-0.15, -0.3, 0.06,
                                 0.15, -0.5, 0.06,),  # shape (3*n,)
            fixed_hand_goal=(0, -0.25, 0.154),
            target_upper_space=(0.25, -0.25, 0.06),
            target_lower_space=(-0.25, -0.55, 0.06),

            # robot params
            random_robot_ee_pos=False,
            hand_init_upper_space=(0 + 0.25, -0.40 + 0.15, 0.154),
            hand_init_lower_space=(0 - 0.25, -0.40 - 0.15, 0.154),

            image=False,
            render_params={
                "target_pos": (0, -0.742, 0.642),
                "distance": 0.15,
                "yaw": 0,
                "pitch": -37,
                "roll": 0,
            },

            # base arguments
            kinova_type='j2s7s300_beam',
            control_mode='push_primitive_consXY',

            ## 10s traj horizon
            timeStep=0.01,
            actionRepeat=100,  # very importrant!!!
            maxSteps=24000,  # per max 1200
            action_scale_conts=0.1,  # m
            isRender=True,
            debug=False,
            multi_view=False,

            skip_first=0,

            robotHomeAngle=(
            4.66983682, 3.09011904, 1.22106735, 1.34376026, 0.09548322, 4.45514805, 1.82119646, 1.263, 1.347, 1.347),
            robot_ee_pos=(-0.3, -0.1, 0.35416687),
            fixed_orn=(-0.32618785, 0.94479543, 0.02328156, -0.02051954),

            fixed_finger=1.263,

            stepTextPosition=[-0.65, 0, 0.3],

            # kinova arguments
            isLimitedWorkSpace=True,
            hand_high=(0 + 0.3, -0.40 + 0.2, 0.4),
            hand_low=(0 - 0.3, -0.40 - 0.2, -0.4),
            useDynamics=True,

            # other
            debug_info=False,

        )
    )

    register(
        id='Jaco2PushPrimitiveXYyawEnv-v0',
        entry_point='multiworld.envs.pybullet.jaco_push_primitive.jaco_push_primitive:Jaco2PushPrimitiveXYyaw',
        max_episode_steps=200,
        reward_threshold=20000.0,
        kwargs=dict(
            # env arguments
            addTable=True,
            table_position=[0.0000000, -0.650000, -0.59000],
            table_name='white',

            isImageObservation=False,

            # obj setting
            isRandomObjects=True,
            fixed_objects_init_pos=(-0.12, -0.3, 0.06,  # x, y, z, shape (3*n,)
                                    -0.05, -0.3, 0.06,
                                     0.05, -0.3, 0.06,
                                     0.12, -0.3, 0.06,),
            obj_name_list=['b_cube_m', 'b_cube_w' , 'b_L1', 'b_L2',
                           'b_cube_m', 'b_cube_w', 'b_L1', 'b_L2'],
            num_movable_bodies=2,
            obj_pos_upper_space=(0 + 0.2, -0.25, 0.06),
            obj_pos_lower_space=(0 - 0.2, -0.55, 0.06),
            obj_max_upper_space=(0 + 0.28, -0.20, 0.4),
            obj_max_lower_space=(0 - 0.28, -0.60, -0.4),

            obj_euler_upper_space=(0, 0, math.pi),
            obj_euler_lower_space=(0, 0, -math.pi),
            obj_safe_margin=0.05,
            obj_scale_range=(1, 1),
            obj_mass=None,
            obj_friction=0.8,
            use_random_rgba=True,
            num_RespawnObjects=5,

            # primitive
            push_delta_scale_x=0.1,
            push_delta_scale_y=0.1,
            steps_check=50,
            gripper_safe_height=0.204,
            offstage_pose=(-0.3, -0.1, 0.35416687,
                           -0.326071, 0.944798, 0.0238332, -0.021595),
            offstage_joint_pos=(4.66983682, 3.09011904, 1.22106735, 1.34376026, 0.09548322, 4.45514805, 1.82119646),
            max_phase_stpes=200,
            max_offstage_steps=900,
            max_motion_steps=200,
            num_goal_stpes=None,
            maxActionSteps=10,

            init_skip_timestep=100,

            # goal setting
            isRandomGoals=True,
            isIgnoreGoalCollision=False,
            fixed_objects_goals=(-0.15, -0.3, 0.0,
                                 0.15, -0.5, 0.0,),  # shape x,y,z,yaw
            fixed_hand_goal=(0, -0.25, 0.154),
            target_upper_space=(0.25, -0.25, 0.06, 0,0, math.pi),   ## size 6 :x,y,z,roll,pitch,yaw
            target_lower_space=(-0.25, -0.55, 0.06,  0,0, -math.pi),

            # robot params
            random_robot_ee_pos=False,
            hand_init_upper_space=(0 + 0.25, -0.40 + 0.15, 0.154),
            hand_init_lower_space=(0 - 0.25, -0.40 - 0.15, 0.154),

            image=False,
            render_params={
                "target_pos": (0, -0.742, 0.642),
                "distance": 0.15,
                "yaw": 0,
                "pitch": -37,
                "roll": 0,
            },

            # base arguments
            kinova_type='j2s7s300_beam',
            control_mode='push_primitive_consXY',

            ## 10s traj horizon
            timeStep=0.01,
            actionRepeat=100,  # very importrant!!!
            maxSteps=24000,  # per max 1200
            action_scale_conts=0.1,  # m
            isRender=True,
            debug=False,
            multi_view=False,

            skip_first=0,

            robotHomeAngle=(
                4.66983682, 3.09011904, 1.22106735, 1.34376026, 0.09548322, 4.45514805, 1.82119646, 1.263, 1.347,
                1.347),
            robot_ee_pos=(-0.3, -0.1, 0.35416687),
            fixed_orn=(-0.32618785, 0.94479543, 0.02328156, -0.02051954),

            fixed_finger=1.263,

            stepTextPosition=[-0.65, 0, 0.3],

            # kinova arguments
            isLimitedWorkSpace=True,
            hand_high=(0 + 0.3, -0.40 + 0.2, 0.4),
            hand_low=(0 - 0.3, -0.40 - 0.2, -0.4),
            useDynamics=True,

            # other
            debug_info=False,
            isRenderGoal=False,

        )
    )

    register(
        id='Jaco2PushPrimitiveOneXYEnv-v0',
        entry_point='multiworld.envs.pybullet.jaco_push_primitive.jaco_push_primitive:Jaco2PushPrimitiveXY',
        max_episode_steps=200,
        reward_threshold=20000.0,
        kwargs=dict(
            # env arguments
            addTable=True,
            table_position=[0.0000000, -0.650000, -0.59000],
            table_name='white',

            isImageObservation=False,

            # obj setting
            isRandomObjects=True,
            fixed_objects_init_pos=(0, -0.3, 0.06,  # shape (3*n,)
                                    0, -0.5, 0.06,),
            obj_name_list=['b_cube_m', 'b_cube_w', 'b_L1', 'b_L2',
                           'b_cube_m', 'b_cube_w', 'b_L1', 'b_L2'],
            num_movable_bodies=1,
            obj_pos_upper_space=(0 + 0.2, -0.25, 0.06),
            obj_pos_lower_space=(0 - 0.2, -0.55, 0.06),
            obj_max_upper_space=(0 + 0.28, -0.20, 0.4),
            obj_max_lower_space=(0 - 0.28, -0.60, -0.4),

            obj_euler_upper_space=(0, 0, 0),
            obj_euler_lower_space=(0, 0, 0),
            obj_safe_margin=0.05,
            obj_scale_range=(1, 1),
            obj_mass=None,
            obj_friction=0.8,
            use_random_rgba=True,
            num_RespawnObjects=5,

            # primitive
            push_delta_scale_x=0.1,
            push_delta_scale_y=0.1,
            steps_check=50,
            gripper_safe_height=0.204,
            offstage_pose=(-0.3, -0.1, 0.35416687,
                           -0.326071, 0.944798, 0.0238332, -0.021595),
            offstage_joint_pos=(4.66983682, 3.09011904, 1.22106735, 1.34376026, 0.09548322, 4.45514805, 1.82119646),
            max_phase_stpes=200,
            max_offstage_steps=900,
            max_motion_steps=200,
            num_goal_stpes=None,
            maxActionSteps=10,

            init_skip_timestep=100,

            # goal setting
            isRandomGoals=True,
            isIgnoreGoalCollision=False,
            fixed_objects_goals=(-0.15, -0.3, 0.06,
                                 0.15, -0.5, 0.06,),  # shape (3*n,)
            fixed_hand_goal=(0, -0.25, 0.154),
            target_upper_space=(0.25, -0.25, 0.06),
            target_lower_space=(-0.25, -0.55, 0.06),

            # robot params
            random_robot_ee_pos=False,
            hand_init_upper_space=(0 + 0.25, -0.40 + 0.15, 0.154),
            hand_init_lower_space=(0 - 0.25, -0.40 - 0.15, 0.154),

            image=False,
            render_params={
                "target_pos": (0, -0.742, 0.642),
                "distance": 0.15,
                "yaw": 0,
                "pitch": -37,
                "roll": 0,
            },

            # base arguments
            kinova_type='j2s7s300_beam',
            control_mode='push_primitive_consXY',

            ## 10s traj horizon
            timeStep=0.01,
            actionRepeat=100,  # very importrant!!!
            maxSteps=24000,  # per max 1200
            action_scale_conts=0.1,  # m
            isRender=False,
            debug=False,
            multi_view=False,

            skip_first=0,

            robotHomeAngle=(
                4.66983682, 3.09011904, 1.22106735, 1.34376026, 0.09548322, 4.45514805, 1.82119646, 1.263, 1.347,
                1.347),
            robot_ee_pos=(-0.3, -0.1, 0.35416687),
            fixed_orn=(-0.32618785, 0.94479543, 0.02328156, -0.02051954),

            fixed_finger=1.263,

            stepTextPosition=[-0.65, 0, 0.3],

            # kinova arguments
            isLimitedWorkSpace=True,
            hand_high=(0 + 0.3, -0.40 + 0.2, 0.4),
            hand_low=(0 - 0.3, -0.40 - 0.2, -0.4),
            useDynamics=True,

            # other
            debug_info=False,

        )
    )


def create_image_pybullet_jaco_push_primitive_xy_env_v1():
    from multiworld.core.image_env import ImageEnv


    wrapped_env = gym.make('Jaco2ObjectsPusherXYEnv-v0', isRender=True)

    camera_params = {"target_pos": (0.0, -0.461, -0.004),
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
    #return wrapped_env
    return ImageEnv(
        wrapped_env,
        64,
        init_camera= camera_params,
        transpose=True,
        normalize=True,
    )

def create_image_pybullet_jaco_push_primitive_xyyaw_env_v1():
    from multiworld.core.image_env import ImageEnv


    wrapped_env = gym.make('Jaco2PushPrimitiveXYyawEnv-v0', isRender=True)

    camera_params = {"target_pos": (0.0, -0.461, -0.004),
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
    #return wrapped_env
    return ImageEnv(
        wrapped_env,
        64,
        init_camera= camera_params,
        transpose=True,
        normalize=True,
    )



