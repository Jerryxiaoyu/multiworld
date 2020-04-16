
import numpy as np
import pybullet as p
import gym

class BasePybulletEnv(gym.Env):
    def __init__(self ):


        self._base_adim, self._base_sdim = None, None  # state/action dimension of Mujoco control
        self.action_dim, self.state_dim = None, None  # state/action dimension presented to agent
        self._ncam = 1


    def reset(self):
        return NotImplementedError


    @property
    def adim(self):
        return self.action_dim

    @property
    def sdim(self):
        return 1

    @property
    def ncam(self):
        return self._ncam



