# Inspired by ray rllib at
# https://github.com/ray-project/ray/blob/master/rllib/examples/env/stateless_pendulum.py

from typing import Optional, Tuple, Union

import gym
import numpy as np
from gym.envs.classic_control import PendulumEnv
from gym.spaces import Box


class StatelessPendulum(PendulumEnv):
    """Partially observable variant of the Pendulum gym environment.
    https://github.com/openai/gym/blob/master/gym/envs/classic_control/
    pendulum.py
    We delete the angular velocity component of the state, so that it
    can only be solved by a memory enhanced model (policy).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Fix our observation-space (remove angular velocity component).
        high = np.array([1.0, 1.0], dtype=np.float32)
        self.observation_space = Box(low=-high, high=high, dtype=np.float32)

    def step(
        self, action: gym.core.ActType
    ) -> Tuple[gym.core.ObsType, float, bool, dict]:
        next_obs, reward, done, info = super().step(action)
        # next_obs is [cos(theta), sin(theta), theta-dot (angular velocity)]
        return next_obs[:-1], reward, done, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ) -> Union[gym.core.ObsType, Tuple[gym.core.ObsType, dict]]:
        if return_info:
            init_obs, info = super().reset()
            return init_obs[:-1], info

        init_obs = super().reset()
        # init_obs is [cos(theta), sin(theta), theta-dot (angular velocity)]
        return init_obs[:-1]
