from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np


class MultiarmedBandit(gym.Env):
    def __init__(self, num_bandits=10, std=1.0, episode_length=100):
        self.num_bandits = num_bandits
        self.std = std
        self.episode_length = episode_length
        self.observation_space = gym.spaces.Box(shape=(1,), low=-1e5, high=1e5)
        self.action_space = gym.spaces.Discrete(num_bandits)

    def step(self, action, increment=True):
        obs = np.random.normal(self.bandits[action], self.std)
        reward = np.array([action, obs])
        done = self.num_steps < self.episode_length
        if increment:
            self.num_steps += 1

        return obs, reward, done, self.info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[gym.core.ObsType, Tuple[gym.core.ObsType, Dict[str, Any]]]:

        if seed is not None:
            np.random.seed(seed)

        self.num_steps = 0
        self.bandits = np.random.rand(self.num_bandits)
        self.info = {"bandits": self.bandits}
        rand_action = np.random.randint(10)
        obs, _, _, info = self.step(rand_action, increment=False)

        if return_info:
            return obs, info
        return obs
