from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np


class RepeatFirst(gym.Env):
    """A game where the agent must repeat the first observation

    Args:
        game_len: The maximum length of an episode in timesteps
        num_buttons: The number of possible observations

    Returns:
        A gym environment
    """

    def __init__(self, num_buttons=4, game_len=64):
        self.num_buttons = num_buttons
        self.game_len = game_len
        self.action_space = gym.spaces.Discrete(num_buttons)
        self.observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(2),
                gym.spaces.Discrete(num_buttons),
            )
        )

    def cycle(self):
        button = self.sys_seq.pop()
        self.watched_seq.append(button)
        return button

    def uncycle(self, action):
        button = self.watched_seq.popleft()
        done = len(self.watched_seq) == 0
        return action == button, done

    def make_obs(self, button, is_start=False):
        return np.array([int(is_start), button])

    def step(self, action):
        done = False
        reward_scale = 1 / self.game_len
        if action == self.button:
            reward = reward_scale
        else:
            reward = 0
            done = True

        obs = self.make_obs(np.random.randint(self.num_buttons))

        info = {}

        return obs, reward, done, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[gym.core.ObsType, Tuple[gym.core.ObsType, Dict[str, Any]]]:
        if seed is not None:
            np.random.seed(seed)
        self.button = np.random.randint(self.num_buttons)
        obs = self.make_obs(self.button, is_start=True)
        if return_info:
            return obs, {}

        return obs
