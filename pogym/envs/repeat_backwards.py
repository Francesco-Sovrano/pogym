import enum
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple, Union

import gym
import numpy as np


class Mode(enum.IntEnum):
    PLAY = 0
    WATCH = 1


class RepeatBackwards(gym.Env):
    """A game where the agent must press buttons in the reverse order it saw
    them pressed. E.g., seeing [1, 2, 3] means I should press them in the order
    [3, 2, 1].

    Args:
        game_len: The maximum number of button presses the agent
            must memorize
        num_buttons: The number of unique buttons for the agent to press

    Returns:
        A gym environment
    """

    def __init__(self, game_len=32, num_buttons=4):
        self.num_buttons = num_buttons
        self.game_len = game_len
        self.action_space = gym.spaces.Discrete(num_buttons)
        self.observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(2),  # Whether listening or playing
                gym.spaces.Discrete(num_buttons),
            )
        )
        self.mode = Mode.WATCH

    def cycle(self):
        button = self.sys_seq.pop()
        self.watched_seq.append(button)
        return button

    def uncycle(self, action):
        button = self.watched_seq.popleft()
        done = len(self.watched_seq) == 0
        return action == button, done

    def make_obs(self, button):
        return np.array([self.mode.value, button])

    def step(self, action):
        done = False
        reward = 0
        if self.mode == Mode.WATCH:
            obs = self.make_obs(self.cycle())
            if len(self.sys_seq) == 0:
                self.mode = Mode.PLAY
        else:
            correct_button, done = self.uncycle(action)
            obs = self.make_obs(0)
            if correct_button:
                reward = 1.0 / self.game_len

        info = {}

        return obs, reward, done, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        if seed is not None:
            np.random.seed(seed)
        self.sys_seq = np.random.randint(
            0, self.num_buttons, size=self.game_len
        ).tolist()

        self.watched_seq: Deque[int] = deque([])
        self.player_seq: Deque[int] = deque([])
        self.num_replayed = 1
        self.mode = Mode.WATCH
        obs = self.make_obs(self.cycle())
        if return_info:
            return obs, {}

        return obs
