from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np

from pogym.core.deck import Deck


class RepeatFirst(gym.Env):
    """A game where the agent must repeat the rank of the first card it saw

    Args:
        num_decks: The number of decks to cycle through, which determines
            episode length

    Returns:
        A gym environment
    """

    def __init__(self, num_decks=1):
        self.deck = Deck(num_decks)
        self.deck.add_players("player")
        self.action_space = self.deck.get_obs_space(["ranks"])
        self.observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(2),
                self.action_space,
            )
        )

    def make_obs(self, card, is_start=False):
        return np.array([int(is_start), card])

    def step(self, action):
        done = False
        reward_scale = 1 / self.deck.num_cards
        if action == self.card:
            reward = reward_scale
        else:
            reward = 0
            done = True

        if len(self.deck) == 1:
            done = True

        self.deck.deal("player", 1)
        card = self.deck.show("player", ["ranks_idx"])[0, -1]
        obs = self.make_obs(card)
        self.deck.discard_hands("player")

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
        self.deck.reset()
        self.deck.deal("player", 1)
        self.card = self.deck.show("player", ["ranks_idx"])[0, -1]
        obs = self.make_obs(self.card, is_start=True)
        if return_info:
            return obs, {}

        return obs
