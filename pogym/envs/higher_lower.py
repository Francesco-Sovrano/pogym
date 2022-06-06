from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np

from pogym.core.deck import CardRepr, Deck


class HigherLower(gym.Env):
    """A game of higher/lower. Given a deck of cards, the agent predicts whether the
    next card drawn from the deck is higher or lower than the last card drawn from
    the deck. A push results in zero reward, while a correct/incorrect guess result
    in 1/deck_size and -1/deck_size reward. The agent can learn to count cards to
    infer which cards are left in the deck, improving accuracy.

    Args:
        num_decks: The number of individual decks combined into a single deck.

    Returns:
        A gym environment
    """

    def __init__(self, num_decks=1):
        self.num_decks = num_decks
        self.deck = Deck(num_decks, card_repr=CardRepr.RANKS)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = self.deck.card_obs_space
        self.value_map = dict(zip(self.deck.ranks, range(len(self.deck.ranks))))
        self.deck_size = len(self.deck)

    def step(self, action):
        guess_higher = action == 0
        if len(self.deck) <= 1:
            done = True
        else:
            done = False

        next_card = self.deck.draw()
        next_value = self.value_map[self.deck.id_to_str(next_card)[1]]
        curr_value = self.value_map[self.deck.id_to_str(self.curr_card)[1]]

        rew_scale = 1 / self.deck_size
        if next_value == curr_value:
            reward = 0
        elif next_value > curr_value and guess_higher:
            reward = rew_scale
        elif next_value < curr_value and guess_higher:
            reward = -rew_scale
        elif next_value < curr_value and not guess_higher:
            reward = rew_scale
        elif next_value > curr_value and not guess_higher:
            reward = -rew_scale
        else:
            raise Exception("Should not reach this point")

        info = {
            "previous": self.deck.id_to_viz(self.curr_card),
            "current": self.deck.id_to_viz(next_card),
        }

        self.curr_card = next_card
        obs = self.deck.id_to_obs(self.curr_card).item()

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
        self.curr_card = self.deck.draw()
        obs = self.deck.id_to_obs(self.curr_card).item()
        if return_info:
            return obs, {}

        return obs
