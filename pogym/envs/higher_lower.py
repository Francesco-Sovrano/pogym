from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np

from pogym.core.deck import Deck


def value_fn(hand):
    if hand[-1] > hand[-2]:
        return 1
    elif hand[-1] == hand[-2]:
        return 0
    else:
        return -1


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
        self.deck = Deck(num_decks)
        self.deck.add_players("player")
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = self.deck.get_obs_space(["ranks"])
        self.value_map = dict(zip(self.deck.ranks, range(len(self.deck.ranks))))
        self.deck_size = len(self.deck)

    def step(self, action):
        guess_higher = action == 0
        if len(self.deck) <= 1:
            done = True
        else:
            done = False

        self.deck.deal("player", 1)
        assert self.deck.hand_size("player") == 2
        curr_idx, next_idx = self.deck["player"]
        curr_value, next_value = self.deck["ranks_idx"][[curr_idx, next_idx]]
        # next_card = self.deck.draw()
        # next_value = self.value_map[self.deck.id_to_str(next_card)[1]]
        # curr_value = self.value_map[self.deck.id_to_str(self.curr_card)[1]]

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

        # info = {
        # "previous": self.deck.id_to_viz(self.curr_card),
        # "current": self.deck.id_to_viz(next_card),
        # }

        viz = np.stack(self.deck.show("player", ["suits", "ranks"])).T
        self.deck.discard("player", 0)
        # self.curr_card = next_card
        # obs = self.deck.id_to_obs(self.curr_card).item()
        obs = self.deck.show("player", ["ranks_idx"]).item()

        return obs, reward, done, {"card": viz}

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
        obs = self.deck.show("player", ["ranks_idx"]).item()
        viz = np.concatenate(self.deck.show("player", ["suits", "ranks"]))
        # self.curr_card = self.deck.draw()
        # obs = self.deck.id_to_obs(self.curr_card).item()
        if return_info:
            return obs, {"card": viz}

        return obs
