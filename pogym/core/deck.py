import copy
import itertools
from enum import Enum
from typing import List

import gym
import numpy as np


def ascii_version_of_card(ranks, suits, return_string=True):
    """Instead of a boring text version of the card we render an ASCII image of
    the card.

    :param cards: One or more card objects
    :param return_string: By default we return the string version
        of the card, but the dealer hide the 1st card and we
    keep it as a list so that the dealer can add a hidden card in front of the list
    """
    # we will use this to prints the appropriate icons for each card
    suits_name = ["s", "d", "h", "c"]
    suits_symbols = ["♠", "♦", "♥", "♣"]

    # create an empty list of list, each sublist is a line
    lines = [[] for i in range(9)]

    for s, r in zip(suits, ranks):
        # "King" should be "K" and "10" should still be "10"
        if r == "10":  # ten is the only one who's rank is 2 char long
            rank = r
            space = ""  # if we write "10" on the card that line will be 1 char to long
        else:
            rank = r
            space = " "  # no "10", we use a blank space to will the void
        # get the cards suit in two steps
        suit = suits_name.index(s)
        suit = suits_symbols[suit]

        # add the individual card on a line by line basis
        lines[0].append("┌─────────┐")
        lines[1].append(
            "│{}{}       │".format(rank, space)
        )  # use two {} one for char, one for space or char
        lines[2].append("│         │")
        lines[3].append("│         │")
        lines[4].append("│    {}    │".format(suit))
        lines[5].append("│         │")
        lines[6].append("│         │")
        lines[7].append("│       {}{}│".format(space, rank))
        lines[8].append("└─────────┘")

    result = []
    for index, line in enumerate(lines):
        result.append("".join(lines[index]))

    # hidden cards do not use string
    if return_string:
        return "\n".join(result)
    else:
        return result


class CardRepr(Enum):
    FULL = 0
    SUITS_AND_RANKS = 1
    RANKS = 2


class DeckEmptyError(Exception):
    pass


class Deck:
    """An object that represents a collection of cards.

    A deck can represent a single deck or multiple decks
    """

    # Spades, hearts, clubs, and diamonds
    colors = ["r", "b"]
    suits = ["s", "d", "h", "c"]
    # ace, 2, 3, .... jack, queen, king
    ranks = ["a", "2", "3", "4", "5", "6", "7", "8", "9", "10", "j", "q", "k"]
    num_unique_cards = len(suits) * len(ranks)
    card_obs_space = gym.spaces.Tuple(
        (
            gym.spaces.Discrete(len(colors)),  # color
            gym.spaces.Discrete(len(suits)),  # suit
            gym.spaces.Discrete(len(ranks)),  # rank
        )
    )
    rank_card_obs_space = gym.spaces.Discrete(len(ranks))

    def get_obs_space(self):
        if self.card_repr == CardRepr.FULL:
            self.card_obs_space = gym.spaces.Tuple(
                (
                    gym.spaces.Discrete(len(self.colors)),  # color
                    gym.spaces.Discrete(len(self.suits)),  # suit
                    gym.spaces.Discrete(len(self.ranks)),  # rank
                )
            )
        elif self.card_repr == CardRepr.SUITS_AND_RANKS:
            self.card_obs_space = gym.spaces.Tuple(
                (
                    gym.spaces.Discrete(len(self.suits)),  # suit
                    gym.spaces.Discrete(len(self.ranks)),  # rank
                )
            )
        elif self.card_repr == CardRepr.RANKS:
            self.card_obs_space = gym.spaces.Discrete(len(self.ranks))

    def __init__(self, num_decks=1, card_repr: CardRepr = CardRepr.FULL):
        self.num_decks = num_decks
        single_deck = list(itertools.product(self.suits, self.ranks))
        self.card_values = list(range((len(single_deck))))
        self.card_ids = list(range(len(single_deck) * num_decks))
        self.card_repr = card_repr

        # Deck is represented as a stack
        self.deck = copy.deepcopy(self.card_ids)
        self.in_play: List[int] = []
        self.discard: List[int] = []

        self._str_to_id = dict(zip(itertools.cycle(single_deck), self.card_ids))
        self._id_to_str = dict(zip(self.card_ids, itertools.cycle(single_deck)))

        self._str_to_val = dict(zip(single_deck, self.card_values))
        self._val_to_str = dict(zip(self.card_values, itertools.cycle(single_deck)))

        self._id_to_val = dict(zip(self.card_ids, itertools.cycle(self.card_values)))

    def __len__(self):
        return len(self.deck)

    def draw(self):
        """Draws a single card from the deck, returning the card id.

        Will error if the deck is empty, make sure to check
        len(Deck.deck) before calling.
        """
        try:
            card_id = self.deck.pop()
            self.in_play.append(card_id)
            return card_id
        except IndexError:
            raise DeckEmptyError()

    def discard_hands(self):
        """Discards all cards that have been drawn by placing them into the
        discard pile."""
        self.discard += self.in_play
        self.in_play.clear()

    def reset(self, shuffle=True):
        """Discards all current cards in play, then adds the discarded pile to
        the back of the deck.

        Set shuffle=True to ensure the deck is shuffled after combining.
        """
        self.discard_hands()
        # self.deck += self.discard
        self.deck = copy.deepcopy(self.card_ids)
        if shuffle:
            np.random.shuffle(self.deck)
        assert len(self.deck) == 52 * self.num_decks

    def str_to_id(self, string):
        return self._str_to_id[string]

    def strs_to_ids(self, strings):
        return [self._str_to_id[s] for s in strings]

    def id_to_str(self, card_id):
        return self._id_to_str[card_id]

    def str_to_val(self, string):
        return self._str_to_val[string]

    def val_to_str(self, card_val):
        return self._val_to_str[card_val]

    def id_to_val(self, card_id):
        return self._id_to_val[card_id]

    def id_to_obs(self, card_id):
        string = self.id_to_str(card_id)
        suit, rank = string
        suit_idx = self.suits.index(suit)
        rank_idx = self.ranks.index(rank)
        color_idx = 0 if suit in ["h", "d"] else 1
        if self.card_repr == CardRepr.FULL:
            return np.array((color_idx, suit_idx, rank_idx))
        elif self.card_repr == CardRepr.SUITS_AND_RANKS:
            return np.array((suit_idx, rank_idx))
        elif self.card_repr == CardRepr.RANKS:
            return np.array((rank_idx))

    def id_to_viz(self, card_id):
        string = self.id_to_str(card_id)
        suit, rank = string
        color = "r" if suit in ["h", "d"] else "b"
        if self.card_repr == CardRepr.FULL:
            return color, suit, rank
        elif self.card_repr == CardRepr.SUITS_AND_RANKS:
            return suit, rank
        elif self.card_repr == CardRepr.RANKS:
            return rank

    def ids_to_renders(self, card_ids):
        strs = [self.id_to_str(c) for c in card_ids]
        if len(strs) == 0:
            return "\n".join([""] * 10)
        suits, ranks = zip(*strs)
        return ascii_version_of_card(ranks, suits)

    def obs_to_viz(self, obs):
        color_idx, suit_idx, rank_idx = obs
        return self.colors[color_idx], self.suits[suit_idx], self.ranks[rank_idx]
