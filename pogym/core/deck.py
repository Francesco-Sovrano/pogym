import itertools
import random
import copy
import gym


class Deck:
    """An object that represents a collection of cards.
    A deck can represent a single deck or multiple decks"""
    # Spades, hearts, clubs, and diamonds
    colors = ["r", "b"]
    suits = ["h", "d", "c", "s"]
    # ace, 2, 3, .... jack, queen, king
    ranks = ["a", "2", "3", "4", "5", "6", "7", "8", "9", "10", "j", "q", "k"]
    num_unique_cards = len(suits) * len(ranks)
    card_obs_space = gym.spaces.Dict({
        "is_null": gym.spaces.Discrete(2), # Whether or not there is a card present
        "color": gym.spaces.Discrete(len(colors)),
        "suit": gym.spaces.Discrete(len(suits)),
        "rank": gym.spaces.Discrete(len(ranks))
    })


    def __init__(self, num_decks=1, shuffled=True):
        self.num_decks = num_decks
        single_deck = list(itertools.product(self.suits, self.ranks))
        self.card_values = list(range((len(single_deck))))
        self.card_ids = list(range(len(single_deck) * num_decks))

        # Deck is represented as a stack
        self.deck = copy.deepcopy(self.card_ids)
        self.in_play = []
        self.discard = []
        if shuffled:
            random.shuffle(self.deck)
        
        self._str_to_id = dict(zip(itertools.cycle(single_deck), self.card_ids))
        self._id_to_str = dict(zip(self.card_ids, itertools.cycle(single_deck)))

        self._str_to_val = dict(zip(single_deck, self.card_values))
        self._val_to_str = dict(zip(self.card_values, itertools.cycle(single_deck)))

        self._id_to_val = dict(zip(self.card_ids, itertools.cycle(self.card_values)))

    def __len__(self):
        return len(self.deck)

    def draw(self):
        """Draws a single card from the deck, returning
        the card id. Will error if the deck is empty, make sure
        to check len(Deck.deck) before calling.""" 
        card_id = self.deck.pop()
        self.in_play.append(card_id)
        return card_id

    def discard_hands(self):
        """Discards all cards that have been drawn by placing them
        into the discard pile"""
        self.discard += self.in_play
        self.in_play.clear()

    def reset(self, shuffle=True):
        """Discards all current cards in play, then adds the discarded pile
        to the back of the deck. Set shuffle=True to ensure the deck is shuffled
        after combining."""
        self.discard_hands()
        self.deck += self.discard
        if shuffle:
            random.shuffle(self.deck)
        assert len(self.deck) == 52 * self.num_decks

    def str_to_id(self, string):
        return self._str_to_id[string]

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
        return color_idx, suit_idx, rank_idx

    def id_to_viz(self, card_id):
        string = self.id_to_str(card_id)
        suit, rank = string
        return color, suit, int(rank)

    def obs_to_viz(self, obs):
        color_idx, suit_idx, rank_idx = obs
        return self.colors[color_idx], self.suits[suit_idx], self.ranks[rank_idx]

