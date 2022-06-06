import copy
import enum
from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np

from pogym.core.deck import CardRepr, Deck, DeckEmptyError


class Phase(enum.IntEnum):
    # The betting phase where the player selects a bet
    BET = 0
    # The cards are dealt to the player and house, player does not act
    DEAL = 1
    # Player chooses to hit or stay
    PLAY = 2
    # Player receives the final cards (for counting) and the reward
    PAYOUT = 3


class BlackJack(gym.Env):
    """A game of blackjack, where card counting is possible. Successful agents
    should learn to count cards, and bet higher/hit less often when the deck
    contains higher cards. Note that splitting is not allowed.

    Args:
        bet_sizes: The bet sizes available to the agent. These correspond to
            the final reward.
        num_decks: The number of individual decks combined into a single deck.
            In Vegas, this is usually between four and eight.
        max_rounds: The maximum number of rounds where the agent and dealer
            can hit/stay. There is no max in real blackjack, however this
            would result in a very large observation space.
        games_per_episode: The number of games per episode. This must be set high
            for card-counting to have an effect. When set to one, the game
            becomes fully observable.

    Returns:
        A gym environment
    """

    card_map = {
        "a": 1,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "10": 10,
        "j": 10,
        "q": 10,
        "k": 10,
    }

    def __init__(
        self,
        bet_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        num_decks=1,
        max_rounds=6,
        games_per_episode=20,
    ):
        self.deck = Deck(num_decks=num_decks, card_repr=CardRepr.RANKS)
        self.bet_sizes = bet_sizes
        self.max_rounds = max_rounds
        # Hit, stay, and bet amount
        self.action_space = gym.spaces.Dict(
            {
                "hit": gym.spaces.Discrete(2),
                "bet_size": gym.spaces.Discrete(len(bet_sizes)),
            }
        )
        self.games_per_episode = games_per_episode
        self.curr_game = 0
        self.curr_round = 0
        self.observation_space = gym.spaces.Dict(
            {
                "phase": gym.spaces.Discrete(3),
                "dealer_hand": gym.spaces.Tuple(
                    max_rounds * [self.deck.card_obs_space]
                ),
                "dealer_hand_cards_in_play": gym.spaces.MultiBinary(max_rounds),
                "player_hand": gym.spaces.Tuple(
                    max_rounds * [self.deck.card_obs_space]
                ),
                "player_hand_cards_in_play": gym.spaces.MultiBinary(max_rounds),
            }
        )
        self.dealer_hand = []
        self.player_hand = []
        self.action_phase = Phase.BET

    def hand_value(self, hand):
        value = 0
        has_ace = False
        for card in hand:
            suit, rank = self.deck.id_to_str(card)
            if rank == "a":
                has_ace = True
            value += self.card_map[rank]

        if value < 11 and has_ace:
            # ace = 1 + 10
            value += 10
        return value

    def bet(self, action):
        # Take the previous action (bet)
        self.curr_bet = self.bet_sizes[action["bet_size"]]
        result = "player and dealer draw cards"
        return result

    def deal(self, action):
        # First round, serve the cards
        self.curr_bet = self.bet_sizes[action["bet_size"]]
        self.player_hand += [self.deck.draw(), self.deck.draw()]
        self.dealer_hand += [self.deck.draw()]

    def play(self, action):
        player_hit = action["hit"]
        if player_hit:
            # Hit
            self.player_hand += [self.deck.draw()]
            result = "player hits"

        player_value = self.hand_value(self.player_hand)
        player_bust = player_value > 21
        player_blackjack = player_value == 21
        player_natural = player_blackjack and len(self.player_hand) == 2
        player_max_cards = len(self.player_hand) == self.max_rounds

        game_done = (
            not player_hit or player_bust or player_blackjack or player_max_cards
        )
        if not game_done:
            return 0, game_done, result

        dealer_plays = (not player_bust and not player_hit) or player_blackjack

        if dealer_plays:
            dealer_value = self.hand_value(self.dealer_hand)
            while dealer_value < 17 and len(self.dealer_hand) < self.max_rounds:
                self.dealer_hand += [self.deck.draw()]
                dealer_value = self.hand_value(self.dealer_hand)
        else:
            dealer_value = self.hand_value(self.dealer_hand)

        dealer_bust = dealer_value > 21
        dealer_blackjack = dealer_value == 21
        dealer_natural = dealer_blackjack and len(self.player_hand) == 2
        player_adv = player_value - dealer_value

        # compare
        if player_adv == 0:
            reward = 0
            result = f"player ({player_value}) and dealer ({dealer_value}) push"
        elif player_adv > 0:
            reward = self.curr_bet
            result = f"player ({player_value}) beats dealer ({dealer_value})"
        elif player_adv < 0:
            reward = -self.curr_bet
            result = f"player ({player_value}) loses to dealer ({dealer_value})"

        # busts
        if player_bust and not dealer_bust:
            reward = -self.curr_bet
            result = f"player ({player_value}) bust"
        elif dealer_bust and not player_bust:
            reward = self.curr_bet
            result = f"dealer ({dealer_value}) bust"
        elif dealer_bust and player_bust:
            reward = 0
            result = f"player ({player_value}) and dealer ({dealer_value}) bust"

        # naturals
        if player_natural and not dealer_natural:
            reward = 1.5 * self.curr_bet
            result = "player natural"
        elif dealer_natural and not player_natural:
            reward = -self.curr_bet
            result = "dealer natural"
        elif dealer_natural and player_natural:
            result = "push: player and dealer naturals"
            reward = 0

        return reward, game_done, result

    def game_reset(self):
        """Resets a game, but not the entire env."""
        self.curr_game += 1
        self.curr_round = 0
        self.deck.discard_hands()
        self.player_hand.clear()
        self.dealer_hand.clear()

    def step(self, action):
        reward = 0
        done = False
        result = ""

        if len(self.deck) < 3:
            done = True
            result = "deck empty, episode over"

        try:
            game_done = False
            if self.action_phase == Phase.BET:
                result = self.bet(action)
                self.obs, self.info = self.build_obs_infos()
                self.action_phase = Phase.DEAL
            elif self.action_phase == Phase.DEAL or self.action_phase == Phase.PAYOUT:
                self.deal(action)
                self.obs, self.info = self.build_obs_infos()
                self.action_phase = Phase.PLAY
            elif self.action_phase == Phase.PLAY:
                reward, game_done, result = self.play(action)
                if game_done:
                    self.action_phase = Phase.PAYOUT
                if reward != 0:
                    assert self.action_phase == Phase.PAYOUT
                self.obs, self.info = self.build_obs_infos()
        except DeckEmptyError:
            done = True
            self.info["result"] += ", deck empty, episode over"

        self.curr_round += 1

        if game_done:
            self.game_reset()
            self.action_phase = Phase.BET

        if self.curr_game == self.games_per_episode - 1:
            done = True

        return self.obs, reward, done, self.info

    def render(self):
        phase = Phase(self.obs["phase"]).name
        print(f"Phase: {phase}")
        print(f"Current Bet: {self.info['current_bet']}")
        dealer = self.deck.ids_to_renders(self.info["dealer_hand"])
        player = self.deck.ids_to_renders(self.info["player_hand"])
        dealer_val = self.hand_value(self.info["dealer_hand"])
        player_val = self.hand_value(self.info["player_hand"])
        print(f"dealer hand (sum={dealer_val}):\n{dealer}")
        print(f"player hand (sum={player_val}):\n{player}")
        print(self.info["result"])
        print("_______________________________")

    def build_obs_infos(self):
        # Convert card ids to color, suit, rank
        dealer_hand = np.zeros(self.max_rounds, dtype=np.int32)
        for i, c in enumerate(self.dealer_hand):
            dealer_hand[i] = self.deck.id_to_obs(c)
        dealer_hand_cards_in_play = np.zeros(self.max_rounds, dtype=np.int8)
        dealer_hand_cards_in_play[: len(self.dealer_hand)] = 1

        # Convert card ids to color, suit, rank
        player_hand = np.zeros(self.max_rounds, dtype=np.int32)
        for i, c in enumerate(self.player_hand):
            player_hand[i] = self.deck.id_to_obs(c)
        player_hand_cards_in_play = np.zeros(self.max_rounds, dtype=np.int8)
        player_hand_cards_in_play[: len(self.player_hand)] = 1

        obs: Dict[str, Any] = {
            "phase": self.action_phase.value,
            "dealer_hand": dealer_hand,
            "dealer_hand_cards_in_play": dealer_hand_cards_in_play,
            "player_hand": player_hand,
            "player_hand_cards_in_play": player_hand_cards_in_play,
        }
        infos: Dict[str, Any] = {
            "phase": self.action_phase,
            "current_bet": self.curr_bet,
            "dealer_hand": copy.deepcopy(self.dealer_hand),
            "player_hand": copy.deepcopy(self.player_hand),
            "result": "",
        }

        return obs, infos

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[gym.core.ObsType, Tuple[gym.core.ObsType, Dict[str, Any]]]:
        if seed is not None:
            np.random.seed(seed)
        self.curr_game = 0
        self.curr_round = 0
        self.action_phase = Phase.BET
        self.curr_bet = -float("inf")
        self.deck.reset()
        self.obs, self.info = self.build_obs_infos()
        self.action_phase = Phase.DEAL
        if return_info:
            return self.obs, self.info

        return self.obs


if __name__ == "__main__":
    game = BlackJack()
    done = False
    obs, info = game.reset(return_info=True)
    reward = -float("inf")
    game.render()
    phase: int = obs["phase"]
    action_dict = {"bet_size": 0, "hit": 0}

    while not done:
        if phase == Phase.BET:
            action = input(f"How much to bet? Input index: {game.bet_sizes} ")
            action_dict = {"bet_size": int(action), "hit": 0}
        elif phase == Phase.PLAY or phase == Phase.DEAL:
            action = input("Stay (0) or hit (1)?")
            action_dict = {"bet_size": 0, "hit": int(action)}
        elif phase == Phase.PAYOUT:
            action = input(f"Received reward of {reward}, any key to continue.")
        else:
            action = input(f"phase {phase}")

        obs, reward, done, info = game.step(action_dict)
        phase = obs["phase"]
        game.render()
