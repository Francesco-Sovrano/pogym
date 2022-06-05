import gym
import numpy as np

from pogym.core.deck import Deck


class BlackJack(gym.Env):
    """A game of blackjack, where card counting is possible. Successful agents
    should learn to count cards, and bet higher/hit less often when the deck contains
    higher cards. Note that splitting is not allowed.

    Args:
        bet_sizes: The bet sizes available to the agent. These correspond to
            the final reward.
        num_decks: The number of individual decks combined into a single deck.
            In Vegas, this is usually between four and eight.
        max_hits: The maximum number of rounds where the agent and dealer
            can hit/stay. There is no max in real blackjack, however this 
            would result in a very large observation space.
        games_per_episode: The number of games per episode. This must be set high
            for card-counting to have an effect. When set to one, the game
            becomes fully observable.

    Returns:
        A gym environment
    """
    def __init__(self, bet_sizes=[0.2, 0.4, 0.6, 0.8, 1.0], num_decks=1, max_rounds=6, games_per_episode=20):
        self.deck = Deck(num_decks=num_decks)
        self.bet_sizes = bet_sizes
        self.max_rounds = max_rounds
        # Hit, stay, and bet amount
        self.action_space = gym.spaces.Dict({
            "hit_or_stay": gym.spaces.Discrete(2),
            "bet_size": gym.spaces.Discrete(len(bet_sizes))
        })
        self.games_per_episode = games_per_episode
        self.curr_game = 0
        self.curr_round = 0
        self.curr_bet = 0
        self.observation_space = gym.spaces.Dict({
            "phase": gym.spaces.Discrete(3),
            "dealer_hand": gym.spaces.Tuple(max_rounds * [Deck.card_obs_space]),
            "player_hand": gym.spaces.Tuple(max_rounds * [Deck.card_obs_space]),
        })
        self.dealer_hand = []
        self.player_hand = []
        self.play_phase = 0

    def hand_value(self, hand):
        card_map = {"a":1, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "j": 10, "q": 10, "k":10}
        value = 0
        has_ace = False
        for card in hand:
            suit, rank = self.deck.id_to_str(card)
            if rank == "a":
                has_ace = True
            value += card_map[rank]

        if value <= 11 and has_ace:
            # ace = 1 + 10
            value += 10
        return value
            
    def step(self, action):
        # First round:
        #   curr_round == 0
        #   input action: select bet
        #   output obs: two player cards and one dealer card
        # 
        # Hit round:
        #   curr_round > 0
        #   input action: hit
        #   output obs: n player cards and one dealer card
        #
        # Terminal round:
        #   curr_round > 0
        #   triggers on: player stay, player bust, natural
        #   input action: hit or stay
        #   output obs: results after busting or staying
        #   output reward: +- bet (or 1.5x bet in case of player natural)

        # In the first round, there are no cards on the table
        # and the player sets a bet
        # In second round, the gets two cards
        # and the dealer gets one
        # We observe
        reward = 0
        done = False
        game_done = False
        dealer_hit = False
        need_obs = True
        player_hit = action["hit_or_stay"] == 0
        # Whether we are betting (0) or playing (1)
        self.play_phase = int(self.curr_round > 0)
        # Potentially two draws, end game if deck is too small
        if len(self.deck) < 2:
            done = True
            result = "deck empty, episode over"
        elif not self.play_phase:
            # Draw two cards for player and one for dealer
            self.curr_bet = self.bet_sizes[action["bet_size"]]
            self.player_hand += [self.deck.draw(), self.deck.draw()]
            self.dealer_hand += [self.deck.draw()]
            # Render here, so we see the results of the last game
            # during the betting phase
            self.obs = self.build_obs()
            need_obs = False
            result = "player and dealer draw cards"
        else:
            # Playing phase
            # Player
            if player_hit:
                # Hit
                self.player_hand += [self.deck.draw()]
                result = "player hits"

            player_value = self.hand_value(self.player_hand)
            player_bust = player_value > 21
            player_blackjack = player_value == 21
            player_natural = player_blackjack and len(self.player_hand) == 2

            # Terminal phases
            if not player_hit or player_bust or player_blackjack:
                # Player done, dealer's turn
                dealer_value = self.hand_value(self.dealer_hand)
                while dealer_value < 17:
                    self.dealer_hand += [self.deck.draw()]
                    dealer_value = self.hand_value(self.dealer_hand)

                game_done = True
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

        if need_obs:
            self.obs = self.build_obs()
            print("round", self.curr_round)
        self.curr_round += 1
        if game_done:
            self.curr_game += 1
            self.curr_round = 0
            self.deck.discard_hands()
            self.player_hand.clear()
            self.dealer_hand.clear()
        if self.curr_game >= self.games_per_episode - 1:
            done = True

        return self.obs, reward, done, {"result": result}

    def render_hand(self, hand):
        cards = []
        for card in hand:
            if card[0] == 0:
                # Placeholder/null card
                continue
            cards.append(self.deck.obs_to_viz(card[1:].tolist()))
        if len(cards) > 0:
            return np.stack(cards)
        else:
            return np.empty(0)


    def render(self):
        phase = "bet" if self.obs["phase"] == 0 else "play"
        print(f"phase: {phase}") 
        dealer = self.render_hand(self.obs["dealer_hand"])
        player = self.render_hand(self.obs["player_hand"])
        dealer_val = self.hand_value(self.dealer_hand)
        player_val = self.hand_value(self.player_hand)
        print(f"dealer hand (sum={dealer_val}):\n {dealer}")
        print(f"player hand (sum={player_val}):\n {player}")


    def build_obs(self):
        phase = np.array(self.play_phase)

        # Convert card ids to color, suit, rank
        dealer_hand = [np.array([1, *self.deck.id_to_obs(c)]) for c in self.dealer_hand]
        # Pad hand with empty cards
        dealer_hand += [np.array([0, 0, 0, 0]) for i in range(self.max_rounds - len(self.dealer_hand))]
        dealer_hand = np.stack(dealer_hand)

        # Convert card ids to color, suit, rank
        player_hand = [np.array([1, *self.deck.id_to_obs(c)]) for c in self.player_hand]
        # Pad hand with empty cards
        player_hand += [np.array([0, 0, 0, 0]) for i in range(self.max_rounds - len(self.player_hand))]
        player_hand = np.stack(player_hand)

        obs = {
            "phase": phase,
            "dealer_hand": dealer_hand,
            "player_hand": player_hand
        }
        return obs


    def reset(self, return_info=False):
        self.curr_game = 0
        self.curr_round = 0
        self.play_phase = 0
        self.deck.reset()
        self.obs = self.build_obs()
        # Due to how reset works, we do not use
        # the action selected directly after reset
        if return_info:
            return self.obs, {}

        return self.obs


if __name__ == '__main__':
    game = BlackJack()
    done = False
    obs = game.reset()
    action_dict = {"bet_size": 0, "hit_or_stay": 0}
    while not done:
        obs, reward, done, info = game.step(action_dict)
        phase = obs['phase']
        if phase == 0:
            action = input(f'How much to bet? Input index: {game.bet_sizes} ')
            action_dict = {"bet_size": int(action), "hit_or_stay": 0}
        else:
            action = input(f'Hit (0) or stay (1)?')
            action_dict = {"bet_size": 0, "hit_or_stay": int(action)}

        #game.render()
        print(obs)
        print(info['result'])
        if reward != 0:
            print(f"Got reward: {reward}")





