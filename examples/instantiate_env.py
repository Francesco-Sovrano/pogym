import gym
import pogym
from pogym.envs.higher_lower import HigherLower

# You can either load them the normal way
env = HigherLower(num_decks=2)
obs = env.reset()

# or the gym way
env = gym.make("pogym-Blackjack-v0")
obs = env.reset()
