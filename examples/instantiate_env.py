import gym

import pogym  # noqa: F401
from pogym.envs.higher_lower import HigherLower

# After import pogym
# You can either load them the normal way
env = HigherLower(num_decks=2)
obs = env.reset()

# or the gym way
env = gym.make("pogym-Blackjack-v0")
obs = env.reset()
