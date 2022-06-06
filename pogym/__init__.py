import gym
import inspect
from pogym.envs.blackjack import BlackJack
from pogym.envs.higher_lower import HigherLower
from pogym.envs.stateless_cartpole import StatelessCartPole
from pogym.envs.stateless_pendulum import StatelessPendulum

ALL_ENVS = {
    BlackJack: {
        "id": "pogym-Blackjack-v0",
    },
    HigherLower: {"id": "pogym-HigherLower-v0"},
    StatelessCartPole: {"id": "pogym-StatelessCartPole-v0"},
    StatelessPendulum: {
        "id": "pogym-StatelessPendulum-v0",
    },
}

for e, v in ALL_ENVS.items():
    gym.envs.register(
        entry_point=":".join([inspect.getmodule(e).__name__, e.__name__]), **v
    )
