import inspect
from typing import Any, Dict

import gym

from pogym.envs.blackjack import BlackJack
from pogym.envs.higher_lower import HigherLower
from pogym.envs.stateless_cartpole import StatelessCartPole
from pogym.envs.stateless_pendulum import StatelessPendulum

ALL_ENVS: Dict[gym.Env, Dict[str, Any]] = {
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
    mod_name = inspect.getmodule(e).__name__  # type: ignore
    gym.envs.register(entry_point=":".join([mod_name, e.__name__]), **v)
