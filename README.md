# Partially Observable Gym
![tests](https://github.com/smorad/pogym/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/smorad/pogym/branch/master/graph/badge.svg?token=I47IDFZXSV)](https://codecov.io/gh/smorad/pogym)

Partially Observable Gym, or `pogym` is a collection of Partially Observable Markov Decision Process (POMDP) environments following the [Openai Gym](https://github.com/openai/gym) interface. The goal of `pogym` is to provide a standard benchmark for memory-based models in reinforcement learning. In other words, we want to provide a place for you to test and compare your models and algorithms to R2D2, recurrent PPO, decision transformers, and so on. `pogym` has a few basics tenets that we will adhere to:
1. **Painless setup** - `pogym` requires only `gym` and `numpy` as dependencies, and can be installed with a single `pip install`.
2. **Laptop-sized tasks** - None of our environments have large observation spaces or require GPUs to render.
3. **No overfitting** - It is possible for memoryless agents to receive high rewards on environments by memorizing the layout of each level. To avoid this, all environments are procedurally generated. 

## Environments
The environments are split into set or sequence tasks. Ordering matters in sequence tasks (e.g. the order of the button presses in simon matters), and does not matter in set tasks (e.g. the "count" in blackjack does not change if you swap o<sub>t-1</sub> and o<sub>t-k</sub>).

### Set Environments
* Memory/Concentration (partially implemented)
* Blackjack
* Baccarat (not implemented yet)
* Higher/Lower
* Battleship (not implemented yet)
* Multiarmed Bandit (not implemented yet)
* Minesweeper (not implemented yet)

### Sequence Environments
* RememberPrevObs (not implemented yet)
* RememberFirstObs (not implemented yet)
* Stateless Cartpole
* Stateless Pendulum (not implemented yet)
* Treasure Hunt (not implemented yet)
* Repeat Backwards

## Contributing
Make sure you run with precommit hooks:
```bash
pip install pre-commit
git clone https://github.com/smorad/pogym
cd pogym
pre-commit install
```


### Environment Descriptions
#### Baccarat
Identical rules to casino baccarat, with betting. The agent should use memory to count cards and increase bets when it is more likely to win.
#### Battleship
One-player battleship. Select a gridsquare to launch an attack, and receive confirmation whether you hit the target. The agent should use memory to remember which gridsquares were hits and which were misses, completing an episode sooner.
#### Multiarmed Bandit
Over an episode, solve a multiarmed bandit problem by maximizing the expected reward. The agent should use memory to keep a running mean and variance of bandits.
#### Minesweeper
Classic minesweeper, but with reduced vision range. The agent only has vision of the surroundings near its last sweep. The agent must use memory to remember where the bombs are
#### RememberPrevObs
Output the t-k<sup>th</sup> observation for a reward
#### RememberFirstObs
Output the zeroth observation for a reward
#### Stateless Cartpole
Classic cartpole, except the velocity and angular velocity magnitudes are hidden. The agent must use memory to differentiate position into velocity.
#### Noisy Pendulum
Classic pendulum, but the observations are corrupted with large amounts of gaussian noise.
#### Treasure Hunt
The agent is placed in an open square and must search for a treasure. With memory, the agent can remember where it has been and complete the episode faster.
#### Repeat Backwards
The agent will receive k observations then must repeat them in reverse order.
