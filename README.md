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
* Multiarmed Bandit
* Minesweeper (not implemented yet)

### Sequence Environments
* Repeat Previous
* Repeat First
* Repeat Backwards
* Stateless Cartpole
* Stateless Pendulum
* Treasure Hunt (not implemented yet)
* Bipedal Walker

## Contributing
Make sure you run with precommit hooks:
```bash
pip install pre-commit
git clone https://github.com/smorad/pogym
cd pogym
pre-commit install
```


### Environment Descriptions
#### BlackJack
Casino blackjack, but unlike other environments the game is not over after the hand is dealt. The game continues until the deck(s) of cards are exhausted. The agent should learn to maintain a "count" of the cards it has seen. Using memory, it can infer what cards remain in the deck, and adjust the bet accordingly to maximize return.
#### Baccarat
Identical rules to casino baccarat, with betting. The agent should use memory to count cards and increase bets when it is more likely to win.
#### Higher/Lower
Guess whether the next card drawn from the deck is higher or lower than the previously drawn card. The agent should keep a count like blackjack and baccarat and modify bets, but this game is significantly simpler than either baccarat or blackjack.
#### Battleship
One-player battleship. Select a gridsquare to launch an attack, and receive confirmation whether you hit the target. The agent should use memory to remember which gridsquares were hits and which were misses, completing an episode sooner.
#### Multiarmed Bandit
Over an episode, solve a multiarmed bandit problem by maximizing the expected reward. The agent should use memory to keep a running mean and variance of bandits.
#### Minesweeper
Classic minesweeper, but with reduced vision range. The agent only has vision of the surroundings near its last sweep. The agent must use memory to remember where the bombs are
#### Repeat Previous
Output the t-k<sup>th</sup> observation for a reward
#### Repeat First
Output the zeroth observation for a reward
#### Repeat Backwards
The agent will receive k observations then must repeat them in reverse order.
#### Stateless Cartpole
Classic cartpole, except the velocity and angular velocity magnitudes are hidden. The agent must use memory to compute rates of change.
#### Stateless Pendulum
Classic pendulum, but the velocity and angular velocity are hidden from the agent. The agent must use memory to compute rates of change.
#### Treasure Hunt
The agent is placed in an open square and must search for a treasure. With memory, the agent can remember where it has been and complete the episode faster.
#### Bipedal Walker
Classic bipedal walker with procedurally-generated levels, but with a single LiDAR ray cast from the head. The agent must move the head and combine single rays over time into a representation of the environment to avoid obstacles.
