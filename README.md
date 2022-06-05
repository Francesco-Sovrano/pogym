# Partially Observable Gym
![tests](https://github.com/smorad/pogym/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/smorad/pogym/branch/master/graph/badge.svg?token=I47IDFZXSV)](https://codecov.io/gh/smorad/pogym)

Partially Observable Gym, or `pogym` is a collection of Partially Observable Markov Decision Process (POMDP) environments following the [Openai Gym](https://github.com/openai/gym) interface. The goal of `pogym` is to provide a standard benchmark for memory-based models in reinforcement learning. In other words, we want to provide a place for you to test and compare your models and algorithms to R2D2, recurrent PPO, decision transformers, and so on. `pogym` has a few basics tenets that we will adhere to:
1. **Painless setup** - `pogym` requires only `gym` and `numpy` as dependencies, and can be installed with a single `pip install`.
2. **Laptop-sized tasks** - None of our environments have large observation spaces or require GPUs to render.
3. **No overfitting** - It is possible for memoryless agents to receive high rewards on environments by memorizing the layout of each level. To avoid this, all environments are procedurally generated. 
