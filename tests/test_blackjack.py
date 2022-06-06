import unittest

from pogym.envs.blackjack import BlackJack


class TestBlackjack(unittest.TestCase):
    def test_reset(self):
        b = BlackJack()
        b.reset()

    def test_step(self):
        b = BlackJack()
        a = {"hit": 1, "bet_size": 1}
        [b.step(a) for i in range(10)]
