import unittest

from pogym.envs.repeat_previous import RepeatPrevious


class TestRepeatPrevious(unittest.TestCase):
    def test_all(self):
        e = RepeatPrevious()
        _ = e.reset()
        for i in range(100):
            _, _, done, _ = e.step(0)
            if done:
                e.reset()

    def test_k(self):
        e = RepeatPrevious(k=2)
        obs0 = e.reset()
        obs1, rew1, done1, info1 = e.step(0)
        self.assertFalse(done1)
        obs2, rew2, done2, info2 = e.step(obs0[1])
        self.assertFalse(done2)
        obs3, rew3, done3, info3 = e.step(obs1[1])
        self.assertFalse(done3)
        obs4, rew4, done4, info4 = e.step(100)
        self.assertTrue(done4)
