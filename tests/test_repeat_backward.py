import unittest

from pogym.envs.repeat_backwards import RepeatBackwards


class TestRepeatBackwards(unittest.TestCase):
    def test_init(self):
        RepeatBackwards().reset()

    def test_step(self):
        b = RepeatBackwards()
        obs = b.reset()
        done = False
        while not done:
            obs, reward, done, info = b.step(0)
