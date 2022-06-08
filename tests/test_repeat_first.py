import unittest

from pogym.envs.repeat_first import RepeatFirst


class TestRepeatFirst(unittest.TestCase):
    def test_all(self):
        e = RepeatFirst()
        _ = e.reset()
        for i in range(100):
            _, _, done, _ = e.step(0)
            if done:
                e.reset()
