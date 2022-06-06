import unittest

from pogym.core import deck


class TestCore(unittest.TestCase):
    def test_deck_size(self):
        d = deck.Deck(num_decks=1)
        self.assertEqual(len(d), 52)
        d = deck.Deck(num_decks=7)
        self.assertEqual(len(d), 7 * 52)

    def test_draw_discard_reset(self):
        d = deck.Deck()
        a, b = d.draw(), d.draw()
        self.assertTrue(a, b not in d.deck)
        self.assertEqual(d.in_play, [a, b])

        d.discard_hands()
        self.assertTrue(a, b not in d.deck)
        self.assertEqual(d.in_play, [])
        self.assertEqual(d.discard, [a, b])

        d.reset()
        self.assertTrue(a, b in d.deck)
        self.assertTrue(a, b not in d.in_play)
        self.assertTrue(a, b not in d.discard)
