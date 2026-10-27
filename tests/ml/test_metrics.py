import unittest
from ml.eval.metrics import Tile, maximum_matches, score_image, summarize


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.hand = [Tile("1m", (0, 0, 10, 20)), Tile("1m", (12, 0, 10, 20))]

    def test_duplicate_same_class_does_not_hide_missing_tile(self):
        duplicated = [self.hand[0], self.hand[0]]
        self.assertFalse(score_image(self.hand, duplicated)["hand_exact_match"])
        self.assertTrue(score_image(self.hand, duplicated)["multiset_exact"])

    def test_wrong_class_extra_and_missing_all_fail(self):
        for predictions in [self.hand[:1], self.hand + [self.hand[0]],
                            [Tile("2m", t.bbox) for t in self.hand]]:
            self.assertFalse(score_image(self.hand, predictions)["hand_exact_match"])

    def test_rejection_and_http_failure_stay_in_denominator(self):
        result = summarize([score_image(self.hand, self.hand),
                            score_image(self.hand, self.hand, accepted=False),
                            score_image(self.hand, [], failed=True)])
        self.assertEqual(result["images"], 3)
        self.assertEqual(result["hand_exact_match"], 1 / 3)
        self.assertEqual(result["accepted_coverage"], 1 / 3)

    def test_matching_recovers_from_greedy_assignment(self):
        truth = [Tile("1m", (2, 0, 10, 10)), Tile("1m", (0, 0, 10, 10))]
        predictions = [Tile("1m", (1, 0, 10, 10)), Tile("1m", (5, 0, 10, 10))]
        self.assertEqual(maximum_matches(truth, predictions, 0.5), 2)

    def test_empty_accepted_background_is_correct_but_not_rejection(self):
        self.assertTrue(score_image([], [])["hand_exact_match"])
        self.assertFalse(score_image([], [], accepted=False)["hand_exact_match"])
        self.assertFalse(score_image([], [], failed=True)["model_exact"])
        self.assertFalse(score_image([], [], failed=True)["multiset_exact"])


if __name__ == "__main__":
    unittest.main()
