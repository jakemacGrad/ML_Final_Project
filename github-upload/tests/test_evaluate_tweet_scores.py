import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_tweet_scores import POLICIES, confusion, decision_metrics, threshold_choices


class EvaluationTests(unittest.TestCase):
    def test_false_positive_budget_counts_tied_scores_together(self):
        labels = np.array([False] * 10 + [True] * 3)
        scores = np.array([.99, .75, .4, .39, .38, .37, .36, .35, .34, .33, .95, .85, .75])
        cutoffs = dict(zip(POLICIES, threshold_choices(labels, scores), strict=True))
        strict = confusion(labels, scores >= cutoffs["train_fpr_cap_10pct"])
        relaxed = confusion(labels, scores >= cutoffs["train_fpr_cap_20pct"])
        np.testing.assert_array_equal(strict, [9, 1, 1, 2])
        np.testing.assert_array_equal(relaxed, [8, 2, 0, 3])

    def test_all_bot_baseline_exposes_zero_human_recall(self):
        metrics = decision_metrics(np.array([0, 40, 0, 60]))
        self.assertEqual(metrics["bot_precision"], 0.6)
        self.assertEqual(metrics["bot_recall"], 1)
        self.assertEqual(metrics["human_recall"], 0)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
