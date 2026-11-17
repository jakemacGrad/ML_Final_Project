import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tweet_patterns import EXTENDED_NAMES, NAMES, TWEET_LIMIT, summarize


class TweetStyleTests(unittest.TestCase):
    def test_style_measures_case_and_punctuation_after_url_removal(self):
        timeline = dict(enumerate([
            "RT @CHANNEL: This update is for you https://example.org/hidden!😃",
            "This is an article for you https://example.org",
            "RT @CHANNEL: This one is for you https://example.org",
            "@friend These are MY words — YES!! 😃…",
            "The Story Is In The Paper",
            "This story is for you https://example.org",
            "I am at the park with you",
            "This story is for everyone",
        ]))
        baseline = summarize(timeline)
        extended = summarize(timeline, extended=True)
        self.assertEqual(extended.dtype, baseline.dtype)
        np.testing.assert_array_equal(extended[:len(NAMES)], baseline)
        values = dict(zip(EXTENDED_NAMES, extended, strict=True))
        for name in (
            "emoji_post_rate", "em_dash_post_rate", "exclamation_post_rate",
            "ellipsis_post_rate", "repeated_punctuation_post_rate", "title_case_post_rate",
        ):
            self.assertAlmostEqual(values[name], 1 / 8, msg=name)
        self.assertGreater(values["uppercase_letter_rate"], 0)
        self.assertGreater(values["uppercase_word_rate"], 0)
        self.assertEqual(values["link_retweet_post_rate"], 2 / 8)
        self.assertAlmostEqual(values["broadcast_switch_rate"], 3 / 7)
        self.assertEqual(values["longest_broadcast_run_share"], 3 / 8)

    def test_extended_features_use_same_cap_and_missing_fallback(self):
        timeline = {
            index: "This message is for you" for index in range(TWEET_LIMIT + 1)
        }
        timeline[TWEET_LIMIT] += " — WOW!! 😃"
        extended = summarize(timeline, extended=True)
        np.testing.assert_array_equal(extended[:len(NAMES)], summarize(timeline))
        values = dict(zip(EXTENDED_NAMES, extended, strict=True))
        self.assertEqual(values["emoji_post_rate"], 0)
        short = summarize({0: "This message is for you"}, extended=True)
        self.assertEqual(short.shape, (len(EXTENDED_NAMES),))
        self.assertTrue(np.isnan(short).all())


if __name__ == "__main__":
    unittest.main()
