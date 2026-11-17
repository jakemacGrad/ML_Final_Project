import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tweet_patterns import NAMES, summarize
from tweet_pca import FEATURE, build, fit_patterns, project, write_csv


class TweetPCATests(unittest.TestCase):
    def test_templates_match_and_short_timeline_uses_fallback(self):
        updates = {
            i: f"RT @channel: This is a local update for item {i}. https://example.com/{i}"
            for i in range(8)
        }
        result = dict(zip(NAMES, summarize(updates), strict=True))
        self.assertEqual(result["retweet_rate"], 1)
        self.assertEqual(result["template_reuse"], 1)
        self.assertAlmostEqual(result["exact_template_rate"], 7 / 8)
        updates.pop(0)
        self.assertTrue(np.isnan(summarize(updates)).all())

    def test_projection_is_independent_of_other_queries(self):
        rng = np.random.default_rng(22)
        train = rng.normal(size=(25, len(NAMES)))
        fitted = fit_patterns(train, "all_patterns")
        query = rng.normal(size=(1, len(NAMES)))
        missing = np.full((1, len(NAMES)), np.nan)
        other = rng.normal(loc=300, size=(1, len(NAMES)))
        batched = project(np.vstack((query, missing, other)), fitted)
        np.testing.assert_allclose(project(query, fitted)[0], batched[0])
        np.testing.assert_allclose(batched[1], np.median(project(train, fitted), axis=0))

    def test_export_uses_original_folds_and_can_score_without_test_labels(self):
        with tempfile.TemporaryDirectory() as temp, threadpool_limits(limits=1):
            root = Path(temp)
            archive, folds, out = root / "data.zip", root / "folds.csv", root / "course"
            accounts = [str(i) for i in range(1, 26)]
            with ZipFile(archive, "w") as source:
                source.writestr("Twibot-20 Data/train_users_cleaned.csv", "ID,account_type\n" + "".join(
                    f"{account},{'bot' if int(account) % 2 else 'human'}\n" for account in accounts
                ))
                for split, keys in (("train", accounts), ("test", ["26", "27"])):
                    tweets = ["ID,tweet_idx,text\n"]
                    for account in keys[:-1]:
                        for i in range(8 + int(account) % 6):
                            ending = (
                                "with a link https://example.com" if (i + int(account)) % 3 else "for you?"
                            )
                            tweets.append(f"{account},{i},This is an English example {ending} {i}\n")
                    source.writestr(f"Twibot-20 Data/{split}_tweets.csv", "".join(tweets))
                source.writestr("Twibot-20 Data/test_users_cleaned.csv", "ID,opaque\n26,na\n27,na\n")
            write_csv(folds, ("fold", "role", "ID", "tweet_knn_score"), (
                (fold + 1, "validation" if int(account) % 5 == fold else "train", account, 0.5)
                for fold in range(5) for account in accounts
            ))
            build(archive, folds, out, held_out=True)
            manifest = json.loads((out / "run_manifest.json").read_text())
            with (out / "test_features.csv").open(newline="") as stream:
                scores = list(csv.DictReader(stream))
            self.assertEqual([row["ID"] for row in scores], ["26", "27"])
            self.assertEqual(float(scores[1][FEATURE]), manifest["fallback"])
            with (out / "cv_features.csv").open(newline="") as stream:
                validation = [row["ID"] for row in csv.DictReader(stream) if row["role"] == "validation"]
            self.assertCountEqual(validation, accounts)
