import json
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tweet_patterns import NAMES
from tweet_pca_knn import (
    COMPONENTS, CONVERSION, UPDATE_MODEL, FittedKNN, build, choose_threshold, fit_with_batches, project_sources,
    read_label_batches, threshold_flags,
)


class PcaKnnTests(unittest.TestCase):
    def test_conversion_mapping_is_read_without_executing_the_notebook(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "project.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr(CONVERSION, json.dumps({"cells": [{
                    "cell_type": "code",
                    "source": ["LABEL_MAP = {0: 'human', 1: 'bot'}\n", "raise RuntimeError('do not run')\n"],
                }]}))
            self.assertEqual(project_sources(path)["account_labels"], {0: "human", 1: "bot"})

    def test_calibration_counts_ineligible_accounts_but_does_not_flag_them(self):
        labels = np.array([False, False, False, True, True, True])
        scores = np.array([.99, .8, .4, .9, .6, .95])
        available = np.array([False, True, True, True, True, False])
        threshold = choose_threshold(labels, scores, 0, available)
        self.assertAlmostEqual(threshold, .9)
        np.testing.assert_array_equal(
            threshold_flags(scores, available, threshold), [False, False, False, True, False, False]
        )

    def test_saved_model_recreates_scores_and_keeps_sparse_accounts_unflagged(self):
        rng = np.random.default_rng(82)
        values = rng.normal(size=(60, len(NAMES)))
        labels = np.arange(len(values)) % 2 == 0
        queries = np.vstack([values[:8], np.full(len(NAMES), np.nan)])
        with tempfile.TemporaryDirectory() as temp, threadpool_limits(limits=1):
            path = Path(temp) / "model.npz"
            original = FittedKNN.fit(values, labels, NAMES, COMPONENTS)
            original.save(path, .95, .85)
            restored, strict, screening = FittedKNN.load(path)
            self.assertEqual((strict, screening), (.95, .85))
            points, coverage = restored.score(queries)
            np.testing.assert_array_equal(points, original.score(queries)[0])
            self.assertEqual(points[-1], restored.prior)
            self.assertFalse(threshold_flags(points, coverage, screening)[-1])

    def test_labeled_update_changes_the_reference_and_guards_benchmark_ids(self):
        rng = np.random.default_rng(87)
        values = rng.normal(size=(60, len(NAMES)))
        truth = np.arange(len(values)) % 2 == 0
        with tempfile.TemporaryDirectory() as temp, threadpool_limits(limits=1):
            path = Path(temp) / "labeled.csv"
            path.write_text("ID,tweet_idx,text,account_type\n" + "".join(
                f"6021,{i},This is an automated alert for our local area {i},bot\n"
                for i in range(8)
            ))
            batch, evidence = read_label_batches([path], set())
            self.assertEqual(evidence[0]["with_measured_score"], 1)
            updated = fit_with_batches(values, truth, batch)
            self.assertEqual(len(updated.labels), 61)
            points, _ = updated.score(batch[0])
            self.assertGreater(points[0], .99)
            with self.assertRaisesRegex(ValueError, "disjoint"):
                read_label_batches([path], {"6021"})

    def test_update_build_exports_before_and_after_metrics(self):
        rng = np.random.default_rng(74)
        with tempfile.TemporaryDirectory() as temp, threadpool_limits(limits=1):
            root = Path(temp)
            project, data, batch = (root / name for name in ("project.zip", "data.zip", "batch.csv"))
            with ZipFile(project, "w") as archive:
                archive.writestr(CONVERSION, json.dumps({"cells": [{
                    "cell_type": "code", "source": "LABEL_MAP = {0: 'human', 1: 'bot'}",
                }]}))
            with ZipFile(data, "w") as archive:
                archive.writestr("Twibot-20 Data/train_users_cleaned.csv", "ID,account_type\n" + "".join(
                    f"{i},{'bot' if i % 2 else 'human'}\n" for i in range(1, 121)
                ))
                archive.writestr("Twibot-20 Data/test_users_cleaned.csv", "ID\n121\n")
            batch.write_text("ID,tweet_idx,text,account_type\n" + "".join(
                f"122,{i},This is an automated alert for our local area {i},bot\n" for i in range(8)
            ))
            out = root / "updated"
            with patch("tweet_pca_knn.extract", return_value=rng.normal(size=(120, len(NAMES)))):
                build(project, data, out, False, [batch])
            manifest = json.loads((out / "run_manifest.json").read_text())
            self.assertEqual(manifest["model"], UPDATE_MODEL)
            self.assertEqual(manifest["release_knn_reference_accounts"], 91)
            self.assertTrue({"PCA + KNN (14 descriptors)", UPDATE_MODEL} <= manifest["development"].keys())
