"""Train and evaluate a supervised account classifier from the two project ZIPs."""

import argparse
import ast
import csv
import json
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from evaluate_tweet_scores import confusion, decision_metrics
from tweet_patterns import MIN_TWEETS, NAMES, SCAN_LIMIT, TWEET_LIMIT, extract, summarize, timelines
from tweet_pca import checksum, component_scores, fit_patterns, profiles, select_candidate, write_csv

CONVERSION = "ML_Final_Project-main/conversion.ipynb"
LABELS = {0: "human", 1: "bot"}
FOCUSED = (
    "retweet_rate", "link_rate", "hashtag_rate", "template_reuse",
    "exact_template_rate", "word_diversity",
)
COMPONENTS = 8
NEIGHBORS = 35
FOLD_SEED = 6030
CALIBRATION_SEED = 6021
CALIBRATION_SHARE = 0.25
STRICT = 0.01
SCREENING = 0.10
BUDGETS = (0, 0.005, STRICT, 0.02, SCREENING, 0.20)
MODEL = "PCA + KNN (14 descriptors)"
SEED_MODEL = "PCA + KNN (simulated 70% seed)"
UPDATE_MODEL = "PCA + KNN (with labeled updates)"
IDENTIFICATION_RULES = {
    "amplification": ("retweet_rate", "link_rate", "hashtag_rate"),
    "repeated_wording": ("template_reuse", "exact_template_rate", "word_diversity"),
    "conversation_and_style": tuple(name for name in NAMES if name not in FOCUSED),
}


def project_sources(path):
    with ZipFile(path) as archive:
        notebook = json.loads(archive.read(CONVERSION))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        for statement in ast.parse(code).body:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "LABEL_MAP"
                for target in statement.targets
            ):
                if ast.literal_eval(statement.value) != LABELS:
                    raise ValueError("Unexpected account label mapping in the conversion notebook.")
                return {"archive_sha256": checksum(path), "conversion_member": CONVERSION,
                        "account_labels": LABELS}
    raise ValueError("The conversion notebook is missing the account label mapping.")


@dataclass
class FittedKNN:
    fields: tuple
    center: np.ndarray
    scale: np.ndarray
    pca_center: np.ndarray
    basis: np.ndarray
    classifier: KNeighborsClassifier
    prior: float
    reference: np.ndarray
    labels: np.ndarray
    variance: float

    @staticmethod
    def transform(values, center, scale, pca_center, basis):
        return ((values - center) / scale - pca_center) @ basis.T

    @classmethod
    def fit(cls, values, labels, fields, components):
        available = np.isfinite(values).all(axis=1)
        usable = values[available][:, [NAMES.index(field) for field in fields]]
        outcomes = labels[available]
        if len(usable) < NEIGHBORS or len(np.unique(outcomes)) != 2:
            raise ValueError("KNN needs at least 35 usable accounts and both classes.")
        scaler = StandardScaler().fit(usable)
        pca = PCA(n_components=components, svd_solver="full").fit(scaler.transform(usable))
        reference = cls.transform(usable, scaler.mean_, scaler.scale_, pca.mean_, pca.components_)
        model = KNeighborsClassifier(n_neighbors=NEIGHBORS, weights="distance").fit(reference, outcomes)
        return cls(fields, scaler.mean_, scaler.scale_, pca.mean_, pca.components_, model,
                   float(outcomes.mean()), reference, outcomes, float(pca.explained_variance_ratio_.sum()))

    def score(self, values):
        available = np.isfinite(values).all(axis=1)
        scores = np.full(len(values), self.prior)
        if available.any():
            usable = values[available][:, [NAMES.index(field) for field in self.fields]]
            projected = self.transform(usable, self.center, self.scale, self.pca_center, self.basis)
            scores[available] = self.classifier.predict_proba(projected)[:, 1]
        return scores, available

    def save(self, path, strict, screening):
        np.savez_compressed(
            path, fields=self.fields, center=self.center, scale=self.scale,
            pca_center=self.pca_center, basis=self.basis, reference=self.reference,
            labels=self.labels, prior=self.prior, variance=self.variance, neighbors=NEIGHBORS,
            strict_threshold=strict, screening_threshold=screening,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            reference, labels = saved["reference"], saved["labels"]
            classifier = KNeighborsClassifier(
                n_neighbors=int(saved["neighbors"]), weights="distance",
            ).fit(reference, labels)
            model = cls(tuple(saved["fields"]), saved["center"], saved["scale"], saved["pca_center"],
                        saved["basis"], classifier, float(saved["prior"]), reference, labels,
                        float(saved["variance"]))
            return model, float(saved["strict_threshold"]), float(saved["screening_threshold"])


def choose_threshold(labels, scores, budget, available):
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("Threshold calibration needs both classes.")
    # A separate lowest score for unavailable accounts keeps the full class denominators.
    sentinel = min(0.0, float(scores.min())) - 1
    fpr, recall, thresholds = roc_curve(
        labels, np.where(available, scores, sentinel), drop_intermediate=False,
    )
    feasible = np.flatnonzero((np.rint(fpr * negatives) <= budget * negatives) & (thresholds > sentinel))
    return float(thresholds[feasible[np.argmax(recall[feasible])]])


def threshold_flags(scores, available, threshold):
    return available & (scores >= threshold)


def calibration_split(indices, labels, seed):
    return train_test_split(indices, test_size=CALIBRATION_SHARE, stratify=labels[indices], random_state=seed)


def read_label_batches(paths, reserved):
    matrices, labels, provenance = [], [], []
    for path in paths:
        labels_by_id = {}

        def checked_rows(reader):
            for row in reader:
                account = (row["ID"] or "").strip()
                label = row["account_type"]
                if not re.fullmatch(r"[1-9][0-9]*", account) or label not in LABELS.values():
                    raise ValueError("Labeled updates need original decimal Twitter IDs and bot/human labels.")
                if account in reserved or labels_by_id.setdefault(account, label) != label:
                    raise ValueError("Labeled updates must be disjoint, with one label per account.")
                yield row

        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = {"ID", "tweet_idx", "text", "account_type"}
            if not fields.issubset(reader.fieldnames or []):
                raise ValueError("A labeled batch needs ID, tweet_idx, text, and account_type columns.")
            values_by_id = {account: summarize(posts) for account, posts in timelines(checked_rows(reader))}
        ids = sorted(values_by_id)
        if not ids:
            raise ValueError("A labeled batch must contain accounts.")
        values = np.vstack([values_by_id[account] for account in ids])
        measured = int(np.isfinite(values).all(axis=1).sum())
        if not measured:
            raise ValueError("A labeled batch must contain at least one account with enough English tweet text.")
        matrices.append(values)
        labels.extend(labels_by_id[account] == "bot" for account in ids)
        reserved.update(ids)
        provenance.append({"name": path.name, "sha256": checksum(path), "accounts": len(ids),
                           "with_measured_score": measured})
    if not matrices:
        return None, provenance
    return (np.vstack(matrices), np.array(labels, dtype=bool)), provenance


def fit_with_batches(values, labels, update):
    if update is not None:
        added_values, added_labels = update
        values = np.vstack((values, added_values))
        labels = np.concatenate((labels, added_labels))
    return FittedKNN.fit(values, labels, NAMES, COMPONENTS)


def development(values, labels, update=None):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    rankings, totals = defaultdict(list), defaultdict(lambda: np.zeros(4, dtype=int))
    complete = np.ones(len(values), dtype=bool)
    rows = []
    for fold, (training, validation) in enumerate(folds.split(values, labels), 1):
        fit, calibration = calibration_split(training, labels, CALIBRATION_SEED + fold)
        candidate = select_candidate(values[fit], labels[fit])
        base = fit_patterns(values[fit], candidate[0])
        predictions = {
            "Course PCA score": (
                (component_scores(values[calibration], candidate, base), complete[calibration]),
                (component_scores(values[validation], candidate, base), complete[validation]),
            ),
        }
        for name, fields, components in (
            ("PCA + KNN (six suggested cues)", FOCUSED, 5),
            (MODEL, NAMES, COMPONENTS),
        ):
            trained = FittedKNN.fit(values[fit], labels[fit], fields, components)
            predictions[name] = trained.score(values[calibration]), trained.score(values[validation])

        seed, _ = train_test_split(fit, test_size=0.30, stratify=labels[fit], random_state=CALIBRATION_SEED + fold)
        starter = FittedKNN.fit(values[seed], labels[seed], NAMES, COMPONENTS)
        predictions[SEED_MODEL] = starter.score(values[calibration]), starter.score(values[validation])
        if update is not None:
            updated = fit_with_batches(values[fit], labels[fit], update)
            predictions[UPDATE_MODEL] = updated.score(values[calibration]), updated.score(values[validation])

        for name, ((cal_scores, cal_available), (scores, available)) in predictions.items():
            truth = labels[validation]
            ap = float(average_precision_score(truth, scores))
            auc = float(roc_auc_score(truth, scores))
            rankings[name].append((ap, auc))
            for budget in BUDGETS:
                cutoff = choose_threshold(labels[calibration], cal_scores, budget, cal_available)
                counts = confusion(truth, threshold_flags(scores, available, cutoff))
                totals[name, budget] += counts
                rows.append({"fold": fold, "model": name, "target_human_fpr": budget,
                             "threshold": cutoff, "average_precision": ap, "roc_auc": auc,
                             **decision_metrics(counts)})
    return rows, {
        name: {
            "mean_average_precision": float(np.mean([item[0] for item in rankings[name]])),
            "mean_roc_auc": float(np.mean([item[1] for item in rankings[name]])),
            "operating_points": {str(budget): decision_metrics(totals[name, budget]) for budget in BUDGETS},
        }
        for name in rankings
    }


def test_predictions(archive, training_ids, model, strict, screening, path):
    ids = sorted(profiles(archive, "test"))
    if set(training_ids).intersection(ids):
        raise ValueError("Training and test account IDs overlap.")
    vectors = extract(archive, ids, "test")
    scores, available = model.score(vectors)
    high = threshold_flags(scores, available, strict)
    review = threshold_flags(scores, available, screening)
    bands = np.select([~available, high, review], ["insufficient_text", "high_priority", "review"],
                      default="below_screening_threshold")
    write_csv(path, ("ID", "bot_score", "review_band"), zip(ids, scores, bands, strict=True))
    return {"accounts": len(ids), "with_measured_score": int(available.sum()),
            "high_priority": int(high.sum()), "review": int((review & ~high).sum())}


def build(ml_archive, data_archive, output, held_out, label_batches=()):
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    source = project_sources(ml_archive)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(data_archive) as archive, threadpool_limits(limits=2):
        outcomes = profiles(archive, "train")
        ids = sorted(outcomes)
        labels = np.array([outcomes[account] == "bot" for account in ids])
        values = extract(archive, ids)
        reserved = set(ids)
        if label_batches:
            reserved.update(profiles(archive, "test"))
        updates, batches = read_label_batches(label_batches, reserved)
        folds, comparisons = development(values, labels, updates)

        fit, calibration = calibration_split(np.arange(len(ids)), labels, CALIBRATION_SEED)
        model = fit_with_batches(values[fit], labels[fit], updates)
        scores, available = model.score(values[calibration])
        strict = choose_threshold(labels[calibration], scores, STRICT, available)
        screening = choose_threshold(labels[calibration], scores, SCREENING, available)

        with tempfile.TemporaryDirectory(dir=output.parent) as temp:
            stage = Path(temp) / "release"
            stage.mkdir()
            model_path = stage / "pca_knn_model.npz"
            model.save(model_path, strict, screening)
            fields = list(folds[0])
            write_csv(stage / "development_metrics.csv", fields, ([row[key] for key in fields] for row in folds))
            metadata = {
                "project_zip": source,
                "twibot_zip_sha256": checksum(data_archive),
                "model": UPDATE_MODEL if updates is not None else MODEL,
                "model_artifact_sha256": checksum(model_path),
                "packages": {package: version(package) for package in ("numpy", "scipy", "scikit-learn")},
                "features": NAMES,
                "identification_rules": IDENTIFICATION_RULES,
                "labeled_batches": batches,
                "focused_comparison_features": FOCUSED,
                "maximum_tweets_scanned": SCAN_LIMIT,
                "maximum_eligible_tweets": TWEET_LIMIT,
                "minimum_eligible_tweets": MIN_TWEETS,
                "pca_components": COMPONENTS,
                "explained_variance_ratio": model.variance,
                "knn_neighbors": NEIGHBORS,
                "knn_weights": "distance",
                "fold_seed": FOLD_SEED,
                "calibration_seed": CALIBRATION_SEED,
                "calibration_share": CALIBRATION_SHARE,
                "development_accounts": len(ids),
                "development_bots": int(labels.sum()),
                "development_with_measured_score": int(np.isfinite(values).all(axis=1).sum()),
                "release_fit_accounts": len(fit),
                "release_knn_reference_accounts": len(model.labels),
                "release_calibration_accounts": len(calibration),
                "release_calibration_humans": int((~labels[calibration]).sum()),
                "release_calibration": {
                    name: {"target_human_fpr": budget,
                           "threshold": cutoff if np.isfinite(cutoff) else None,
                           **decision_metrics(confusion(labels[calibration], threshold_flags(scores, available, cutoff)))}
                    for name, budget, cutoff in (("strict", STRICT, strict), ("screening", SCREENING, screening))
                },
                "development": comparisons,
                "test_labels_used": False,
            }
            if held_out:
                restored, strict, screening = FittedKNN.load(model_path)
                metadata["test_predictions"] = test_predictions(
                    archive, ids, restored, strict, screening, stage / "test_predictions.csv"
                )
            (stage / "run_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-archive", required=True, type=Path, help="ML_Final_Project-main.zip")
    parser.add_argument("--archive", required=True, type=Path, help="TwiBot-20 data ZIP")
    parser.add_argument("--output", required=True, type=Path, help="new release directory")
    parser.add_argument("--held-out", action="store_true", help="export test predictions without test labels")
    parser.add_argument("--labeled-batch", type=Path, action="append", default=[],
                        help="add a disjoint, verified bot/human-labeled group of accounts")
    args = parser.parse_args()
    try:
        build(args.ml_archive, args.archive, args.output, args.held_out, args.labeled_batch)
    except (BadZipFile, KeyError, OSError, SyntaxError, ValueError) as error:
        parser.exit(1, f"PCA + KNN build failed: {error}\n")


if __name__ == "__main__":
    main()
