"""Build a DS 6021 tweet component using feature standardization and PCA."""

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from tweet_patterns import NAMES, csv_rows, extract

FEATURE = "tweet_pca_score"
SEED = 6021
GROUPS = {
    "all_patterns": NAMES,
    "writing_style": (
        "mention_prefix_rate", "mean_length", "punctuation_rate", "personal_word_rate",
        "question_rate", "function_word_rate", "mean_word_length",
    ),
    "posting_patterns": (
        "retweet_rate", "link_rate", "hashtag_rate", "template_reuse", "exact_template_rate",
        "word_diversity", "length_cv",
    ),
}
COMPONENTS = 3


def fit_patterns(values, group, groups=GROUPS, feature_names=NAMES):
    indexes = [feature_names.index(name) for name in groups[group]]
    usable = values[np.isfinite(values).all(axis=1)][:, indexes]
    if len(usable) < COMPONENTS:
        raise ValueError("Too few accounts with usable tweet text for PCA.")
    scale = StandardScaler().fit(usable)
    standardized = scale.transform(usable)
    pca = PCA(n_components=COMPONENTS, svd_solver="full").fit(standardized)
    for weights in pca.components_:
        if weights[np.argmax(np.abs(weights))] < 0:
            weights *= -1
    fallback = np.median(pca.transform(standardized), axis=0)
    return indexes, scale, pca, fallback


def project(values, fitted):
    indexes, scale, pca, fallback = fitted
    coverage = np.isfinite(values).all(axis=1)
    scores = np.tile(fallback, (len(values), 1))
    if coverage.any():
        scores[coverage] = pca.transform(scale.transform(values[coverage][:, indexes]))
    return scores


def component_scores(values, candidate, fitted):
    _, component, sign = candidate
    return sign * project(values, fitted)[:, component]


def select_candidate(values, labels, groups=GROUPS, feature_names=NAMES):
    """Rank fixed PCA projections with inner validation labels only."""
    splits = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    metrics = {
        (group, component, sign): []
        for group in groups for component in range(COMPONENTS) for sign in (1, -1)
    }
    for train, validation in splits.split(values, labels):
        for group in groups:
            fitted = fit_patterns(values[train], group, groups, feature_names)
            projections = project(values[validation], fitted)
            for component in range(COMPONENTS):
                for sign in (1, -1):
                    scores = sign * projections[:, component]
                    metrics[group, component, sign].append(
                        average_precision_score(labels[validation], scores)
                    )
    return max(metrics, key=lambda candidate: np.mean(metrics[candidate]))


def write_csv(path, headers, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def read_reference(path, ids):
    positions = dict(zip(ids, range(len(ids)), strict=True))
    roles = {fold: {"train": {}, "validation": {}} for fold in range(1, 6)}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            group = roles[int(row["fold"])][row["role"]]
            index = positions[row["ID"]]
            if index in group:
                raise ValueError("Duplicate account in a version 1 fold role.")
            group[index] = float(row["tweet_knn_score"])
    seen = set()
    for fold, role in roles.items():
        train, validation = role["train"], role["validation"]
        if train.keys() & validation.keys() or len(train) + len(validation) != len(ids):
            raise ValueError("Invalid version 1 train or validation partition.")
        if seen.intersection(validation):
            raise ValueError("An account is in multiple version 1 validation folds.")
        seen.update(validation)
        indices = sorted(validation)
        yield fold, sorted(train), indices, [validation[index] for index in indices]
    if seen != set(range(len(ids))):
        raise ValueError("Historic folds do not validate every development account exactly once.")


def development(ids, values, labels, folds, frozen=None, groups=GROUPS, feature_names=NAMES):
    rows, comparisons = [], []
    for fold, train, validation, baseline in folds:
        candidate = select_candidate(values[train], labels[train], groups, feature_names)
        fitted = fit_patterns(values[train], candidate[0], groups, feature_names)
        validation_scores = component_scores(values[validation], candidate, fitted)
        actual = labels[validation]
        comparisons.append((
            fold, candidate[0], candidate[1] + 1, candidate[2],
            average_precision_score(actual, baseline), average_precision_score(actual, validation_scores),
            roc_auc_score(actual, baseline), roc_auc_score(actual, validation_scores),
        ))
        if frozen is not None:
            frozen_fit = (
                fitted if frozen[0] == candidate[0]
                else fit_patterns(values[train], frozen[0], groups, feature_names)
            )
            frozen_scores = component_scores(values[validation], frozen, frozen_fit)
            comparisons[-1] += (
                average_precision_score(actual, frozen_scores), roc_auc_score(actual, frozen_scores),
            )
        for role, index, scores in (
            ("train", train, component_scores(values[train], candidate, fitted)),
            ("validation", validation, validation_scores),
        ):
            rows.extend((fold, role, ids[i], score) for i, score in zip(index, scores, strict=True))
    return comparisons, rows


def profiles(archive, split):
    label = split == "train"
    names = ("ID", "account_type") if label else ("ID",)
    accounts = {}
    for row in csv_rows(archive, f"{split}_users_cleaned.csv", names):
        account = (row["ID"] or "").strip()
        if not account or account in accounts:
            raise ValueError(f"Blank or duplicate account in {split} profiles.")
        outcome = row["account_type"] if label else None
        if label and outcome not in {"human", "bot"}:
            raise ValueError("Invalid label in training profiles.")
        accounts[account] = outcome
    return accounts


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(archive_path, reference, out, held_out, *, extended=False):
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    groups, feature_names = GROUPS, NAMES
    if extended:
        from tweet_patterns import ACTIVITY_PROXY_NAMES, EXTENDED_NAMES, STYLE_NAMES

        groups = {
            "all_patterns": EXTENDED_NAMES,
            "writing_style": GROUPS["writing_style"] + STYLE_NAMES,
            "posting_patterns": GROUPS["posting_patterns"] + ACTIVITY_PROXY_NAMES,
        }
        feature_names = EXTENDED_NAMES
    with ZipFile(archive_path) as archive, threadpool_limits(limits=2):
        labels_by_id = profiles(archive, "train")
        ids = sorted(labels_by_id)
        labels = np.array([labels_by_id[account] == "bot" for account in ids])
        values = extract(archive, ids, extended=extended)
        folds = list(read_reference(reference, ids))
        candidate = select_candidate(values, labels, groups, feature_names)
        comparisons, fold_rows = development(ids, values, labels, folds, candidate, groups, feature_names)
        fitted = fit_patterns(values, candidate[0], groups, feature_names)
        train_scores = component_scores(values, candidate, fitted)
        coverage = np.isfinite(values).all(axis=1)
        _, _, pca, fallback = fitted
        component = candidate[1]

        metadata = {
            "schema_version": 1,
            "archive_sha256": checksum(archive_path),
            "v1_folds_sha256": checksum(reference),
            "inner_cv_seed": SEED,
            "group": candidate[0],
            "component": component + 1,
            "sign": candidate[2],
            "explained_variance_ratio": float(pca.explained_variance_ratio_[component]),
            "loadings": dict(zip(groups[candidate[0]], map(float, candidate[2] * pca.components_[component]), strict=True)),
            "fallback": float(candidate[2] * fallback[component]),
            "train_accounts": len(ids),
            "train_accounts_with_measured_score": int(coverage.sum()),
            "held_out_features_exported": held_out,
        }
        if extended:
            metadata["extended_descriptors"] = feature_names[len(NAMES):]

        with tempfile.TemporaryDirectory(dir=out.parent) as temp:
            stage = Path(temp) / "features"
            stage.mkdir()
            write_csv(stage / "cv_features.csv", ("fold", "role", "ID", FEATURE), fold_rows)
            write_csv(stage / "development_metrics.csv", (
                "fold", "group", "component", "sign", "v1_ap", "pca_ap", "v1_roc_auc", "pca_roc_auc",
                "frozen_recipe_ap", "frozen_recipe_roc_auc",
            ), comparisons)
            write_csv(stage / "train_features.csv", ("ID", FEATURE), zip(ids, train_scores, strict=True))
            if held_out:
                test_ids = sorted(profiles(archive, "test"))
                if labels_by_id.keys() & set(test_ids):
                    raise ValueError("Training and test profile IDs overlap.")
                test_values = extract(archive, test_ids, "test", extended=extended)
                test_scores = component_scores(test_values, candidate, fitted)
                write_csv(stage / "test_features.csv", ("ID", FEATURE), zip(test_ids, test_scores, strict=True))
                metadata["test_accounts"] = len(test_ids)
                metadata["test_accounts_with_measured_score"] = int(np.isfinite(test_values).all(axis=1).sum())
            (stage / "run_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            stage.rename(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--reference-folds", type=Path, default=Path("WORKING/twibot20-knn/cv_features.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--held-out", action="store_true", help="also export frozen test scores")
    parser.add_argument("--extended", action="store_true", help="test additional style and post-sequence descriptors")
    args = parser.parse_args()
    try:
        build(args.archive, args.reference_folds, args.output, args.held_out, extended=args.extended)
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"Tweet PCA build failed: {error}\n")


if __name__ == "__main__":
    main()
