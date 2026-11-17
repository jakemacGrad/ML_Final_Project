"""Audit paired development folds and training-selected decision thresholds."""

import argparse
import csv
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from tweet_patterns import ACTIVITY_PROXY_NAMES, EXTENDED_NAMES, STYLE_NAMES, extract
from tweet_pca import GROUPS, development, profiles, read_reference, write_csv

MODELS = ("v1_knn", "course_pca", "style_pca")
POLICIES = ("max_f1", "train_fpr_cap_10pct", "train_fpr_cap_20pct", "train_recall_floor_80pct")
COMPARISONS = (("course_pca", "v1_knn"), ("style_pca", "course_pca"))
BOOTSTRAP_SAMPLES = 1000
SEED = 6021


def keyed_scores(path, column):
    values = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["fold"]), row["role"], row["ID"])
            score = float(row[column])
            if key in values or not np.isfinite(score):
                raise ValueError(f"Duplicate key or invalid score in {path.name}.")
            values[key] = score
    return values


def threshold_choices(labels, scores):
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("Decision thresholds require both training classes.")
    fpr, recall, cutoffs = roc_curve(labels, scores, drop_intermediate=False)
    # Integer confusion counts keep tied-score cutoffs at the intended budget.
    tp = np.rint(recall * positives).astype(int)
    fp = np.rint(fpr * negatives).astype(int)
    f1 = 2 * tp / (positives + tp + fp)
    precision = np.divide(tp, tp + fp, out=np.zeros(len(tp), dtype=float), where=(tp + fp) > 0)
    strict = np.flatnonzero(10 * fp <= negatives)
    capped = np.flatnonzero(5 * fp <= negatives)
    recall_floor = np.flatnonzero(5 * tp >= 4 * positives)
    return (
        cutoffs[int(np.argmax(f1))],
        cutoffs[strict[int(np.argmax(tp[strict]))]],
        cutoffs[capped[int(np.argmax(tp[capped]))]],
        cutoffs[recall_floor[int(np.argmax(precision[recall_floor]))]],
    )


def confusion(labels, decisions):
    positive = decisions.astype(bool)
    return np.array([
        ((~labels) & (~positive)).sum(),
        ((~labels) & positive).sum(),
        (labels & (~positive)).sum(),
        (labels & positive).sum(),
    ], dtype=int)


def decision_metrics(counts):
    tn, fp, fn, tp = map(int, counts)

    def share(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    recall = share(tp, tp + fn)
    specificity = share(tn, tn + fp)
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "bot_precision": share(tp, tp + fp),
        "bot_recall": recall,
        "bot_f1": share(2 * tp, 2 * tp + fp + fn),
        "human_precision": share(tn, tn + fn),
        "human_recall": specificity,
        "human_f1": share(2 * tn, 2 * tn + fp + fn),
        "false_positive_rate": share(fp, fp + tn),
        "balanced_accuracy": (recall + specificity) / 2,
        "mcc": share(tp * tn - fp * fn, ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5),
    }


def rank_metrics(labels, scores):
    if len(np.unique(labels)) < 2:
        return None
    return {
        "ap": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def assemble_folds(scores, truth):
    known_keys = set(scores[MODELS[0]])
    if any(set(scores[model]) != known_keys for model in MODELS[1:]):
        raise ValueError("All models must use identical fold, role, and account keys.")
    fold_roles = defaultdict(lambda: defaultdict(list))
    for fold, role, account in sorted(known_keys):
        if role not in {"train", "validation"}:
            raise ValueError("Unexpected fold role.")
        fold_roles[fold][role].append(account)
    if set(fold_roles) != set(range(1, 6)):
        raise ValueError("Expected the five original folds.")

    seen = set()
    for fold, roles in sorted(fold_roles.items()):
        train, validation = roles["train"], roles["validation"]
        if set(train) & set(validation) or set(train) | set(validation) != truth.keys():
            raise ValueError("Invalid partition of development accounts.")
        if seen.intersection(validation):
            raise ValueError("An account is validated more than once.")
        seen.update(validation)
        train_labels = np.array([truth[account] for account in train], dtype=bool)
        labels = np.array([truth[account] for account in validation], dtype=bool)
        ranked, predictions, thresholds = {}, {}, {}
        for model in MODELS:
            train_scores = np.array([scores[model][fold, "train", account] for account in train])
            ranked[model] = np.array([scores[model][fold, "validation", account] for account in validation])
            thresholds[model] = threshold_choices(train_labels, train_scores)
            predictions[model] = np.stack([ranked[model] >= cutoff for cutoff in thresholds[model]])
        yield fold, validation, labels, ranked, predictions, thresholds
    if seen != truth.keys():
        raise ValueError("Some accounts have no validation score.")


def interval(samples):
    return list(map(float, np.percentile(samples, [2.5, 97.5])))


def paired_intervals(folds):
    rng = np.random.default_rng(SEED)
    ranking, decisions = defaultdict(list), defaultdict(list)
    strata = [
        tuple(np.flatnonzero(labels == label) for label in (False, True))
        for _, _, labels, _, _, _ in folds
    ]

    for _ in range(BOOTSTRAP_SAMPLES):
        ap = defaultdict(list)
        auc = defaultdict(list)
        counts = np.zeros((len(MODELS), len(POLICIES), 4), dtype=int)
        for (_, _, labels, ranked, predicted, _), groups in zip(folds, strata, strict=True):
            selected = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
            truth = labels[selected]
            for model_index, model in enumerate(MODELS):
                metrics = rank_metrics(truth, ranked[model][selected])
                ap[model].append(metrics["ap"])
                auc[model].append(metrics["roc_auc"])
                for policy_index, pred in enumerate(predicted[model]):
                    counts[model_index, policy_index] += confusion(truth, pred[selected])
        for candidate, baseline in COMPARISONS:
            contrast = f"{candidate}_minus_{baseline}"
            ranking[contrast, "ap"].append(np.mean(ap[candidate]) - np.mean(ap[baseline]))
            ranking[contrast, "roc_auc"].append(np.mean(auc[candidate]) - np.mean(auc[baseline]))
            for i, policy in enumerate(POLICIES):
                a = decision_metrics(counts[MODELS.index(candidate), i])
                b = decision_metrics(counts[MODELS.index(baseline), i])
                for metric in ("bot_precision", "bot_recall", "bot_f1", "false_positive_rate"):
                    decisions[contrast, policy, metric].append(a[metric] - b[metric])

    return {
        contrast: {
            "ranking": {metric: interval(ranking[contrast, metric]) for metric in ("ap", "roc_auc")},
            "decisions": {
                policy: {metric: interval(decisions[contrast, policy, metric])
                         for metric in ("bot_precision", "bot_recall", "bot_f1", "false_positive_rate")}
                for policy in POLICIES
            },
        }
        for contrast in (f"{candidate}_minus_{baseline}" for candidate, baseline in COMPARISONS)
    }


def paired_swaps(folds):
    result = {}
    for candidate, baseline in COMPARISONS:
        changes = defaultdict(lambda: np.zeros(4, dtype=int))
        for _, _, truth, _, predictions, _ in folds:
            for index, policy in enumerate(POLICIES):
                gained = predictions[candidate][index] & ~predictions[baseline][index]
                lost = predictions[baseline][index] & ~predictions[candidate][index]
                changes[policy] += (
                    (truth & gained).sum(), (truth & lost).sum(),
                    (~truth & gained).sum(), (~truth & lost).sum(),
                )
        fields = ("additional_bots_detected", "previously_detected_bots_missed",
                  "additional_humans_flagged", "previously_flagged_humans_cleared")
        result[f"{candidate}_minus_{baseline}"] = {
            policy: dict(zip(fields, map(int, change), strict=True)) for policy, change in changes.items()
        }
    return result


def coverage_slices(folds, shared_measured):
    slices = {}
    for name, include in (
        ("both_measured", lambda account: account in shared_measured),
        ("either_fallback", lambda account: account not in shared_measured),
    ):
        by_model = defaultdict(list)
        total, bots = 0, 0
        for _, ids, labels, ranked, _, _ in folds:
            mask = np.array([include(account) for account in ids])
            total += int(mask.sum())
            bots += int(labels[mask].sum())
            for model in MODELS:
                metrics = rank_metrics(labels[mask], ranked[model][mask])
                if metrics:
                    by_model[model].append(metrics)
        slices[name] = {"accounts": total, "bots": bots, "ranking": macro_ranking(by_model)}
    return slices


def macro_ranking(by_model):
    return {
        model: {metric: float(np.mean([fold[metric] for fold in metrics])) for metric in ("ap", "roc_auc")}
        for model, metrics in by_model.items()
    }


def cue_ablation(ids, patterns, labels, reference):
    folds = list(read_reference(reference, ids))
    result = {}
    for name, extras in (("typography_only_added", STYLE_NAMES), ("sequence_only_added", ACTIVITY_PROXY_NAMES)):
        groups = {group: fields + extras for group, fields in GROUPS.items()}
        runs, _ = development(ids, patterns, labels, folds, groups=groups, feature_names=EXTENDED_NAMES)
        result[name] = {
            "ap": float(np.mean([row[5] for row in runs])),
            "roc_auc": float(np.mean([row[7] for row in runs])),
            "folds": [
                {"fold": fold, "group": group, "component": axis, "sign": sign, "ap": ap, "roc_auc": auc}
                for fold, group, axis, sign, _, ap, _, auc in runs
            ],
        }
    return result


def audit(archive, v1_dir, pca_dir, style_dir, output):
    scores = {
        "v1_knn": keyed_scores(v1_dir / "cv_features.csv", "tweet_knn_score"),
        "course_pca": keyed_scores(pca_dir / "cv_features.csv", "tweet_pca_score"),
        "style_pca": keyed_scores(style_dir / "cv_features.csv", "tweet_pca_score"),
    }
    with ZipFile(archive) as source:
        truth = {account: label == "bot" for account, label in profiles(source, "train").items()}
        ids = sorted(truth)
        patterns = extract(source, ids, extended=True)
        coverage = np.isfinite(patterns).all(axis=1)
    pca_measured = {account for account, measured in zip(ids, coverage, strict=True) if measured}
    with (v1_dir / "tweet_coverage.csv").open(newline="", encoding="utf-8") as stream:
        v1_measured = {row["ID"] for row in csv.DictReader(stream)
                       if row["split"] == "train" and row["score_available"] == "1"}
    folds = list(assemble_folds(scores, truth))
    totals = defaultdict(lambda: np.zeros(4, dtype=int))
    by_model = defaultdict(list)
    fold_decisions = []
    for fold, _, labels, ranked, predictions, thresholds in folds:
        for model in MODELS:
            by_model[model].append(rank_metrics(labels, ranked[model]))
            for policy, pred, threshold in zip(POLICIES, predictions[model], thresholds[model], strict=True):
                counts = confusion(labels, pred)
                totals[model, policy] += counts
                fold_decisions.append({"fold": fold, "model": model, "policy": policy,
                                       "train_threshold": float(threshold), **decision_metrics(counts)})

    outcome = np.array(list(truth.values()))
    labels_by_id = np.array([truth[account] for account in ids], dtype=bool)
    cue_prevalence = {}
    for cue in ("emoji_post_rate", "em_dash_post_rate"):
        cue_values = patterns[:, EXTENDED_NAMES.index(cue)]
        cue_prevalence[cue] = {}
        for label, label_mask in (("bot", labels_by_id), ("human", ~labels_by_id)):
            observed = cue_values[coverage & label_mask]
            cue_prevalence[cue][label] = {
                "measured_accounts": len(observed),
                "accounts_with_cue": int((observed > 0).sum()),
                "mean_account_rate": float(observed.mean()) if len(observed) else None,
            }
    summary = {
        "accounts": len(outcome),
        "bots": int(outcome.sum()),
        "humans": int((~outcome).sum()),
        "v1_measured": len(v1_measured),
        "pca_measured": len(pca_measured),
        "ranking": macro_ranking(by_model),
        "decisions": {
            model: {policy: decision_metrics(totals[model, policy]) for policy in POLICIES}
            for model in MODELS
        },
        "always_flag_bot": decision_metrics(confusion(outcome, np.ones(len(outcome), dtype=bool))),
        "coverage_slices": coverage_slices(folds, v1_measured & pca_measured),
        "cue_prevalence": cue_prevalence,
        "cue_ablation": cue_ablation(ids, patterns, labels_by_id, v1_dir / "cv_features.csv"),
        "paired_account_swaps": paired_swaps(folds),
        "paired_bootstrap_95pct": paired_intervals(folds),
        "bootstrap_replicates": BOOTSTRAP_SAMPLES,
    }
    fields = list(fold_decisions[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "audit"
        stage.mkdir()
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        write_csv(stage / "fold_decisions.csv", fields, ([row[field] for field in fields] for row in fold_decisions))
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("archive", "v1-dir", "pca-dir", "style-dir", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory.")
    audit(args.archive, args.v1_dir, args.pca_dir, args.style_dir, args.output)


if __name__ == "__main__":
    main()
