# DS 6021 tweet-pattern PCA

This is the course model for one additional account-level predictor. It uses the PCA method listed in the Fall 2026 DS 6021 syllabus, Week 13. The [original V1 handoff](README-v1.md) documents the historical semantic-KNN comparison.

The separately trained [supervised PCA + KNN model and full report](PCA-KNN-REPORT.md) use the supplied bot and human labels and support labeled updates.

## How the feature works

The scorer examines the first 96 timeline records by tweet index. It uses up to 24 posts that pass a deterministic English-text heuristic based on common words. It measures repeated wording, retweet and link rates, and other writing characteristics. Twitter URLs and handles are stripped from the text used for wording comparisons. The scorer keeps all account IDs and uses the training median when an account has fewer than eight eligible posts.

The final fit standardizes 14 numeric tweet features and projects them onto PCA component 3. This signed coordinate is the exported `tweet_pca_score`. In the full training fit, its strongest positive loadings were retweet rate, punctuation rate, and exact-template repetition. The loadings describe a posting pattern. The teammate’s profile classifier estimates how that pattern relates to bot labels.

The feature fit uses numeric tweet descriptors without bot labels. During development, inner cross-validation uses training labels to select one of the first three components and its orientation from three fixed feature groups. Each outer validation fold is scored with the recipe selected within that fold's training data. The archive's test labels are unused.

## Results and team handoff

See the [paired evaluation](EVALUATION.md) for bot precision, recall, human false alerts, and the expanded emoji and posting-pattern experiment.

| Matched comparison | Mean average precision | Mean ROC AUC |
| --- | ---: | ---: |
| Historical V1 | 0.5911 | 0.5104 |
| PCA with nested development selection | **0.6627** | **0.5982** |
| Frozen recipe refitted on the development folds | 0.6643 | 0.5961 |

The nested comparison measures the selection procedure on the five V1 validation folds. The frozen-recipe row checks the chosen recipe on development data already used for selection. The nested procedure and frozen recipe both improved average precision over V1 in all five folds. These results measure the tweet score alone. [Per-fold metrics](WORKING/twibot20-course-pca/development_metrics.csv)

- Join the [development features](WORKING/twibot20-course-pca/cv_features.csv) one-to-one by string `ID` within a fold and role. Use that fold's `train` rows to fit and its `validation` rows to compare the same profile-model recipe with and without `tweet_pca_score`.
- Use [final train features](WORKING/twibot20-course-pca/train_features.csv) to fit the frozen recipe and [test features](WORKING/twibot20-course-pca/test_features.csv) for the group's final holdout comparison. Fit any classifier thresholds on development data. Use only `tweet_pca_score` as the added predictor.
- The archived timeline count `n_tweets` is derived from the collected tweets. Use the original profile field `statuses_count` if the profile baseline needs lifetime post count. [Run provenance and loadings](WORKING/twibot20-course-pca/run_manifest.json)

## Reproduce locally

Run from this project folder using Python 3.14. The course model reads the supplied archive and the V1 fold definitions.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -p test_tweet_pca.py
.venv/bin/python src/tweet_pca.py \
  --archive '<path-to-twibot-zip>' \
  --output WORKING/twibot20-course-pca-rerun --held-out
```

Omit `--held-out` to generate development and training exports alone.
