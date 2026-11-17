# Twitter bot detection with PCA and KNN

## Result

The model uses a timeline to score whether its **account** is automated. It standardizes 14 tweet descriptors, reduces them to eight principal components, and uses the labels of 35 nearby training accounts to score new accounts. PCA learns the representation from the tweet descriptors. The KNN vote uses bot and human labels, making the combined model a supervised classifier. The project's [course PCA feature](README.md) remains available for the group's unsupervised contribution.

On 8,278 matched development accounts, the combined model averaged **0.7433 average precision** and **0.6994 ROC AUC**. At the strict threshold it detected 355 bots and falsely flagged 38 humans. At the broader screening threshold it detected 1,595 bots and flagged 409 humans for review. The tradeoff appears in the tables below. The official test export contains scores and review bands with the labels reserved for the group's final comparison.

## How we reached this model

| Stage | What we tested | What the result showed |
| --- | --- | --- |
| Historical V1 | Semantic KNN measuring repeated meaning within an account | Average precision 0.5911 and ROC AUC 0.5104. Repetition in meaning alone provided limited separation. [V1 handoff](README-v1.md) |
| Course PCA | A signed direction over standardized, directly measured posting and writing features | Average precision 0.6627 and ROC AUC 0.5982 on the original folds. The released direction was strongest on retweets, punctuation, and exact template repetition. |
| Expanded PCA trial | Emoji-like symbols, em dashes, capitalization, punctuation, and link/retweet sequencing | Average precision 0.6593 and ROC AUC 0.5947. The larger feature set produced closely matched tradeoffs at the tested budgets. [Earlier paired audit](EVALUATION.md) |
| Supervised PCA + KNN | A focused six-cue model and a 14-descriptor model, both training on the supplied account labels | The wider model ranked accounts better and caught more bots at the tested thresholds. The focused model achieved zero observed false flags at its strictest point and detected just 27 bots. |

The supervised comparison below uses a fresh fit and calibration partition inside each original development fold. Its course-PCA baseline therefore has slightly different metrics from the earlier course-PCA audit.

## Initial rules and data provenance

The bot-identification write-up guided which quantities to measure. Training evidence came from the supplied account labels. A declared automated weather feed can be a positive example of automation, and the target is independent of whether the account is useful or harmful. Prior account-level bot research similarly combines content, retweet behavior, timing, and profile information. [TwiBot-20 schema](https://github.com/bunsenfeng/twibot-20#twibot-20-sample) · [Primary bot-detection study](https://ojs.aaai.org/index.php/icwsm/article/view/14871)

| Identification signal | Use in this model |
| --- | --- |
| Amplification | Fraction of eligible tweets containing a retweet marker, link, or hashtag. |
| Repetition and narrow wording | Exact template reuse after removing links and normalizing numbers, nearest character four-gram overlap within the account, and word diversity. |
| Conversation and style | Mention-prefix rate, personal and function words, questions, punctuation, and word and post length. These complement the six focused indicators. |
| Extreme posting frequency or timed bursts | This requires per-tweet timestamps and comparable observation windows. The supplied tweet table gives text and tweet order. |
| Avatars, bios, and follower ratios | These belong to the separate account-profile model. |

The program opens **ML_Final_Project-main.zip** and checks the label mapping in `conversion.ipynb` as source text. The notebook maps 0 to human and 1 to bot, preserves the account ID, and emits one row per tweet as `ID,tweet_idx,text,account_type`. The runner streams the supplied **TwiBot-20 data ZIP** for the cleaned user and tweet tables. It reads labels once per training account from `train_users_cleaned.csv`. Each tweet is matched to its account by the original string ID. [Archive provenance and checksums](SOURCE-MANIFEST.md)

## Fit and validation

1. Scan up to 96 records per account by tweet index and summarize up to 24 tweets passing the deterministic English-text filter. An account needs at least eight eligible tweets for a measured score. Sparse accounts remain in metric denominators and receive an `insufficient_text` review band.
2. Recreate the five account-stratified folds with seed 6030. Inside each fold, fit on 75% of its training accounts and reserve 25% for threshold calibration. Both groups are disjoint from the outer validation accounts.
3. Standardize the descriptors using the fit partition, fit PCA and retain eight components, then fit distance-weighted Euclidean KNN with 35 neighbors. The focused comparison retains five components from the six suggested cues. The full release’s eight components explain 87.8% of the standardized training variance.
4. For each false-positive budget, choose the calibration threshold that catches the most bots within that budget. Apply it once to the outer validation accounts. Report precision, recall, F1, human false-positive rate, and raw confusion counts. Average precision and ROC AUC are means across the five validation folds. Decision counts are pooled. [Threshold calibration method](https://scikit-learn.org/stable/modules/classification_threshold.html)

| Model | Mean average precision | Mean ROC AUC |
| --- | ---: | ---: |
| Course PCA, with the same fit/calibration split | 0.6624 | 0.5993 |
| Six suggested cues + PCA + KNN | 0.6878 | 0.6288 |
| **14 descriptors + PCA + KNN** | **0.7433** | **0.6994** |

| Calibration budget | Model | Observed human FPR | Bot precision | Bot recall | Bots caught | Humans flagged |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1% | Course PCA | 1.18% | 77.2% | 3.1% | 146 | 43 |
| 1% | Six cues + PCA + KNN | 0.91% | 82.8% | 3.4% | 159 | 33 |
| **1%** | **Full PCA + KNN** | **1.05%** | **90.3%** | **7.6%** | **355** | **38** |
| 10% | Course PCA | 10.35% | 73.8% | 22.8% | 1,058 | 376 |
| 10% | Six cues + PCA + KNN | 9.39% | 77.2% | 24.8% | 1,152 | 341 |
| **10%** | **Full PCA + KNN** | **11.26%** | **79.6%** | **34.3%** | **1,595** | **409** |

At a 1% calibration budget, the full model added **209 bot detections while flagging five fewer humans** than course PCA. The broader setting increases coverage, with a correspondingly larger human review queue. At a 20% calibration budget, the full model caught 49.8% of bots and produced an observed human FPR of 22.0%. Observed FPR can exceed the calibration budget on new accounts. Precision also depends on the bot prevalence in the evaluated population. The development population was 56.1% bots. [Per-fold decisions and metrics](WORKING/twibot20-supervised-pca-knn/development_metrics.csv)

## Saved release and interpretation

The saved model fits on 6,208 development accounts, of which 5,566 have enough text for the KNN reference. The other 2,070 development accounts set the release thresholds. The strict cutoff is 0.8692 and the screening cutoff is 0.6985. These are neighbor vote scores. Treat the resulting bands as priority for review.

- **High priority** meets the strict threshold.
- **Review** meets the screening threshold.
- **Below screening threshold** has a measured score below both thresholds.
- **Insufficient text** has too few eligible posts and carries the fit population's bot-rate fallback for ranking.

The original 1,183 test accounts have one score each. The exported queue has 61 high-priority accounts, 203 additional review accounts, and 142 with insufficient text. Those counts describe model outputs. Official test precision and recall require the reserved labels after the team freezes the final comparison.

## How the model adapts

Use verified new accounts with `bot` or `human` labels. Useful additions include human-run news, sports, or event accounts whose posts resemble automated updates, and verified bots that write varied or conversational posts. Those examples let KNN learn where the initial cues overlap. Keep the labels grounded in evidence about the accounts. The model refits scaling, PCA, and the neighbor reference from the enlarged labeled set, then recalibrates both thresholds and compares the update on the same development accounts. A new run records the batch hashes and the original and updated precision/recall results. This makes it possible to keep the older release if an added batch makes the tradeoff worse.

In a development-only simulation, each fold used 70% of its fit partition as the starting reference and then added the other 30% as newly labeled examples. The calibration and validation accounts stayed fixed. Average precision rose from **0.7407 to 0.7433**. At the 10% screening calibration budget, precision rose from **78.4% to 79.6%**, recall from **33.6% to 34.3%**, and human false alerts fell from **431 to 409**. At the 1% calibration budget, false alerts fell from **43 to 38** and precision rose from **89.8% to 90.3%**, while detections fell from **378 to 355**. The simulation uses labels from the supplied archive. An independent update batch may have a different effect.

Additional account batches use CSV columns `ID,tweet_idx,text,account_type`, grouped by original decimal account ID. Supply a consistent verified label on every row for each account. Use accounts outside the original train/test archive and any other batch. For each release, pass all labeled batches intended to be in its reference set. Compare precision and recall at the same false-positive budget before selecting it for the team model. A separately labeled set from the intended population would support an external check.

## Files and reproduction

- [Model entry point](src/tweet_pca_knn.py) and [tweet feature extraction](src/tweet_patterns.py)
- [Fitted model](WORKING/twibot20-supervised-pca-knn/pca_knn_model.npz) and [run manifest](WORKING/twibot20-supervised-pca-knn/run_manifest.json)
- [Held-out scores and review bands](WORKING/twibot20-supervised-pca-knn/test_predictions.csv)

Run from the project folder with Python 3.14 and the pinned local dependencies.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/tweet_pca_knn.py \
  --ml-archive '<path-to-ml-project-zip>' \
  --archive '<path-to-twibot-zip>' \
  --output WORKING/twibot20-supervised-pca-knn-rerun --held-out
```

For an update, use a fresh output directory and add `--labeled-batch /path/to/verified-accounts.csv`. Repeat the flag for additional batches. Use the development results to review the update before requesting held-out scores. The model file stores the fitted transforms, thresholds, and numerical neighbor reference for local use.
