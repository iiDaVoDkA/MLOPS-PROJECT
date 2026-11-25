# Experiments

Tracking server and Model Registry: https://dagshub.com/iiDaVoDkA/MLOPS-PROJECT.mlflow
(experiment `sms-spam`, registered model `sms-spam-classifier`).

## Primary metric

**`f1_spam`: F1 score of the `spam` class on the held-out evaluation set. Higher is better.**

The app tells a user whether a message is spam. Both kinds of error cost something: a missed spam
message reaches the user (low recall), and a legitimate message flagged as spam may be ignored (low
precision). Only about 13% of messages are spam, so accuracy is misleading: a model that answers
"ham" every time is 87% accurate and useless. F1 on the spam class balances the two errors and
ignores the easy majority class.

Flagging a real message is the costlier mistake for the user, so precision also has its own
guardrail below.

## Quality gates

Defined in [`gates.yaml`](gates.yaml), committed in `922820f` before any model was trained
(the first training commit is `fbd4e23`).

| Gate | Value | Meaning |
|---|---|---|
| `primary_metric` | `f1_spam` | compared on the same 991 held-out messages for champion and candidate |
| `greater_is_better` | `true` | |
| `quality_floor` | `0.85` | a candidate below this is never promoted, even with no champion |
| `min_improvement` | `0.005` | a candidate must beat the champion by at least this much; a tie is rejected |
| `guardrails.precision_spam_min` | `0.90` | a candidate with lower spam precision is rejected whatever its F1 |
| `retrain.max_f1_drop` | `0.03` | the CT pipeline retrains when the champion's F1 falls this far below the value recorded at approval |

The CT pipeline (`ml/pipeline.py`) scores the champion first and retrains only when the data
version is newer than the champion's, the champion is below the quality floor, or its F1 dropped
by at least `max_f1_drop`. Otherwise it keeps the champion without training
(`retain_no_retraining`). The retraining rule was added in its own commit before any CT run.

The rule is the pure function `should_promote` in [`promote.py`](promote.py). Checks run in this
order: valid score, quality floor, guardrails, improvement over the champion. Anything that cannot
be verified (a missing guardrail metric, a non-numeric score) is a rejection, and the champion is
retained. [`train.py`](train.py) applies the rule and logs the decision and its reason on the run
(`promotion_decision`, `promotion_reason`, `champion_version_before`, `registered_version`).

## How a run is produced

```bash
python -m ml.train --data-version v1 --run-name baseline-v1
```

Each run logs:

- **Provenance:** `git_sha`, `git_dirty`, `dataset_version`, `dvc_hash` (tags and parameters).
  Training refuses to register a model from uncommitted code.
- **Hyperparameters:** `C`, `class_weight`, `ngram_max`, `min_df`, `seed`.
- **Metrics:** `cv_f1_spam_mean` / `cv_f1_spam_std` (5-fold cross-validation on the training
  split) and `f1_spam`, `precision_spam`, `recall_spam`, `accuracy` on the held-out set. When a
  champion exists, its scores on the same held-out set are logged as `champion_*`.
- **Model:** the scikit-learn pipeline with its signature (`text: string` → label) and an input example.
- **Artifacts:** `evaluation/confusion_matrix.png`, `evaluation/misclassified.csv`,
  `evaluation/eval_metrics.json`, `evaluation/gates.yaml`, `data/manifest.json`.

A model that passes the gates is registered with the tags `approved`, `git_sha`,
`dataset_version`, `dvc_hash` and `f1_spam`, and the `champion` alias moves to it.

## Model

TF-IDF over lowercased word tokens (sublinear term frequency) followed by logistic regression.
The pipeline takes a table with one `text` column, so the served model needs no preprocessing code.

### Choosing the baseline settings

Settings were compared by 5-fold cross-validation **on the v1 training split only**
(1,580 messages). The held-out set was not read during this step.

| Settings | CV F1 (spam) | CV precision | CV recall |
|---|---|---|---|
| C=1 | 0.646 ± 0.040 | 1.000 | 0.479 |
| C=1, balanced class weights | 0.927 ± 0.026 | 0.921 | 0.933 |
| **C=10 (baseline)** | **0.896 ± 0.018** | **0.979** | **0.828** |
| C=10, balanced class weights | 0.921 ± 0.025 | 0.924 | 0.919 |
| C=10, balanced, uni+bigrams | 0.937 ± 0.028 | 0.951 | 0.923 |

`C=10` with unigrams and no class weighting was chosen as the baseline. It is the simplest
setting that clears the F1 floor, and it has the highest precision of those that do, well clear
of the 0.90 guardrail. The balanced settings score a higher F1 but their precision sits within
fold-to-fold noise of the guardrail. They remain candidates for a tuned run.

## Results

Held-out evaluation set of data `v1`: 991 messages, 129 spam.

| Run | Data | Settings | F1 (spam) | Precision | Recall | Accuracy | Decision | Registry |
|---|---|---|---|---|---|---|---|---|
| `baseline-v1` | v1 | C=10, unigrams | **0.9129** | 0.9821 | 0.8527 | 0.9788 | promoted | version 1, `champion` |

Confusion matrix of `baseline-v1`: 110 spam caught, 19 spam missed, 2 legitimate messages
flagged, 860 legitimate messages passed.

| | |
|---|---|
| MLflow run ID | `9a660843b5bf4046a9e8b92b534d8f41` |
| Git SHA | `fbd4e234416ae3f7634734ca642604dd8e300751` |
| Dataset | `v1`, DVC hash `5a5df1283f2ddd03bb210cc9db4ff898.dir` |
| Decision reason | no champion yet and candidate f1_spam 0.9129 passes the gates |

## Production model choice

`sms-spam-classifier` version 1 is the champion. It is the first model to pass every gate:
F1 0.9129 against a floor of 0.85, and spam precision 0.9821 against a guardrail of 0.90.
No other version has been registered yet.

Its weakness is recall: 19 of 129 spam messages get through. Training on more data (`v2`) or
a tuned setting are the two planned candidates. Each must beat 0.9129 by at least 0.005 on the
same held-out messages, without dropping below 0.90 precision, to replace it.
