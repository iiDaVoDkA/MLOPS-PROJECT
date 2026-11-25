# Data

The model classifies an SMS message as `spam` or `ham` (legitimate).

## Source and license

| | |
|---|---|
| Dataset | SMS Spam Collection, UCI Machine Learning Repository (dataset 228) |
| Authors | Tiago Almeida, José María Gómez Hidalgo |
| Citation | Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84 |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Download URL | https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip |
| Snapshot obtained | 2026-11-24 |
| Snapshot sha256 | `1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3` |

The archive holds one file, `SMSSpamCollection`: 5,574 English messages, one per line, as
`label<TAB>text` (4,827 ham, 747 spam). `ml/prepare_data.py` refuses to run if the downloaded
archive does not match the sha256 above, so a silent upstream change cannot alter a dataset version.

## From source to dataset

`python -m ml.prepare_data --version v1` rebuilds `data/v1` byte for byte:

1. **Deduplicate.** Two messages are the same if their text matches after lowercasing and
   collapsing whitespace. The first occurrence is kept. 5,574 lines become 5,159 unique messages.
   No message in the snapshot carries conflicting labels (the script would drop such messages).
2. **Identify.** Each message gets `id` = first 16 hex characters of the sha256 of its normalized text.
3. **Assign a bucket.** `bucket = int(id[:8], 16) % 100`. The bucket depends only on the text,
   not on row order, a random seed or a library version.
4. **Write** `train.csv`, `eval.csv` (columns `id,label,text`, UTF-8, sorted by `id`) and
   `manifest.json` (source hash, bucket ranges, row counts).

## Split design

| Buckets | Role | Rows | ham | spam |
|---|---|---|---|---|
| 0–19 | Held-out evaluation set, **identical in every version** | 991 | 862 | 129 |
| 20–49 | Training batch 1 → `v1` | 1,580 | 1,371 | 209 |
| 50–99 | Training batch 2 → added in `v2` | 2,588 | 2,284 | 304 |

- The evaluation set is fixed before any model exists and is never used for fitting or tuning.
  Champion and candidate are always scored on these same 991 messages. Any tuning must use
  cross-validation or a validation split taken from `train.csv`.
- Deduplication happens before the split, so no message (even with different casing or spacing)
  can sit on both sides.
- The messages carry no timestamp, so there is no chronology to preserve.
- The split is not stratified, but the spam share is close across splits (13.0% eval, 13.2% v1 train).

## Versions

| Version | Content | Git tag | DVC hash (`data/<version>.dvc`) |
|---|---|---|---|
| `v1` | eval + training batch 1 (1,580 training rows) | `data-v1` | `5a5df1283f2ddd03bb210cc9db4ff898.dir` |
| `v2` | eval + training batches 1 and 2 (4,168 training rows) | `data-v2` | `aa465d0a6458dad7b057af7261765687.dir` |

Each Git tag points at the merge commit on `dev` that added the version's `.dvc` pointer.

**What changes between v1 and v2:** v2 simulates a new delivery of labelled messages. It adds
training batch 2 (buckets 50–99: 2,588 messages, 2,284 ham and 304 spam) to the v1 training set,
which grows from 1,580 to 4,168 rows (12.3% spam). Nothing else changes: the evaluation file is
byte-identical to v1's, every v1 training row is still present, labels are untouched, and the
source snapshot and processing are the same. Because only training data grows, a champion trained
on v1 and a candidate trained on v2 can be compared directly on the same 991 held-out messages.

v2 was built with `python -m ml.prepare_data --version v2` from the same snapshot and passes the
data contract. It has not been used for training yet: the continuous-training workflow trains on it.

## Data contract

`python -m ml.data_contract data/<version>` exits non-zero and lists every violation if any check fails:

- both files exist with exactly the columns `id,label,text`, all strings;
- no missing or blank values;
- text length within 1..1000 characters (the app's input schema uses the same bound);
- labels are `ham` or `spam`, both present, spam share within 5%..40%;
- at least 500 training rows and 200 evaluation rows;
- every `id` is well-formed, matches its text, and is unique within its split;
- train and eval share no `id` and no normalized text.

## Getting the data

```bash
dvc pull data/v1.dvc data/v2.dvc
```

The DVC remote is Azure Blob Storage. Credentials are never committed: provide them through the
environment (`AZURE_STORAGE_CONNECTION_STRING`, or `AZURE_STORAGE_KEY` / `AZURE_STORAGE_SAS_TOKEN`)
or with `dvc remote modify --local`, which writes to the git-ignored `.dvc/config.local`.
