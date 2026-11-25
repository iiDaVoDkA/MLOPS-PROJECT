# Deployment

Live app: https://sms-spam-detector-gdlg.onrender.com (Render, free tier, runs a Docker image).

## Deployment pipeline

| Branch / event | Workflow | Job | What happens |
|---|---|---|---|
| PR to `dev`, `staging`, `main` | `ci.yml` | `lint-test`, `docker` | ruff + pytest; image build + container smoke test. Nothing is published. |
| push to `staging` | `staging.yml` | `validate-release-candidate` | pytest, image build, container smoke test against the offline test registry, then push the image to GHCR. |
| push to `main` | `deploy.yml` | `deploy` | Same verification, push the image, deploy **that exact image** to Render, then smoke-test the live URL. |

Both `staging.yml` and `deploy.yml` can also be started by hand (`workflow_dispatch`).
`ct.yml` (continuous training and model rollback) only runs by hand: see
[Continuous training](#continuous-training).
`deploy.yml` runs in the `production` concurrency group. The continuous-training workflow
(`ct.yml`) uses the same group, so a deploy and a model promotion never run at the same time.

### Image tags

Images live at `ghcr.io/iidavodka/mlops-project` and are tagged with the **full commit SHA**
they were built from (`ghcr.io/iidavodka/mlops-project:<sha>`). The same SHA is baked into the
image as `GIT_SHA` (reported by `GET /model-info` as `app_git_sha`) and as the
`org.opencontainers.image.revision` label. A tag is never reused for a different build.
(`:bootstrap` was the one-off image used to create the Render service.)

### Runtime configuration and secrets

The image contains no secrets, no data and no model. Everything is set as Render environment
variables and read at start-up:

| Variable | Purpose |
|---|---|
| `MLFLOW_TRACKING_URI` | DagsHub MLflow registry |
| `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | registry credentials (password is secret) |
| `MODEL_NAME` | `sms-spam-classifier` |
| `MODEL_ALIAS` | `champion` |
| `PORT` | set by Render |

On start-up the app resolves `MODEL_NAME@MODEL_ALIAS` to a version, loads it, and only reports
ready once a test prediction has succeeded. The alias is resolved at start-up only: after the
alias moves, the app serves the new version from its next restart or deploy.

GitHub configuration used by `deploy.yml` (environment `production`):

| Name | Kind | Purpose |
|---|---|---|
| `RENDER_DEPLOY_HOOK_URL` | secret | Render deploy hook (already contains `?key=...`) |
| `MLFLOW_TRACKING_PASSWORD` | secret | read the champion version from the registry |
| `PROD_URL` | variable | live URL to smoke-test |
| `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MODEL_NAME` | variables | registry access |

Images are pushed with the workflow's `GITHUB_TOKEN` (`packages: write`).

### Deploying by hand

From a clean checkout of the commit to deploy (Render runs `linux/amd64`, so the platform flag is
required on Apple Silicon):

```bash
SHA=$(git rev-parse HEAD)
IMAGE=ghcr.io/iidavodka/mlops-project:$SHA

docker login ghcr.io -u <github-user>        # token with write:packages, typed at the prompt
docker buildx build --platform linux/amd64 \
  --build-arg GIT_SHA=$SHA \
  --label org.opencontainers.image.source=https://github.com/iiDaVoDkA/MLOPS-PROJECT \
  -t $IMAGE --push .

# Deploy that exact image. RENDER_DEPLOY_HOOK_URL comes from the Render dashboard.
ENCODED=$(python -c "import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=''))" "$IMAGE")
curl -fsS -X POST -o /dev/null "${RENDER_DEPLOY_HOOK_URL}&imgURL=${ENCODED}"

# Which model must it serve? (needs the MLflow variables in the environment)
EXPECTED_VERSION=$(python -m scripts.champion_version)

python -m scripts.smoke_test --url "$PROD_URL" \
  --expect-version "$EXPECTED_VERSION" --expect-app-sha "$SHA" --timeout 900
```

### If the app does not become ready

Render keeps the previous instance serving until the new one passes `/health/ready`. If the new
image crashes, or cannot load the model, users stay on the old version, and the live smoke test
never sees the new `app_git_sha`, so it fails after its timeout and the `deploy` job fails.
The job summary says so. Check the Render logs, then fix forward or roll back.

## Rollback

**App rollback** (a bad image): redeploy the last good image by its commit SHA. Every deployed
image stays in GHCR, so nothing needs rebuilding:

```bash
GOOD_SHA=<last good commit>
ENCODED=$(python -c "import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=''))" \
  "ghcr.io/iidavodka/mlops-project:$GOOD_SHA")
curl -fsS -X POST -o /dev/null "${RENDER_DEPLOY_HOOK_URL}&imgURL=${ENCODED}"
python -m scripts.smoke_test --url "$PROD_URL" --expect-app-sha "$GOOD_SHA" --timeout 900
```

Then revert the bad commit on `main` through a PR, so the next deploy does not bring it back.

**Model rollback** (a bad champion): run the `ct` workflow (Actions → ct → Run workflow) with
`operation: rollback`, `rollback_version: <previous approved version>` and `target: production`.
It moves the `champion` alias back (only to a version that was approved), restarts the live app
by re-deploying the image that is live now, and waits until `/model-info` reports that version.
See [Continuous training](#continuous-training).

By hand, with the MLflow variables in the environment:

```bash
python -m ml.pipeline --rollback <previous version>
# restart: re-deploy the live image (app rollback commands above, with the current app_git_sha)
python -m scripts.smoke_test --url "$PROD_URL" --expect-version <previous version> --timeout 900
```

## Continuous training

`.github/workflows/ct.yml` is started by hand (Actions → ct → Run workflow). It runs
`ml/pipeline.py` and then refreshes and verifies whatever serves the model. It uses the
`production` concurrency group, like `deploy.yml`, so a retrain and a deploy never overlap.

### Inputs

| Input | Values | Used by |
|---|---|---|
| `operation` | `pipeline` (default), `rollback` | — |
| `dataset_version` | `v1` (default), `v2` | pipeline: pulled with DVC from the Azure remote, validated, used for scoring and training |
| `rollback_version` | a positive integer | rollback: the approved version to make champion again (checked before anything runs) |
| `target` | `staging-demo` (default), `production` | `production` → `vars.MODEL_NAME` (served by the live app); `staging-demo` → `vars.STAGING_MODEL_NAME` (no app) |

Secrets and variables (environment `production`): `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`,
`MLFLOW_TRACKING_PASSWORD`, `MODEL_NAME`, `STAGING_MODEL_NAME`, `AZURE_STORAGE_SAS_TOKEN` (DVC pull),
`RENDER_DEPLOY_HOOK_URL` and `PROD_URL`.

### Decisions

The pipeline writes `results/decision.json` (uploaded as the `ct-results` artifact and shown in
the job summary).

| Decision | Meaning | Refresh |
|---|---|---|
| `bootstrapped` | no champion existed; the first model passed the gates and became champion | yes |
| `promoted` | retraining was triggered and the candidate beat the champion under `ml/gates.yaml` | yes |
| `rolled_back` | the alias moved back to the requested approved version | yes |
| `retain_no_retraining` | the champion is healthy on this data; nothing trained | no |
| `retain_rejected` | a candidate was trained and registered but failed the gates; champion unchanged | no |
| `rollback_unchanged` | the alias already pointed to the requested version | no |
| `failed` | data contract, registry or rollback check failed; champion unchanged; the job fails | no |

### How the app is refreshed

The app resolves `MODEL_NAME@champion` only at start-up. When the champion moves on the
`production` target, the workflow:

1. reads the live image commit from `GET $PROD_URL/model-info` (`app_git_sha`), retrying for
   about 3 minutes in case the free instance is asleep;
2. re-deploys that same image `ghcr.io/iidavodka/mlops-project:<app_git_sha>` with the deploy hook
   (a restart: the code does not change);
3. runs `scripts.smoke_test --expect-version <champion_after> --expect-app-sha <app_git_sha>`,
   which waits (up to 15 minutes) until the new instance serves the new champion, because Render
   keeps the old instance answering until the new one is ready.

On `staging-demo` no app serves the model, so the workflow instead resolves
`models:/<STAGING_MODEL_NAME>@champion`, checks it is `champion_after`, loads it and predicts three
messages from `data/<dataset_version>/eval.csv` (or a fixed ham and spam message after a rollback).

### Running the demos

Use `target: staging-demo` to rehearse without touching production.

1. **Bootstrap**: `operation: pipeline`, `dataset_version: v1`. With no champion yet, the decision
   is `bootstrapped` (version 1).
2. **No retraining needed**: run the same again. The champion is healthy on `v1`:
   `retain_no_retraining`.
3. **Retrain on new data**: `dataset_version: v2`. `v2` is newer than the champion's data, so the
   pipeline retrains; the decision is `promoted` if the candidate passes the gates, otherwise
   `retain_rejected`.
4. **Rollback**: `operation: rollback`, `rollback_version: 1`. The decision is `rolled_back`
   (or `rollback_unchanged` if version 1 is already champion).

For production, run the same with `target: production` and check the job summary: the
"live verification" row shows the live smoke-test result.

## Twelve-factor app

| # | Factor | How this project implements it |
|---|---|---|
| I | Codebase | One Git repo (`MLOPS-PROJECT`) holds the app, ML code, tests, docs and workflows. The same codebase runs in several deploys: CI containers, the staging validation, and production on Render. Each image is tagged with the commit SHA it was built from. |
| II | Dependencies | Every dependency is pinned: `requirements.txt` (runtime, the only file installed in the image) and `requirements-dev.txt` (training, DVC, tests). The image installs them into an isolated virtualenv. The health check uses only the Python standard library, so it needs no system tool. |
| III | Config | Everything that changes between deploys comes from environment variables: `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `MODEL_NAME`, `MODEL_ALIAS`, `PORT`. Secrets live in Render's environment and in GitHub secrets, never in the code or the image. `.env.example` lists every variable, with the secret values left empty. |
| IV | Backing services | The MLflow registry (DagsHub) and the data store (Azure Blob through DVC) are attached resources, reached by URL and credentials from config. Swapping one needs no code change: CI points `MLFLOW_TRACKING_URI` at a local SQLite test registry instead of DagsHub. |
| V | Build, release, run | **Build:** CI builds the Docker image and tags it with the commit SHA. **Release:** that image plus Render's environment config, deployed by `deploy.yml` through the deploy hook (`imgURL` = the exact SHA tag). **Run:** Render runs the container. A release is never edited in place; a fix is a new commit and a new image. The model has its own release step: the CT workflow moves the `champion` alias, and the app picks it up at its next start. |
| VI | Processes | The app is stateless. It keeps only the loaded model in memory; the only files it writes are the model artifacts MLflow downloads to a temporary folder at start-up, which a new instance simply downloads again. Any instance can be killed and replaced without losing data. |
| VII | Port binding | The app is self-contained: uvicorn serves HTTP on the port given in `$PORT` (set by Render, default 8000). No external web server is needed. |
| VIII | Concurrency | **Limited relevance here.** The free Render plan runs a single instance with one uvicorn worker, which is enough for this traffic. Because processes are stateless, scaling out would only mean running more instances behind Render's load balancer. |
| IX | Disposability | The process is live immediately and loads the model in the background. `/health/ready` reports ready only after a test prediction succeeds, and Render routes traffic to a new instance only then. The container runs uvicorn with `exec`, so it receives `SIGTERM` directly and shuts down cleanly. |
| X | Dev/prod parity | The same Dockerfile builds the image in CI, in the staging validation and in production, and production runs exactly the image that `deploy.yml` has just verified. The tools are the same everywhere (pinned MLflow 3.5.1, the same Python 3.11). The one deliberate difference: CI uses an offline SQLite registry with a small test model, production uses DagsHub with the real champion. |
| XI | Logs | The app writes its logs to standard output as an event stream and never manages log files. Render collects and shows them. |
| XII | Admin processes | One-off tasks run as separate commands from the same repo and pinned dependencies, never inside the web container: `ml/prepare_data.py` (build a dataset version), `ml/train.py` (baseline), `ml/pipeline.py` (continuous training and rollback, run by `ct.yml`), `scripts/smoke_test.py` (live verification). |
