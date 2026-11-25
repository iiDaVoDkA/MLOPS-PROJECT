# Deployment

Live app: https://sms-spam-detector-gdlg.onrender.com (Render, free tier, runs a Docker image).

## Deployment pipeline

| Branch / event | Workflow | Job | What happens |
|---|---|---|---|
| PR to `dev`, `staging`, `main` | `ci.yml` | `lint-test`, `docker` | ruff + pytest; image build + container smoke test. Nothing is published. |
| push to `staging` | `staging.yml` | `validate-release-candidate` | pytest, image build, container smoke test against the offline test registry, then push the image to GHCR. |
| push to `main` | `deploy.yml` | `deploy` | Same verification, push the image, deploy **that exact image** to Render, then smoke-test the live URL. |

Both `staging.yml` and `deploy.yml` can also be started by hand (`workflow_dispatch`).
`deploy.yml` runs in the `production` concurrency group. The continuous-training workflow
(to be added) will use the same group, so a deploy and a model promotion never run at the same time.

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

**Model rollback** (a bad champion): move the `champion` alias back to the previous version with
the continuous-training pipeline (MLflow variables in the environment), then restart the app so
it resolves the alias again (redeploy the current image with the hook above):

```bash
python -m ml.pipeline --rollback <previous version>
python -m scripts.smoke_test --url "$PROD_URL" --expect-version <previous version> --timeout 900
```
