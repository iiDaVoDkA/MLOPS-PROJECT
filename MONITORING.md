# Monitoring

The app writes one JSON object per log line on stdout and exposes Prometheus metrics at
`GET /metrics`. A local stack (API, Prometheus, Grafana) runs with Docker Compose.

## Start the stack

The API reads the same variables as in production from `.env` (`MLFLOW_TRACKING_URI`,
`MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `MODEL_NAME`, `MODEL_ALIAS`, `LOG_LEVEL`).

```bash
docker compose up --build -d
```

| Service | URL |
|---|---|
| API | http://localhost:8000 (metrics at http://localhost:8000/metrics) |
| Prometheus | http://localhost:9090 (target `api` under Status > Targets) |
| Grafana | http://localhost:3000, login `admin` / `admin` unless `GRAFANA_ADMIN_PASSWORD` is set |

The dashboard "SMS spam API" is in the "SMS spam" folder. Stop everything with `docker compose down`.

## Logs

Every request carries a `request_id`, taken from the `X-Request-ID` header or generated, and
returned in the response header. Each request logs one `request` line (method, path, status,
duration_ms), at level `error` when the status is 500 or more. Each prediction logs a
`prediction_made` line with the model version, label, probability, latency and text length.
The message text itself is never logged.

Errors only:

```bash
docker compose logs api --no-log-prefix | jq -cR 'fromjson? | select(.level == "error")'
```

All lines of one request:

```bash
docker compose logs api --no-log-prefix | jq -cR --arg id abc123 'fromjson? | select(.request_id == $id)'
```

Predictions per model version:

```bash
docker compose logs api --no-log-prefix | jq -rR 'fromjson? | select(.event == "prediction_made") | .model_version' | sort | uniq -c
```

## Metrics

| Metric | Type | Labels |
|---|---|---|
| `http_requests_total` | counter | method, path, status |
| `http_request_duration_seconds` | histogram | path |
| `predictions_total` | counter | prediction, model_version |
| `model_info` | gauge, always 1 | model_version |
| `spam_prediction_share` | gauge | |
| `input_length_mean` | gauge | |

The path label is the route template, and `/metrics` itself is not counted. The last two metrics
are simple drift signals over the last 500 predictions: the share predicted as spam (about 0.13
in the training data) and the mean message length in characters.

## Alerts

Grafana provisions three rules from `monitoring/grafana/alerts.yml`:

| Alert | Condition |
|---|---|
| API down | Prometheus cannot scrape the API for 1 minute |
| 5xx on /predict | more than 5% of /predict requests return 5xx over 2 minutes |
| Spam share | `spam_prediction_share` at or above 0.40 for 2 minutes |

The two traffic-based rules stay Normal when there is no traffic. A spam share alert means the
incoming traffic should be investigated. It does not trigger a retraining by itself.

Alerts go to an email contact point. To actually send emails, set `SMTP_ENABLED=true`,
`SMTP_HOST` (host:port), `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM_ADDRESS` and `ALERT_EMAIL` in
`.env`. Without SMTP the alerts are still visible in Grafana under Alerting.

## Generating traffic

`scripts/replay.py` sends the `text` column of a CSV to `/predict` and prints the responses by
status.

```bash
python -m scripts.replay --csv data/v1/eval.csv --limit 100 --rate 5 --shuffle
```

```bash
python -m scripts.replay --broken --limit 50
```

```bash
python -m scripts.replay --spammy --limit 300 --rate 10
```

`--broken` sends `{"msg": ...}` instead of `{"text": ...}`, so every request fails with a 422.
`--spammy` only sends spam rows, which pushes `spam_prediction_share` above 0.40 and fires the
drift alert after 2 minutes. `--url` points the script at another deployment
(default http://localhost:8000).
