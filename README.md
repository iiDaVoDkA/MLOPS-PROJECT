# MLOPS-PROJECT

Final-year project: machine learning in production.
_Topic and dataset: to be decided._

**Live app:** _coming soon_ · **MLflow:** https://dagshub.com/iiDaVoDkA/MLOPS-PROJECT.mlflow

## Team

- Hassen Ben Amor
- Dalil Adimi

## Branch flow

```mermaid
gitGraph
  commit id: "init"
  branch staging
  branch dev
  branch feature/x
  commit id: "work"
  checkout dev
  merge feature/x id: "squash PR"
  checkout staging
  merge dev id: "release PR"
  checkout main
  merge staging id: "prod PR"
```

- `feature/*`: short-lived, branched from `dev`, merged into `dev` by reviewed PR (**merge commit**).
- `dev` → `staging` → `main`: promoted by PR (**merge commit**).
- `main`, `staging` and `dev` are protected: changes go through a PR with 1 approval, and force pushes and deletions are blocked. Required CI checks will be added with the CI workflow.

## Workflow triggers

| Workflow | Trigger | What it does |
|---|---|---|
| _CI_ | _PR to dev / staging / main_ | _to be added_ |