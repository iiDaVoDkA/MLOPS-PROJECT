"""Print the registry version behind models:/$MODEL_NAME@$MODEL_ALIAS.

The deploy workflow uses it to know which model version the live app must serve:

    EXPECTED_VERSION=$(python -m scripts.champion_version)

Configuration comes from the environment, like the app: MLFLOW_TRACKING_URI (with
MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD, which MLflow reads itself),
MODEL_NAME and MODEL_ALIAS (default "champion"). Exits 1 with a message on stderr
if the configuration is incomplete or the alias does not exist.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping

DEFAULT_ALIAS = "champion"


class ChampionError(RuntimeError):
    pass


def champion_version(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    missing = [key for key in ("MLFLOW_TRACKING_URI", "MODEL_NAME") if not env.get(key)]
    if missing:
        raise ChampionError(f"missing environment variable(s): {', '.join(missing)}")
    uri, name = env["MLFLOW_TRACKING_URI"], env["MODEL_NAME"]
    alias = env.get("MODEL_ALIAS") or DEFAULT_ALIAS

    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    try:
        version = client.get_model_version_by_alias(name, alias)
    except MlflowException as error:
        raise ChampionError(f"no version of {name!r} has the alias {alias!r}: {error}") from error
    return str(version.version)


def main() -> int:
    try:
        print(champion_version())
    except ChampionError as error:
        print(f"champion_version: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
