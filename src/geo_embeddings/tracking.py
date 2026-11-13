"""Log cross-validation runs to MLflow in the Databricks workspace.

Training runs locally (the data is small). MLflow records each run in the
workspace so runs can be compared in the Databricks UI. The experiment lives
under the signed-in user's workspace folder, found at run time, so no user
name or workspace URL is written in the code.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Any

EXPERIMENT_SUBPATH = "geo-embeddings-agents/landcover-classification"


def metric_key(name: str) -> str:
    """Turn a class name into an MLflow-safe metric key fragment ("Corn+Sorghum" -> "corn_sorghum")."""
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def flatten_metrics(results: dict[str, Any]) -> dict[str, float]:
    """Pooled metrics plus per-class F1 and 95% bounds as a flat {key: value} dict."""
    names = {int(k): v for k, v in results["class_names"].items()}
    pooled, boot = results["pooled"], results["block_bootstrap"]
    out = {k: float(pooled[k]) for k in ("overall_accuracy", "kappa", "macro_f1")}
    for k in ("overall_accuracy", "kappa"):
        out[f"{k}_ci_lo"], out[f"{k}_ci_hi"] = (float(v) for v in boot[k]["ci"])
    for c, f in boot["f1_per_class"].items():
        key = metric_key(names[int(c)])
        out[f"f1_{key}"] = float(f["value"])
        out[f"f1_{key}_ci_lo"], out[f"f1_{key}_ci_hi"] = (float(v) for v in f["ci"])
        out[f"blocks_{key}"] = float(f["n_blocks"])
    return out


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, to tie a run to the exact sample data it used."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    """Current git commit, with '-dirty' if there are uncommitted changes."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"]).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def setup_experiment(profile: str | None = None) -> str:
    """Point MLflow at the Databricks workspace and create/select the experiment.

    Uses the Databricks CLI profile from the argument, DATABRICKS_CONFIG_PROFILE,
    or "geo-embeddings". Returns the experiment path.
    """
    import mlflow
    from databricks.sdk import WorkspaceClient

    profile = profile or os.environ.get("DATABRICKS_CONFIG_PROFILE", "geo-embeddings")
    w = WorkspaceClient(profile=profile)
    folder = f"/Users/{w.current_user.me().user_name}/{EXPERIMENT_SUBPATH.rsplit('/', 1)[0]}"
    w.workspace.mkdirs(folder)  # set_experiment does not create parent folders
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")
    path = f"{folder}/{EXPERIMENT_SUBPATH.rsplit('/', 1)[1]}"
    mlflow.set_experiment(path)
    return path


def log_cv_run(results: dict[str, Any], run_name: str, data_path: Path,
               artifacts: list[Path], profile: str | None = None) -> str:
    """Log one cross-validation result to MLflow and return the run id."""
    import mlflow

    setup_experiment(profile)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({f"rf_{k}": v for k, v in results["params"].items()})
        mlflow.log_params({"cv": results["cv"], "block_km": results["block_km"],
                           "n_folds": results["n_folds"], "seed": results["seed"],
                           "n_rows": results["n_rows"], "n_classes": len(results["class_names"]),
                           "bootstrap_n": results["block_bootstrap"]["n_boot"]})
        mlflow.log_metrics(flatten_metrics(results))
        for row in results["per_fold"]:
            for k in ("overall_accuracy", "kappa", "macro_f1"):
                mlflow.log_metric(f"fold_{k}", row[k], step=row["fold"])
        mlflow.set_tags({"git_commit": git_commit(), "data_file": data_path.name,
                         "data_sha256": file_sha256(data_path), "stage": "3",
                         "region": "Pinal County AZ", "years": "2023,2024"})
        for a in artifacts:
            if a.exists():
                mlflow.log_artifact(str(a))
        return run.info.run_id
