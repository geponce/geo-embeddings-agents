"""Stage 3, step 2: Random Forest baseline with spatial block cross-validation.

Run with:  python -m geo_embeddings.train_baseline [--cv spatial|random] [--block-km 10] [--folds 5]

The random option ignores location. It exists to show how much a random
split overstates accuracy when nearby pixels land in both train and test.

Writes out-of-fold predictions (data/processed, gitignored), a metrics JSON
(docs/results) and a confusion matrix figure (docs/figures).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from geo_embeddings.classes import CLASS_NAMES
from geo_embeddings.embeddings import band_names
from geo_embeddings.validation import cross_validate, evaluate, random_folds, spatial_folds

SAMPLES = Path("data/processed/pinal_samples_2023_2024_n2000.parquet")
RF_PARAMS = {"n_estimators": 300, "min_samples_leaf": 1, "max_features": "sqrt", "n_jobs": -1}


def plot_confusion(cm: np.ndarray, names: list[str], title: str, path: Path) -> None:
    """Row-normalized confusion matrix (each row sums to 1, so the diagonal is recall)."""
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            if norm[i, j] >= 0.01:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted class")
    ax.set_ylabel("CDL class (reference)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="share of reference class")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main() -> None:
    """Run spatial CV, print and save metrics, predictions and the confusion matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", choices=["spatial", "random"], default="spatial")
    parser.add_argument("--block-km", type=float, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(SAMPLES)
    X = df[band_names()].to_numpy()
    y = df["label"].to_numpy()
    labels = sorted(CLASS_NAMES)
    names = [CLASS_NAMES[c] for c in labels]

    folds, blocks = spatial_folds(df.lon.to_numpy(), df.lat.to_numpy(), y,
                                  args.block_km, args.folds, args.seed)
    if args.cv == "random":
        folds = random_folds(y, args.folds, args.seed)
        print(f"random stratified folds, {args.folds} folds (block ids kept for analysis only)")
        tag, cv_desc, title_cv = "rf_random", "random stratified", "random split"
    else:
        print(f"{len(np.unique(blocks))} blocks of {args.block_km:g} km, {args.folds} folds")
        tag = f"rf_spatial{args.block_km:g}km"
        cv_desc, title_cv = "spatial blocks", f"{args.block_km:g} km blocks"
    print("rows per class per fold:")
    print(pd.crosstab(df.class_name, folds).to_string(), "\n")

    oof, per_fold = cross_validate(
        lambda: RandomForestClassifier(**RF_PARAMS, random_state=args.seed), X, y, folds, labels
    )
    m = evaluate(y, oof, labels)

    print("per-fold metrics:")
    print(per_fold.round(3).to_string(index=False))
    print(f"\npooled out-of-fold: OA {m['overall_accuracy']:.3f} | Kappa {m['kappa']:.3f}"
          f" | macro F1 {m['macro_f1']:.3f}")
    print(f"across folds: OA {per_fold.overall_accuracy.mean():.3f} ± {per_fold.overall_accuracy.std():.3f}"
          f" | Kappa {per_fold.kappa.mean():.3f} ± {per_fold.kappa.std():.3f}")
    print("\nF1 per class:")
    for c, f in sorted(m["f1_per_class"].items(), key=lambda kv: -kv[1]):
        print(f"  {CLASS_NAMES[c]:<14} {f:.3f}")

    out = df[["year", "label", "class_name", "cdl_code", "cdl_conf", "lon", "lat"]].assign(
        block=blocks, fold=folds, pred=oof, pred_name=pd.Series(oof).map(CLASS_NAMES).to_numpy())
    pred_path = Path(f"data/processed/oof_{tag}.parquet")
    out.to_parquet(pred_path, index=False)

    results = {"model": "RandomForestClassifier", "params": RF_PARAMS, "cv": cv_desc,
               "block_km": args.block_km if args.cv == "spatial" else None, "n_folds": args.folds, "seed": args.seed,
               "n_rows": int(len(df)), "class_names": {int(k): v for k, v in CLASS_NAMES.items()},
               "pooled": m, "per_fold": per_fold.to_dict(orient="records")}
    res_path = Path(f"docs/results/{tag}.json")
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(results, indent=2))

    fig_path = Path(f"docs/figures/confusion_{tag}.png")
    plot_confusion(np.array(m["confusion_matrix"]), names,
                   f"Random Forest, {args.folds}-fold CV, {title_cv} "
                   f"(OA {m['overall_accuracy']:.2f}, Kappa {m['kappa']:.2f})", fig_path)
    print(f"\nwrote {pred_path}, {res_path}, {fig_path}")


if __name__ == "__main__":
    main()
