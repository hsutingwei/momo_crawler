"""Batch runner for label/sampling configuration sweeps.

Usage:
    python experiments/label_sweep.py

The script executes multiple configurations by calling Model/train.py with the
current v1 pipeline baseline (XGBoost product-level) while varying label/sampling
parameters. Each configuration writes outputs under Model/outputs_v1_sweep/<name>.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = REPO_ROOT / "Model" / "train.py"
OUTPUT_ROOT = REPO_ROOT / "Model" / "outputs_v1_sweep"

# Base arguments shared by every sweep run.
BASE_ARGS: List[str] = [
    "--mode", "product_level",
    "--date-cutoff", "2025-06-25",
    "--vocab-mode", "global",
    "--top-n", "80",
    "--algorithms", "xgboost",
    "--fs-methods", "no_fs",
    "--cv", "5",
    "--oversample", "xgb_scale_pos_weight",
    "--threshold-min-recall", "0.4",
]

# Each config only lists the varying arguments.
CONFIGS: List[Dict[str, List[str]]] = [
    {
        "name": "delta8_gap21",
        "args": [
            "--label-delta-threshold", "8",
            "--label-max-gap-days", "21",
            "--min-comments", "5",
            "--keyword-blacklist", "口罩",
        ],
    },
    {
        "name": "delta15_health",
        "args": [
            "--label-delta-threshold", "15",
            "--label-max-gap-days", "14",
            "--min-comments", "5",
            "--keyword-whitelist", "膠原蛋白,益生菌,維他命,葉黃素",
            "--keyword-blacklist", "",
        ],
    },
    {
        "name": "delta12_ratio20_min10",
        "args": [
            "--label-delta-threshold", "12",
            "--label-ratio-threshold", "0.2",
            "--label-max-gap-days", "14",
            "--min-comments", "10",
            "--keyword-blacklist", "口罩",
        ],
    },
    {
        "name": "delta5_gap10_all",
        "args": [
            "--label-delta-threshold", "5",
            "--label-max-gap-days", "10",
            "--min-comments", "3",
            "--keyword-blacklist", "",
        ],
    },
    {
        "name": "delta20_gap21_highvalue",
        "args": [
            "--label-delta-threshold", "20",
            "--label-max-gap-days", "21",
            "--min-comments", "8",
            "--keyword-whitelist", "膠原蛋白,維他命,雞精,寵物",
            "--keyword-blacklist", "",
        ],
    },
]


def run_config(config: Dict[str, List[str]]) -> None:
    outdir = OUTPUT_ROOT / config["name"]
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(TRAIN_PY)]
    cmd += BASE_ARGS
    cmd += config["args"]
    cmd += ["--outdir", str(outdir)]
    print(f"\n[SWEEP] Running {config['name']} ...")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    for cfg in CONFIGS:
        run_config(cfg)
    print("\n[SWEEP] All configurations completed.")


if __name__ == "__main__":
    main()
