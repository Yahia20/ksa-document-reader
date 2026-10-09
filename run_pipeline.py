"""Rebuild everything with one command.

    python run_pipeline.py            # data -> tests -> dev + test evaluation -> MIDV-500 (if downloaded) -> site
    python run_pipeline.py --skip-data

Steps:
  1. generate the synthetic dev and test sets (headless Chrome)
  2. run the unit tests
  3. score the reader on dev and on the frozen test split
  4. score MRZ reading on MIDV-500 frames, if scripts/fetch_midv500.py was run
  5. build site/index.html
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)]), "PYTHONUTF8": "1"}


def step(title: str, *args: str) -> None:
    print(f"\n=== {title}", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, env=ENV, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true", help="reuse data/synthetic")
    ap.add_argument("--skip-midv", action="store_true")
    a = ap.parse_args()
    if not a.skip_data:
        step("Generate synthetic documents", "-m", "synth.make_dataset")
    step("Unit tests", "-m", "pytest", "-q", "tests")
    step("Evaluate on dev", "-m", "docreader.evaluate", "--split", "dev", "--workers", "1")
    step("Evaluate on test", "-m", "docreader.evaluate", "--split", "test", "--workers", "1")
    if not a.skip_midv and (ROOT / "data" / "raw" / "midv500" / "images").exists():
        step("Evaluate on MIDV-500", "-m", "docreader.evaluate_midv")
    step("Build the results page", "-m", "docreader.report")


if __name__ == "__main__":
    main()
