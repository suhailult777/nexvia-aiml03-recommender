"""Run the whole pipeline in order, then the tests.

    C:/Python312/python.exe run_all.py

Each step is a separate process so an import error in one model does not hide
behind another. Stop at the first failure.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("download and split", "src/data_prep.py"),
    ("build features", "src/features.py"),
    ("train two-tower", "src/two_tower.py"),
    ("build faiss index", "src/faiss_index.py"),
    ("train ranker", "src/ranker.py"),
    ("evaluate", "src/evaluate.py"),
    ("latency benchmark", "src/latency.py"),
    ("plots", "src/plots.py"),
]


def main() -> int:
    t0 = time.time()
    for name, script in STEPS:
        print(f"\n=== {name} ({script}) ===", flush=True)
        start = time.time()
        result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
        if result.returncode != 0:
            print(f"FAILED at {script} with exit code {result.returncode}")
            return result.returncode
        print(f"--- {name} finished in {time.time() - start:.1f}s", flush=True)

    print("\n=== tests ===", flush=True)
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"],
                            cwd=ROOT)
    print(f"\nwhole pipeline ran in {time.time() - t0:.1f}s")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
