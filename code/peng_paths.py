"""Single place that defines where inputs and outputs live.

Layout (relative to the repository root):
    data/       design table, external benchmark data, LCR measurements
    results/    every derived number (JSON, CSV, LaTeX macro files, tables)
    results/baseline/   outputs of the scripts in code/baseline/
    figures/    figure files written by the scripts
    code/       analysis scripts

The root defaults to the parent of this file's directory. Set the environment
variable PENG_ROOT to use a different checkout or a scratch copy.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PENG_ROOT", Path(__file__).resolve().parents[1])).resolve()
ROOT_STR = str(ROOT)
DATA = ROOT / "data"
RESULTS = ROOT / "results"
BASELINE = RESULTS / "baseline"
FIGURES = ROOT / "figures"
CODE = ROOT / "code"
