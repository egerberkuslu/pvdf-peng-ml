"""Run the baseline analyses (code/baseline/*.py) and write their JSON files to results/baseline/.

The baseline scripts read targets_design.parquet and external_data/ from the current
directory, so this wrapper runs each of them inside results/baseline/ with those two
inputs linked in. Usage:
    python code/run_baseline.py              # all scripts, in dependency order
    python code/run_baseline.py physgp_analysis ga_search
"""
import os
import shutil
import subprocess
import sys

from peng_paths import BASELINE, CODE, DATA

ORDER = [
    "batchD_analysis",
    "activeL_analysis",
    "conformal_analysis",
    "calibrated_conformal",
    "mlp_tuned_analysis",
    "anova_detail",
    "physgp_analysis",
    "protocol_replay",
    "protocol_replay_extended",
    "groupwise_conformal",
    "revision_experiments",
    "ga_search",
]


def link(src, dst):
    if os.path.lexists(dst):
        return False
    try:
        os.symlink(src, dst)
    except OSError:
        (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, dst)
    return True


def main(names):
    BASELINE.mkdir(parents=True, exist_ok=True)
    made = [
        p
        for p, s in (
            (BASELINE / "targets_design.parquet", DATA / "targets_design.parquet"),
            (BASELINE / "external_data", DATA / "external"),
        )
        if link(str(s), str(p))
    ]
    try:
        for n in names or ORDER:
            print(f"== {n}", flush=True)
            subprocess.run([sys.executable, str(CODE / "baseline" / f"{n}.py")], cwd=BASELINE, check=True)
    finally:
        for p in made:
            if os.path.islink(p) or os.path.isfile(p):
                os.remove(p)
            else:
                shutil.rmtree(p)


if __name__ == "__main__":
    main(sys.argv[1:])
