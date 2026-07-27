# pvdf-peng-ml — Uncertainty-Aware, Experiment-Efficient Surrogate Modeling of Electrospun PVDF/BaTiO3/MWCNT Piezoelectric Nanogenerators

Analysis code, derived data, and per-analysis results for the manuscript

> **Uncertainty-Aware, Experiment-Efficient Surrogate Modeling of Electrospun
> PVDF/BaTiO₃/MWCNT Piezoelectric Nanogenerators**
> Candan Akça (ORCID 0000-0001-5305-0296), Ege Erberk Uslu (ORCID 0000-0001-9119-8574),
> Levent Paralı (ORCID 0000-0002-4462-7628)
> Manisa Celal Bayar University and Ege University, Türkiye
> Prepared for submission to *Advanced Engineering Informatics*.

**Code repository:** <https://github.com/egerberkuslu/pvdf-peng-ml>

## Description

The manuscript builds an uncertainty-aware Gaussian-process surrogate for the
open-circuit voltage of an electrospun poly(vinylidene fluoride)/barium
titanate/multi-walled carbon nanotube (PVDF/BaTiO3/MWCNT) piezoelectric
nanogenerator, trained on 75 full-factorial recordings spanning five
compositions, three tapping forces, and five excitation frequencies. The
Gaussian process is chosen in advance from fourteen benchmarked regressors
under leakage-safe leave-one-out cross-validation, and every prediction ships
with a measured validity domain, coverage-checked uncertainty, and an explicit
experiment-budget audit.

This repository reproduces every result in the paper: the 14-model benchmark
under leave-one-out cross-validation, the group-wise (leave-one-level-out)
cross-validation that draws the interpolation-versus-generalization boundary,
empirical interval-coverage measurement, CV+/jackknife+ conformal prediction,
the fully specified factorial ANOVA, the retrospective active-learning audit,
and the external replication of the whole evaluation protocol on two public
UCI engineering benchmarks. Results ship as one JSON file per analysis so
that any number in the paper can be checked without rerunning anything.

## Dataset Information

| File | Contents | Source and license |
|---|---|---|
| `data/targets_design.parquet` | Derived, analysis-ready design table: 75 conditions (composition, CNT wt%, force, frequency) with four voltage targets (RMS, peak-to-peak, peak, squared-voltage proxy) | Derived from raw waveforms measured by Koç et al. (2025), used with the originators' permission; raw waveforms available from those authors on reasonable request |
| `data/external/` | UCI benchmarks for the external replication: Airfoil Self-Noise (N=1503) and Energy Efficiency (N=768) | UCI Machine Learning Repository, CC BY 4.0; provenance details in `data/external/NOTE.txt` |
| `results/*.json` | One JSON per analysis (see the script table below) plus the canonical benchmark outputs `reg_stats.json` and `reg_models.json` | Produced by this repository |

Where `reg_stats.json` and `reg_models.json` overlap, **`reg_stats.json` is the
canonical source** for the numbers quoted in the paper (the files differ only
in the ARD-GP score on the peak-voltage target, 0.783 vs 0.777, from a minor
refit between runs).

## Code Information

```
code/           analysis and figure scripts (Python 3)
results/        JSON outputs, one per analysis
data/           derived 75-condition design table (targets_design.parquet)
data/external/  UCI benchmark datasets (see NOTE.txt for provenance)
```

| Script | Output | Paper section |
|---|---|---|
| `batchD_analysis.py` | `batchD_results.json` | group-wise CV, coverage, calibration slope |
| `activeL_analysis.py` | `activeL_results.json` | active-learning audit |
| `conformal_analysis.py` | `conformal_results.json` | CV+/jackknife+ intervals |
| `mlp_tuned_analysis.py` | `mlp_tuned_results.json` | nested-CV MLP tuning |
| `anova_detail.py` | `anova_results.json` | full ANOVA specification |
| `protocol_replay.py` | `protocol_replay_results.json` | external replication |
| `make_graphical_abstract.py` | figure | overview figure (early draft) |
| `make_model_comparison.py` | `fig05` bar chart | fourteen-regressor comparison (Fig. 5) |
| `make_calibration_figure.py` | `fig05_calibration` | LOO calibration scatter (Fig. 6) |
| `make_uncertainty_map.py` | `fig07_uncertainty_map` | GP reliability map + quoted std stats (Fig. 9) |
| `make_eda_figure.py` | `fig00_eda` | six-panel exploratory-data-analysis figure (Fig. 2) |
| `make_response_surface.py` | `fig06_response_surface` | GP/RF response surfaces at the 2 wt% slice (Fig. 8) |

## Usage Instructions

```bash
git clone https://github.com/egerberkuslu/pvdf-peng-ml.git
cd pvdf-peng-ml
python3 -m venv .venv && source .venv/bin/activate
pip install scikit-learn pandas numpy scipy statsmodels joblib matplotlib pyarrow
```

Run each script from a directory containing `targets_design.parquet` (scripts
read it from the current working directory and write their JSON next to it;
they also fall back to the repository's `data/` folder automatically):

```bash
cd data
python ../code/batchD_analysis.py
python ../code/make_calibration_figure.py   # prints the LOO R2 per target and
                                            # asserts they match the published values
```

`protocol_replay.py` expects the UCI files under `external_data/`; from
`data/`, create a symlink first: `ln -s external external_data`.

## Requirements

Python 3.12; scikit-learn 1.7, pandas 2.2, numpy 2.3, scipy 1.17,
statsmodels 0.14, joblib 1.5, pyarrow, and matplotlib (figure scripts only).
All results are deterministic (fixed seeds); the figure scripts assert their
recomputed cross-validated scores against the published values before writing
any output.

## Methodology

1. **Design table.** The 75 open-circuit-voltage waveforms (one second at
   1 kHz per condition) are summarized into per-condition amplitude targets;
   the squared-voltage proxy is the exact deterministic function
   E = 1000 x RMS^2.
2. **Benchmark.** Fourteen regressors, from linear baselines to tree
   ensembles and an ARD Matern-5/2 Gaussian process, are scored by
   leave-one-out cross-validation with all preprocessing fitted inside each
   fold; in-sample scores are reported next to cross-validated ones.
3. **Generalization boundary.** The cross-validation is repeated group-wise,
   holding out every condition that shares one factor level, which separates
   within-grid interpolation from generalization to unseen levels.
4. **Uncertainty.** Empirical coverage of the Gaussian-process intervals is
   measured under leave-one-out, and distribution-free CV+/jackknife+
   conformal intervals are supplied for the tree ensembles.
5. **Physics checks.** A fully specified factorial ANOVA, an interpretable
   closed-form voltage law, and per-material Lorentzian resonance fits.
6. **Experiment efficiency.** A retrospective expected-improvement replay of
   the measurement campaign counts the experiments the surrogate would have
   saved.
7. **External replication.** The identical recipe is replayed on the two UCI
   benchmarks, reproducing the same qualitative findings.

## Citation

If you use this code or the derived design table, please cite the manuscript:

```bibtex
@article{akca2026pvdf,
  author = {Ak{\c{c}}a, Candan and Uslu, Ege Erberk and Paral{\i}, Levent},
  title  = {Uncertainty-Aware, Experiment-Efficient Surrogate Modeling of
            Electrospun {PVDF}/{BaTiO}$_3$/{MWCNT} Piezoelectric Nanogenerators},
  year   = {2026},
  note   = {Prepared for submission to Advanced Engineering Informatics},
  url    = {https://github.com/egerberkuslu/pvdf-peng-ml}
}
```

Please also cite the data sources you use:

- **Raw waveform measurements** — Koç M., Guluzade S., Tatardar F., Musayeva N. N., Sarı A., Paralı L. *Piezoelectric nanogenerators based on PVDF fibers.* Journal of Materials Science 60(48):25481-25503, 2025. DOI 10.1007/s10853-025-11872-9.
- **Airfoil Self-Noise** — Brooks T. F., Pope D. S., Marcolini M. A. *Airfoil self-noise and prediction.* NASA RP-1218, 1989.
- **Energy Efficiency** — Tsanas A., Xifara A. *Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools.* Energy and Buildings 49:560-567, 2012.

A machine-readable [`CITATION.cff`](CITATION.cff) is included, so GitHub's
"Cite this repository" button gives the same reference.

## License

Code: MIT (see [`LICENSE`](LICENSE)). Data files keep their original
licenses and terms as described under *Dataset Information*.

## Getting help and contributing

For questions or reproduction problems, please open an issue on this
repository and include the failing command together with the output of
`pip freeze`. Pull requests are welcome, particularly additional regressors
for the benchmark; a new model only needs to follow the fixed-hyperparameter,
fold-internal-preprocessing protocol used by the existing scripts.
