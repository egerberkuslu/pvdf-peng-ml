# pvdf-peng-ml

Analysis code and derived results for the manuscript *"Uncertainty-Aware
Surrogate Modeling of Composition–Force–Frequency Effects in Electrospun
PVDF/BaTiO3/MWCNT Piezoelectric Nanogenerators"* (E. E. Uslu, Ege University).

Reproduces every result in the paper: the 14-model benchmark under
leave-one-out cross-validation, group-wise (leave-one-level-out)
cross-validation, empirical interval-coverage measurement, CV+/jackknife+
conformal prediction, the full factorial ANOVA, the retrospective
active-learning audit, and the external replication of the evaluation protocol
on two public UCI engineering benchmarks.

## Layout

```
code/           analysis scripts (Python 3)
results/        JSON outputs, one per analysis
data/           derived 75-condition design table (targets_design.parquet)
data/external/  UCI benchmark datasets (see NOTE.txt for provenance)
```

## Reproducing

Run each script from a directory containing `targets_design.parquet` (scripts
read it from the current working directory and write their JSON next to it).
`protocol_replay.py` expects the UCI files under `external_data/`; from
`data/`, create a symlink first: `ln -s external external_data`.

| Script | Output | Paper section |
|---|---|---|
| `batchD_analysis.py` | `batchD_results.json` | group-wise CV, coverage, calibration slope |
| `activeL_analysis.py` | `activeL_results.json` | active-learning audit |
| `conformal_analysis.py` | `conformal_results.json` | CV+/jackknife+ intervals |
| `mlp_tuned_analysis.py` | `mlp_tuned_results.json` | nested-CV MLP tuning |
| `anova_detail.py` | `anova_results.json` | full ANOVA specification |
| `protocol_replay.py` | `protocol_replay_results.json` | external replication |
| `make_graphical_abstract.py` | figure | overview figure (early draft) |

`reg_models.json` and `reg_stats.json` are the authoritative outputs of the
original DVC benchmark pipeline (per-model LOO-CV scores and the
bootstrap/Wilcoxon significance statistics reported in Tables 3-4 of the
paper).

## Dependencies

Python 3.12, scikit-learn 1.7, pandas 2.2, numpy 2.3, scipy 1.17,
statsmodels 0.14, joblib 1.5, matplotlib (figure script only). All results are
deterministic (fixed seeds).

## Data provenance

The raw open-circuit-voltage waveforms were measured by Koç et al. for an
electrospun single-needle PVDF/BaTiO3/MWCNT nanogenerator system (Koç et al.,
2025, *Journal of Materials Science* 60(48):25481–25503, DOI
[10.1007/s10853-025-11872-9](https://doi.org/10.1007/s10853-025-11872-9)) and
are available from those authors on reasonable request. This repository
contains only the derived, analysis-ready design table
(`data/targets_design.parquet`), used with the originators' permission.

The external benchmarks in `data/external/` come from the UCI Machine Learning
Repository and keep their original terms (CC BY 4.0): Airfoil Self-Noise
(Brooks, Pope, Marcolini, NASA RP-1218, 1989) and Energy Efficiency (Tsanas &
Xifara, *Energy and Buildings* 49:560–567, 2012).

## License

Code: MIT (see LICENSE). Data files keep their original licenses/terms as
described above.
