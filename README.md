# pvdf-peng-ml

Code, derived data and derived results for the manuscript

> **Machine Learning Prediction of the Voltage Response of Electrospun PVDF/BaTiO3/MWCNT
> Piezoelectric Nanogenerators Across Composition, Force and Frequency**
> Candan Akça (ORCID 0000-0001-5305-0296), Ege Erberk Uslu (ORCID 0000-0001-9119-8574),
> Levent Paralı (ORCID 0000-0002-4462-7628)

**Repository:** <https://github.com/egerberkuslu/pvdf-peng-ml>

## Purpose

The study predicts the voltage response of electrospun poly(vinylidene fluoride)
(PVDF), barium titanate (BaTiO3) and multi-walled carbon nanotube (MWCNT)
piezoelectric nanogenerators (PENG) from composition, tapping force and tapping
frequency. A fourteen-regressor benchmark under leave-one-out cross-validation
selects an ARD Gaussian process, a closed-form voltage law (PhysGP and the
five-parameter LawGP) is tested as a mean function, and the evaluation separates
interpolation inside the measured grid from generalization to unseen composition,
force or frequency levels. The remaining analyses cover calibrated and conformal
intervals, specimen variability, frequency-axis and K-mode extensions of the law,
an RC read-out model, a simulated generator with known ground truth, active
learning, an external check on digitized literature sweeps and on public UCI
benchmarks, and the LCR dielectric measurements of the five compositions.
Every number in the manuscript is written by one of the scripts below into a
LaTeX macro file (`results/numbers_*.tex`) with a JSON file that holds the
full-precision value.

## Folder layout

```
code/                 analysis scripts (Python); peng_paths.py defines all paths
code/baseline/        first-stage analyses whose JSON outputs the later scripts read
code/figures/         scripts of the manuscript's result figures
data/                 design table, LCR measurements, external data (no raw waveforms)
results/              derived numbers: JSON, CSV, LaTeX macros, table bodies
results/baseline/     outputs of code/baseline/ (and of the fourteen-regressor benchmark)
results/tables/       table bodies (booktabs) written by the scripts
results/lcr/          LCR summary, RC chain, figure data and macros
figures/              PDFs of the five manuscript result figures; scripts write further figure files here
```

## Data description

The experiment is a full factorial of **5 compositions x 3 forces x 5 frequencies = 75
conditions**, one recording per condition. The compositions are PVDF, PVDF/BaTiO3, and
PVDF/BaTiO3 with 1, 2 and 3 wt% MWCNT; the tapping force is 1, 2 and 3 N; the tapping
frequency is 5, 10, 15, 20 and 25 Hz. The open-circuit voltage was recorded for one
second at 1 kHz per condition. The measurements and the fabrication follow
Koç et al. (2025), *The Piezoelectric Nanogenerators Based on PVDF/BaTiO3/MWCNT
Ternary Composite Prepared by the Electrospinning Method*, Journal of Materials
Science 60(48):25481-25503, doi:10.1007/s10853-025-11872-9.

| File | Contents |
|---|---|
| `data/design_table.csv` | One row per condition (75 rows): `condition_id` (0-74), `composition`, `cnt_pct`, `force_N`, `freq_Hz`, `is_pristine`, targets `Vrms_V`, `Vpp_V`, `Vmax_V` and `energy_proxy` (1000 x Vrms^2) |
| `data/targets_design.parquet` | The same table in the form the scripts read (`rms_Voc`, `Vpp`, `Vmax`, `energy` columns; row index = `condition_id`) |
| `data/lcr/lcr_tidy.csv` | LCR measurements of the five compositions, 201 frequencies from 1 kHz to 1 MHz each: `Cs_F`, `Cp_F`, loss tangent `D`, stated electrode area and thickness, relative permittivity recomputed from them. The stated geometry gives a relative permittivity below one for four compositions, so it is flagged TO CONFIRM and the permittivity is not used as a material property |
| `data/external_peng/*.csv`, `*.provenance.json` | Values read from figures of four open-access nanogenerator papers (Salama 2024, Shi 2025, Li 2022, Park 2017) with axis calibration and reading error per study; cite the original articles when using them. The source PDFs and annotated overlays are not redistributed |
| `data/external/` | UCI benchmarks (Airfoil Self-Noise, Energy Efficiency, Concrete, Yacht, Combined Cycle Power Plant), unchanged; provenance in `data/external/NOTE.txt` |
| `results/lcr/lcr_summary.csv` | Per-composition summary of the LCR measurements |

**Raw waveforms are not part of this repository. The sampled voltage waveforms are
available from the authors on reasonable request (TO CONFIRM by the authors).** The
design table is derived from them; the scripts that start from the waveforms
(`a1_segment.py`, `a1_specimen.py`, `a4_figures.py`, `a6_twin.py`, `a7_dynforce.py`,
`a11_freq.py`, `figures/make_cycles.py`) need `data/long.parquet` and `data/series_index.parquet` and cannot be rerun
without them; their outputs are shipped under `results/`. The original LCR workbook is
likewise available on request; `data/lcr/lcr_tidy.csv` holds every value of its five
labelled blocks.

## How to reproduce

Python 3.12.4 on Linux. Package versions of the environment the results were produced in are in
`requirements.txt` (numpy 2.3.5, pandas 2.2.3, scipy 1.17.1, scikit-learn 1.7.2,
statsmodels 0.14.6, xgboost 3.3.0, matplotlib 3.11.0).

```bash
git clone https://github.com/egerberkuslu/pvdf-peng-ml.git
cd pvdf-peng-ml
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All scripts are run from the repository root as `python code/<script>.py`; paths come
from `code/peng_paths.py` (set the environment variable `PENG_ROOT` to work on another
copy). Seeds are fixed, so a rerun on the same library versions rewrites identical
files; other versions change low-order digits (see Notes).

Run order (each step reads the files written by the earlier ones, which are already
in `results/`, so every script can also be run on its own):

1. `python code/run_baseline.py` (optional): the twelve scripts of `code/baseline/`
   into `results/baseline/`; they overwrite the shipped first-stage files (see Notes).
2. `python code/a2_physgp.py` (about 1 minute on 16 cores): LawGP identifiability,
   per-level axes, selection rule; writes `a2_physgp.json`, `numbers_a2.tex`, tables.
3. `python code/a3_tables.py` (5 to 10 minutes) and `python code/a3_sensitivity.py`:
   benchmark, coverage, active-learning and replication tables.
4. `python code/a5_structured.py`, `a8_specvar.py`, `a9_external.py`, `a10_methods.py`,
   `a12_axisrule.py`, `a13_rankset.py`, `a14_kmode.py` (a few minutes to about half
   an hour each; `a12` and `a14` accept `--smoke` for a reduced run and `--n-jobs`).
5. `python code/lcr_analysis.py` (seconds): LCR summary, RC chain and macros.
6. `python code/figures/make_{axes,coverage,law,lcr,twin}.py`: manuscript result
   figures into `figures/`.

## Which file regenerates what

| Script | Writes | Content (macro prefix) |
|---|---|---|
| `a1_specimen.py`, `a1_segment.py` | `results/a1_*.csv/json`, `numbers_a1.tex`, `tables/tab_spec_*.tex` | cycle segmentation, repeatability, split-half, specimen bootstrap (`spec`); needs raw waveforms |
| `a2_physgp.py`, `a2_report.py` | `a2_physgp.json`, `numbers_a2.tex`, `tables/tab_phys_*.tex` | LawGP/PhysGP against the plain GP, per-level and axis-wise results (`phys`) |
| `a3_tables.py`, `a3_bench.py`, `a3_conformal.py`, `a3_sensitivity.py` | `a3_tables.json`, `a3_groupwise_coverage.json`, `numbers_a3.tex`, `tables/tab_bench_*.tex`, `tab_cov_*.tex`, `tab_al_*.tex`, `tab_rep_*.tex` | benchmark of fourteen regressors, interval coverage, McNemar tests, replication (`bench`, `cov`) |
| `a4_figures.py` | `a4_desc.json`, `numbers_a4.tex`, `tables/tab_desc_optima.tex` | descriptive figures and optima (`desc`); needs raw waveforms |
| `a5_structured.py` | `a5_structured.json`, `numbers_a5.tex`, `tables/tab_alt_*.tex` | structured kernels against plain GP and PhysGP |
| `a6_twin.py` | `a6_twin.json`, `a6_*.csv`, `numbers_a6.tex`, `tables/tab_a6_*.tex` | simulated generator with known ground truth; needs raw waveforms for its first block |
| `a7_dynforce.py` | `a7_dynforce.json`, `a7_*.csv`, `numbers_a7.tex` | RC deconvolution and open-circuit sensitivity; needs raw waveforms |
| `a8_specvar.py` | `a8_specvar.json`, `numbers_a8.tex` | literature-calibrated specimen variability |
| `a9_external.py` | `a9_external.json`, `a9_*.csv`, `numbers_a9.tex` | protocol applied to the digitized literature sweeps |
| `a10_methods.py` | `a10_methods.json`, `a10_*.csv`, `numbers_a10.tex` | fixed against tuned models, active-learning stopping, public datasets |
| `a11_freq.py` | `a11_freq.json`, `numbers_a11.tex` | frequency-axis extensions; needs raw waveforms |
| `a12_axisrule.py` | `a12_axisrule.json`, `a12_units.csv`, `numbers_a12.tex` | inner-fold audit of the axis rule |
| `a13_rankset.py` | `a13_rankset.json`, `numbers_a13.tex` | composition statements that survive specimen variability |
| `a14_kmode.py` | `a14_kmode.json`, `numbers_a14.tex` | K-mode law chosen by BIC |
| `lcr_analysis.py` | `results/lcr/*` | LCR summary, RC high-pass chain, macros (`lcr`) |
| `code/baseline/*.py` | `results/baseline/*.json` | first-stage analyses; `reg_stats.json` and `reg_models.json` (benchmark scores) are shipped, their pipeline is not |
| `code/reg_common.py` | | model factories of the fourteen regressors |
| `code/figures/make_*.py` | `figures/*.pdf/png` | manuscript result figures (axes, coverage, law, lcr, twin; cycles needs raw waveforms) |

## Notes

The recordings 52, 54 and 73 (zero-based `condition_id`) are scaled copies of other
recordings; every headline is also reported without them (n = 72, `*_n72` files and
macros).

The first-stage JSON files in `results/baseline/` are the inputs the later scripts read.
They were produced with earlier library versions; rerunning them with the pinned
versions reproduces the numbers to about the third decimal (for example the
leave-one-out R2 of the PhysGP changes by up to 0.002).

## Citation

See `CITATION.cff`. The manuscript is under preparation; no DOI exists yet.
Please also cite the data sources: Koç et al. (2025) for the measurements, and the
original articles of the digitized sweeps and the UCI benchmarks listed in
`results/a9_provenance.csv` and `data/external/NOTE.txt`.

## License

Code: MIT (see `LICENSE`). Data files keep the licenses of their sources.

## Contact

Open an issue on this repository with the failing command and the output of `pip freeze`.
