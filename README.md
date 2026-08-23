# behavioral_decoding

Multimodal decoding of **individual choice** and **aggregate market behaviour**
from fMRI, EEG, facial video, and self-report.

AIxBio Africa project. Aayush Gandhi and Gowthaam Gopalakrishnan.

---

## The idea in one paragraph

A small group of people in a scanner can forecast what a large population will
do, and the neural signal that does the forecasting is not the same signal that
predicts what any one of those people will choose. Genevsky, Yoon and Knutson
(2017) found that nucleus accumbens and medial prefrontal activity both
predicted individual crowdfunding choices, but only NAcc generalised to forecast
market funding outcomes weeks later, and the scanned sample's own behavioural
measures forecast nothing. This repository is a framework for testing that
dissociation across four modalities at once, with the leakage guards and
baselines that a claim like it requires. Full citations, with DOIs, in
[`docs/literature.md`](docs/literature.md).

---

## Status

Framework and validation harness, working end to end on synthetic data. Two
real-dataset loaders are built and tested against their actual file formats:

- **DEAP** (`io/deap.py`, [`docs/deap.md`](docs/deap.md)): EEG + peripheral +
  behaviour; needs the licensed download; YouTube stimuli give it a real
  aggregate outcome via view counts.
- **NARPS ds001734** (`io/narps.py`, [`docs/narps.md`](docs/narps.md)): the
  fMRI mixed-gambles reward task; reuses `FMRILoader`'s NAcc/vmPFC/AIns sphere
  extraction; public OpenNeuro download, no licence. NARPS is an
  *individual-level* fMRI validation: on gambles the economic baseline dominates
  the aggregate arm by construction, so it is the fMRI plumbing check, not a
  brain-beats-behaviour demonstration.

> **NARPS is not evidence for the market-forecasting claim.** The
> neuroforecasting thesis (§6, §Interpretation) is that NAcc predicts an
> *aggregate market outcome* which behaviour fails to forecast. NARPS has no
> such market outcome: its "aggregate" arm is the population acceptance rate of
> each gamble, and gain/loss forecast that rate almost by construction, so
> behaviour is *expected* to win there. NARPS earns its place only as an
> individual-level check that the NAcc/vmPFC/AIns extraction recovers choice.
> The market-forecasting evidence comes from datasets with a genuine external
> outcome (DEAP view counts; the crowdfunding/microloan/video studies in
> `docs/literature.md`), and the two are kept in separate arms of the run record
> so they are never conflated.

The MNE/OpenCV loader paths still have not been run against real recordings;
expect to fix things. See [`docs/design.md`](docs/design.md) §12 for the full
list of what is not built.

```bash
python scripts/run_deap.py --demo    # whole DEAP path on a synthetic fixture, no download
python scripts/run_narps.py --demo   # whole NARPS BIDS path on a synthetic fixture (needs .[fmri])
```

---

## Quick start

```bash
git clone https://github.com/aaygan29/behavioral_decoding.git
cd behavioral_decoding
pip install -e ".[dev]"
python scripts/run_demo.py --quick
```

The demo generates synthetic multimodal data with the Genevsky/Knutson
dissociation built in by construction, runs the full pipeline, and checks that
it recovers what was planted. Output ends with:

```
checks
  [PASS] individual choice beats chance out of fold       balanced accuracy = 0.633
  [PASS] individual result survives label permutation     permutation p = 0.0050
  [PASS] brain forecasts the market better than self-report brain oos R2 = 0.467 vs behaviour 0.254
  [PASS] brain market forecast is above the mean baseline brain oos R2 = 0.467
```

Those four checks are the framework's positive control. A pipeline that misses
an effect present by construction cannot be trusted to find it in real data.

Optional extras, installed only if you need them:

```bash
pip install -e ".[fmri]"    # nilearn, nibabel
pip install -e ".[eeg]"     # mne
pip install -e ".[face]"    # opencv-python
pip install -e ".[vision]"  # torch, transformers (the real ViT encoder)
```

---

## What it does

### 1. Four modalities, one trial contract

Every loader returns a `ModalityBlock`: a trial-by-feature matrix keyed twice,
by `subject_id` and by `stimulus_id`.

| Modality | Default features | Loader |
|---|---|---|
| fMRI | NAcc, MPFC, and anterior insula spheres, the anticipatory-affect ROIs from the neuroforecasting papers; fMRIPrep confounds selected and cleaned before extraction | `io/fmri.py`, `io/confounds.py` |
| EEG | Per-epoch log band power (delta to gamma) plus early-frontal and late-parietal ERP windows, or inter-channel covariance for the Riemannian path | `io/eeg.py` |
| Face | Vision-transformer frame embeddings, or interpretable action-unit and landmark features | `io/face.py` |
| Behaviour | Ratings, derived choice features (response time is available but **off by default** for choice prediction: it is measured after the decision, so using it leaks the outcome) | `io/behavior.py` |

Subject keys keep cross-validation honest. Stimulus keys are what let individual
responses pool into a group-level market forecast.

Blocks are aligned by explicit `(subject, stimulus)` join, not by zipping arrays
and hoping. `features/align.py` refuses duplicate keys and reports what an inner
join costs before you pay it.

### 2. Bagging, with resampling nested correctly

```
BaggingClassifier
  └── Pipeline(StandardScaler → AdaptiveOverSampler → base estimator)
```

Bagging draws the bootstrap replicate first; scaling and SMOTE happen inside it.
Every bag therefore sees a different synthetic minority set, which dilutes the
idiosyncrasies of any single SMOTE draw. Resample-then-bag would give every bag
identical synthetic points and would also leak.

`AdaptiveOverSampler` resolves `k_neighbors` against the minority count in the
slice it actually receives, and steps aside when there is not enough minority
data to interpolate meaningfully. Without it, a bagging bootstrap that happens
to draw two minority rows takes the whole run down.

`recommend_strategy` also declines to use SMOTE when it is not warranted: class
weights below roughly 3:1 imbalance, Borderline-SMOTE for high-dimensional face
embeddings, plain SMOTE elsewhere, and nothing at all below six minority
samples.

### 3. Majority voting and accuracy-weighted reconciliation

One model per modality, reconciled three ways:

- **`majority`**: hard vote, ties broken by mean confidence. No weighting to
  overfit, which makes it the honest choice when subject count is low.
- **`soft`**: mean of predicted probabilities.
- **`accuracy_weighted`**: weight each modality by how far its **out-of-fold**
  balanced accuracy exceeds chance.

*Out-of-fold* is the load-bearing word. Weighting by training accuracy would
hand the largest weight to whichever model overfits hardest, which for a
1536-dimensional face embedding is guaranteed. Modalities that cannot beat
chance on held-out subjects get weight zero, not a floor.

```
reconciliation: accuracy_weighted (weights from out-of-fold balanced_accuracy)
  modality   oof_balacc    oof_auc    oof_ap   weight
  fmri           0.6328     0.6926    0.4117   0.6585
  behavior       0.5378     0.5673    0.2719   0.1873
  face           0.5311     0.5585    0.2630   0.1542
  eeg            0.4944     0.5185    0.2192   0.0000  (below chance, dropped)
```

### 4. Per-modality estimators

Each modality's base learner is chosen for its feature structure, and is a
one-line config change ([`docs/estimators.md`](docs/estimators.md)):

- **fMRI, EEG, face**: `elasticnet` (L1+L2 logistic) by default. The L2 term
  shares weight across correlated features (NAcc_L/NAcc_R, collinear band-power
  columns, high-dimensional embeddings); the L1 term still drops dead ones.
- **face**: `linear_svm` is a supported alternative (comparable, slower).
- **behaviour**: `gradient_boosting` for its low-dimensional, mixed, monotone
  features.
- **EEG, Riemannian**: `riemann` classifies in the **log-Euclidean tangent
  space** of the channel-covariance manifold. Much of EEG's signal is in
  inter-channel covariance, which lives on the manifold of SPD matrices, not a
  flat vector space, so the pipeline projects to the tangent space *first*:
  `Pipeline(RiemannianTangentSpace → StandardScaler → SMOTE → logistic)`. Feed
  it covariance features (`EEGLoader(include_covariance=True)`); the tangent
  reference is computed on training data only, so it stays leakage-safe. The
  default metric is the closed-form **log-Euclidean** map (NumPy/SciPy only). An
  optional **affine-invariant** backend (`riemann_metric="riemann"`, via
  `pip install '.[riemann]'` for PyRiemann) iterates to the true geometric mean
  and whitens by it, which is the more principled projection on ill-conditioned
  real EEG; selecting it without PyRiemann installed raises rather than silently
  falling back, so an "affine-invariant" result is always the real thing. Both
  paths are validated on noisy and short-epoch (rank-deficient) covariances in
  `tests/test_riemann.py`, not just clean synthetic data.

Two learners the docs deliberately do **not** ship as drop-ins, with the honest
reasons: mixed-effects (random intercepts do not transfer under subject-grouped
CV) and, previously, Riemannian (now built, above).

### 7. fMRIPrep confound preprocessing

A raw `*_desc-confounds_timeseries.tsv` is never handed to Nilearn as-is.
`io/confounds.select_confounds` picks an explicit, named nuisance set (default
`motion12+physio`: six motion parameters, their derivatives, CSF, white matter)
rather than regressing out all hundred-plus columns, and it handles the reality
of the files: the leading-row NaNs on every `*_derivative1` and
`framewise_displacement` column are filled (column mean by default), and
non-numeric or all-NaN columns are dropped by name. What was kept, what was
requested-but-missing, what was dropped, and how many NaN cells were filled all
land in the block provenance and the run record. `FMRILoader.load` exposes
`confound_strategy` / `confound_columns` / `confound_fill`.

### 8. Nested hyperparameter tuning

`MultimodalEnsemble(tune=True)` runs a subject-grouped hyperparameter search per
modality (elastic-net `C` and `l1_ratio`, resampling `k_neighbors`, and bagging
`n_bags` / `max_samples`). The search is **nested**: it runs entirely inside each
outer training fold on that fold's subjects, scored by grouped out-of-fold
balanced accuracy, so the outer test subjects never influence the chosen
hyperparameters. The grids are small on purpose (the outer CV re-runs the whole
search in every fold) and overridable via `tune_grid`; the params chosen in each
fold are written to the run record so their stability can be inspected. Off by
default because it multiplies fit cost by the grid size.

### 9. Probability calibration

Because the ensemble reconciles *probabilities*, calibration is part of the
claim, not just ranking. Every `classification_report` now carries the **Brier
score** and **Expected Calibration Error**, and the run record stores
**reliability-curve** data (per-bin confidence vs observed frequency) for the
pooled out-of-fold predictions and for each modality
(`evaluation/metrics.calibration_curve_points`). A modality that is accurate but
overconfident is visible here before it distorts a soft or weighted vote.

### 5. Vision transformers for the face arm

`ViTEncoder` wraps a Hugging Face ViT for frame embedding. If torch and
transformers are missing it falls back to a fixed random projection so the
pipeline stays runnable, and it says so: `backend="random_projection_fallback"`,
`reportable: False` in provenance, and `assert_real_encoder()` raises. Call that
method in any script that produces numbers you intend to report.

### 6. Both outcome levels, evaluated separately

`compare_forecast_arms` runs brain-only, behaviour-only, face-only, and combined
arms against the market outcome, holding out whole stimuli:

```
aggregate market forecast (held-out stimuli)
  arm                  n_stim     oos_R2  pearson_r          p
  fmri_only                40     0.4741     0.6896     0.0000
  brain_only               40     0.4674     0.6935     0.0000
  all_modalities           40     0.3904     0.6434     0.0000
  behavior_only            40     0.2537     0.5074     0.0008
```

Reporting only the combined arm would hide the dissociation the whole design
exists to detect.

---

## Guards

The two leaks below both produce better-looking numbers, neither raises an
error, and both are checked in CI rather than remembered.

**Subject leakage.** All individual-level CV groups by subject. A random
trial-level split puts the same person on both sides of the fold and the model
learns to recognise the person instead of the choice.

**Resampling leakage.** SMOTE runs inside the fold, on training data only.
`tests/test_leakage.py` reproduces the classic mistake on pure noise, where the
only honest answer is chance:

| | mean balanced accuracy on noise |
|---|---|
| resample, then split | 0.59 |
| split, then resample inside the pipeline | 0.47 |

Also enforced: nested CV for anything reported, subject-level bootstrap
intervals (a trial-level interval on 12 subjects is roughly three times too
narrow), permutation tests that can never return p = 0, and a metrics report
that prints the majority-class baseline next to accuracy.

---

## Layout

```
src/behavioral_decoding/
├── io/              loaders, one per modality, plus the ModalityBlock contract
├── features/        ViT encoding, cross-modality alignment
├── balance/         SMOTE variants, adaptive resampling, strategy selection
├── models/          per-modality bagged learners (incl. Riemannian tangent-space), ensemble reconciliation
├── evaluation/      grouped CV, metrics, aggregate forecasting
├── pipelines/       end-to-end run and run-record writing
└── synthetic.py     ground-truth generator (the positive control)

docs/
├── literature.md    the neuroforecasting canon, with verified DOIs
├── design.md        each decision, its rejected alternative, and what would falsify it
├── estimators.md    per-modality base learners incl. the Riemannian EEG path; why mixed-effects is not a drop-in
├── deap.md          DEAP loader: format traps, circularity, the market route
├── narps.md         NARPS loader: format, the individual-vs-aggregate honesty point
└── data_sources.md  candidate datasets per modality, and the gap between them
```

---

## Usage

```python
from behavioral_decoding.config import ExperimentConfig
from behavioral_decoding.features.align import build_dataset
from behavioral_decoding.io import BehaviorLoader, EEGLoader, FMRILoader
from behavioral_decoding.pipelines.train import run_experiment

blocks = {
    "fmri": FMRILoader().load(func_paths, events, subject_ids, t_r=2.0),
    "eeg": EEGLoader().load(raw_paths, subject_ids),
    "behavior": BehaviorLoader().load(table, outcome_column="choice"),
}

dataset = build_dataset(
    blocks,
    y_individual={(subject, stimulus): choice, ...},
    y_aggregate={stimulus: market_outcome, ...},
)

config = ExperimentConfig.load("configs/experiment_default.yaml")
record = run_experiment(dataset, config)
```

`run_experiment` writes a JSON run record containing the resolved config, the
git commit and whether the tree was dirty, per-modality provenance, both
evaluation levels, bootstrap intervals, permutation p-values, and the final
weights. A result whose configuration is not recorded cannot be reproduced.

---

## Development

```bash
pytest                    # 66 tests, about 4 minutes
pytest tests/test_leakage.py -v
ruff check src tests scripts
```

---

## Interpretation limits

Worth reading before writing any of this up. Expanded in
[`docs/literature.md`](docs/literature.md) §3.

- **Forecasting is not causation.** Every published result here is predictive.
  None establishes that NAcc activity causes market outcomes.
- **This does not read intent.** Reverse inference from a small ROI to a
  specific mental state is not supported by these findings. No function in this
  repo should be described as decoding what someone wants.
- **EEG and face do not inherit the fMRI evidence.** The published dissociation
  is an fMRI result. Whether a scalp or facial proxy carries the same
  stimulus-general affective component is an open question, and one of the more
  interesting things this project could actually test. Until it is tested here,
  those arms are exploratory and should be labelled that way.
- **Individual and aggregate levels have different effective sample sizes.**
  Thirty subjects is adequate for aggregate forecasting because the unit of
  analysis becomes the stimulus. It is not adequate for strong individual-level
  decoding claims.

---

## Licence

MIT. See [LICENSE](LICENSE).
