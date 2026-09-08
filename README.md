# behavioral_decoding

Multimodal decoding of **individual choice** and **aggregate market behaviour**
from fMRI, EEG, facial video, and self-report.

AIxBio Africa project. Aayush Gandhi and Gowthaam Gokulakrishnan.

---

## The idea

A small group of people in a scanner can forecast what a large population will
do, and the neural signal that does the forecasting is *not* the same signal that
predicts what any one of those people will choose. Genevsky, Yoon and Knutson
(2017) found that nucleus accumbens and medial prefrontal activity both predicted
individual crowdfunding choices, but only NAcc generalised to forecast market
funding outcomes weeks later, and the scanned sample's own behavioural measures
forecast nothing.

This repository is a framework for testing that **individual-vs-aggregate
dissociation** across four modalities at once, with the leakage guards, baselines,
and calibration checks a claim like it requires. Full citations, with DOIs, in
[`docs/literature.md`](docs/literature.md).

---

## Quick start

```bash
git clone https://github.com/aaygan29/behavioral_decoding.git
cd behavioral_decoding
pip install -e ".[dev]"
python scripts/run_demo.py --quick
```

The demo generates synthetic multimodal data with the Genevsky/Knutson
dissociation built in by construction, runs the full pipeline, and checks that it
recovers what was planted:

```
checks
  [PASS] individual choice beats chance out of fold        balanced accuracy = 0.633
  [PASS] individual result survives label permutation      permutation p = 0.0050
  [PASS] brain forecasts the market better than self-report brain oos R2 = 0.467 vs behaviour 0.254
  [PASS] brain market forecast is above the mean baseline  brain oos R2 = 0.467
```

These four checks are the framework's **positive control**: a pipeline that
misses an effect present by construction cannot be trusted to find one in real
data.

### Optional extras

Installed only when you need the corresponding modality or backend:

```bash
pip install -e ".[fmri]"     # nilearn, nibabel:    fMRI loading + ROI extraction
pip install -e ".[eeg]"      # mne:                 EEG loading
pip install -e ".[riemann]"  # pyriemann:           affine-invariant EEG backend
pip install -e ".[face]"     # opencv-python:       action-unit / landmark face features
pip install -e ".[vision]"   # torch, transformers: the real ViT face encoder
```

---

## Status

Framework and validation harness, working end to end on synthetic data. Two
real-dataset loaders are built and tested against their actual file formats:

- **DEAP** (`io/deap.py`, [`docs/deap.md`](docs/deap.md)): EEG + peripheral +
  behaviour. Needs the licensed download; its YouTube stimuli give it a real
  aggregate outcome via view counts.
- **NARPS ds001734** (`io/narps.py`, [`docs/narps.md`](docs/narps.md)): the fMRI
  mixed-gambles reward task, reusing `FMRILoader`'s NAcc/vmPFC/AIns sphere
  extraction. Public OpenNeuro download, no licence.

Run either end to end on a synthetic, format-real fixture without any download:

```bash
python scripts/run_deap.py --demo    # whole DEAP path on a synthetic fixture
python scripts/run_narps.py --demo   # whole NARPS BIDS path (needs .[fmri])
```

The MNE/OpenCV loader paths have not yet been run against real recordings; expect
to fix things. [`docs/design.md`](docs/design.md) §12 lists what is not built.

**African cohort (behaviour + aggregate arm): built and run on real data.**
DEAP and NARPS are both Western-collected. `io/behavior.py`'s loader has been
run, unmodified, against 3,023 real Kenyan Kiva microloan records
([`scripts/run_africa_cohort.py`](scripts/run_africa_cohort.py),
[`docs/africa_cohort.md`](docs/africa_cohort.md)) as a same-pipeline control:
individual-level loan repayment (balanced accuracy 0.630, 95% CI
0.615–0.646) and an aggregate arm forecasting sector-level repayment rate from
held-out sectors (R² 0.406, n=14 sectors, noisy at that n, flagged as such by
the pipeline itself). No open African fMRI/EEG dataset with this project's
modality coverage exists yet, so there is no neural arm here, see
`docs/africa_cohort.md` for exactly what was dropped as confounded (a
near-leakage column) and what statistic breaks (in an informative way) on
single-observation-per-subject data. Remaining candidate datasets and gaps are
tracked in [`docs/data_sources.md`](docs/data_sources.md).

> **NARPS validates the plumbing, not the market claim.** NARPS is an
> *individual-level* fMRI check that the reward ROIs recover accept/reject. Its
> "aggregate" arm is the population acceptance rate of each gamble, which gain and
> loss forecast almost by construction, so behaviour is *expected* to win there.
> The market-forecasting evidence comes only from datasets with a genuine external
> outcome (DEAP view counts; the crowdfunding / microloan / video studies in
> `docs/literature.md`). The two levels are kept in separate arms of the run
> record so they are never conflated.

---

## How it works

### Modalities and the trial contract

Every loader returns a `ModalityBlock`: a trial-by-feature matrix keyed twice, by
`subject_id` and by `stimulus_id`.

| Modality | Default features | Loader |
|---|---|---|
| fMRI | NAcc, MPFC, and anterior insula spheres (the anticipatory-affect ROIs from the neuroforecasting papers) | `io/fmri.py` |
| EEG | Per-epoch log band power (delta–gamma) + early-frontal / late-parietal ERP windows, or inter-channel covariance for the Riemannian path | `io/eeg.py` |
| Face | Vision-transformer frame embeddings, or interpretable action-unit and landmark features | `io/face.py` |
| Behaviour | Ratings and derived choice features. Response time is **off by default** for choice prediction: it is measured *after* the decision, so using it to predict that decision leaks the outcome | `io/behavior.py` |

Subject keys keep cross-validation honest. Stimulus keys let individual responses
pool into a group-level market forecast. Blocks are aligned by an explicit
`(subject, stimulus)` join. `features/align.py` refuses duplicate keys and
reports what an inner join costs before you pay it, rather than zipping arrays and
hoping.

### Cross-validation and leakage guards

Two leaks below both produce better-looking numbers and neither raises an error,
so both are enforced in code and checked in CI rather than remembered.

- **Subject leakage.** All individual-level CV groups by subject. A random
  trial-level split puts the same person on both sides of a fold, and the model
  learns to recognise the person instead of the choice.
- **Resampling leakage.** SMOTE runs *inside* the fold, on training data only.
  `tests/test_leakage.py` reproduces the classic mistake on pure noise, where the
  only honest answer is chance:

  | | mean balanced accuracy on noise |
  |---|---|
  | resample, then split | 0.59 |
  | split, then resample inside the pipeline | 0.47 |

Also enforced: nested CV for anything reported, subject-level bootstrap intervals
(a trial-level interval on 12 subjects is roughly three times too narrow),
permutation tests that can never return p = 0, and a metrics report that prints
the majority-class baseline next to accuracy.

### fMRI confound preprocessing

A raw fMRIPrep `*_desc-confounds_timeseries.tsv` is never handed to Nilearn as-is.
`io/confounds.select_confounds` picks an explicit, named nuisance set (default
`motion12+physio`: six motion parameters, their derivatives, CSF, white matter)
rather than regressing out all hundred-plus columns, and it handles the reality of
the files: the leading-row NaNs on every `*_derivative1` and
`framewise_displacement` column are filled (column mean by default), and
non-numeric or all-NaN columns are dropped by name. What was kept, requested but
missing, dropped, and how many NaN cells were filled all land in the block
provenance. `FMRILoader.load` exposes `confound_strategy`, `confound_columns`, and
`confound_fill`.

### Class imbalance and bagging

```
BaggingClassifier
  └── Pipeline(StandardScaler → AdaptiveOverSampler → base estimator)
```

Bagging draws the bootstrap replicate first; scaling and SMOTE happen inside it,
so every bag sees a different synthetic minority set and no single SMOTE draw
dominates. Resample-then-bag would give every bag identical synthetic points, and
would also leak.

`AdaptiveOverSampler` resolves `k_neighbors` against the minority count in the
slice it actually receives and steps aside when there is too little minority data
to interpolate. `recommend_strategy` declines SMOTE when it is not warranted:
class weights below roughly 3:1 imbalance, Borderline-SMOTE for high-dimensional
face embeddings, plain SMOTE elsewhere, and nothing at all below six minority
samples.

### Per-modality estimators

Each modality's base learner is chosen for its feature structure and is a one-line
config change ([`docs/estimators.md`](docs/estimators.md)):

- **fMRI, EEG, face**: `elasticnet` (L1+L2 logistic) by default. The L2 term
  shares weight across correlated features (NAcc_L/NAcc_R, collinear band-power
  columns, high-dimensional embeddings); the L1 term still drops dead ones.
- **face**: `linear_svm` is a supported alternative (comparable, slower).
- **behaviour**: `gradient_boosting` for its low-dimensional, mixed, monotone
  features.
- **EEG, Riemannian**: `riemann` classifies in the tangent space of the
  channel-covariance manifold. Much of EEG's signal lives in inter-channel
  covariance, which sits on the manifold of SPD matrices, not a flat vector space,
  so the pipeline projects to the tangent space *first*:
  `Pipeline(RiemannianTangentSpace → StandardScaler → SMOTE → logistic)`. Feed it
  covariance features (`EEGLoader(include_covariance=True)`); the tangent
  reference is fitted on training data only, so it stays leakage-safe.

  The default metric is the closed-form **log-Euclidean** map (NumPy/SciPy only).
  An optional **affine-invariant** backend (`riemann_metric="riemann"`, via
  `pip install '.[riemann]'`) iterates to the true geometric mean and whitens by
  it (the more principled projection on ill-conditioned real EEG). Selecting it
  without PyRiemann installed raises rather than silently falling back, so an
  "affine-invariant" result is always the real thing. Both paths are validated on
  noisy and short-epoch (rank-deficient) covariances, not just clean synthetic
  data.

Mixed-effects is deliberately *not* shipped as a drop-in: random intercepts do not
transfer to unseen subjects under subject-grouped CV. The reasoning is in
[`docs/estimators.md`](docs/estimators.md).

### Vision transformer for the face arm

`ViTEncoder` wraps a Hugging Face ViT for frame embedding. If torch and
transformers are missing it falls back to a fixed random projection so the
pipeline stays runnable, and it says so: `backend="random_projection_fallback"`,
`reportable: False` in provenance, and `assert_real_encoder()` raises. Call that
method in any script that produces numbers you intend to report.

### Ensemble reconciliation

One model per modality, reconciled three ways:

- **`majority`**: hard vote, ties broken by mean confidence. No weighting to
  overfit, which makes it the honest choice when subject count is low.
- **`soft`**: mean of predicted probabilities.
- **`accuracy_weighted`**: weight each modality by how far its **out-of-fold**
  balanced accuracy exceeds chance.

*Out-of-fold* is the load-bearing word. Weighting by training accuracy would hand
the largest weight to whichever model overfits hardest, which for a
1536-dimensional face embedding is guaranteed. Modalities that cannot beat chance
on held-out subjects get weight zero, not a floor.

```
reconciliation: accuracy_weighted (weights from out-of-fold balanced_accuracy)
  modality   oof_balacc    oof_auc    oof_ap   weight
  fmri           0.6328     0.6926    0.4117   0.6585
  behavior       0.5378     0.5673    0.2719   0.1873
  face           0.5311     0.5585    0.2630   0.1542
  eeg            0.4944     0.5185    0.2192   0.0000  (below chance, dropped)
```

### Nested hyperparameter tuning

`MultimodalEnsemble(tune=True)` runs a subject-grouped hyperparameter search per
modality: elastic-net `C` and `l1_ratio`, resampling `k_neighbors`, and bagging
`n_bags` / `max_samples`. The search is **nested**: it runs entirely inside each
outer training fold, on that fold's subjects, scored by grouped out-of-fold
balanced accuracy, so the outer test subjects never influence the chosen
hyperparameters. Grids are small on purpose (the outer CV re-runs the whole search
in every fold) and overridable via `tune_grid`; the params chosen in each fold are
written to the run record so their stability can be inspected. Off by default
because it multiplies fit cost by the grid size.

### Probability calibration

Because the ensemble reconciles *probabilities*, calibration is part of the claim,
not just ranking. Every `classification_report` carries the **Brier score** and
**Expected Calibration Error**, and the run record stores **reliability-curve**
data (per-bin confidence vs observed frequency) for the pooled out-of-fold
predictions and each modality (`evaluation/metrics.calibration_curve_points`). A
modality that is accurate but overconfident is visible here before it distorts a
soft or weighted vote.

### Two outcome levels, evaluated separately

`compare_forecast_arms` runs brain-only, behaviour-only, face-only, and combined
arms against the aggregate market outcome, holding out whole stimuli:

```
aggregate market forecast (held-out stimuli)
  arm                  n_stim     oos_R2  pearson_r          p
  fmri_only                40     0.4741     0.6896     0.0000
  brain_only               40     0.4674     0.6935     0.0000
  all_modalities           40     0.3904     0.6434     0.0000
  behavior_only            40     0.2537     0.5074     0.0008
```

Reporting only the combined arm would hide the dissociation the whole design
exists to detect. Individual-level CV groups by subject; aggregate-level CV groups
by stimulus.

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

`run_experiment` writes a JSON run record with the resolved config, the git commit
and whether the tree was dirty, per-modality provenance, both evaluation levels,
bootstrap intervals, permutation p-values, calibration curves, and the final
weights. A result whose configuration is not recorded cannot be reproduced.

---

## Layout

```
src/behavioral_decoding/
├── io/              loaders, one per modality, + the ModalityBlock contract; confound cleaning
├── features/        ViT encoding, cross-modality alignment
├── balance/         SMOTE variants, adaptive resampling, strategy selection
├── models/          per-modality bagged learners (incl. Riemannian tangent-space), ensemble reconciliation, tuning
├── evaluation/      grouped CV, metrics + calibration, aggregate forecasting
├── pipelines/       end-to-end run and run-record writing
└── synthetic.py     ground-truth generator (the positive control)

docs/
├── literature.md    the neuroforecasting canon, with verified DOIs
├── design.md        each decision, its rejected alternative, and what would falsify it
├── estimators.md    per-modality base learners incl. the Riemannian EEG path
├── deap.md          DEAP loader: format traps, circularity, the market route
├── narps.md         NARPS loader: format, the individual-vs-aggregate honesty point
└── data_sources.md  candidate datasets per modality, and the gaps between them
```

---

## Development

```bash
pytest                        # full suite (a few minutes)
pytest tests/test_leakage.py -v
ruff check src tests scripts
```

---

## Interpretation limits

Worth reading before writing any of this up; expanded in
[`docs/literature.md`](docs/literature.md) §3.

- **Forecasting is not causation.** Every result here is predictive. None
  establishes that NAcc activity *causes* market outcomes.
- **This does not read intent.** Reverse inference from a small ROI to a specific
  mental state is not supported by these findings. No function in this repo should
  be described as decoding what someone wants.
- **EEG and face do not inherit the fMRI evidence.** The published dissociation is
  an fMRI result. Whether a scalp or facial proxy carries the same
  stimulus-general affective component is open, and one of the more interesting
  things this project could test. Until it is tested here, those arms are
  exploratory and labelled that way.
- **Individual and aggregate levels have different effective sample sizes.** Thirty
  subjects is adequate for aggregate forecasting, where the unit of analysis is the
  stimulus. It is not adequate for strong individual-level decoding claims.

---

## Neuroprivacy risk demonstration (ASZED-153, Nigeria)

A different arm from the rest of this repo, run and documented separately:
[`docs/africa_neuroprivacy.md`](docs/africa_neuroprivacy.md),
[`scripts/run_africa_neuroprivacy.py`](scripts/run_africa_neuroprivacy.py).

Real 19-channel EEG from 76 subjects (schizophrenia patients and matched
controls, Ile-Ife/Ilesa, Nigeria, [ASZED-153](https://zenodo.org/records/14178398),
CC-BY), MD5-verified on download. Each subject's four recorded EEG segments
are treated as four modalities and run through the same accuracy-weighted
multibagging ensemble used everywhere else in this repo, same band-power
feature family, same nested subject-grouped CV, same calibration and
bootstrap reporting, but predicting diagnostic category instead of a market
outcome. Literature-grounded EEG correlates of schizophrenia (resting
frontal delta/theta, reduced alpha, reduced auditory gamma phase-locking)
motivate why this label should live in exactly this kind of short, cheap
recording.

Result: balanced accuracy 0.705 (95% bootstrap CI 0.616–0.799, chance = 0.5)
recovering diagnosis from four short EEG segments never designed to disclose
it. The permutation test this repo normally reports is degenerate for a
subject-level label (documented, not hidden, see the doc). This is the
empirical basis for the risk claim in the section below: an off-the-shelf
pipeline, real African clinical EEG, and a handful of short recordings are
enough to leak a protected health attribute the recording session was not
about.

**The risk is not EEG-specific.**
[`docs/biosignal_privacy_generalization.md`](docs/biosignal_privacy_generalization.md)
shows why the same pipeline absorbs cardiac, electrodermal, pupillometric, and
endocrine signals as additional modality blocks with no change to the
mathematics, lists the parameters that set the size of the risk, and reports a
second real-data run on a non-neural biosignal: this repo's ensemble applied to
the [PPG-BP Database](https://doi.org/10.1038/sdata.2018.20) (219 fingertip
photoplethysmograms, Guilin, China), decoding undisclosed cardiovascular and
metabolic status
([`scripts/run_biosignal_privacy.py`](scripts/run_biosignal_privacy.py)). That
arm is a *weaker* effect than ASZED-153 and is reported as such: the point of
running it was to test the generalization honestly, not to manufacture a second
alarming number.

**Fusion is what makes it dangerous.**
[`docs/biosignal_fusion.md`](docs/biosignal_fusion.md) closes the harder question:
whether reconciling several biosignal families at once beats the best single one.
On a generator with planted structure, run through this repo's ensemble unchanged,
fused balanced accuracy reaches 0.62 against a best-single-family 0.58 (ROC AUC
0.60 to 0.68), an ablation ladder rises as each independent family joins and is
flat when a pure-noise family is appended, and five runtime gates verify the
result ([`scripts/run_biosignal_fusion.py`](scripts/run_biosignal_fusion.py),
figure in [`docs/figures/`](docs/figures/biosignal_fusion.png)). The inverse-variance
argument (independent weak leaks add up) is the reason aggregate access to cheap
biosignals is a biosecurity concern, not just a privacy one.

---

## Relevance to the Global South and Africa

*Context and motivation, kept separate from the method above. Nothing in this
section changes what the pipeline does or how a result is validated, see
[Interpretation limits](#interpretation-limits) for the actual scientific
caveats.*

This is an **AIxBio Africa** project. The neuroforecasting literature it builds
on (Genevsky, Yoon, Knutson) comes almost entirely from WEIRD-population fMRI
samples, and open neuroimaging data and hardware carry the same skew. Doing
this work as an African-led effort, rather than importing a finished pipeline
and a foreign dataset, is part of the point: African researchers set and own
the methodology here, the eventual test of whether the effect holds in African
populations and markets is treated as an open question rather than assumed, and
the reproducibility machinery (run records, provenance, calibration reporting)
exists partly so results can't be quietly overclaimed or exported without an
audit trail, a real risk in a research environment with weaker regulatory
oversight than where this literature originated. That gap is checked, not
assumed: as of 2026-08-17 neither Nigeria nor Kenya has a binding,
AI-specific national law, per the
[Global AI Governance Map](https://global-ai-governance-map.vercel.app/) (see
the sourced table in
[`docs/africa_neuroprivacy.md`](docs/africa_neuroprivacy.md#governance-context-why-this-risk-isnt-hypothetical)).
The
[neuroprivacy demonstration above](#neuroprivacy-risk-demonstration-aszed-153-nigeria)
is what that risk looks like when it isn't hypothetical: real African
clinical EEG, an off-the-shelf pipeline, and a decodable protected attribute
the recording wasn't about. None of the rest of this section is a scientific
claim; it's why the project exists and who it's built to serve.

---

## Licence

MIT. See [LICENSE](LICENSE).
