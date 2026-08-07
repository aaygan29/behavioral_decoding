# Design

Why the code is shaped the way it is. Each section states a decision, the
alternative it rejected, and what would make it wrong.

---

## 1. Two outcome levels, never merged

`MultimodalDataset` carries `y_individual` (trial-level choice) and
`y_aggregate` (per-stimulus market outcome) as separate fields, and the pipeline
evaluates them separately.

**Rejected alternative:** a single target. It would be simpler and it would
destroy the finding. The published result is that these two levels come apart
(see [literature.md](literature.md) §1). A framework that collapses them cannot
express the claim it exists to test.

**Consequence:** two units of analysis, two grouping keys, two ways to leak.
Individual-level CV groups by subject. Aggregate-level CV groups by stimulus.

---

## 2. Modalities stay as separate blocks until the very end

Each modality is a `ModalityBlock` with its own feature matrix. They are
concatenated only inside a forecasting arm that explicitly asks for it.

**Rejected alternative:** early fusion, concatenating all features into one
matrix and fitting one model. That is a reasonable default in many settings, and
it is wrong here for three reasons.

1. It makes per-modality reliability unmeasurable, so accuracy-weighted
   reconciliation has nothing to weight with.
2. Dimensionalities differ by two orders of magnitude. A 4-dimensional ROI
   vector concatenated with a 1536-dimensional embedding is not a fusion; it is
   the embedding with four extra columns.
3. Modalities have different missingness. Early fusion forces one imputation
   policy across all of them.

**What would change this:** if the sample grew to thousands of subjects, a
jointly-trained multimodal network would likely beat late fusion. At the sample
sizes this literature works with, it will not.

---

## 3. Bagging wraps the resampling pipeline, not the other way around

```
BaggingClassifier
  └── Pipeline(StandardScaler → AdaptiveOverSampler → base estimator)
```

Bagging draws the bootstrap replicate first. Scaling and resampling then happen
*inside* that replicate.

**Why this order:** every bag sees a different synthetic minority set. That
variation is the point. A single SMOTE draw has idiosyncrasies, and with a small
minority class those idiosyncrasies are large; averaging over bags dilutes them.
Resample-then-bag would give every bag identical synthetic points, throwing that
away, and would also leak.

**Cost:** slower, by roughly the bag count. Acceptable at these data sizes.

---

## 4. Reconciliation weights come from out-of-fold estimates only

`MultimodalEnsemble.fit` runs an inner subject-grouped CV per modality, scores
each on its out-of-fold predictions, and weights by how far that score exceeds
chance.

**Rejected alternative:** weight by training accuracy. This is the obvious
implementation and it is actively harmful. The face block has more features than
trials, so its training accuracy is near 1.0 regardless of whether it carries
signal. Weighting by it would hand the ensemble to whichever modality overfits
hardest.

**Rejected alternative:** learn the weights with a stacking meta-learner. Better
in principle, and it needs more data than this design has. A meta-learner over
four modalities with 20 subjects is fitting four parameters on 20 effective
observations. The excess-over-chance rule is a fixed function with no free
parameters, so it cannot overfit the weighting.

**Below-chance modalities get weight zero.** Not a floor, zero. A modality that
cannot beat chance on held-out subjects contributes variance and no signal.
`weight_floor` exists as an override for the case where a strong prior says a
modality matters and the sample is too small to show it, but using it means the
weighting is no longer purely data-driven and the run record says so.

---

## 5. Three reconciliation rules, and majority vote is not a fallback

`majority`, `soft`, and `accuracy_weighted` are all first-class. Majority vote is
the right choice when subject count is low, because it has no weighting to
overfit. Report it alongside the weighted rule rather than only when the weighted
rule disappoints. Picking whichever rule scored best after seeing the results is
selection on the test set.

The cleanest practice: fix the rule in the config before the run, and report the
others as a robustness check with the selection rule stated.

---

## 6. Nested CV for anything reported

`cross_validate_ensemble` estimates reconciliation weights by an inner CV
*inside each outer training set*. Outer test subjects contribute nothing to the
weights, the resampling, or the scaling.

**What this protects against:** fitting the ensemble once on everything and
reporting its internal out-of-fold score. That number looks like an honest
out-of-fold estimate and is not, because weight selection saw every subject.

The function also returns per-fold weights so weight stability can be inspected.
A modality whose weight swings between 0.0 and 0.6 across folds is not reliably
useful; that is a small sample talking, and the spread is reported rather than
averaged away.

---

## 7. Resampling adapts or steps aside, and never crashes a run

`AdaptiveOverSampler` resolves `k_neighbors` against the minority count in the
slice it actually receives, and returns the data untouched when even that is not
enough.

**The bug this fixes** was real and was caught by the test suite during
development. `recommend_strategy` picks `k` from the full training set, but the
resampler then runs on each inner CV fold, and inside that on each bagging
bootstrap. A block with 40 minority trials overall routinely hands a single bag
2 of them, and `SMOTE(k_neighbors=5)` raises on that.

**Why stepping aside is the right behaviour, not a fudge:** interpolating a
minority class from two examples does not describe a class, it draws a line
segment between two points. `recommend_strategy` would have chosen class weights
for a sample that small anyway, and the base estimator still carries
`class_weight="balanced"`. Skips are logged at DEBUG, because under bagging they
are expected and frequent.

---

## 8. SMOTE is not the default answer to imbalance

`recommend_strategy` returns `sampler="none"` with class weights below roughly
3:1 imbalance, and refuses SMOTE entirely below 6 minority samples. Above that
it recommends `borderline` for high-dimensional face embeddings and plain
`smote` elsewhere.

**Why:** SMOTE interpolates linearly between minority neighbours. That is
defensible for ROI betas, where the space is low-dimensional and roughly
continuous. It is much less defensible for ViT embeddings, where the midpoint
between two subjects' faces is a face belonging to no one. Borderline-SMOTE
synthesises only near the decision boundary, which limits the damage.

Below 3:1, class weights achieve the same effect without synthesising anything,
and there is no reason to prefer the riskier tool.

---

## 9. The ViT fallback announces itself

`ViTEncoder` falls back to a fixed random projection when torch and transformers
are missing, so the pipeline stays runnable. It sets
`backend="random_projection_fallback"`, reports `reportable: False` in its
provenance, and `assert_real_encoder()` raises.

**Why not just require torch:** it makes the whole repo uninstallable for
someone who only wants the fMRI arm.

**Why not fall back silently:** a run that produced numbers from a random
projection would be indistinguishable from a real one in the output. Any script
that generates reportable numbers should call `assert_real_encoder()` first.

---

## 10. Loaders do not clean data

`BaseLoader` subclasses turn files into a `ModalityBlock` and nothing else. No
filtering, no imputation, no standardisation.

**Why:** anything that estimates parameters across trials must happen inside the
CV fold. A loader that z-scores across all trials has already leaked, and the
leak is invisible downstream. Per-recording signal conditioning (bandpass
filtering, epoching, motion regression) is fine in a loader because it does not
look across the train/test boundary.

`FMRILoader(standardize=True)` exists for the case where you know what you are
doing, and is off by default with a comment saying why.

---

## 11. Synthetic data is a positive control, not a placeholder

`synthetic.py` builds the Genevsky/Knutson dissociation in by construction: a
generalisable channel that forecasts the market, an integrative channel that
predicts the individual, and self-report dominated by subject-specific taste.

`scripts/run_demo.py` then checks that the pipeline *recovers* it, and
`tests/test_neuroforecast.py::test_the_dissociation_holds_across_seeds` checks
that the recovery is not a lucky seed.

A framework that misses an effect present by construction cannot be trusted to
find it in real data. This check should stay green, and if it goes red the
correct response is to fix the pipeline, not to relax the threshold.

---

## 12. What is not built yet

Stated plainly so nobody assumes otherwise.

- **No real data is wired in.** The loaders have BIDS/MNE/OpenCV paths, and
  none of them have been run against a real dataset. Expect to fix things.
- **The fMRI extractor is a peak-window average, not a GLM.** Fine for getting
  the pipeline running. Replace with `nilearn.glm.first_level` before reporting.
- **No hyperparameter search.** Defaults are literature-motivated and
  untuned. Any search must be nested inside the outer CV or it leaks.
- **No multiple-comparison correction across arms.** `compare_forecast_arms`
  runs six arms and reports six p-values. Correct them, or preregister one
  primary arm.
- **The EEG and face arms have no published precedent** in this paradigm. See
  [literature.md](literature.md) §3.
