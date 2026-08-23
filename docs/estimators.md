# Per-modality estimators

Which base learner each modality uses, why, and how to change it. Also an honest
account of two stronger models that were suggested and do **not** drop into this
framework's pipeline unchanged, and what it would take to add them properly.

## The pipeline each estimator lives in

Every modality model is a bag of identical pipelines:

```
BaggingClassifier
  └── Pipeline(StandardScaler -> AdaptiveOverSampler(SMOTE) -> base estimator)
```

So a base estimator here always sees **standardised, class-balanced, flat
feature vectors**. Two consequences that decide what fits:

- The estimator must expose `predict_proba`. The ensemble weights modalities on
  out-of-fold probabilities (`evaluation/cv.out_of_fold_proba`), so a learner
  without calibrated probabilities is unusable as-is.
- The features are a flat `(n_trials, n_features)` matrix, column-standardised.
  Anything whose feature space is *not* a flat Euclidean vector (a covariance
  matrix on its manifold, say) is distorted by per-column scaling and by SMOTE's
  straight-line interpolation before the estimator ever sees it.

## Defaults

| modality | default | why |
|---|---|---|
| fMRI | `elasticnet` | ROI betas are correlated (NAcc_L/NAcc_R move together); L2 shares weight across them, L1 drops dead ROIs |
| EEG | `elasticnet` | band-power and ERP columns are collinear; regularised logistic is the right baseline |
| face | `elasticnet` | high-dimensional embeddings; `linear_svm` is an equally good, slower alternative |
| behaviour | `gradient_boosting` | low-dimensional, mixed, monotone economic/self-report features |

Set them per experiment in the config:

```yaml
model:
  base_learner:
    fmri: elasticnet     # or logistic, linear_svm, svm, random_forest
    face: linear_svm
```

## The learners

- **`elasticnet`**: L1+L2 logistic (`saga` solver, `l1_ratio=0.5`). The default
  for every dense neural/embedding block. The L2 term shares weight across
  correlated features instead of arbitrarily picking one, which plain L1 (lasso)
  does badly with collinear ROIs; the L1 term still zeroes out uninformative
  features, which plain L2 never does. Needs standardised input, which the
  pipeline provides.
- **`linear_svm`**: linear-kernel SVC with Platt-scaled probabilities. A strong
  baseline for high-dimensional embeddings and often within noise of elastic-net
  on the face block. Slower, because `probability=True` fits an internal CV for
  calibration; that is why elastic-net is the face default and this is the
  documented alternative.
- **`logistic`**: plain L2 logistic. Kept as a simple, fast reference.
- **`svm`**: RBF-kernel SVC. Non-linear; rarely the right call at these sample
  sizes, but available.
- **`random_forest`**, **`gradient_boosting`**: tree ensembles for the tabular
  behaviour block.
- **`riemann`**: logistic regression in the log-Euclidean tangent space of EEG
  channel covariances. Needs covariance features, not band power. Detailed in
  its own section below.

## Riemannian tangent-space classification for EEG (implemented)

`riemann` is a real estimator now, not a base-learner swap: it is logistic
regression in the **log-Euclidean tangent space** of the EEG channel-covariance
manifold. Use it like any other learner, but feed it covariance features:

```python
from behavioral_decoding.io.eeg import EEGLoader

# Covariance mode is mutually exclusive with band power / ERP.
block = EEGLoader(
    include_bandpower=False, include_erp=False, include_covariance=True
).from_arrays(epochs, sfreq, times, subject_ids, stimulus_ids)
```

```yaml
model:
  base_learner:
    eeg: riemann      # expects covariance features, not band power
```

How it stays correct inside the existing pipeline:

```
BaggingClassifier
  └── Pipeline(RiemannianTangentSpace -> StandardScaler -> SMOTE -> logistic)
```

The tangent projection goes *first*, so scaling and SMOTE only ever touch the
flat Euclidean tangent vectors, never the raw covariance entries. The reference
point (the log-Euclidean mean) is computed in `fit` on training data only, inside
each fold and each bag, so it is leakage-safe. Rank-deficient covariances from
short epochs are regularised toward a scaled identity before the matrix log, and
`max_features` is forced to 1.0 because a column subset of a flattened covariance
is not a covariance.

Two honest caveats:

- **On clean data the tangent map does not beat a plain logistic**, because the
  raw covariance entries are already linearly separable there. Its advantage
  shows on ill-conditioned real EEG, which is the regime the method was built
  for. The tests assert the path recovers covariance structure *above chance*,
  not that it beats logistic on synthetic data, which would be cherry-picking.
- The default is the **log-Euclidean** metric (closed-form, scipy-only). The
  **affine-invariant** metric, which iterates to a geometric mean and whitens by
  it, is now available as an optional backend:
  `build_modality_model("eeg", base_learner="riemann", riemann_metric="riemann")`
  (or `RiemannianTangentSpace(metric="riemann")`), which uses `pyriemann`
  (`pip install '.[riemann]'`). It is the more principled projection on
  ill-conditioned real EEG. Selecting it without `pyriemann` installed raises a
  clear `ImportError` instead of silently falling back, so a reported
  affine-invariant result is always the real thing.
- **Validation is not limited to clean data.** `tests/test_riemann.py` exercises
  the tangent map on noisy covariances (heavy additive channel noise) and on
  short-epoch, rank-deficient covariances, asserting finite output and
  above-chance recovery under noise, which is the regime the method is for.

## The other suggested model, and why it is not a drop-in

Recorded so the decision is deliberate, not forgotten.

### Hierarchical / mixed-effects model (fMRI)

A random-intercept-per-subject logistic model is the textbook way to handle
subject-to-subject differences, and within a single sample it usually beats a
pooled model. The catch here is the **cross-subject cross-validation**. This
framework groups folds by subject, so every test subject is unseen at fit time.
A per-subject random intercept has no estimate for a subject the model never
saw; the best it can do at test is fall back to the population mean, which is
what a pooled model already gives. So the partial-pooling benefit that makes
mixed models shine largely evaporates for *held-out-subject* generalisation,
which is the quantity this framework reports.

Mixed effects would help if the evaluation were within-subject (predicting new
trials for subjects already seen). It is not, on purpose: individuating a person
from their own repeated trials is a much easier and less interesting claim than
generalising across people. Elastic-net with `class_weight="balanced"` plus
subject-grouped CV is the honest default here.

If you still want it: fit `statsmodels` `BinomialBayesMixedGLM` (a random
intercept per subject) inside each training fold, and at test time drop the
random-effect term and predict from fixed effects only. Wrap it to expose
`predict_proba`, and report it next to elastic-net rather than replacing it. Do
not expect a cross-subject gain.

(The other model previously in this section, Riemannian tangent-space
classification for EEG, is now implemented; see the section above.)

## What did change

`elasticnet` and `linear_svm` were added to the learner factory and the
dense-block defaults moved from plain logistic to elastic-net. The `riemann`
path was then added for EEG: a covariance feature family in the loader and a
log-Euclidean tangent-space transformer prepended to the pipeline. Everything
else about the pipeline (bagging, in-fold SMOTE, out-of-fold weighting, subject-
grouped CV) is unchanged. The dense-block swaps are honest swaps of the base
estimator; the Riemannian path adds a manifold-correct front-end without
changing how anything is evaluated.
