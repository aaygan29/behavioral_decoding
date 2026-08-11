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

## Two stronger models, and why they are not drop-ins

Both were suggested and both are genuinely better *for the right setup*. Neither
fits the flat-feature, subject-grouped-CV pipeline without a dedicated path, and
bolting them in naively would produce numbers that look principled and are not.
Recorded here so the decision is deliberate, not forgotten.

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

### Riemannian methods (EEG)

For EEG built from **channel covariance / connectivity** features, Riemannian
tangent-space classification is a real upgrade over band power: it respects the
geometry of the space of covariance matrices instead of treating their entries
as independent numbers. But two steps in this pipeline are wrong for covariance
features:

- **StandardScaler** z-scores each covariance entry independently, which
  destroys the positive-definite structure the Riemannian method depends on.
- **SMOTE** interpolates along straight lines between covariance vectors. The
  geodesic between two covariance matrices is *not* a straight line in entry
  space, so the synthetic minority points are off-manifold.

A correct Riemannian EEG arm therefore needs its own pipeline: emit per-trial
covariance matrices (not band power), project to the tangent space at the
Riemannian mean of the *training* fold, and only then standardise / resample /
classify in that tangent space. That is a separate feature family and a separate
pipeline, best added as a distinct `riemann` path (with `pyriemann`, or a
scipy-`logm` tangent map for a dependency-free version) rather than a base
learner slotted into the existing flat pipeline. It is a good next step; it is
not a one-line default change, and pretending it were would corrupt the manifold
structure it exists to exploit.

## What did change

`elasticnet` and `linear_svm` were added to the learner factory and the
dense-block defaults moved from plain logistic to elastic-net. Everything else
about the pipeline (bagging, in-fold SMOTE, out-of-fold weighting, subject-
grouped CV) is unchanged, so these are honest swaps of the base estimator, not a
change to how anything is evaluated.
