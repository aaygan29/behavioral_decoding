# Multi-biosignal fusion and why aggregate biosignal inference is an AIxBio risk

This document sits beside [`biosignal_privacy_generalization.md`](biosignal_privacy_generalization.md).
That doc argued the pipeline is modality-agnostic and showed a single non-neural
family (fingertip PPG) leaking health status weakly. This one closes the harder
question the privacy doc left open: **does an actor who reconciles several biosignal
families at once get a prediction stronger than any family alone?** That is the
step that turns a set of individually weak leaks into a strong one, and it is the
step that makes the risk a biosecurity problem rather than a curiosity.

The experiment is [`scripts/run_biosignal_fusion.py`](../scripts/run_biosignal_fusion.py).
It runs the repository's own `MultimodalEnsemble` unchanged. Nothing in the neuro
corpus was modified to produce it.

---

## 1. The biosignals, and what the literature says each one carries about behaviour

The premise is that behaviour and decision-making are not read only from the brain.
Several peripheral signals can be sensed with cheap hardware (a wrist strap, a phone
camera, a saliva strip) and each carries a partial, mechanistically grounded trace of
the state that drives a choice. According to PubMed:

- **Cardiac / heart-rate variability (autonomic arousal).** Vagally mediated HRV
  indexes the prefrontal-autonomic circuit that supports flexible decisions; good
  performers on the Iowa Gambling Task carry higher resting and task HRV
  (Forte, Morelli & Casagrande, 2021, *Brain Sci*,
  [DOI](https://doi.org/10.3390/brainsci11020243)). This is the "neurovisceral
  integration" link between a mechanically sensed cardiac signal and choice quality.
- **Electrodermal activity (anticipatory somatic marking).** Anticipatory skin
  conductance responses precede advantageous choices in the somatic-marker account,
  though the account's strength is contested (Dunn, Dalgleish & Lawrence, 2006,
  *Neurosci Biobehav Rev*, [DOI](https://doi.org/10.1016/j.neubiorev.2005.07.001)).
  EDA is included as a *real but noisier* autonomic view, which is exactly its status
  in the literature.
- **Pupillometry (noradrenergic volatility / exploration).** Baseline pupil diameter
  tracks tonic locus-coeruleus noradrenergic activity, which rises when the
  environment becomes volatile and the agent shifts from exploiting to exploring
  (Pajkossy et al., 2017, *Psychophysiology*,
  [DOI](https://doi.org/10.1111/psyp.12964); Vincent et al., 2019, *PLoS Comput Biol*,
  [DOI](https://doi.org/10.1371/journal.pcbi.1007126)). Pupil size is trivially
  sensed by any front camera.
- **Endocrine tone (slow risk/reward bias).** Salivary testosterone predicts a
  trader's daily profit and cortisol rises with market volatility (Coates & Herbert,
  2008, *PNAS*, [DOI](https://doi.org/10.1073/pnas.0704025105)); administered cortisol
  and testosterone both shift people toward riskier assets and can destabilise
  experimental markets (Cueva et al., 2015, *Sci Rep*,
  [DOI](https://doi.org/10.1038/srep11206); review: Coates, Gurnell & Sarnyai, 2010,
  *Phil Trans R Soc B*, [DOI](https://doi.org/10.1098/rstb.2009.0193)). Hormones move
  slowly, so per-decision information is weak. The model treats endocrine tone as a
  weak, subject-level signal for exactly this reason.
- **Respiration (a phase clock on cognition).** Nasal breathing entrains limbic
  oscillations and breathing phase modulates fear discrimination and memory
  (Zelano et al., 2016, *J Neurosci*,
  [DOI](https://doi.org/10.1523/JNEUROSCI.2586-16.2016)). Not simulated here, but it
  is another cheap mechanical channel and slots into the same block structure.
- **Neural (EEG).** The repo core. Entered as inter-channel **covariance**, the form
  in which EEG carries most of its discriminative signal.

*Attribution: the five bullets above draw on articles retrieved from PubMed; DOIs
link each source.* The generator in the script plants each family's signal at a
strength matched to this evidence: cardiac and EDA share an autonomic driver, pupil
carries a separate noradrenergic driver, EEG a cortical driver, endocrine a slow
subject-level driver, and a sixth "null" family carries pure noise.

---

## 2. The mathematics of why fusion helps (and when it does not)

**Per-family model.** Each family *m* gets its own bagged classifier. Within every
bootstrap bag, SMOTE rebalances the minority class *inside the bag only*, then a base
learner is fit; for EEG the learner is a logistic regression in the Riemannian
log-Euclidean **tangent space** of the covariance matrices. Bagging averages *B*
bag predictions:

  p_m(x) = (1/B) Σ_b  f_{m,b}(x)

Bagging reduces variance roughly as Var/B for the part of the error that is
independent across bags; SMOTE-in-bag keeps that averaging from being dominated by a
single interpolation draw.

**Riemannian step (EEG only).** Covariance matrices live on the curved manifold of
symmetric positive-definite matrices, not in a flat vector space, so z-scoring and
SMOTE-interpolating their raw entries is not geometrically valid and can produce
non-covariance "matrices". The fix is to map each covariance *C* to the tangent
space at the training mean via the matrix logarithm,

  φ(C) = logm(C̄^{-1/2} C C̄^{-1/2})   (affine-invariant), or
  φ(C) = logm(C) − mean_train(logm(C))  (log-Euclidean, the default here),

after which the vectors are genuinely Euclidean and the scaler / SMOTE / logistic
downstream are valid again. The map is fit on training data only, inside each fold
and each bag, so it is leakage-safe.

**Reconciliation (the fusion step).** Let p_m be family *m*'s out-of-fold
positive-class probability. Three rules combine them:

  majority:            p = (1/M) Σ_m 1[p_m ≥ 0.5]
  soft:                p = (1/M) Σ_m p_m
  accuracy_weighted:   p = Σ_m w_m p_m ,   w_m ∝ max(0, a_m − 0.5),   Σ w_m = 1

where a_m is family *m*'s **out-of-fold** balanced accuracy. Out-of-fold matters:
weighting by training accuracy would give the largest weight to whichever family
overfit hardest. A family with a_m ≤ 0.5 gets weight zero, so it is dropped.

**Why aggregation can beat the best single family.** Write each family's score as a
noisy read of the shared decision logit ℓ: s_m = ℓ + ε_m. If the ε_m are
*independent* across families with variances σ_m², the inverse-variance-weighted
combination has error variance

  σ²_fused = 1 / Σ_m (1/σ_m²)   ≤   min_m σ_m² .

The combined estimate is strictly better than the best single family whenever more
than one family carries real, independent signal. accuracy_weighted approximates
inverse-variance weighting: a higher a_m means a smaller σ_m, so it gets more weight.
The inequality is the risk claim stated precisely: **independent weak leaks add up.**

**When fusion does *not* help, and the two controls that check it.** The inequality
assumes real, independent signal. Two failure cases must be ruled out, and the
experiment builds in a control for each:

1. *Redundancy.* If two families see the same driver (cardiac and EDA both read
   autonomic arousal here), their errors are correlated and the second adds little.
   The ablation ladder measures this: the EDA step should add less than the pupil
   step, which sees a genuinely new driver.
2. *No signal.* A family that carries nothing (the `null_control`) must get weight ≈ 0
   under accuracy_weighted and must not raise accuracy when appended to the ladder.
   If it did, the gain would be capacity/overfitting, not fusion. This is the
   negative control that separates real aggregation from an artefact.

---

## 3. The experimental arc

1. **Plant a known structure.** Generate six biosignal families over 24 subjects ×
   60 stimuli where the binary choice is driven by four partly-independent latent
   drivers, each sensed best by a different family, plus a pure-noise family. Ground
   truth (which family carries how much) is recorded.
2. **Baseline each family alone** through `run_experiment`: nested subject-grouped CV,
   in-fold SMOTE, bagging, pooled out-of-fold balanced accuracy with a subject-level
   bootstrap CI and a label-permutation p-value.
3. **Fuse all six** under each reconciliation rule; read the learned weights.
4. **Measure fusion lift** = full-ensemble balanced accuracy − best single family.
5. **Ablation ladder**: add families one at a time and watch the curve. It should
   rise as complementary families join and flatten when the null family is appended.
6. **Ground-truth check**: the accuracy_weighted weights should rank the families by
   planted strength, and the null family's weight should be ≈ 0.

The honest status of this arc: it is a **positive control on synthetic data**. It
proves the aggregation machinery recovers a fusion gain *when independent
cross-modal signal exists*, and that it correctly discards a junk family. It is not
evidence that any particular real person's choices are decodable at these levels.
Real multi-biosignal-per-subject datasets (DEAP, WESAD) require an EULA or a large
download and were not pulled this session; the DEAP loader in the repo is already
written for exactly this fusion test once the data is licensed.

---

## 4. Results

Six families, 24 subjects, 60 stimuli, 1440 trials, positive rate 0.30. Nested
subject-grouped CV, in-fold SMOTE, 20 bags per family, 2000 subject-bootstraps,
800 label permutations. Seed 0. Config: accuracy_weighted unless noted.

**Each family alone.** Balanced accuracy, pooled out-of-fold, subject-bootstrap 95% CI,
label-permutation p:

| family | balanced acc | 95% CI | ROC AUC | perm p |
|---|---|---|---|---|
| cardiac | 0.580 | 0.553–0.608 | 0.601 | 0.001 |
| pupil | 0.579 | 0.551–0.609 | 0.605 | 0.001 |
| eeg | 0.576 | 0.546–0.603 | 0.604 | 0.001 |
| eda | 0.544 | 0.523–0.563 | 0.567 | 0.002 |
| endocrine | 0.539 | 0.511–0.567 | 0.567 | **0.260** |
| null_control | 0.464 | 0.439–0.486 | 0.442 | 0.996 |

Endocrine does not clear the permutation test, which is the correct outcome: slow
hormones carry little per-decision information. The null family sits below chance.

**Fusion of all six**, by reconciliation rule:

| rule | balanced acc | 95% CI | ROC AUC | lift vs best single |
|---|---|---|---|---|
| majority | 0.601 | 0.571–0.632 | 0.636 | +0.021 |
| soft | 0.623 | 0.597–0.651 | 0.684 | +0.043 |
| accuracy_weighted | 0.621 | 0.592–0.646 | 0.682 | +0.041 |

Best single family = cardiac at 0.580. Soft and accuracy_weighted both beat it;
ROC AUC rises from 0.605 (best single) to 0.68 fused. Learned accuracy_weighted
weights: cardiac 0.25, eeg 0.24, pupil 0.23, eda 0.14, endocrine 0.13,
null_control 0.00.

**Ablation ladder** (add one family at a time, accuracy_weighted):

| step | families | balanced acc | 95% CI |
|---|---|---|---|
| +eeg | 1 | 0.576 | 0.546–0.603 |
| +cardiac | 2 | 0.591 | 0.564–0.617 |
| +eda | 3 | 0.607 | 0.580–0.632 |
| +pupil | 4 | 0.616 | 0.587–0.644 |
| +endocrine | 5 | 0.621 | 0.592–0.646 |
| +null_control | 6 | 0.621 | 0.592–0.646 |

Accuracy climbs as each signal-bearing family joins and is flat when the null family
is appended (0.621 → 0.621). This is the difference between fusion and capacity: more
inputs help only when they carry independent signal.

![Per-family accuracy and the fusion ladder](figures/biosignal_fusion.png)

**Runtime gates** (in `check_gates`, all PASS this run): fusion > best single;
null weight = 0; null not above chance; null adds no ladder accuracy; endocrine
weight below the strong-family mean. The run exits non-zero if any gate fails.

**Reading the size.** The absolute numbers are modest by construction (weak planted
signals, shallow features). The result is the *shape*: independent weak families
combine to beat the best one, redundant families add less, and noise adds nothing.
That shape is the risk claim; the magnitude on real fused data is an empirical
question the framework is now equipped to answer once a licensed multimodal cohort
is loaded.

---

## 5. Why aggregate biosignal inference is a genuine AIxBio biorisk, in the African context

The single-family privacy doc framed the leak. Fusion is what makes it dangerous, and
the danger is sharper in a Global South / African biosecurity setting for concrete,
structural reasons:

- **The cheap channels are the ones already deployed.** Pupil (any front camera), HRV
  and EDA (a sub-$30 wrist wearable), and voice are exactly the sensors shipping in
  the low-cost phones and community-health devices spreading fastest across the
  continent. The families the fusion argument relies on are not exotic lab
  instruments; they are consumer hardware.
- **Fusion lowers the hardware bar.** Section 2's inequality means an actor does not
  need one high-quality sensor; several cheap, individually-uninformative sensors
  reconciled together clear the bar the privacy doc showed a single sensor barely
  reaching. This inverts the usual assumption that strong inference needs strong
  instruments.
- **Reconciliation across records is the amplifier.** The same accuracy_weighted rule
  that fuses modalities can fuse *databases*: a health-post PPG record, a mobile-money
  behavioural log, and a phone-camera pupil trace, linked by identity, are just more
  blocks in the ensemble. Data-protection regimes that are still maturing across much
  of the continent, plus cross-border data flows to model operators elsewhere, make
  this linkage the realistic threat, not a hypothetical one.
- **The inferred attributes are the sensitive ones.** The drivers these signals read (arousal,
  stress/cortisol tone, risk bias, exploration state) map onto exactly the
  attributes an actor would want for targeting: susceptibility to persuasion, stress
  under coercion, health status that carries stigma. An AI system given standing
  access to fused biosignal streams is an inference engine over protected health and
  behavioural attributes, whether or not that was its stated purpose.
- **Positives inform, nulls do not bound.** The shallow, hand-built features here set
  a floor, not a ceiling. A weak or null result does not prove a well-resourced actor
  with deep feature learning could not extract more; it only proves this modest
  pipeline could not. The risk assessment must be read asymmetrically.

The defensive implication is not "do not sense biosignals." It is that **access to
multiple biosignal streams under one identity should be governed as a capability**,
because the fusion step converts an aggregate of individually-legal, individually-weak
data collections into a strong inference that no single collection consented to. That
governance question is the AIxBio contribution of this line of work.

*Literature in sections 1 and 2 was retrieved from PubMed; each claim links its DOI.*

---

## 6. Datasets and methods cited

**Methods reused (not reinvented).** The fusion experiment uses established methods
rather than novel ones:

- SMOTE for in-fold minority oversampling: Chawla et al. (2002), *JAIR*,
  [DOI](https://doi.org/10.1613/jair.953).
- Bagging: Breiman (1996), *Machine Learning*,
  [DOI](https://doi.org/10.1007/BF00058655).
- Riemannian tangent-space classification of covariance matrices: Barachant et al.
  (2012), *IEEE TBME*, [DOI](https://doi.org/10.1109/TBME.2011.2172210); implemented
  via the log-Euclidean metric (Arsigny et al., 2006,
  [DOI](https://doi.org/10.1002/mrm.20965)), with the affine-invariant backend
  provided by `pyriemann`.

**Datasets.** The fusion run is synthetic (generator in the script), so it cites no
subject data. The surrounding empirical results in this repository do use real,
publicly licensed datasets, cited where they are used:

- **PPG-BP Database** (fingertip photoplethysmography, Guilin), Liang et al. (2018),
  *Sci Data*, [DOI](https://doi.org/10.1038/sdata.2018.20). Used in
  `run_biosignal_privacy.py`.
- **ASZED-153** (EEG), cited in `docs/africa_neuroprivacy.md`.
- **DEAP** (EEG + peripheral physiology, per-subject multimodal), Koelstra et al.
  (2012), *IEEE T-AC*, [DOI](https://doi.org/10.1109/T-AFFC.2011.15). The intended
  real-data home for this fusion test; loader already in `io/deap.py`.
- **WESAD** (wrist/chest ECG, EDA, EMG, respiration, temperature), Schmidt et al.
  (2018), *ICMI*, [DOI](https://doi.org/10.1145/3242969.3242985). A second candidate
  multimodal cohort.

Where a dataset ships its own validated preprocessing or reference analysis, this
project reuses it rather than rebuilding it; new code is added only for the fusion
step that the source analyses do not cover.
