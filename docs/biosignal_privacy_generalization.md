# Biosignal privacy: the neuroprivacy risk is not EEG-specific

A companion to [`africa_neuroprivacy.md`](africa_neuroprivacy.md). That document
runs one empirical demonstration: this repo's own accuracy-weighted multibagging
ensemble, given four short EEG segments from a real Nigerian clinical cohort,
recovers a protected health attribute (schizophrenia diagnosis) the recording was
not about, at balanced accuracy 0.705.

This document makes a narrower, structural claim and adds no new empirical run:
**the same result follows by construction for a broad family of consumer-
collectable biosignals, because nothing in the pipeline's mathematics is neural.**
It also lays out the parameters that decide how large the risk is in a given
deployment, and why several of those parameters sit at their worst values in a
Global South context.

---

## 1. Why the framework is modality-agnostic

The data contract ([`io/base.py`](../src/behavioral_decoding/io/base.py)) is a
`ModalityBlock`: a `(n_trials, n_features)` matrix, each row keyed by
`subject_id` and `stimulus_id`, plus a provenance record. Every downstream stage
operates on that contract and only that contract:

```
ModalityBlock  ->  BaggingClassifier(Pipeline(StandardScaler -> AdaptiveOverSampler -> base estimator))
               ->  out-of-fold balanced accuracy per block
               ->  reconciliation: weight each block by how far its OOF score exceeds chance
```

The ensemble ([`models/ensemble.py`](../src/behavioral_decoding/models/ensemble.py))
never inspects what a block *is*. `EEG`, `FMRI`, `FACE`, `BEHAVIOR` are string
labels in `KNOWN_MODALITIES`, not typed code paths. The ASZED-153 script already
exploits this: it registers four EEG recording phases as four independent
"modalities" and reconciles them. Any signal that can be reduced to a per-window
feature vector and keyed by subject is, for this code, a modality block, and it
rides the identical bagging, in-fold resampling, subject-grouped nested CV,
calibration reporting, and OOF reconciliation.

The consequence: the ASZED-153 demonstration is not a fact about EEG. It is a
fact about *weak, cheap, per-window biosignal features plus a reliability-weighted
ensemble plus a protected attribute that has a physiological correlate in the
literature*. EEG is one instance. The sections below enumerate the others.

---

## 2. The biosignal family that slots in unchanged

Each row is a candidate `ModalityBlock`. "Feature family" is the low-dimensional
reduction that would go in `X`. "Literature-linked attributes" are protected or
sensitive properties with a documented physiological correlate, i.e. the kind of
label a decode could target the way ASZED-153 targeted diagnosis. None of this is
a claim that a decode *succeeds* at a given attribute in a given dataset. It is a
map of where the same pipeline would be pointed.

| Signal | Consumer collection route | Feature family (per window) | Literature-linked attributes |
|---|---|---|---|
| **Cardiac (ECG / PPG / HRV)** | Any wrist band, ring, phone-camera PPG, chest strap | Time-domain (mean HR, SDNN, RMSSD, pNN50), frequency-domain (LF, HF, LF/HF power), nonlinear (Poincare SD1/SD2, sample entropy) | Atrial fibrillation and other arrhythmia, diabetes / glycemic state, sleep apnea, hypertension, acute psychological stress, anxiety and depressive disorders, PTSD, pregnancy, biological sex, age, recent alcohol / caffeine / stimulant use, acute infection and fever, emotional arousal |
| **Electrodermal (EDA / GSR)** | Wrist band, palmar patch, some VR controllers | Tonic skin-conductance level, phasic response rate / amplitude / latency (cvxEDA or Ledalab tonic-phasic split) | Sympathetic arousal, chronic stress load, anxiety disorders, some seizure types, pain state, deception-correlated arousal (contested, flag as such) |
| **Pupillometry / oculomotor** | Any front camera that resolves the iris, eye-tracker in a headset or laptop | Baseline diameter, task-evoked dilation amplitude and latency, constriction velocity, blink rate, microsaccade rate, gaze dispersion and fixation entropy | Cognitive load, arousal, stimulus preference and interest (directly choice-relevant), fatigue and drowsiness, opioid use (miosis) and other drug effects, concussion / TBI, Parkinson's, schizophrenia (smooth-pursuit and antisaccade markers), autism-linked oculomotor patterns, migraine, autonomic dysfunction |
| **Endocrine / salivary / sweat assay** | Saliva swab kits, sweat patches, continuous glucose monitors, at-home hormone tests | Cortisol diurnal slope, cortisol awakening response, acute cortisol reactivity, salivary alpha-amylase (sympathetic proxy), testosterone, estradiol / progesterone, DHEA, glucose level and variability, inflammatory markers | Chronic stress and allostatic load, pregnancy, menstrual-cycle phase and fertility window, PCOS, hormone-therapy / transition status, metabolic disease and prediabetes, depression (blunted CAR), PTSD, recent acute stressor |
| **Respiration** | Chest strap, radar / camera plethysmography, phone accelerometer on the chest | Rate, inspiration / expiration ratio, breath-to-breath variability, sigh rate | Anxiety and panic, acute stress, apnea, respiratory infection, opioid effect |
| **Skin temperature** | Ring, wrist band, forehead sensor | Mean, circadian phase, distal-proximal gradient | Circadian disruption, menstrual-cycle phase, fever / infection, thermoregulatory dysfunction |
| **Voice / acoustic** | Any microphone, any voice assistant, any call | Pitch mean and range, jitter, shimmer, harmonics-to-noise ratio, speaking rate, spectral tilt | Depression, Parkinson's, cognitive decline, intoxication, acute stress, respiratory illness, biological sex, approximate age |
| **Gait / accelerometry** | Phone in a pocket, wrist band | Cadence, stride variability, step regularity, activity-bout structure, sleep-wake fragmentation | Parkinson's and other movement disorders, frailty and fall risk, depression (psychomotor change), sleep disorders, intoxication, pregnancy (late) |

DEAP already carries several of these. Its channels 33 to 40 are GSR, respiration,
blood-volume pulse (plethysmograph), skin temperature, and facial EMG, and
[`io/deap.py`](../src/behavioral_decoding/io/deap.py) already routes them into a
non-EEG `peripheral` block with level / variability / trend / range features. The
generalization is not hypothetical plumbing. The peripheral block is in the repo
today: what is absent is only a script that points it at a held-out protected
label instead of a market outcome.

---

## 3. Reconciliation is the risk amplifier

The ASZED-153 number (0.705) did not come from one strong signal. Its four EEG
phases scored OOF balanced accuracy 0.711, 0.643, 0.633, 0.624 individually. None
is what a regulator would call a diagnostic test. The out-of-fold reconciliation
combined them, and the shortest, cheapest segment carried the largest weight.

That is the mechanism to worry about here. A device with a PPG sensor, a camera
that resolves the pupil, and a paired sweat patch supplies three weak biosignal
blocks. Each on its own may sit just above chance for, say, pregnancy or acute
depression or cycle phase. The reconciliation step is designed to find and
up-weight whichever weak blocks carry genuine out-of-fold signal and to zero the
ones that do not. Its output is a single decode that can clear chance for a
protected attribute while every input, examined alone, looks like consumer
wellness telemetry. Risk is monotone in the number of fused weak modalities, and
consumer hardware trends toward more sensors per device, not fewer.

---

## 4. Parameters that set the size of the risk

These are the knobs to reason with when assessing a real deployment. They are
written so a reader can locate a specific product on each axis.

1. **Stated collection purpose vs. inferable attribute set.** A wellness app, a
   mobile game, a hiring assessment, a telehealth triage form, and a loan-
   onboarding flow each state a narrow purpose. The gap between that purpose and
   the attribute table in section 2 is the exposure.
2. **Number of weak biosignal blocks fused (k).** Each added sensor stream is
   another out-of-fold-weighted vote. See section 3. Track the sensor count of
   the collecting device and any linked accounts.
3. **Recording length and unit cost.** ASZED-153's most informative segment was
   its shortest. A risk that needed a 40-minute scan would be self-limiting. One
   that needs 90 seconds of wrist PPG is not.
4. **Base rate of the target attribute.** Class imbalance inflates raw accuracy;
   balanced accuracy against chance is the honest read (the repo enforces this
   everywhere). A rare attribute is not safe, it just needs the balanced metric
   to see.
5. **Attribute sensitivity in context.** Cycle phase, HIV status, a psychiatric
   diagnosis, or transition status carry very different real-world consequences
   depending on who holds the decode and what they gate on it.
6. **Legal protection in the jurisdiction of collection.** Whether a binding rule
   attaches to health or biometric inference from non-medical sensor data, and
   whether an enforcement body has the capacity to test it. See section 5.
7. **Recourse available to the data subject.** Whether the person can see, contest,
   or withhold the inference, and whether refusing collection costs them access
   to credit, work, or care.
8. **Decoder portability.** Whether a model trained on one population is applied
   to another with different base rates and physiology, which both mis-calibrates
   the decode and, where it still works, exposes locally stigmatized attributes.

---

## 5. Why these parameters sit at their worst in a Global South context

This section extends the governance snapshot already sourced in
[`africa_neuroprivacy.md`](africa_neuroprivacy.md#governance-context-why-this-risk-isnt-hypothetical)
(Nigeria and Kenya: no binding AI-specific national law as of 2026-08-17, only
non-binding national strategies). Added parameters:

- **Data-protection law exists but health-inference enforcement is thin.**
  Nigeria's NDPA (2023) and Kenya's Data Protection Act (2019) establish consent,
  purpose-limitation, and data-protection-impact-assessment duties on paper. What
  is not established in practice is enforcement capacity for the specific case of
  *inferred* health or biometric attributes from consumer sensor streams,
  case law testing DPIA adequacy for that use, or routine scrutiny of cross-
  border transfer of the resulting model or its outputs.
- **Consented-for-research data is legally reusable for AI.** ASZED-153 is CC-BY.
  It was collected with proper ethics oversight for computational-psychiatry
  research and is now, by licence, available for exactly the secondary use this
  repo demonstrates. That is the intended point of the demonstration: the
  extractive step is not a breach, it is permitted. Weaker IRB coverage for
  secondary AI reuse of clinically-consented data widens this.
- **Helicopter-research pattern.** Datasets are collected in-region; the models,
  the inferences, and the commercial value are exported. The methodology in this
  repo (African-led, with run records and provenance so results cannot be
  quietly overclaimed or exported without an audit trail) is a deliberate
  counter to that pattern, described in the README.
- **Sensor penetration on cheap hardware.** Camera-based pupillometry and PPG
  heart-rate do not require premium devices. Rising low-cost Android and paired-
  wearable adoption means the collection front is widening where the legal
  backstop is weakest.
- **Collection fronts with leverage over the data subject.** Microloan
  onboarding, gig-platform worker assessment, insurance rollout, and public-
  health screening are all plausible places a biosignal stream gets collected,
  and all are contexts where a decode of diagnosis, pregnancy, or stress load
  would materially change a person's access to credit, work, or care, with
  parameter 7 (recourse) near zero.
- **Population base-rate mismatch.** A decoder trained on a WEIRD cohort, applied
  to an African population with different disease base rates and physiology, is
  both mis-calibrated (parameter 4) and, where it does carry signal, aimed at
  whatever attribute is locally consequential.
- **Asymmetry of cost and consequence.** The marginal decode is cheap to run.
  Its output is disproportionately consequential where social safety nets are
  thinner, so the same false or true positive lands harder.

---

## 6. What would make this empirical rather than structural

Stated plainly, matching the honesty standard the rest of this repo holds to.

This document runs nothing. Its claim is that the ASZED-153 result generalizes by
construction because the pipeline is modality-agnostic (section 1) and the
biosignals in section 2 have documented attribute correlates. To turn the
structural claim into an empirical one:

- Run the existing pipeline on **AMIGOS** (EEG, ECG, GSR, and face video on the
  same subjects) or on **DEAP peripheral-only**, predicting a held-out sensitive
  label (for AMIGOS, personality or affect-disorder proxies; the datasets were
  not built for this, which is itself the caveat), with the same nested subject-
  grouped CV, bootstrap CI, and calibration reporting used everywhere else.
- Both datasets are Western-collected and consented for affect research, so a
  positive result there speaks to *feasibility of the method*, not to a Global
  South population specifically. ASZED-153 remains the only arm on a real African
  cohort.
- Report the same degeneracy warnings the neuroprivacy and Kenya arms already
  carry: a within-subject label permutation is a no-op for a subject-constant
  trait, so the bootstrap CI over resampled subjects is the statistic to trust.

Until such a run exists, the honest framing is: the risk demonstrated on
ASZED-153 is not specific to EEG, the code to extend it to autonomic and
endocrine signals is largely already in the repo, and the parameters in
sections 4 and 5 are where a specific deployment should be assessed.

---

## 7. See also

- [`africa_neuroprivacy.md`](africa_neuroprivacy.md): the ASZED-153 empirical
  demonstration and the sourced governance table.
- [`africa_cohort.md`](africa_cohort.md): the Kenyan Kiva behaviour arm and its
  dropped confound.
- [`design.md`](design.md) section 2: why modalities stay as separate blocks,
  which is the same property that makes this generalization mechanical.
- README, [Interpretation limits](../README.md#interpretation-limits): the
  standing caveats (forecasting is not causation, this does not read intent)
  apply here without change.
