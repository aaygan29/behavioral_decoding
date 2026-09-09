# Neuroprivacy risk demonstration: ASZED-153 (Nigeria)

A different kind of arm from the rest of this repo, and deliberately kept
separate. Everywhere else here, the question is "does brain signal forecast a
market/choice outcome the subject knew they were part of." This arm asks a
narrower, sharper, safety-relevant question: **given a few minutes of routine
EEG, collected under one stated purpose, can this repo's own multibagging
pipeline recover a sensitive personal attribute the recording was not
ostensibly about?**

## What was pulled

[**ASZED-153**](https://zenodo.org/records/14178398), the African
Schizophrenia EEG Dataset. 153 subjects (76 schizophrenia patients, 77
matched controls), recruited at Obafemi Awolowo University Teaching Hospital
Complex (Ile-Ife) and its Wesley Guild Hospital unit (Ilesa), Nigeria.
19-channel international 10/20 EEG, four recording phases per session (per
the dataset's own session metadata: two short baseline-length segments and
two longer task/passive-listening segments). CC-BY on Zenodo, MD5-verified on
download by [`scripts/fetch_africa_neuroprivacy.py`](../scripts/fetch_africa_neuroprivacy.py).

This is real clinical EEG from a real African cohort, collected and released
for legitimate computational-psychiatry research, with proper ethics
oversight and consent for that stated research purpose. Nothing here
re-identifies or targets any individual; every number below is a pooled,
de-identified, out-of-fold statistic on data already public. The point of
running it is what such data makes *possible* once collected, not to expose
anyone here.

## Why this design, not the market-forecast design

There is no aggregate/market arm in this script, on purpose. Diagnosis is a
property of a person, not of a stimulus, there's no group-level outcome to
hold out and forecast the way DEAP's view counts or the Kiva sector repayment
rate are. What *is* structurally identical to the rest of this repo: four
EEG segments recorded per subject (Phase 1-4) are treated as four separate
modalities and reconciled through the same accuracy-weighted multibagging
ensemble that combines fMRI/EEG/face/behaviour elsewhere,
[`scripts/run_africa_neuroprivacy.py`](../scripts/run_africa_neuroprivacy.py),
same `EEGLoader` band-power feature family (`io/eeg.py`), same nested
subject-grouped CV, same calibration and bootstrap reporting.

The label predicted is diagnostic category (patient vs. control),
**not** any observable the recording session was designed around. That
substitution is the whole demonstration: a battery of short EEG segments,
of the kind a wellness app, a consumer neurotech headset, or an unrelated
study could plausibly collect, is shown here to carry a decodable protected
health attribute.

## Literature grounding

This is not a fishing expedition. The features used (log band power,
delta-gamma, per channel, the same `bandpower()` this repo already ships) are
chosen because the schizophrenia-EEG literature has repeatedly tied specific
spectral signatures to this diagnosis: reduced 40 Hz auditory steady-state
gamma-band phase-locking, elevated resting frontal delta/theta, and reduced
alpha are among the most replicated findings in the field. The four phases
here plausibly span baseline and passive-auditory segments where exactly
those signatures should live. No attempt is made to identify *which* literature
mechanism the model actually leans on per phase, see Limits below, the claim
is narrower: known, literature-motivated regions of the spectrum are enough to
make this decodable with an off-the-shelf pipeline.

## Method notes

- **Sample rate.** The archive mixes two recording devices at different
  native rates (Contec KT-2400 at 200 Hz, BrainMaster Discovery24-E at
  256 Hz); every file is resampled to 200 Hz before epoching so epoch length
  and feature-column count are identical across subjects and devices.
- **Channels.** Normalised to a fixed, named 19-channel 10/20 montage
  (`Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Pz,
  Cz`), raw files carry bracketed index suffixes and inconsistent extra
  reference/EOG/EMG channels, so subjects/devices are only comparable after
  this normalisation.
- **Epoching.** Each phase recording is cut into fixed 2-second,
  non-overlapping windows (no stimulus-locking available at this level, so no
  ERP-window features are used, band power only). Each subject contributes
  as many epochs as the *shortest* of their four phases allows, so every
  phase-modality is aligned trial-for-trial per subject.
- **Cohort cut.** Subjects whose first recorded session has exactly four
  `Phase *.edf` files (76 subjects, the count in the run record) are used, so the
  four phase-modalities are structurally identical across subjects. This is a
  scope decision, not a data-quality filter, the excluded sessions have a
  different phase count, not worse data.
- **CV.** Grouped by subject (an epoch's fold membership never splits within a
  subject), same as every other arm in this repo.

## Results

Full record: `results/africa_neuroprivacy_aszed/run_record.json`.
76 subjects (a complete first-session, four-phase recording per subject), 236
epochs total, pooled out-of-fold:

| metric | value |
|---|---|
| n subjects / n epochs | 76 / 236 |
| majority-class accuracy baseline | 0.784 |
| balanced accuracy | **0.705** (95% CI 0.616–0.799) |
| ROC AUC | 0.792 |
| Brier score | 0.140 |
| ECE | 0.085 |

The 95% bootstrap interval sits entirely above chance (0.5) for a balanced
metric. Per-phase reliability, from the ensemble's own accuracy weighting
(out-of-fold balanced accuracy per phase-modality):

| phase | oof balanced accuracy | ensemble weight |
|---|---|---|
| phase1 (short) | 0.711 | 0.348 |
| phase2 (long) | 0.643 | 0.278 |
| phase3 (short) | 0.633 | 0.252 |
| phase4 (long) | 0.624 | 0.122 |

All four phases individually clear chance; none dominates, and the ensemble's
own reliability-weighting (not a hand-picked "best" phase) is what combines
them. That the shortest, cheapest segment (phase1) carries the most weight is
itself part of the risk point: this is not a result that requires a long,
expensive scan.

**Read the balanced accuracy, not the raw accuracy, against chance.** The
0.784 majority-baseline-accuracy figure above is inflated by class imbalance
in the epoch pool (185 patient epochs vs. 51 control epochs, some subjects'
phases contribute more usable epochs than others); balanced accuracy (0.705,
chance = 0.5) is the number that actually says whether diagnosis is
decodable.

**The permutation test is degenerate here too, for a different reason than
the Kenya cohort's, and should not be read as p=1.0 evidence against the
effect.** `permutation_test` shuffles labels *within subject group*. Diagnosis
is a subject-level trait: every epoch from the same subject carries the
*identical* label. Shuffling a set of identical values among themselves is a
no-op no matter how many epochs a subject has, so the "null" collapses onto
the observed statistic and reports p=1.0, a structural property of applying
a within-subject label permutation to a label that is constant within
subject, not a finding about the data. This is worth stating loudly: it is
exactly the kind of tool/data mismatch that would let someone quietly
overclaim "not significant" by citing a broken statistic instead of the valid
one. The bootstrap CI above, which resamples subjects rather than reshuffling
labels, is the number to trust, and it excludes chance.

## Governance context: why this risk isn't hypothetical

The "weaker legal protection" claim above is a factual claim, not a rhetorical
one, so it should be checked rather than asserted. Cross-referenced against
the [Global AI Governance Map](https://global-ai-governance-map.vercel.app/)
(Bekhzod Alikhanov; aggregates EUR-Lex, OECD, UNESCO, CoE, ISO, national
regulator sources, Oxford Insights, CAIDP, Stanford HAI; status as of
2026-08-17):

| | Nigeria | Kenya |
|---|---|---|
| Binding AI-specific law | **None confirmed** | **None confirmed** |
| National AI governance instrument | National AI Strategy (non-binding guidance) | National AI Strategy 2025–2030 (non-binding guidance) |
| International AI-instrument coverage | Indirect only, UNESCO Recommendation, AU Continental AI Strategy, Bletchley Declaration, etc.; all membership-coverage or political-endorsement, none binding | Same pattern |
| Oxford Insights Gov AI Readiness 2025 | 50.79/100, rank 70 | 52.55/100, rank 68 |
| CAIDP AI & Democratic Values Index 2026 | 6/12, Tier IV | 6/12, Tier IV |
| UNESCO Readiness Assessment (RAM) | In process | Completed |

Neither the country the EEG cohort was recruited in (Nigeria) nor the country
the market cohort was recruited in (Kenya, see
[`africa_cohort.md`](africa_cohort.md)) has a binding, AI-specific national
law on the books as of this writing, only non-binding strategy documents and
indirect coverage through international soft-law instruments. That is the
concrete version of "an environment with weaker regulatory oversight than
where this literature originated": there is currently no binding domestic
rule in either country that would govern a use case like covert diagnostic
inference from routine EEG, the way an AI Act-covered jurisdiction's biometric
and health-inference provisions would attempt to.

This is a governance-landscape snapshot from a third-party tracker, not a
legal opinion, and it will go stale, re-check it before citing in anything
external. It answers a factual question ("is there a binding rule against
this"), not the ethical one, this project is itself an example of the kind
of research that should proceed carefully and transparently precisely because
that binding rule does not yet exist.

## Limits, stated plainly

- **n is small.** 76 subjects is workable for a pooled out-of-fold estimate
  with a bootstrap interval, but not for strong claims about which spectral
  band or channel is doing the work.
- **Phase identity is not fully certain.** The dataset's public session
  metadata gives phase *durations* (two short, two long) consistent with the
  published protocol (rest / arithmetic / oddball / ASSR) but does not label
  each phase file with its paradigm name per session. This doc does not claim
  "phase 4 is the ASSR arm", it says four literature-plausible EEG segments,
  reconciled by an ensemble that weights each by its own out-of-fold
  reliability, were enough to decode the trait. That is a weaker, more
  defensible claim, and it is the one being made.
- **This is not proof of a mechanism.** A positive result here says the
  *information* is present and extractable with an off-the-shelf pipeline. It
  does not localise which spectral signature carries it, and does not
  establish that any real-world actor is currently doing this. The relevance
  is to feasibility and risk, not to a causal or diagnostic claim.
- **Confirms nothing about causation, and reads no one's intent**, same
  caveats as the rest of this repo's [Interpretation limits](../README.md#interpretation-limits)
  apply here as well.
