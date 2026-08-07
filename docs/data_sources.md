# Candidate data sources

Starting points for each modality, plus an honest statement of where the gap is.

**Verify before you download.** Accession numbers and access conditions change.
Every entry below should be confirmed on the hosting portal before it is written
into a config or a methods section. Nothing here has been downloaded or run
against the loaders yet.

---

## The gap, stated first

**No public dataset has simultaneous fMRI, EEG, facial video, and real
population-level market outcomes.** That combination does not exist off the
shelf, and any project plan that assumes it does will stall at week two.

Three viable ways around it:

1. **Modality-by-modality validation.** Validate each arm on the best available
   dataset for that modality, then collect the combined dataset yourself. Slow,
   and it produces publishable intermediate results.
2. **Start from the affective-computing datasets.** DEAP, MAHNOB-HCI, and AMIGOS
   already pair EEG with facial video and self-report on the same subjects and
   stimuli. They lack fMRI and lack a real market outcome, but they support the
   full EEG + face + behaviour pipeline today, including the ensemble and the
   reconciliation logic.
3. **Synthesise the aggregate outcome.** For stimuli that exist in the wild
   (music videos, ads, film clips), a real population measure sometimes can be
   scraped: view counts, sales ranks, campaign funding totals. This is the move
   Genevsky and Knutson made, and it is the cheapest route to a genuine
   aggregate outcome.

Option 2 is the fastest path to a working end-to-end result. Option 3 is what
makes it a neuroforecasting result rather than an affect-decoding result.

---

## fMRI

### Portals
- **OpenNeuro** ([openneuro.org](https://openneuro.org)) is the main source.
  BIDS-formatted, mostly CC0, downloadable with the `openneuro-py` client or
  DataLad. Search by task rather than by accession.
- **Human Connectome Project** ([humanconnectome.org](https://www.humanconnectome.org))
  includes a gambling/reward task across a large sample. Requires a data use
  agreement.
- **Natural Scenes Dataset** ([naturalscenesdataset.org](https://naturalscenesdataset.org))
  is 7T, high per-subject trial counts, and built for encoding-model work rather
  than choice, but it is the right shape for individual-level modelling.

### Specific datasets worth checking
- **NARPS mixed-gambles** (OpenNeuro `ds001734`). Around 108 subjects deciding
  whether to accept mixed gain/loss gambles. Reward-circuit engagement is the
  whole point of the task, and the sample size is unusual for fMRI. Also the
  dataset behind the NARPS analytic-variability study, so the analytic
  degrees of freedom are unusually well documented.
- **Tom, Fox, Trepel & Poldrack mixed gambles** (OpenNeuro `ds000005`). Small
  (around 16 subjects) but a clean, well-understood loss-aversion design. Good
  for smoke-testing the fMRI loader against real BIDS files.

Neither has a market-level outcome. Both are useful for validating that
`FMRILoader` plus the ROI extraction produces trial-level features that predict
individual choice above chance, which is a real milestone on its own.

---

## EEG

### Portals
- **NEMAR** ([nemar.org](https://nemar.org)) indexes BIDS-formatted EEG and MEG
  datasets hosted on OpenNeuro, with compute attached.
- **OpenNeuro** hosts EEG as well as MRI.
- **PhysioNet** ([physionet.org](https://physionet.org)) for physiological
  signals more broadly.

### Datasets pairing EEG with other modalities
These are the most valuable entries on this page for this project, because they
give you two or three of the four modalities on the same subjects and stimuli.

- **DEAP** (Database for Emotion Analysis using Physiological signals). 32
  subjects, 40 music-video trials each, EEG plus peripheral physiology, with
  frontal **face video for a subset of participants**, and self-reported valence,
  arousal, dominance, and liking per trial. Requires signing an end-user
  licence agreement. The stimuli are real music videos, which means a public
  popularity measure may be obtainable for the aggregate arm.
- **MAHNOB-HCI**. EEG, face video from multiple angles, eye gaze, and
  peripheral physiology while watching emotional film clips, with self-report.
  Requires registration.
- **AMIGOS**. EEG, ECG, GSR, and face video, in both individual and group
  viewing conditions, with affect and personality measures. Requires
  registration.
- **SEED / SEED-IV** (Shanghai Jiao Tong University). EEG during emotional film
  clips. Application required.

DEAP is probably the best single starting point: it exercises `EEGLoader`,
`FaceLoader`, and `BehaviorLoader` at once, has genuine class imbalance in the
liking ratings, and its stimuli are real-world items with a plausible public
popularity signal.

---

## Facial expression and video

- **Aff-Wild2** / the ABAW challenge series. Large in-the-wild video with
  valence-arousal, action-unit, and expression annotations. The standard
  benchmark for training or fine-tuning the ViT arm.
- **DISFA**, **BP4D**, **CK+**. Lab-recorded, FACS-coded action units. Smaller
  and cleaner; good for validating that the geometric feature family is wired
  correctly.
- **AffectNet**. Large, in-the-wild, still images with expression and
  valence-arousal labels. Static rather than temporal, so it suits pretraining
  the frame encoder rather than modelling dynamics.

Access for all of these is by request or licence agreement. None is CC0.

Practical note: for this project the face arm is a *response* measure recorded
while a participant views a stimulus, not an expression-recognition benchmark.
A model pretrained on AffectNet gives you a feature extractor, and the
downstream task is still trial-level choice prediction.

---

## Behaviour and aggregate market outcomes

This is the arm most people forget to plan for, and it is the one that turns a
decoding project into a forecasting project.

- **Kiva** ([kiva.org](https://www.kiva.org)) publishes microloan data snapshots.
  This is the market Genevsky and Knutson (2015) forecast. Loan requests carry
  photographs, text, funding amounts, and funding outcomes, which is exactly the
  stimulus-plus-outcome structure the aggregate arm needs.
- **Kickstarter** scrapes (several public archives, updated monthly) give project
  descriptions, images, funding goals, and success or failure. The market in
  Genevsky, Yoon & Knutson (2017).
- **YouTube Data API** gives view counts and engagement for video stimuli. The
  market in Tong et al. (2020).
- **Spotify / Billboard charts** for music stimuli.

The pattern in all four: pick stimuli that already exist in a public market,
show them in the lab, and pull the market outcome afterwards. The aggregate
outcome is then a real measurement rather than a proxy, which is the whole
methodological trick.

**Check the terms of service before scraping anything**, and check whether your
institution's ethics approval covers linking lab data to public platform data.

---

## Suggested sequence

1. **DEAP** to get EEG + peripheral + behaviour running end to end against real
   recordings. **The loader for this is built** (`io/deap.py`,
   [`docs/deap.md`](deap.md)); it needs only the EULA'd download. DEAP's YouTube
   stimuli also give a route to a real aggregate outcome via view counts
   (`io/deap_market.py`), so it can exercise *both* levels, not just the
   individual one. Validates the EEG and behaviour paths and the whole ensemble.
2. **NARPS `ds001734`** to validate the fMRI loader and ROI extraction on real
   BIDS data with a reward task and a decent sample. **The loader for this is
   built** (`io/narps.py`, [`docs/narps.md`](narps.md)); it reuses `FMRILoader`'s
   NAcc/vmPFC/AIns sphere extraction and needs only the OpenNeuro download (no
   licence). Note NARPS is an *individual-level* validation: on gambles the
   economic baseline dominates the aggregate arm by construction, so it is the
   fMRI plumbing check, not a brain-beats-behaviour demonstration.
3. **A market-linked stimulus set** (Kiva or Kickstarter items) for a
   purpose-built aggregate arm. This is the step that requires collecting your
   own scans, and it is the one that produces the fully novel result.

Steps 1 and 2 are parallelisable and neither depends on the other. Step 1 is the
one with a working loader today.
