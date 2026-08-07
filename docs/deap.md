# DEAP

How this framework loads DEAP, every trap the format sets, and the honest limits
of what a DEAP result can claim.

DEAP (Koelstra et al., 2012, *IEEE Transactions on Affective Computing*,
[10.1109/T-AFFC.2011.15](https://doi.org/10.1109/T-AFFC.2011.15)) is the fastest
route from this framework to real recordings. 32 participants each watched 40
one-minute music videos while 32-channel EEG and 8 peripheral physiological
channels were recorded; 22 of the 32 also have frontal face video. Every
participant saw every video, so the subject-by-stimulus design is fully crossed,
which is exactly the shape the aggregate-forecasting arm needs.

**Access.** DEAP requires an end-user licence agreement with Queen Mary
University of London. The loader downloads nothing. Request the
`data_preprocessed_python` archive (the 128 Hz pickles) plus `metadata_csv/` and
`video_list.csv`.

---

## Quick start

```bash
# No download: run the whole path on a synthetic DEAP-format fixture.
python scripts/run_deap.py --demo

# Real data, individual level only.
python scripts/run_deap.py --root /path/to/DEAP --target liking

# Real data, both levels, once you have YouTube view counts.
python scripts/run_deap.py --root /path/to/DEAP --target liking --counts counts.csv
```

```python
from behavioral_decoding.io.deap import DEAPLoader, load_deap

dataset = load_deap(
    "/path/to/DEAP",
    target="liking",          # valence | arousal | dominance | liking
    binarise="fixed",         # or "subject_median"
    threshold=5.0,
    loader=DEAPLoader(behavior_mode="noncircular"),
)
```

`load_deap` returns a `MultimodalDataset` with three blocks: `eeg`,
`peripheral`, and `behavior`. Face video is deliberately left out; see below.

---

## The six traps

Each of these silently corrupts a result if ignored. Each is handled in the
loader and pinned by a test in `tests/test_deap.py`.

**1. The preprocessed data is bandpass filtered 4.0 to 45.0 Hz.** Delta (1 to
4 Hz) is not in the data. The default bands (`DEAP_BANDS`) omit delta, and
`DEAPLoader` raises if you request a band outside the passband, rather than
returning filter roll-off dressed up as a feature. If you need delta, you need
the raw `.bdf` release, not the preprocessed pickles.

**2. The label column order is `(valence, arousal, dominance, liking)`.**
Swapping valence and arousal produces a wrong, completely plausible result. The
order lives in one constant, `DEAP_LABEL_NAMES`, asserted in a test.

**3. Each trial is 8064 samples = 384 baseline + 7680 trial** (3 s of pre-trial
rest at 128 Hz, then 60 s of video). The loader splits the baseline off and, by
default, subtracts each trial's own baseline band power. That correction is
per-trial and self-referential, so it cannot leak across the CV split.

**4. The pickles were written under Python 2** and need `encoding="latin1"`.
Without it you get a `UnicodeDecodeError` that reads like a corrupt download.

**5. Channels 33 to 40 are not EEG.** hEOG, vEOG, zEMG, tEMG, GSR, respiration,
plethysmograph, temperature. Band power in the EEG sense is meaningless for slow
autonomic and muscle signals, so they get their own `peripheral` block with
level/variability/trend/range features.

**6. This is not an event-related design.** A 60-second music video has no
stimulus-locked ERP, so the framework's ERP-window features are not used here.

---

## The circularity problem, and why it matters here

DEAP's behaviour block deserves more suspicion than the EEG or peripheral blocks,
and the reason is specific to what this framework is for.

The four SAM ratings (valence, arousal, dominance, liking) were collected on the
same screen, seconds apart, from the same person. Predicting `liking` from
`valence` and `arousal` is nearly trivial and says nothing about neural signal.
It is self-report predicting self-report.

That is a problem because in the neuroforecasting paradigm the behavioural
comparator is a **choice** (fund it or not, keep watching or not), which is a
different kind of measurement from the brain signal. Genevsky, Yoon and Knutson
(2017) showed brain beating *that* comparator. DEAP has no choice measure. So a
DEAP plot where behaviour beats EEG is not the inverse of their result, and a
plot where EEG beats behaviour is not a replication of it. It is a different
comparison.

To keep the comparison meaningful, `DEAPLoader` defaults to
`behavior_mode="noncircular"`, which uses only **familiarity** and **trial
order** (fatigue and habituation) as behavioural features. Neither is a
same-instrument rating of the target. `behavior_mode="ratings"` gives you the
circular version if you want it, and stamps `circular: True` into the block's
provenance so it cannot quietly become the headline.

See [literature.md](literature.md) §1 for what the choice-based comparator
actually was.

---

## Turning DEAP into a neuroforecasting dataset

DEAP ships no market outcome, which is what would otherwise make it an affect
*decoding* dataset rather than a *forecasting* one. But its 40 stimuli are real,
public YouTube music videos, and `video_list.csv` has the links. A view count is
a genuine population behaviour measured outside the lab. That is the move
Genevsky and Knutson made, and it is available here.

`behavioral_decoding.io.deap_market` does the bookkeeping:

```python
from behavioral_decoding.io import deap_market

ids = deap_market.youtube_ids_from_video_list("DEAP/video_list.csv")
# -> {"exp-01": {"youtube_id": "...", "url": "..."}, ...}

# Fetch counts yourself via the YouTube Data API (videos.list, part=statistics),
# save them, then:
counts = deap_market.load_counts_file("counts.csv")
y_aggregate, prov = deap_market.market_outcome_from_counts(
    counts,
    id_to_experiment=deap_market.invert_id_map(ids),
    log_transform=True,
    fetch_date="2026-08-07",
)
dataset = load_deap("DEAP", target="liking", y_aggregate=y_aggregate)
```

The module does **not** fetch counts. Scraping YouTube or calling its API is a
network operation with terms-of-service and rate-limit implications, and it does
not belong inside a loader.

Two warnings it enforces:

- **Temporal gap.** DEAP is from 2012. A 2026 view count reflects a decade of
  accumulation the EEG could not have anticipated. Pass `fetch_date` so the gap
  is on the record; better, find an archived contemporaneous count or frame the
  outcome explicitly as "durable popularity".
- **Skew.** View counts are extremely skewed; `log_transform=True` is the
  default, and the forecasting arm reports Spearman alongside Pearson because
  rank is what survives the skew.

---

## The theory-specified feature: frontal alpha asymmetry

On the fMRI side, the framework starts from theory-specified ROIs (NAcc, MPFC)
rather than whole-brain, because that is where the neuroforecasting signal lives.
DEAP's analogue is **frontal alpha asymmetry**. Greater relative left-frontal
activity indexes approach motivation, and because alpha power is inversely
related to cortical activity, the conventional index is `log(right) - log(left)`
alpha. `DEAPLoader` computes it for five frontal pairs (`DEAP_ASYMMETRY_PAIRS`)
and appends it to the EEG block. It is the closest DEAP gets to a motivated,
low-dimensional feature rather than a whole-scalp dump.

This is a motivated default, not a validated biomarker in this paradigm. Treat
it as the starting hypothesis, not the answer.

---

## Face video is not auto-loaded

`load_deap` builds `eeg`, `peripheral`, and `behavior`, but not `face`. Three
reasons:

1. The face videos are distributed in separate archives from the signals.
2. Only 22 of 32 participants have them. Auto-including the face block would
   force an inner join that discards a third of the participants for *every*
   analysis, whether or not the face arm is in use.
3. Frame extraction plus ViT encoding is heavy, and should be an explicit,
   cached step.

`available_face_videos(root)` reports which participants have video on disk.
Build the face block separately with `FaceLoader`, key it by the same
`(subject, exp-NN)` pairs, and join it deliberately when you want it. On DEAP the
`zEMG` channel (zygomaticus, the smile muscle) is already in the peripheral
block and is a cheaper first proxy for facial affect than the video.

---

## Analytic choices that change the numbers

Published DEAP accuracies are frequently not comparable, because these choices
move them and are not always reported. Fix each before looking at results.

- **Binarisation.** `fixed` (split at 5.0) keeps labels comparable across
  participants and produces real per-participant imbalance. `subject_median`
  guarantees balanced classes but changes the label's meaning to "high for this
  person" and lets a model score by learning response style. Use it only with
  subject-grouped CV, which this framework enforces regardless.
- **Target.** valence, arousal, dominance, or liking are four different
  problems with four different difficulties. Pick one a priori.
- **Baseline correction.** On by default. It changes the features; it is not a
  free cleanup step.
- **Which bands, and asymmetry on or off.** All recorded in the EEG block's
  provenance.

None of this is corrected for multiple comparisons across the arms in
`compare_forecast_arms`. Preregister one primary arm, or correct across them.
