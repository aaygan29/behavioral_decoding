# NARPS (ds001734)

How this framework loads the NARPS mixed-gambles dataset, the format details that
matter, and an honest statement of what NARPS can and cannot show.

NARPS (Botvinik-Nezer et al., 2019, *Scientific Data*,
[10.1038/s41597-019-0113-7](https://doi.org/10.1038/s41597-019-0113-7); dataset
[ds001734 on OpenNeuro](https://openneuro.org/datasets/ds001734), DOI
[10.18112/openneuro.ds001734.v1.0.5](https://doi.org/10.18112/openneuro.ds001734.v1.0.5))
is the fMRI counterpart to DEAP in this project: a reward-circuit task with
enough subjects to matter. On each trial a participant sees a 50/50 gamble with a
possible **gain** and a possible **loss** and decides whether to accept it. This
is exactly the anticipatory-affect paradigm the framework's fMRI ROIs come from,
so it is the right dataset to validate the fMRI loader and the NAcc / vmPFC / AIns
sphere extraction on real data.

**Access.** NARPS is public on OpenNeuro; no licence agreement. It is large
(4-D BOLD across ~108 subjects), so the loader downloads nothing and expects a
local BIDS tree, ideally with fMRIPrep-preprocessed derivatives.

---

## Verified format

Confirmed against the dataset's own files on OpenNeuro (2026-08-07), not from
memory. A real events file looks like:

```
onset   duration  gain  loss  RT     participant_response
4.071   4         14    6     2.388  weakly_accept
11.834  4         34    14    2.289  strongly_accept
27.535  4         10    10    1.457  weakly_reject
36.435  4         12    19    1.973  strongly_reject
```

- **Task**: `MGT` (mixed gambles task).
- **TR**: 1.0 s (`task-MGT_bold.json`, `RepetitionTime: 1`).
- **Trials**: 4 runs per subject, 64 trials per run, 4 s cue per trial.
- **`events.tsv` columns**: `onset`, `duration`, `gain`, `loss`, `RT`,
  `participant_response`.
- **Response values**: `strongly_accept`, `weakly_accept`, `weakly_reject`,
  `strongly_reject`, and `NoResp` for no-response trials (RT coded 0).
- **`participants.tsv` columns**: `participant_id`, `group`, `gender`, `age`.
- **Groups**: `equalIndifference` (gains 10-40, losses 5-20) and `equalRange`
  (gains and losses both 5-20). The two groups saw different gamble matrices;
  this is a real design factor, not noise (see below).

---

## How NARPS maps onto the framework

| framework concept | NARPS realisation |
|---|---|
| individual outcome `y_individual` | accept vs reject the gamble |
| stimulus key `stimulus_id` | the gamble, keyed by `(gain, loss)` |
| aggregate outcome `y_aggregate` | population acceptance rate per gamble |
| fMRI block | NAcc / vmPFC / AIns sphere betas per trial |
| behaviour block | gain, loss, expected value (RT off by default, see below) |

**Individual choice.** `strongly_accept` and `weakly_accept` collapse to accept
(1); `strongly_reject` and `weakly_reject` to reject (0). `NoResp` trials are
dropped, not imputed: a missing response is missing data, and guessing it would
manufacture labels. The strong/weak distinction is preserved in a separate
confidence field for anyone who wants a 4-level analysis, but the binary accept
is the default outcome.

**Stimulus key.** The gamble is defined by its `(gain, loss)` pair, and those
pairs recur across subjects, which is what makes a stimulus key and an aggregate
outcome possible at all. Keys look like `g14_l06`.

---

## The traps

Fewer than DEAP, but the ones here are sharp.

**1. The behaviour arm is *supposed* to be strong, and that inverts the usual
story.** In the crowdfunding and video paradigms the neural signal beats the
behavioural measure. Gambles are different: acceptance is largely a function of
gain and loss through expected value, so an economic model built from `gain` and
`loss` forecasts aggregate acceptance almost by construction. On NARPS, expect
**behaviour to win the aggregate arm**. That is not a failure of the neural arm;
it is what gambles are. NARPS is primarily an **individual-level** validation
(can NAcc/vmPFC betas predict this person's accept/reject), and the aggregate arm
is a secondary check where the economic baseline is expected to dominate. Do not
report a forced brain-beats-behaviour headline on NARPS; it would be dishonest
about the task.

**2. The two groups saw different gamble matrices.** `equalIndifference` and
`equalRange` do not share the same `(gain, loss)` grid. Pooling them for the
aggregate arm mixes two stimulus spaces, and a gamble present in one group may be
absent in the other. The loader keeps the group label and, by default, warns when
you pool across groups for aggregate forecasting. Analyse within group unless you
have a reason not to.

**3. Slice-timing and the haemodynamic lag are real analytic choices.** The
default trial feature is a peak-window mean of the BOLD 4 to 8 s after cue onset,
which is a crude stand-in for a proper first-level GLM. It is fine for getting the
pipeline running and wrong for a publishable estimate. Replace it with
`nilearn.glm.first_level` before reporting. The loader records
`extraction="peak_window_mean"` in provenance so this is never hidden.

**4. Confounds must be regressed, and the loader now cleans them for you.**
Motion, framewise displacement, and the aCompCor components in the fMRIPrep
`*_desc-confounds_timeseries.tsv` are not optional for a reward-ROI analysis:
head motion correlates with task events and with individual differences. The
loader no longer passes the raw table to the masker: `io/confounds.py` selects an
explicit nuisance set (default `motion12+physio`), fills the leading-row NaNs on
the derivative and framewise-displacement columns, and drops non-numeric columns,
recording what it kept and dropped in provenance. If you supply no confounds it
still warns.

**5. No-response trials and RT = 0.** `NoResp` rows carry `RT = 0`. Feeding that
zero into an RT feature as if it were a fast response is wrong. The loader drops
`NoResp` trials before building any block.

**6. RT is a post-decision variable, so it is off by default.** Response time is
recorded *after* the choice is made; it is a consequence of the decision, not a
cue available before it. Including it as a feature when predicting accept/reject
leaks the outcome (fast confident accepts vs slow conflicted rejects separate on
RT alone) and inflates accuracy. `NARPSLoader(include_rt=False)` is the default;
turn it on only for an explicit RT/confidence analysis, never for the headline
choice-prediction number.

---

## The theory-specified ROI

On the crowdfunding side the signal lived in NAcc (gain anticipation) and MPFC.
NARPS is built for exactly this circuit: the canonical mixed-gambles result is
that NAcc tracks potential gain and a vmPFC/MPFC region tracks the net expected
value, with anterior insula tracking potential loss. The framework's default
ROIs (`NAcc_L/R`, `MPFC`, `AIns_L/R`) are therefore already the right set, and
the NARPS loader reuses `FMRILoader`'s sphere extraction rather than defining its
own. Confirm the exact sphere coordinates against whatever prior you are
replicating before reporting; sphere placement is a genuine analytic degree of
freedom.

---

## What NARPS validates, and what it does not

**Validates**: that the fMRI loader reads a real BIDS reward task, extracts
trial-wise ROI features, and that those features predict individual accept/reject
above chance with subject-grouped CV. That is milestone 2 in
[data_sources.md](data_sources.md), and it is a real result on its own.

**Does not validate**: the brain-beats-behaviour aggregate claim. Gambles are the
wrong task for that, for the reason in trap 1. NARPS is the fMRI plumbing check
and the individual-decoding check. The aggregate neuroforecasting story needs a
stimulus set with a market outcome that is *not* a deterministic function of the
stimulus parameters, which is what the crowdfunding/video/microlending paradigms
provide and gambles do not.

---

## Usage sketch

```python
from behavioral_decoding.io.narps import load_narps

# Individual-level, one group, from fMRIPrep derivatives.
dataset = load_narps(
    root="/path/to/ds001734",
    derivatives="/path/to/fmriprep",
    group="equalIndifference",
    space="MNI152NLin2009cAsym",
)
```

See `scripts/run_narps.py --demo` for the whole path on a synthetic BIDS fixture
that needs no download.
