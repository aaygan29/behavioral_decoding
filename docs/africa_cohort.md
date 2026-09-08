# African cohort: Kenyan Kiva microloans

A real, individual-level African dataset with a real, non-synthetic market
outcome, run through the same `BehaviorLoader` -> `build_dataset` ->
`run_experiment` path as everything else in this repo, same leakage guards,
same subject/stimulus keying, same calibration reporting.

## What it is not

Not a neuroimaging cohort, and not paired with one. There is no open-access
African fMRI/EEG dataset with the reward-task or market-linked structure this
project's *market-forecasting* design needs (see
[`data_sources.md`](data_sources.md), that gap is real and unresolved). This
is the behaviour + aggregate-market arm only: option 3 from that document
("synthesise the aggregate outcome" from a stimulus that already exists in a
public market), applied to a real African market instead of a Western one.
(A different, real African EEG cohort, ASZED-153, Nigeria, is used
separately for the neuroprivacy risk demonstration; see
[`africa_neuroprivacy.md`](africa_neuroprivacy.md). The two are not linked:
different countries, different subjects, different purpose.)

## Governance context

Kenya has no binding, AI-specific national law as of this writing (only a
non-binding National AI Strategy 2025–2030), see the governance table in
[`africa_neuroprivacy.md`](africa_neuroprivacy.md#governance-context-why-this-risk-isnt-hypothetical)
for the sourced comparison against Nigeria and the citation.

## Source and provenance

- **Origin**: Kiva ([kiva.org](https://www.kiva.org)) microloan listings, the
  "data science for good" loan release used widely in the crowdfunding /
  microfinance ML literature.
- **Access path used here**: Kiva's own snapshot host
  (`s3.kiva.org/snapshots/...`) no longer resolves as of 2026-08. The data was
  pulled instead from a public re-host,
  `raw.githubusercontent.com/pycaret/pycaret/master/datasets/kiva.csv`
  (PyCaret's example-datasets directory), fetched by
  [`scripts/fetch_africa_cohort.py`](../scripts/fetch_africa_cohort.py).
- **Cohort cut**: filtered to `country == "Kenya"`, 3,023 individual loan
  records out of the full mirror (which also contains Dominican Republic and
  Ecuador rows; those are out of scope here).
- **Columns used**: `loan_amount`, `gender` (F/M), `nonpayment` (whether the
  lender or the field partner bears default risk), `en` (the borrower's loan
  description, free text), `sector` (Agriculture, Food, Retail, Services,
  Clothing, ...), `status` (binary: loan repaid vs. defaulted/not funded).
- **License**: no explicit license is attached to the mirror; Kiva's terms of
  use govern the underlying data. Re-verify before any external publication,
  see the standing note in [`data_sources.md`](data_sources.md).

## How it's wired into the pipeline

[`scripts/run_africa_cohort.py`](../scripts/run_africa_cohort.py):

- **Trial keys**: `subject_id` = one synthetic id per loan record (`KE-00000`,
  ...); `stimulus_id` = `sector`. Sectors play the role stimuli play everywhere
  else in this repo, the unit the aggregate arm holds out and forecasts.
- **Behaviour features**: `loan_amount`, `gender_female`, `lender_bears_risk`,
  and `description_word_count` (a plain word count of the borrower's own
  description, stripped of HTML, a cheap proxy for how much a borrower wrote
  about their business, nothing more). `status` is excluded from the feature
  matrix via `outcome_column`, same as every other loader in this repo.
- **Individual-level target**: `status` (repaid vs. not), ordinary
  classification, same nested CV and calibration reporting as the other arms.
- **Aggregate-level target**: per-sector repayment rate, `forecast_task:
  regression`, held out by sector. This tests whether loan-level features
  aggregated to the sector level forecast that sector's real repayment rate,
  the same "individual signal -> group-level forecast" structure as the
  brain/market comparison elsewhere in this repo, but on Kenyan borrowers
  instead of Stanford crowdfunding subjects or DEAP music-video viewers.

## Reproducing it

```bash
python scripts/fetch_africa_cohort.py   # downloads data/raw/kiva_kenya.csv
python scripts/run_africa_cohort.py     # writes results/africa_cohort_kenya/run_record.json
```

## Reading the result

This is a market/behaviour control, not a neural one, there is no brain arm to
compare it against, so it cannot test the Genevsky/Knutson dissociation
directly. What it does test: whether the pipeline's leakage guards,
calibration reporting, and aggregate-forecasting design produce sane, honest
numbers on real (not synthetic, not Western) individual-level data with a real
market outcome.

### What was dropped, and why

The raw table also has a `nonpayment` column (whether the *lender* or Kiva's
*field partner* absorbs a default). It is **not** used as a feature: Kiva's
field partner sets that flag from their own risk assessment of the borrower,
so it is a near-perfect proxy for the outcome (2,100/2,165 "lender"-risk loans
repay; 772/838 "partner"-risk loans do not) rather than an independent
behavioural signal. A first pass that included it produced balanced accuracy
0.95, the same shape of mistake the leakage tests in this repo exist to
catch, just arriving through a confounded column instead of a temporal leak.
It was removed; the numbers below are from the corrected run.

### Results (`results/africa_cohort_kenya/run_record.json`)

3,023 individual loan records, 14 sectors, features = `loan_amount`,
`gender_female`, `description_word_count`.

**Individual level** (does a loan's own features predict whether *it* is
repaid?), pooled out-of-fold:

| metric | value |
|---|---|
| majority-class baseline | 0.723 |
| balanced accuracy | 0.630 (95% CI 0.615–0.646) |
| ROC AUC | 0.726 |
| Brier score | 0.168 |
| ECE | 0.020 |

A real but modest individual-level signal, well above the majority baseline
on balanced accuracy, calibration is good (ECE 0.02), and the confidence
interval is tight because n=3,023.

**Permutation test is degenerate here and should not be read as a null
result.** `permutation_test` shuffles labels *within subject group*; every
subject in this cohort is a singleton (one loan = one row = one subject), so
permuting a group of size 1 is a no-op and the "null" distribution collapses
onto the observed value, reporting p=1.0. That is an artifact of applying a
repeated-measures-style permutation to non-repeated data, not evidence against
the individual-level effect, the bootstrap CI above is the number to trust
for this cohort. This is left as-is on purpose rather than patched, because it
is a real limitation of applying this framework's default statistics to a
single-observation-per-subject dataset, and hiding it would be the same
mistake the confound above was.

**Aggregate level** (does the mean of those features, by sector, forecast that
sector's real repayment rate?), held out by sector:

| arm | n sectors | out-of-sample R² | Pearson r | p |
|---|---|---|---|---|
| behavior_only | 14 | 0.406 | 0.661 | 0.010 |

Real signal, but `n_stim=14`, the pipeline's own warning fires
("out-of-sample estimates at this size are extremely noisy") and it should be
read as suggestive, not as a headline number. This is the same caution the
DEAP and NARPS arms apply to their own small-n forecasts; it applies here too.
