# Literature basis

Every citation below was checked against PubMed on 2026-08-07. DOIs are linked
so nothing here has to be taken on trust. Where a claim is a summary rather than
a quote, the abstract is the source.

---

## 1. The core result: brain beats behaviour at the population level

This is the finding the repository is organised around, and it is a
*dissociation*, not a correlation. The neural signal that predicts what one
person will choose is not necessarily the signal that forecasts what a market
will do, and the behavioural measure that predicts the individual can fail at
the market completely.

### Genevsky, Yoon & Knutson (2017), *Journal of Neuroscience*
**When Brain Beats Behavior: Neuroforecasting Crowdfunding Outcomes.**
[10.1523/JNEUROSCI.1633-16.2017](https://doi.org/10.1523/JNEUROSCI.1633-16.2017)
(PMID 28821681)

Thirty subjects viewed crowdfunding project descriptions during fMRI and decided
whether to fund each one. Both nucleus accumbens (NAcc) and medial prefrontal
cortex (MPFC) activity predicted individual funding choices trial by trial. Only
NAcc activity generalised to forecast market-level funding outcomes on the
internet weeks later. Behavioural measures from the scanned sample did not
forecast the market at all. The pattern replicated in a second study.

This paper is the reason the framework separates `y_individual` from
`y_aggregate`, and the reason `compare_forecast_arms` always runs a
behaviour-only arm next to the brain-only arm. Reporting a single combined
number would hide precisely the effect that makes this line of work interesting.

### Genevsky & Knutson (2015), *Psychological Science*
**Neural Affective Mechanisms Predict Market-Level Microlending.**
[10.1177/0956797615588467](https://doi.org/10.1177/0956797615588467)
(PMID 26187248)

Two studies. First, in a database of 13,500 real microloan requests, positive
affective features of the borrower photographs predicted request success.
Second, in a 28-subject neuroimaging sample, NAcc activity and self-reported
positive arousal predicted the success of those loan requests on the internet
*above and beyond the scanned sample's own lending choices*.

The "above and beyond own choices" phrasing matters. The neural measure is not
just a noisy proxy for the behaviour; it carries information the behaviour does
not.

### Knutson & Genevsky (2018), *Current Directions in Psychological Science*
**Neuroforecasting Aggregate Choice.**
[10.1177/0963721417737877](https://doi.org/10.1177/0963721417737877)
(PMID 29706726)

The review that names the phenomenon and proposes the mechanism: **affective**
choice components generalise across individuals and so forecast aggregate
choice, while **integrative** choice components confer consistency within an
individual and so predict individual choice. Not every neural process that
predicts the individual forecasts the aggregate to the same degree.

That affective/integrative split is written directly into
`src/behavioral_decoding/synthetic.py` as two channels with different
subject-specific loadings. It is the generative model the framework's positive
control tests against.

### Tong, Acikalin, Genevsky, Shiv & Knutson (2020), *PNAS*
**Brain activity forecasts video engagement in an internet attention market.**
[10.1073/pnas.1905178117](https://doi.org/10.1073/pnas.1905178117)
(PMID 32152105)

At video onset, increased NAcc, increased MPFC, and decreased anterior insula
(AIns) predicted individual choices to start and stop watching. At the group
level, only a subset (increased NAcc, decreased AIns) forecast aggregate view
frequency and duration on YouTube, and did so beyond conventional measures.

This is the source of the default ROI set in `io/fmri.py`: NAcc, MPFC, and AIns,
with the expectation that they behave differently across the two levels.

### Genevsky, Tong & Knutson (2025), *PNAS Nexus*
**Neuroforecasting reveals generalizable components of choice.**
[10.1093/pnasnexus/pgaf029](https://doi.org/10.1093/pnasnexus/pgaf029)
(PMID 40007578)

The most directly relevant paper for anyone worried about sampling. Across two
experiments, forecast accuracy from *behaviour* varied with how demographically
representative the scanned sample was. Forecasts from *brain activity* stayed
significant regardless of representativeness.

The practical implication for a project that may recruit from a narrow local
sample: the neural arm may generalise where the behavioural arm does not. That
is a testable prediction, and it is the reason
`compare_forecast_arms` is structured to make the arms comparable rather than
merged.

---

## 2. The parallel line: neural focus groups for population media effects

### Falk, Berkman & Lieberman (2012), *Psychological Science*
**From neural responses to population behavior: neural focus group predicts
population-level media effects.**
[10.1177/0956797611434964](https://doi.org/10.1177/0956797611434964)
(PMID 22510393)

Smokers viewed three anti-smoking television campaigns during fMRI and also
self-reported which campaign they expected to be most effective. Population
success was measured as the change in call volume to the 1-800-QUIT-NOW hotline
before and after each campaign launched. Medial prefrontal activity predicted
the population response. The self-reports did not.

### Falk et al. (2015), *Social Cognitive and Affective Neuroscience*
**Functional brain imaging predicts public health campaign success.**
[10.1093/scan/nsv108](https://doi.org/10.1093/scan/nsv108)
(PMID 26400858)

Fifty smokers viewed anti-smoking messages; the outcome was the population
response to the same messages across roughly 400,000 emails. Aggregated MPFC
activity in a self-localiser region *complemented* existing self-report data,
with a combined model R² up to 0.65. The relationship depended on message
content: it held for strong negative arguments against smoking, not for
compositionally similar neutral images.

Two design lessons taken from this paper. First, "brain instead of behaviour" is
the wrong framing; the combined model is what performed best here, which is why
`compare_forecast_arms` includes an `all_modalities` arm. Second, the effect was
content-dependent, so stimulus category belongs in the design as a factor rather
than being pooled away.

---

## 3. What this literature does *not* license

Worth stating explicitly, because the temptation runs the other way.

- **Sample sizes are small.** Genevsky et al. (2017) scanned 30 subjects;
  Genevsky & Knutson (2015) scanned 28. These are adequately powered for
  aggregate forecasting precisely because the unit of analysis becomes the
  *stimulus*, not the subject. It does not follow that individual-level
  decoding is well powered at n = 30. The two levels have different effective
  sample sizes, and `forecast_market` warns below 20 stimuli for this reason.

- **NAcc activity is not a "buy signal".** The inference runs forward
  (this manipulation produced this activity) far better than backward (this
  activity means this psychological state). Reverse inference from a small ROI
  to a specific mental state is not supported by these findings, and no function
  in this repo should be described as reading intent.

- **Forecasting is not causation.** Every result above is predictive. None of it
  establishes that NAcc activity causes market outcomes, and none of it licenses
  a claim about manipulating outcomes by targeting a region.

- **EEG and facial tracking do not inherit this evidence.** The published
  dissociation is an fMRI result. Whether a scalp EEG or facial-expression proxy
  carries the same stimulus-general affective component is an open question and
  one of the more interesting things this project could actually test. Until it
  is tested here, the EEG and face arms are exploratory, and the repo should say
  so wherever their numbers appear.

---

## 4. Method references

These are the standard sources for the machine-learning components. They are
methods citations, not evidence about brains.

- **SMOTE.** Chawla, Bowyer, Hall & Kegelmeyer (2002), "SMOTE: Synthetic
  Minority Over-sampling Technique," *Journal of Artificial Intelligence
  Research* 16:321-357. [10.1613/jair.953](https://doi.org/10.1613/jair.953)
- **Borderline-SMOTE.** Han, Wang & Mao (2005), in *Advances in Intelligent
  Computing*, 878-887.
  [10.1007/11538059_91](https://doi.org/10.1007/11538059_91)
- **Bagging.** Breiman (1996), "Bagging predictors," *Machine Learning*
  24:123-140. [10.1007/BF00058655](https://doi.org/10.1007/BF00058655)
- **Vision Transformer.** Dosovitskiy et al. (2021), "An Image is Worth 16x16
  Words: Transformers for Image Recognition at Scale," ICLR.
  [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- **imbalanced-learn.** Lemaître, Nogueira & Aridas (2017), *JMLR* 18(17):1-5.
  [jmlr.org/papers/v18/16-365.html](https://jmlr.org/papers/v18/16-365.html)

On resampling and leakage specifically, the failure mode this repo guards
against is documented in the imbalanced-learn user guide's section on combining
samplers with cross-validation: resampling must happen inside the fold. The test
`tests/test_leakage.py::test_smote_before_split_inflates_accuracy_and_the_pipeline_does_not`
reproduces the inflation on pure noise so the magnitude is on record rather than
assumed.
