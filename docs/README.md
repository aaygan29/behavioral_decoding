# Documentation index

Grouped for readability. Files stay flat so existing links keep working.

## Methods and design
- [design.md](design.md) - each design decision, its rejected alternative, and what would falsify it.
- [estimators.md](estimators.md) - per-modality base learners, including the Riemannian EEG path.
- [literature.md](literature.md) - the neuroforecasting canon, with verified DOIs.

## Datasets and loaders
- [deap.md](deap.md) - DEAP loader: format traps, circularity, the market route.
- [narps.md](narps.md) - NARPS loader: format, the individual-vs-aggregate honesty point.
- [data_sources.md](data_sources.md) - candidate datasets per modality and the gaps between them.

## Biosignal fusion, privacy, and the Africa / AIxBio risk
- [biosignal_fusion.md](biosignal_fusion.md) - does combining biosignal families beat the best single one? Literature, the inverse-variance math, results, runtime gates, and the Africa framing. Figure in [figures/](figures/biosignal_fusion.png).
- [biosignal_privacy_generalization.md](biosignal_privacy_generalization.md) - why the pipeline absorbs any biosignal family unchanged, and the parameters that set the size of the risk.
- [africa_neuroprivacy.md](africa_neuroprivacy.md) - real-EEG neuroprivacy demonstration (ASZED-153, Nigeria).
- [africa_cohort.md](africa_cohort.md) - real behavioural cohort run (Kenyan Kiva microloans).

## Related
- Fusion math is validated empirically in [`tests/test_fusion_math.py`](../tests/test_fusion_math.py).
