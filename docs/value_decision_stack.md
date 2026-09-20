# The value-decision stack

A shared, neurally-grounded generative model of a single value-based decision, and the rule for aggregating it into population prediction. decision-phenotype, behavioral_decoding, and the Sapient prediction work are each modeling a slice of this stack; this document is the common backbone so they stop reinventing it and so every layer has to earn its place.

Two disciplines are enforced throughout:
1. Every layer names a circuit, a fitted parameter, and a published result showing the computation can be used this way. No layer is included on vibes.
2. Every layer must survive ablation. A layer that does not improve out-of-sample per-person prediction over the baseline is removed, however biologically pretty. This is the program's own hard-won rule: a "brain-like" component that is present but functionally redundant is worse than no component, because it looks principled while carrying dead weight.

## The stack

Read top to bottom, this is the path from "what are my options" to "I acted."

| # | Layer | Computation | Circuit | Fitted parameter | Grounding paper | Neuroeconomics anchor |
|---|-------|-------------|---------|------------------|-----------------|-----------------------|
| L0 | State representation | how the world is carved into states and features before value applies | OFC as cognitive map of task space | state/feature set, latent-cause partition | Niv, "Learning task-state representations" (Nat Neurosci 2019); Schuck et al., OFC cognitive map (Neuron 2016) | Yael Niv |
| L1 | Value / anticipation | expected value of each option, especially anticipated reward | ventral striatum / NAcc, mPFC | value weights per feature | Knutson et al. anticipatory affect; Bartra et al. value meta-analysis | Knutson, Genevsky |
| L2 | Evidence comparison | hold and compare options, commit to one | LPFC population geometry | drift rate, comparison gain | Li, Chrysanthidis, Brincat, Rose, Miller (iScience 2026) | Rangel, Padoa-Schioppa |
| L3 | Action pressure | turn a chosen value into a motor bias, separate from evidence | superior colliculus | additive per-action bias term | Takács, Bimbard, ..., Carandini (bioRxiv 730072) | logistic/softmax choice |
| L4 | Gain / learning rate | arousal, exploration, how fast beliefs and values move | locus coeruleus norepinephrine | gain, learning rate, exploration temperature | Su et al., LC-NE (bioRxiv 717727) | Niv (learning-rate adaptation), dopamine RPE |
| L5 | Belief / prior update | the precision that makes a belief stick or move | precision-weighting (hierarchical inference) | prior precision | Novelli, Stoliker, Razi et al., PsiConnect (Sci Data 2026); REBUS | Friston, active inference |

### The layer that was missing: L0, state representation (Niv)
None of the handed-in papers cover L0, and it is the one most likely to be the true source of between-person differences. Niv's line shows the brain infers a latent state space (which features matter, how the task decomposes into states) and only then runs value learning over it, with OFC representing that task-state map. The consequence for any per-person model is a specific and avoidable error: if two people carve the task into different states, a model without L0 will absorb that difference into a value or drift parameter and mislabel a representational difference as a preference or an impulsivity difference. So L0 is not optional decoration for a phenotype model; it is a confound-control layer. Fit it, or explicitly assume it is shared and test that assumption.

## The aggregation bridge: individual model to population prediction

The deepest neuroeconomics result in this stack is not "NAcc predicts choice." It is that individual neural signals predict aggregate behavior even when they only weakly predict the individual. Genevsky and Knutson showed the sample-average NAcc response forecasts real-world outcome (microloan funding, crowdfunding success) better than the sample's own choices do, and the effect extends to market-level dynamics. This is what lets a per-person model become a population predictor, which is exactly what Sapient needs and what the cognitive-security collective-scale work needs.

### Why it works, formally verified
The mechanism is statistical, not mystical: averaging over people shrinks idiosyncratic noise, so a fixed signal correlates more strongly with the denoised aggregate than with any noisy individual. Writing the shared signal variance as `a` and idiosyncratic noise variance as `v`, the squared correlation is `rho^2 = a / (a + v)`, and averaging over `N` people replaces `v` by `v / N`.

This kernel is machine-checked in Lean 4 (Mathlib), see `proofs/NeuroforecastProof.lean`:
- `rhoSq_antitone`: less idiosyncratic noise gives strictly higher correlation.
- `aggregation_improves`: for any group size `N > 1`, the signal correlates strictly more with the `N`-averaged outcome than with one individual. This is the neuroforecasting kernel.
- `more_data_is_better`: correlation strictly increases with sample size.
- `aggregate_below_one`: it never reaches 1 while any idiosyncratic noise remains (no free lunch; aggregation denoises but does not manufacture signal).

This is the correlation-form sibling of the inverse-variance fusion inequality already proved in behavioral_decoding (`proofs/FusionMath.lean`, `1/(sum_i 1/sigma_i^2) <= sigma_j^2`). Both are the same variance-reduction principle: combining independent noisy sources beats any single source. Fusion combines modalities within a person; aggregation combines people. The stack uses both.

What the proof does and does not establish. It establishes the aggregation identity that makes neuroforecasting mathematically possible and quantifies it. It does not establish that NAcc is the signal, that the noise is independent, or that any specific dataset behaves this way. Those are empirical and belong to L1 and to each project's validation. Stated plainly so we do not overclaim a theorem into a neuroscience result.

## How each project uses the stack

- decision-phenotype. Currently fits roughly L2 to L4 (AIM-DDM). The upgrade is to add L0 on top (so phenotype differences are not misattributed state-representation differences) and L5 underneath (precision), then run the ablation ladder to keep only the layers that improve out-of-sample per-person prediction. L0 is the highest-value addition and the strongest reviewer defense.
- behavioral_decoding. Each modality maps to a layer: fMRI reward signatures to L1, pupil/latency to L4, choice to L3, task-structure probes to L0. This turns multimodal fusion from "everything in a bag" into "each stream reports a different stage of one model," a much stronger inductive bias and a cleaner story.
- Sapient. Uses the full stack plus the aggregation bridge: fit the per-person stack, then aggregate via the neuroforecasting result to predict population or market response. The Lean-verified kernel is the formal backing for the aggregate product claim, and the individual-to-aggregate move is the actual moat, not any single encoder.

## The ablation ladder (mandatory build order)

Do not ship the six-layer model as a unit. Build it one layer at a time and keep a layer only if it earns its slot.
1. Baseline: plain DDM (L2 + L3 collapsed), out-of-sample per-person accuracy recorded.
2. Add one layer, refit, measure the out-of-sample gain and its confidence interval across seeds.
3. Keep the layer only if the gain is real (survives seed variance and a specification check). Otherwise drop it.
4. Report the final model as the set of load-bearing layers, with the ablation table, not as the full stack asserted a priori.

This is also the honesty guardrail: an elegant modular model whose SC term or precision layer adds nothing over a DDM is a Gate 3 / Gate 5 failure waiting to happen. The ablation table is what makes the stack publishable rather than merely tidy.

## Calibration notes
- In-silico and correlational throughout. None of these layers is a causal claim in humans without a lesion/perturbation test; language stays correlational unless a manipulation is run.
- The grounding papers are mostly rodent or NHP electrophysiology. Their use here is as computational-role evidence (this circuit can compute this term), not as proof the human implementation is identical.
- L5 (precision) and the bounded-influence bifurcation model share the prior-precision coordinate; that link is developed in the bounded-influence repo and is the same math viewed as defense.

## Verification status
- Aggregation kernel: machine-checked in Lean 4 / Mathlib (`proofs/NeuroforecastProof.lean`), builds clean.
- Applicability of each layer: literature-verified (citations above); neuroforecasting and the Niv state-representation layer confirmed against primary sources.
- Empirical fit and ablation: not yet run. That is each project's job and is where the real result is won or lost.
