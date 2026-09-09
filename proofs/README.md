# Formal proofs

## FusionMath.lean

A Lean 4 + Mathlib proof of the inverse-variance fusion inequality that the
biosignal-fusion argument rests on (see
[`../docs/biosignal_fusion.md`](../docs/biosignal_fusion.md) section 2):

    1 / (Sum_i 1/sigma_i^2)  <=  sigma_j^2   for every family j,

proved in two forms:

- `two_variance` / `two_variance_closed_form`: the two-family case, with the
  closed form `a*b/(a+b)`.
- `fused_le_min`: the general finite-family case over a `Finset`.

The same inequality is also checked empirically in
[`../tests/test_fusion_math.py`](../tests/test_fusion_math.py) (algebraic sweep,
Monte-Carlo operational check, and a correlated-noise negative control).

### Verify

Not run in CI (it would pull all of Mathlib). To check locally:

```bash
# in a Lean project with mathlib as a dependency and the cache fetched
lake exe cache get
lake build FusionMath        # exits 0 iff every theorem type-checks
```

Verified against Mathlib on Lean 4 (`lake build` exit 0, no `sorry`).
