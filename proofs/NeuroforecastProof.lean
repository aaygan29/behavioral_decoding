import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic

/-!
# The neuroforecasting aggregation kernel, formally verified

This file proves the one non-obvious mathematical claim underneath the
value-decision stack: a fixed predictor (the denoised neural / value signal) is
strictly more correlated with an *aggregate* outcome than with an *individual*
outcome, purely because averaging shrinks idiosyncratic noise. This is the
statistical mechanism that makes the Knutson and Genevsky "neuroforecasting"
result possible (a signal that weakly predicts one person can strongly predict
the crowd). It does NOT by itself prove any empirical neuroscience claim; it
proves the aggregation identity those claims rely on.

Model. Individual outcome `Y = S + ε`, where `S` is the shared, predictable
signal component (variance `a > 0`) and `ε` is idiosyncratic noise (variance
`v ≥ 0`), independent of `S`. Take the predictor to be `S` itself (best case).
Then Cov(S, Y) = a, Var S = a, Var Y = a + v, so the squared correlation is

    ρ²(a, v) = a² / (a (a + v)) = a / (a + v).

Averaging the outcome over `N` independent individuals replaces the noise
variance `v` by `v / N` (variance of a mean of `N` iid terms), so the aggregate
squared correlation is `ρ²(a, v / N)`.
-/

namespace Neuroforecast

/-- Squared correlation between the denoised signal (variance `a`) and a target
`Y = S + ε` whose idiosyncratic-noise variance is `v`. -/
noncomputable def rhoSq (a v : ℝ) : ℝ := a / (a + v)

/-- A correlation squared never reaches 1 while any idiosyncratic noise remains. -/
lemma rhoSq_lt_one {a v : ℝ} (ha : 0 < a) (hv : 0 < v) : rhoSq a v < 1 := by
  unfold rhoSq
  rw [div_lt_one (by positivity)]
  linarith

/-- Less idiosyncratic noise means a higher (squared) correlation. -/
lemma rhoSq_antitone {a v₁ v₂ : ℝ} (ha : 0 < a) (hv₁ : 0 ≤ v₁) (h : v₁ < v₂) :
    rhoSq a v₂ < rhoSq a v₁ := by
  unfold rhoSq
  gcongr

/-- **Aggregation strictly improves prediction.** For any real group size `N > 1`,
the signal correlates more strongly with the `N`-averaged outcome than with a
single individual's outcome. This is the neuroforecasting kernel. -/
theorem aggregation_improves {a v N : ℝ} (ha : 0 < a) (hv : 0 < v) (hN : 1 < N) :
    rhoSq a v < rhoSq a (v / N) := by
  have hN0 : 0 < N := by linarith
  have hlt : v / N < v := by
    have h2 : v / N < v / 1 := by gcongr
    simpa using h2
  exact rhoSq_antitone ha (le_of_lt (div_pos hv hN0)) hlt

/-- **More data is monotonically better.** A larger sample yields a strictly
higher aggregate correlation. -/
theorem more_data_is_better {a v N₁ N₂ : ℝ}
    (ha : 0 < a) (hv : 0 < v) (h1 : 0 < N₁) (h : N₁ < N₂) :
    rhoSq a (v / N₁) < rhoSq a (v / N₂) := by
  have h2 : 0 < N₂ := by linarith
  have hlt : v / N₂ < v / N₁ := by gcongr
  exact rhoSq_antitone ha (le_of_lt (div_pos hv h2)) hlt

/-- The aggregate correlation stays below 1 for every finite sample: aggregation
denoises but never fully removes the ceiling while `v > 0`. -/
theorem aggregate_below_one {a v N : ℝ} (ha : 0 < a) (hv : 0 < v) (hN : 0 < N) :
    rhoSq a (v / N) < 1 :=
  rhoSq_lt_one ha (div_pos hv hN)

end Neuroforecast
