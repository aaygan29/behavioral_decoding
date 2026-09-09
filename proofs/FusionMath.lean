/-
  Formal proof of the inverse-variance fusion inequality used in
  docs/biosignal_fusion.md.

  Claim: combining independent noisy estimators by inverse-variance weights gives
  a fused variance no larger than the smallest single-estimator variance:

      1 / (Σ_i 1/σ_i²)  ≤  σ_j²        for every j in the family.

  Two forms are proved:
    * `two_variance` / `two_variance_closed_form` : the two-family case.
    * `fused_le_min` : the general finite-family case over a Finset.
-/
import Mathlib

open Finset

/-- Two-family case: the harmonic-style combination never exceeds either input. -/
theorem two_variance (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    1 / (1 / a + 1 / b) ≤ a := by
  have h : 1 / a + 1 / b = (a + b) / (a * b) := by field_simp; ring
  rw [h, one_div_div, div_le_iff₀ (by positivity)]
  nlinarith [sq_nonneg a, mul_pos ha hb]

/-- Closed form of the two-family fused variance. -/
theorem two_variance_closed_form (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    1 / (1 / a + 1 / b) = a * b / (a + b) := by
  have h : 1 / a + 1 / b = (a + b) / (a * b) := by field_simp; ring
  rw [h, one_div_div]

/-- General finite-family case.  For a nonempty family of strictly positive
    variances `σ : ι → ℝ`, the inverse-variance-fused variance is at most the
    variance of any single member `j`. -/
theorem fused_le_min {ι : Type*} (s : Finset ι) (σ : ι → ℝ)
    (hσ : ∀ i ∈ s, 0 < σ i) {j : ι} (hj : j ∈ s) :
    1 / (∑ i ∈ s, 1 / σ i) ≤ σ j := by
  have hjpos : 0 < σ j := hσ j hj
  have hpos : ∀ i ∈ s, 0 ≤ 1 / σ i := fun i hi => (one_div_pos.mpr (hσ i hi)).le
  have hterm : 1 / σ j ≤ ∑ i ∈ s, 1 / σ i := Finset.single_le_sum hpos hj
  have hsum : 0 < ∑ i ∈ s, 1 / σ i := lt_of_lt_of_le (one_div_pos.mpr hjpos) hterm
  rw [div_le_iff₀ hsum]
  calc (1 : ℝ) = σ j * (1 / σ j) := by field_simp
    _ ≤ σ j * ∑ i ∈ s, 1 / σ i := mul_le_mul_of_nonneg_left hterm hjpos.le
