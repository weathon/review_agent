Now I have all the information needed to write the final consolidated review.

## Summary

This paper proposes a tighter auditing procedure for f-differential privacy that requires only a single training run. The core technical contribution is Theorem 9, which derives a novel recursive bound on the probability of an adversary making a given number of correct guesses under f-DP, combining trade-off function analysis with a shuffling argument to handle dependencies between guesses. Algorithm 3 numerically computes the bound, and Theorem 10 provides a formal correctness guarantee. Experiments on Gaussian mechanisms and CIFAR-10 with DP-SGD demonstrate consistently tighter empirical privacy estimates than Steinke et al. (2023).

## Strengths

- **Theorem 9 is a genuine and novel technical contribution.** The recursive bound combining trade-off function analysis with a shuffling argument to handle dependent adversary guesses is, as the authors claim, a new technique with potential broader applications (Section 1, Step 2). The insight that conditioning on the first canary's outcome requires different analyses for the two cases (correct vs. incorrect) — and that using Step 1's bound for both cannot be simultaneously tight — is elegant and non-trivial.

- **Clear, consistent improvement over the only direct baseline.** Figures 1–4 demonstrate that the method yields tighter empirical ε values than Steinke et al. (2023) across all tested settings. Importantly, Figure 1 shows that the proposed bounds *improve* with more canaries while Steinke et al.'s *degrade* — a qualitatively different behavior that directly validates the core claim that the O(m·δ) degradation is avoided.

- **Avoids the O(m·δ) degradation via the full f-DP curve.** The paper clearly identifies the concrete mechanism for Steinke et al.'s sub-optimality — the linear (ε,δ) approximation of the privacy curve is tight at only a single point, forcing an O(m·δ) penalty when used for all events simultaneously (Section 4.1, "Why is our bound better?"). Using the exact f-DP curve avoids this, which is both intuitive and well-supported experimentally.

- **Honest about limitations and specific about sources of sub-optimality.** The paper acknowledges the remaining empirical-theoretical gap, identifies Equations 6–7 in the proof of Theorem 10 as the concrete source, and correctly notes that the method "does not provide a strict upper bound on privacy guarantees but instead offers an estimate" (Section 5). This transparency is commendable.

- **Formal correctness guarantee for the numerical algorithm.** Theorem 10 proves that if Algorithm 3 returns True, then the probability of ≥c correct guesses is ≤τ for any f-DP mechanism and any bounded-guess adversary, providing a provable false-negative rate guarantee essential for auditing.

## Weaknesses

### Fatal
None.

### Major

- **The reconstruction game generalization (k > 2) is presented as a contribution but receives zero experimental validation.** The paper explicitly motivates the generalization as allowing "auditing mechanisms with fewer canaries" (Section 1, Step 3; Section 2.1). Algorithm 2 is defined, Theorem 9 and Algorithm 3 depend on k, and the promised benefit (fewer canaries needed for equivalent statistical power) is a concrete practical claim. Yet every experiment uses k = 2 (membership inference). Whether the recursive bounds remain tight for larger k, whether the numerical algorithm is stable, and whether the promised benefit materializes are all untested. A claimed contribution with no experimental backing is a structural gap — it is presented as a feature but is essentially an unchecked conjecture.

- **Experimental validation is limited in scale and diversity.** The real-data experiments (Figures 2–4) use a single small model (WRN16-4 on CIFAR-10). The single-run efficiency of the method matters most for large model training where multiple runs are prohibitively expensive — precisely the setting where the paper's contribution would be most impactful. Without at least one experiment at larger scale or on a different architecture, it is unclear whether the improvements hold where they matter most. The idealized Gaussian mechanism experiments (Figure 1) validate the theoretical bound but do not test the end-to-end auditing procedure under realistic conditions where attack quality varies and the f-DP hypothesis may be misspecified.

### Minor

- **The remaining gap between empirical and theoretical ε is acknowledged but uncharacterized.** The paper identifies the source of the gap (Equations 6–7 in Theorem 10's proof) but provides no analysis of how the gap scales with m, k, or noise level. Without such characterization, practitioners cannot predict when the method will give useful results. This is especially relevant because the gap appears most pronounced at higher privacy levels (larger noise), which are precisely the settings where auditing is most needed.

- **Sensitivity of empirical privacy to the choice of family F is not analyzed.** Definition 7 depends on the choice of trade-off function family F, and the paper briefly discusses this ("How to choose the family of trade-off functions?"), but provides no analysis of how the empirical privacy estimate changes when F is misspecified — e.g., using Gaussian f-DP curves for a mechanism that is not well-approximated by a Gaussian.

- **Canary injection causes measurable utility loss (80% → 77% accuracy) without discussion of implications.** The paper reports this fact (Section 4.1) but does not analyze whether the altered training dynamics affect the audit's validity, particularly for mechanisms where canary injection could interact with the privacy mechanism in non-trivial ways.

### Trivial

- **The title "Auditing f-DP in One Run" could mislead regarding novelty attribution.** The single-run efficiency is inherited from Steinke et al. (2023); this paper's contribution is the tighter f-DP analysis. The abstract credits Steinke et al., but the title does not. This is a minor framing issue.

## Nice-to-Haves

- Experiments with k > 2 (even a simple synthetic experiment on a Gaussian mechanism with k = 4 or k = 10) to validate the reconstruction game generalization.
- At least one experiment on a larger model or different task (e.g., language model fine-tuning) to demonstrate practical applicability at scale.
- A comparison with multi-run auditing methods at equivalent computational budgets to clarify the efficiency-accuracy tradeoff.
- Visualization of the f-DP curves themselves (hypothesized vs. empirical boundary of F_o) beyond the scalar ε comparison, to reveal where and how the audit tightens the estimate.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Proofs deferred to appendix" as a weakness.** This is standard practice for ICLR submissions; the main text provides a detailed technical overview (Section 1) explaining the key steps. Removed as a formatting/presentation nitpick.

- **"Comparison with multi-run auditing methods" as a major weakness.** The paper explicitly builds on and improves the single-run paradigm of Steinke et al. (2023). Demanding comparison with a fundamentally different paradigm is scope creep. Moved to Nice-to-Have.

- **"Algorithm 3 requires oracle access to f̄⁻¹, may be numerically challenging" as a weakness.** For the Gaussian and sub-sampled Gaussian mechanisms used in all experiments, f̄⁻¹ is straightforward to compute. This is a generic concern that does not affect the paper's demonstrated results.

- **"Missing proof sketches for Theorems 9 and 10 in the main text" as a weakness.** The main text contains a detailed 3-step technical overview (Section 1) explaining the key ideas behind both theorems. While proof sketches would be helpful, their absence does not prevent evaluation.

- **"The title overclaims the efficiency contribution" as a significant weakness.** The abstract clearly attributes single-run efficiency to Steinke et al. This is a trivial framing issue at most.

- **"The idealized setting is not an audit" as a fatal/flaw-level criticism.** The idealized setting (Figure 1) validates that the theoretical bound is tighter than Steinke et al.'s — this is a meaningful contribution even if not a full end-to-end audit. The paper also includes real experiments (Figures 2–4) with actual attacks on CIFAR-10. The concern is valid but overstated; it belongs at the major level (limited experimental scale), not fatal.

## Novel Insights

The most insightful observation from the reviews is the asymmetry in how the two sub-analyses in Theorem 9's proof interact: the paper deliberately avoids using Step 1's bound for both the "first canary guessed correctly" and "first canary guessed incorrectly" cases because that bound *cannot be simultaneously tight* for both. This non-obvious constraint — that a single inequality cannot serve double duty — is the key to understanding both the technique's power and its remaining sub-optimality (Equations 6–7), and it suggests that future tightening may require a more granular case analysis.

## Suggestions

- Add at least one experiment with k > 2, even on a synthetic Gaussian mechanism, to validate whether the reconstruction game generalization delivers its promised benefit of fewer canaries.
- Provide a brief empirical analysis of how the empirical-theoretical ε gap scales with m and noise level (even a table or supplementary figure), so practitioners can assess when the method gives useful results.
- Consider adding one experiment at larger scale (e.g., fine-tuning a language model) to demonstrate the method's value in the setting where single-run efficiency matters most.

## Calibration

**High-scoring anchors (avg > 7):**
- EUSkm2sVJ6 (avg 7.60, Oral): Quantitative data usage inference (DUCI) — novel algorithm with strong empirical validation across settings. This paper under review has comparable theoretical novelty but less thorough empirical validation.
- xUzWmFdglP (avg 7.50, Spotlight): MMCC for tighter DP guarantees — nearly tight bounds with solid validation. Similar profile to this paper (tighter analysis of existing framework) but with more complete validation.
- HMe5CJv9dQ (avg 7.50, Poster): Privacy-utility trade-offs via low-dimensional structure — theoretical + experimental. Comparable contribution level.

**Medium-scoring anchors (avg 4–6):**
- C8niXBHjfO (avg 6.0, Poster): Synthetic data privacy auditing — systematic evaluation but limited technical novelty. This paper under review has stronger theoretical contribution.
- BXMoS69LLR (avg 4.5, Reject): Blind baselines beat MI attacks — interesting finding but methodology concerns. This paper under review is clearly stronger.

**Low-scoring anchors (avg < 3):**
- XH3OiIhtvf (avg 2.0, Reject): FL for face recognition — no contribution, one dataset/model, poor writing. This paper under review is far above this level.
- TbOcySs6g8 (avg 2.50, Withdrawn): DP synthetic data — overclaimed, incomplete proofs. This paper under review has genuine, verified contributions.

This paper sits above the medium anchors (stronger theoretical contribution than C8niXBHjfO, which scored 6) but below the high anchors (less complete empirical validation than xUzWmFdglP, which scored 7.5). The k > 2 untested generalization and limited experimental scale prevent it from reaching the 7+ tier, but the core Theorem 9 and consistent improvement over the baseline place it solidly above borderline.

**Score: 6.5**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>