## Summary

This paper studies how the initialization of neural networks affects the learnability of high-degree (almost-full) parity functions by gradient descent. It shows that almost-full parities can be efficiently learned to perfect accuracy with Rademacher (±1) initialization (and σ-perturbed Rademacher with σ = O(d⁻¹)), but learning provably fails under Gaussian initialization or perturbed Rademacher with large enough constant σ. The negative result relies on a novel measure called "initial gradient alignment" (GAL) and a "junk-flow" coupling argument, providing a general hardness criterion that applies beyond parity and beyond Boolean inputs.

## Strengths

- **Settles an important open case in the parity learning landscape.** Prior work characterized parities when both k and d−k grow (hard) and when k = O(1) (learnable), leaving k = d − O(1) unsettled. The cross-predictability hardness of Abbe & Sandon (2020) does not apply here since the function class is too small. This paper closes the gap, providing both positive and negative results.

- **Introduces a novel and generalizable hardness measure (GAL).** The initial gradient alignment (Definition 2) and the general negative result (Theorem 6) constitute a genuine conceptual contribution. Unlike SQ dimension, cross-predictability, or information exponent, GAL applies to a *single* target function, is loss-dependent, and extends beyond Boolean inputs. The junk-flow coupling technique is elegant and potentially useful in other settings.

- **Provides a clean and memorable separation.** The result that the full parity—trivially learnable in the SQ framework—is hard for noisy-GD with Gaussian initialization on neural networks is a crisp conceptual insight. The Rademacher-gaussian dichotomy is well-demonstrated both theoretically and empirically.

- **Goes beyond weak learning in the positive direction.** While prior work (Abbe & Boix-Adsera, 2022) showed only weak learnability for almost-full parities, this paper establishes strong/perfect learning (Theorems 4 and 5), which is a meaningful improvement.

- **Honest and informative experimental section.** The experiments probe beyond the theoretical setting (hinge loss, multi-layer networks, alternative initializations) and honestly report phenomena that the theory does not yet explain (e.g., GAL behavior for small constant σ).

## Weaknesses

### Major

- **The negative result for perturbed Rademacher initialization (Theorem 8) is incomplete.** The proof requires bounding GAL for both hidden and output layer weights under the perturbed initialization. Section 5.2.2 explicitly states: "Together with a similar bound for the output layer weights (which we omit from this version of the paper)." Without this bound, Theorem 8—which underpins the claim that learning fails for σ ≥ σ₀ under perturbed Rademacher—is not fully proved. This is a significant gap in the paper's central claimed separation.

- **The negative result applies only to noisy-GD with correlation loss, not to standard training regimes.** Theorem 6 and its corollaries are proved for the correlation loss L(y, ŷ) = −yŷ. As acknowledged in Remark 3, the key "junk-flow" coupling step (step 3) "is currently limited to the correlation loss." Since practitioners virtually never use correlation loss, and all experiments use hinge loss, this is a fundamental limitation of the hardness result. The empirical evidence for hinge loss (Figure 2, single-neuron computation of GAL) is suggestive but not rigorous.

- **There is a wide gap between the positive σ regime (σ = O(d⁻¹)) and the negative σ regime (σ ≥ σ₀ = constant).** The intermediate regime d⁻¹ ≪ σ ≪ 1 is entirely uncharacterized. The paper's framing of a "threshold phenomenon" (abstract, conclusion) and the claim that Rademacher initialization is a "special case" both suggest a sharp transition, but the theoretical results leave the most interesting regime open. The numerical GAL estimates in Figure 2 (right panel) hint at super-polynomially small GAL even for small constant σ, but this is left inconclusive.

### Minor

- **The positive result requires large width and only trains the output layer in the strongest version.** Corollary 1 requires Ω(d⁴) hidden neurons for standard ReLU (though Ω(d²) for clipped ReLU). More importantly, Theorems 4–5 (correlation loss) only train the output layer weights while freezing hidden weights at initialization. The hinge loss result (Section 4.2) trains both layers but is deferred to the appendix without a crisp in-text theorem statement. The informal Theorem 1 does not mention these restrictions, which may mislead readers about the generality of the positive claim.

- **The experimental setting does not match the theoretical setup.** Experiments use a 4-layer MLP with SGD on hinge loss, while the theory covers 2-layer ReLU networks with noisy-GD and correlation loss. The qualitative agreement between experiments and theory is encouraging, but the mismatch limits direct empirical validation of the theoretical claims.

- **Specific bias choices in the positive result.** Corollaries 1–2 require particular bias settings (e.g., b_i = 0 for even d, b_i = −1 for odd d). While the paper notes that "the precise values are not crucial except for unlucky choices," no general characterization of which bias distributions work is provided.

### Trivial

- None.

## Nice-to-Haves

- Experiments varying dimension d to probe whether the σ-threshold scales as predicted.
- A complete proof of Theorem 8 (output layer GAL bound for perturbed Rademacher).
- Investigation of whether the negative result extends to hinge loss, even in a restricted model.
- Tighter width bounds for the positive result (could d² be improved?).

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"The SQ separation is merely about algorithm classes, not intrinsic function hardness."** — This is partially valid as a framing concern, but the paper's actual claim is accurate: the full parity is trivially SQ-learnable yet hard for (noisy) GD with most initializations. The fact that this is about algorithm classes rather than intrinsic function complexity is exactly what makes it interesting—the function is easy for one natural computational model and hard for another. This is similar to how all computational separations work. Downgraded to a minor interpretive note rather than a weakness.

- **"The experiments don't test correlation loss."** — While true, this is a nice-to-have rather than a core flaw. The experiments are designed to validate the *qualitative* prediction, and using the more natural hinge loss makes the experiments more relevant to practice.

- **"Experiments where only the output layer is trained."** — While it would strengthen the paper to include such experiments, the current experiments already show that learning succeeds with Rademacher initialization even when all layers are trained, which is a stronger demonstration than the theory requires.

- **"Large polynomial width requirements make this impractical."** — While d⁴ is large, this is typical for provable learning results in the parity literature (compare to the Õ(d^{k−1}) bounds for k-parities). The paper also shows d² is achievable with clipped ReLU. This is a known limitation of the approach, not a fatal flaw.

## Novel Insights

The key conceptual insight—that initialization determines learnability of high-degree parities not through statistical complexity (which is trivial for a singleton class) but through structural alignment captured by GAL—reframes the "hardness of learning" question as one about the *initial state* of the optimization rather than the function class. The junk-flow coupling argument is technically novel: rather than showing that *all functions in a class* look similar from the optimization trajectory (as in cross-predictability arguments), it shows that for a *single* target with small initial alignment, the optimization trajectory remains indistinguishable from training on random labels. This technique may be exportable to other settings where the target presents symmetries (e.g., arithmetic, graph tasks).

## Suggestions

- Complete the proof of Theorem 8 by including the output layer GAL bound, or clearly state Theorem 8 as conditional on this bound rather than as a definitive result.
- Investigate GAL numerically for multi-layer networks (not just single neurons) to assess whether it predicts learning behavior in the experimental setting.
- Even without fully resolving the intermediate σ regime, provide a clear conjecture backed by additional numerical evidence about where the transition occurs.
- Strengthen the informal Theorem 1 by mentioning the required width scaling and output-layer-only training caveat, to align the informal statement more closely with the rigorous results.

## Score and Decision

**Calibration against similar papers:**

- The XOR/parity learning paper (HgOJlxzB16) received scores of 8,8,8,6 — it had similarly novel techniques but more complete results.
- The symmetry-hardness paper (ARPrtuzAnQ) received scores of 8,6,8 — novel framework with some limitations, similar profile.
- The early neuron alignment paper (QibPzdVrRu) received scores of 5,8,5,8 — criticized for strong assumptions and limited generality.

This paper has genuine novelty (GAL measure, junk-flow coupling), settles an open case, and provides a clean conceptual separation. However, it has significant gaps: the incomplete Theorem 8, the correlation-loss-only hardness result, and the wide gap between the positive and negative σ regimes. These are not cosmetic issues—they concern the completeness of the central claimed separation and the generality of the hardness framework. On the other hand, the paper makes a real conceptual contribution that would advance the field even with these limitations acknowledged. Compared to the symmetry-hardness paper (ARPrtuzAnQ) which received a median of ~7, this paper's contributions are somewhat offset by the incomplete proof and the limited scope of the hardness result.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Reject</orange>