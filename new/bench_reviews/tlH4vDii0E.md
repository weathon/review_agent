## Summary

This paper proposes Causal Transfer Learning (CTL), a method that leverages pre-trained and fine-tuned language model representations as paired "environments" to identify invariant causal features via a decomposition assumption, then applies a causal front-door adjustment over token-level local features to construct an OOD-robust predictor from single-domain data. The approach is validated on semi-synthetic sentiment analysis datasets where spurious correlations are artificially injected.

## Strengths

- **Novel and creative conceptual framework.** The idea of using a pre-trained model and its fine-tuned counterpart as two "environments" sharing causal factors but differing in spurious factors (Assumption 2) is an elegant way to circumvent the need for multi-domain data, which is a genuine practical constraint. This opens a potentially interesting research direction connecting PLMs to causal identification.

- **Clear theoretical motivation from propositions to theorems.** The paper systematically motivates the problem (Propositions 1–3 showing why P(Y|X) is not transportable and P(Y|do(X)) is needed but nonparametrically unidentifiable), then builds from assumptions to identification results (Theorem 1 for C, Theorem 2 for P(Y|do(X))), providing a coherent narrative from causality theory to an implementable algorithm.

- **Consistent empirical improvements under controlled shifts.** Tables 1–2 show that CTL consistently outperforms SFT, SWA, and WISE across all OOD spurious correlation levels, with the gap widening as distribution shift intensifies (e.g., Amazon OOD 10%: 58.40 vs. 49.24 for SFT). The ablation variants (CTL-N, CTL-C, CTL-Φ) provide useful dissection of component contributions.

- **Transparent and honest about limitations.** The paper states upfront (Section 7) that PLMs are "already highly resilient to perturbations" and that "the mechanisms through which spurious correlations emerge in complex, real-world environments remain unclear."

## Weaknesses

### Major

- **Critical gap between Assumption 2 and the actual implementation.** Assumption 2 requires "for each input text X, we can obtain a pair of variations of its representations, R₀ and R₁," implying the *same* input processed through two models. However, Algorithm 1 (steps 6–8) uses *different* inputs x̃ᵢ and x̄ᵢ that merely share a label yᵢ, computing r̄₀ = M₀(x̃ᵢ) and r̄₁ = M₁(x̄ᵢ). This is a fundamentally different operation—same-label different texts do not satisfy the "same data point" condition of Von Kügelgen et al. (2021, Theorem 4.4) that the authors invoke. If examples with the same label share the *same* C (as the implementation implicitly assumes), this would mean all texts with the same sentiment have identical causal features, which is clearly false for realistic sentiment data. This disconnect between the stated assumption and the code undermines Theorem 1's applicability.

- **Front-door adjustment validity is questionable and the derivation has gaps.** The proof of Theorem 2 invokes the "Frontdoor Criterion & Assumption 3 and 4" to go from P(y|do(c)) to Σ P(y|Φ',c) P(Φ'). In the standard front-door criterion, the mediator must satisfy specific conditions: (1) all directed paths from treatment to outcome go through the mediator, (2) no unblocked back-door from treatment to mediator, and (3) no unblocked back-door from mediator to outcome after conditioning on treatment. Assumption 4 states C *fully mediates* Φ→Y, which is the *inverse* of what front-door requires (that Φ mediates C→Y). The claim that Assumptions 3–4 together satisfy front-door conditions is not established. Moreover, the final step of the proof—marginalizing to P(Φ'|x')P(x')—assumes P(Φ') factors through P(Φ'|x')P(x') under the observational distribution, but the relationship between this and the interventional distribution P(Y|do(X)) is not rigorously justified. The shuffling of Φ within minibatches (Algorithm 1, step 11; Algorithm 2, step 4) is a heuristic resampling that resembles ensemble averaging rather than a valid approximation of the front-door integral.

- **CTL-C performs nearly as well as full CTL, undermining the necessity of the front-door adjustment.** In Table 1 (Amazon), CTL achieves [93.03, 84.16, 75.83, 67.06, 58.40] while CTL-C achieves [92.99, 84.07, 75.51, 66.62, 57.75]—a difference of only 0.04–0.65 F1 points. In the Yelp table, the gap is similarly small (0.04–1.0). If the front-door adjustment via Φ is the paper's core methodological contribution, this near-equivalence raises the question of whether the added complexity is justified. The paper does not address this.

- **Experiments only evaluate artificially injected spurious correlations, not natural distribution shifts.** Both the semi-synthetic experiments (stopwords "and"/"the" correlated with labels) and the "real-world" experiment (strings "amazon.xxx"/"yelp.yyy" appended to text) inject spurious cues that are local, easily detectable, and directly aligned with the token-patching mechanism used to construct Φ (10 patches of token embeddings). This means the evaluation is essentially testing whether a method designed to adjust for token-level spurious features can handle token-level spurious features. No experiments on naturally occurring distribution shifts (e.g., cross-domain sentiment, CivilComments, MNLI→HANS) are presented. The paper's title and abstract claim "robust causal representation learning" and "practical real-world scenarios," but the evaluation does not substantiate this generality claim.

- **Missing comparisons with domain generalization baselines.** The baselines (SFT0, SFT, SWA, WISE) are general fine-tuning strategies, not methods designed for OOD robustness. Standard single-domain DG baselines such as IRM, GroupDRO, CORAL, or environment-inference methods are absent. Without these comparisons, it is unclear whether the improvements come from the specific causal construction or from any form of regularization that reduces reliance on spurious features.

### Minor

- **Assumption 2 is strong and empirically unverified.** The assumption that fine-tuning only changes spurious features S while preserving causal features C has no empirical support—no probing experiments test whether R₀ and R₁ actually share invariant content. Given that fine-tuning PLMs demonstrably alters task-relevant representations, this assumption requires explicit validation or at least discussion of failure modes.

- **The sampling step in Eq. (2) is under-explained.** Training on (x̃, y) where x̃ is a random same-label example rather than the original x is not standard ERM. No ablation tests whether this step affects results, and its interaction with the causal story is unclear.

- **No analysis of inference cost.** Algorithm 2 requires K samples of shuffled Φ per prediction, but no analysis of how performance scales with K or the computational overhead compared to standard inference.

### Trivial

- The patching scheme (10 non-overlapping patches, mean-averaged) is somewhat arbitrary, and no sensitivity analysis is provided. However, this is a standard design choice and not a core conceptual issue.

## Nice-to-Haves

- Evaluate on at least one benchmark with naturally occurring distribution shifts (e.g., CivilComments-Wilds, MNLI→HNLI) to substantiate the "real-world" claim.
- Add probing experiments to verify that learned C is invariant to spurious attributes while Φ captures them.
- Compare with IRM-style or GroupDRO baselines in the single-domain setting.
- Explicitly analyze and explain why CTL-C ≈ CTL, and whether the front-door adjustment is truly necessary.

## Removed Points

- **"The method is at best a regularized ensemble-like predictor, not an identified causal effect."** While this is the harsh critic's conclusion, it overstates the case—the theoretical framework does connect to causal identification, even if the implementation deviates from the ideal conditions. The core concern is the theory-practice gap, which is already captured above.
- **"Confidence intervals and statistical tests are not reported."** Single-run evaluation is the norm in this field for large models; reporting 5-seed means and boxplots is adequate.
- **"The paper claims applicability to NLI and QA but provides no evidence."** The paper explicitly scopes to text classification and mentions extension as future work; demanding additional tasks is scope creep.
- **"No comparison with other front-door adjustment methods (Li et al., 2021; Mao et al., 2022; Nguyen et al., 2023)."** These cited methods require multi-domain data, which the paper explicitly avoids—comparing under asymmetric conditions (single-domain method vs. multi-domain method) would be an unfair comparison in the other direction. The lack of single-domain OOD baselines is already noted.
- **"Entropy terms in Eq. 3 could encourage high-variance meaningless representations."** This is a generic concern about contrastive/invariant learning that is not demonstrated to be an actual failure mode here.

## Novel Insights

The paper's most novel insight is the framing of pre-trained and fine-tuned PLM representations as "paired environments" for causal identification—a conceptually attractive way to extract invariance from single-domain data. However, this insight remains more promising than proven: the implementation uses same-label different inputs rather than the same input (breaking the theoretical conditions), and the front-door adjustment adds minimal gain over using C alone. The real contribution may be the invariant feature learning via aligned representations (CTL-C), rather than the front-door mechanism the paper emphasizes.

## Suggestions

- **Close the theory-practice gap for Assumption 2:** Either use the same input X through both models (as the theory requires) and justify why same-input representations preserve C, or provide theoretical analysis for the same-label approximation actually used.
- **Explain the CTL-C vs. CTL parity:** If the front-door adjustment contributes <1 F1, either the method's primary value is in learning C alone (which should be the focus), or the front-door implementation needs reconsideration.
- **Test on natural distribution shifts:** Even one additional experiment on a real cross-domain benchmark would significantly strengthen the practical relevance claim.

## Score and Decision

**Calibration:** I compared against papers with similar strength/weakness patterns:
- OatZMyMuIo (causal representation learning, strong theory but weak assumptions): scores 3, 3, 5, 5 (Reject)
- wFf9m4v7oC (front-door adjustment, more validated): scores 5, 6, 6, 6 (Accept poster)
- fHZ04oyEed (representation learning from interventional data, limited experiments): scores 3, 3, 3, 5 (Reject)
- 22ywev7zMt (OOD generalization of SSL): scores 5, 6, 6 (Reject)

This paper has a creative conceptual framing (using PLMs as paired environments for causal identification) but suffers from: (1) a significant theory-practice disconnect in Assumption 2, (2) a front-door derivation with unclear validity, (3) experiments that only validate on artificially injected, method-aligned spurious correlations, and (4) the core claimed mechanism (front-door adjustment) adding negligible improvement over the simpler component (C alone). These issues are similar in severity to papers scored 3–5 in the calibration set. The paper is above the lowest-scoring papers (which have fatal theoretical errors with no empirical support) because it does show consistent improvements and has an interesting conceptual framework, but below the accept threshold because the causal claims are not well-supported by either theory or experiments.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>