Now let me check the calibration papers to help score this work.Now I have enough information to write the final review. Let me synthesize it carefully.---

## Summary
This paper addresses a gap in the high-probability convergence theory of Adam and AdaGrad under heavy-tailed stochastic gradient noise. The authors prove a negative result (Theorem 1): these methods can have high-probability complexity with inverse-power dependence on the confidence level δ, rather than the desirable polylogarithmic dependence. They then show that gradient clipping repairs this (Theorems 2–4), yielding first-of-kind high-probability convergence bounds with polylogarithmic δ-dependence for Clip-Adam(D)/Clip-M-AdaGrad(D) in convex and non-convex settings with bounded α-th moment noise. Experiments on a synthetic quadratic and ALBERT fine-tuning are provided, with the theoretical predictions qualitatively reflected in the results.

---

## Strengths

- **Novel and decisive negative result (Theorem 1).** The construction—using a Huber loss with a carefully designed discrete noise sequence—directly attacks the polylogarithmic-δ property rather than a proxy. The fact that the pathology appears even for α=2 (bounded variance) is surprising given the folk intuition that Adam acts like implicit clipping. This is not achievable with routine proof-from-prior-work.

- **First high-probability guarantees with polylogarithmic δ-dependence for Adam/AdaGrad under heavy-tailed noise without sub-Gaussian or bounded-noise assumptions (Theorems 2–4).** Prior work either required sub-Gaussian/bounded noise (Li & Orabona, 2020; Li et al., 2023) or had inverse-power δ-dependence (Wang et al., 2023). The paper explicitly documents this gap in Table/Section 1.3. The proof that the iterates remain in Q with high probability is an additional technical contribution.

- **Delayed-method results avoid the restrictive Assumption 4.** Theorems 2 and 3 cover Clip-AdamD/Clip-M-AdaGradD without requiring bounded function values, while Theorem 4 (non-delayed) requires Assumption 4. The paper correctly notes this separation and contextualizes it against Li & Liu (2023), whose stronger bounded-empirical-risk assumption in the worst case implies sub-Gaussian noise, effectively trivializing the problem.

- **Insightful identification of two distinct roles of clipping.** The paper carefully distinguishes clipping in m_t (controlling step size in the presence of heavy-tailed noise, as in Clip-SGD) from clipping in b_t (preventing the adaptive scale from growing catastrophically). The connection of the latter role to the mechanism in Theorem 1's failure example is clear and adds methodological insight beyond simply "add clipping."

- **Empirical results qualitatively aligned with theory.** The differential behavior on CoLa (heavy-tailed noise, large benefit from clipping) vs. RTE (lighter tails, similar performance) is exactly what the theory predicts and constitutes meaningful, non-trivial validation.

---

## Weaknesses

### Fatal
*None identified.*

### Major

- **Experiment-theory gap on clipping type.** The theoretical guarantees cover global norm clipping (clip(x, λ) = min{1, λ/‖x‖}x), yet the ALBERT experiments use coordinate-wise and layer-wise clipping—explicitly acknowledged in footnote 6: *"We did not consider the global/norm clipping (the considered in theory), since typically coordinate-wise or layer-wise clipping work better in training neural networks."* This is a significant disconnect: the theoretical guarantees may not apply to the experimentally used methods, and the paper never discusses whether the proofs extend or shows even one experiment with norm clipping to allow a direct check. As a result, the positive theory and positive experiments are loosely coupled.

- **Negative result proved only for α=2 (bounded variance), leaving the core heavy-tail regime (α<2) as a conjecture.** Section 2 states: *"we also conjecture that for α<2 one can show even worse dependence on ε and δ for Adam/AdaGrad... since b_t will grow with high probability even faster in this case."* The paper does prove non-convergence for AdamD/M-AdaGradD when α<2 citing a reference analogy, but the formal lower bound of Theorem 1 covers only α=2. Since the primary motivation of the paper is heavy-tailed noise (α<2), the case most important to the narrative is left as a conjecture.

### Minor

- **Asymmetric hyperparameter tuning in ALBERT experiments.** Batchsize and learning rate are tuned for plain Adam, and these are reused for clipped variants with only the clipping threshold tuned. The paper states: *"For the methods with clipping, we used the same batchsize and stepsize as for Adam and tuned the clipping level."* It is not clear whether this biases toward or against clipping methods (different lrs may be optimal for clipped variants), but it introduces confounding and makes the claim of practical "superiority" harder to sustain rigorously.

- **Restrictive β₂ = K/(K+1) requirement.** The convergence results require β₂ = K/(K+1) (equivalently, β₂ → 1 as K → ∞), which the paper acknowledges makes Adam a "twin" of AdaGrad. As stated in Section 1.3: *"the standard choice of β₂ in theory is β₂ = 1 − 1/K... that is why, as noticed by Défossez et al. (2022), AdaGrad and Adam are 'twins'."* The negative result in Theorem 1 also covers only this parameterization. The practical default β₂ = 0.999 is not covered theoretically, and the paper does not discuss whether the positive or negative results would extend to fixed β₂ < 1.

- **Assumption 4 (bounded function values) for non-delayed methods is restrictive.** Theorem 4 requires f(x) − f* ≤ M for all x ∈ ℝ^d. The paper provides appropriate context (this is weaker than Li & Liu's assumption), but the practical applicability of Theorem 4 for neural network training—where function values are not globally bounded—is left unaddressed. The delayed variants in Theorems 2-3 are the more practically relevant results.

- **The "match Clip-SGD in the convex case" claim is overstated.** The paper itself notes (page 7/8): *"the first term in (10) is not optimal... The optimality of the second term in (10) is still an open question."* This is an appropriate acknowledgment in the main text, but the abstract and Contributions section (Section 1.1) state "match the complexity of Clip-SGD in the convex case up to logarithmic factors" without qualification, which is too broad given that one term is known to be suboptimal.

### Trivial

- **Non-convex rate expressions are complex.** The exponents (3α−2)/(2α−1) and (3α−2)/(2α−2) in Theorems 3 and 4 are difficult to parse without concrete examples. A brief table showing values at α ∈ {1.2, 1.5, 2.0} alongside Clip-SGD rates would help readers assess significance.

---

## Nice-to-Haves

- **Extend the negative result to α < 2.** Even a partial result (e.g., non-convergence or a lower bound on the complexity growth) for α < 2 would complete the theoretical story and strengthen the paper's motivational narrative significantly.

- **Ablation isolating clipping in b_t vs. m_t only.** The paper makes a specific design argument about why clipping b_t matters (Section 3), but this is not isolated experimentally or theoretically. A theorem or even a synthetic experiment showing that clipping only m_t is insufficient for high-probability guarantees would substantiate this insight.

- **Add Clip-SGD as an experimental baseline.** The paper positions Clip-Adam as analogous to Clip-SGD, but never benchmarks against it. This would clarify whether the benefit comes from clipping per se or from the interaction between clipping and adaptive step sizes.

- **Practical guidance on setting λ.** The formulas for γ and λ in Theorems 2–4 depend on σ, R, Δ, L, which are usually unknown. A brief discussion of adaptive or heuristic rules, or a sensitivity analysis in the experiments, would help practitioners.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **[Harsh Critic] "Match Clip-SGD broadly overstated"** — partially valid and preserved in Weaknesses (minor), but the paper does explicitly acknowledge the known suboptimality of the first convex term; the issue is one of abstract/intro framing, not a technical error.

- **[Spark] "No comparison with coordinate-wise AdaGrad/Adam (standard practical versions)"** — The paper's theory covers scalar-norm versions (AdaGrad-Norm, not AdaGrad-CW), which is explicitly stated and is standard in the convergence theory literature. Criticizing the absence of coordinate-wise theoretical results is scope creep.

- **[Spark/Neutral] "No pre-training or larger-scale LLM experiments"** — The paper frames its practical relevance around the observation that LLM training exhibits heavy-tailed noise. Fine-tuning is the experimental domain; demanding full LLM pre-training is beyond the scope of a theory paper. This would be desirable but is not a methodological flaw.

- **[Harsh Critic] Practical generalization from analyzed Adam variants to default Adam is misleading** — The paper clearly scopes its claims to the analyzed scalar-norm variants throughout the main text (Section 1.3 explains the "twin" relationship explicitly). The abstract phrasing could be tightened, but this is a writing suggestion already captured in Minor weaknesses.

- **[Generic Strengths removed]** The following were identified as too generic: "well-motivated and timely research question" (applies to any paper on LLM training), "the paper is well-written," "well-organized." These have been replaced with specific evidenced strengths above.

---

## Novel Insights

The paper's most underappreciated contribution may be the identification that the failure mode of Adam/AdaGrad under heavy-tailed noise is not the step size magnitude per se, but the growth of the adaptive scale accumulator b_t, which—if left unclipped—can absorb a single early catastrophic gradient and permanently degrade all subsequent steps. This is structurally different from the failure mode of SGD under heavy-tailed noise (a single bad step of large magnitude), and explains why the "implicit clipping" intuition for Adam is misleading: the division by b_t controls the step norm, but a large b_t also ensures the division is always small, trapping the iterates near their post-shock location. Clipping b_t is thus not merely "following Clip-SGD's logic" but addressing a genuinely distinct pathology that appears only in adaptive methods.

---

## Suggestions

1. Prove the negative result (or at minimum a non-convergence result) for α < 2 to close the gap between the motivational narrative and the formal result.
2. Include at least one synthetic experiment comparing norm clipping (as analyzed) and coordinate-wise clipping to bridge the theory-practice gap, and add a brief discussion of whether the proof technique could extend to coordinate-wise variants.
3. Narrow the abstract's "match Clip-SGD in the convex case" claim to the second term in (10), which is the only term where the claim holds under current theory.
4. Add a sensitivity plot for the clipping level λ in the ALBERT experiments to help calibrate the robustness of the improvement.

---

## Score and Decision

**Calibration anchors:**
- *High-Probability Convergence for Composite/Distributed* (qOFLn0pMoe, scores 5/5/5, rejected): Comparable theoretical template (clipping for high-prob under heavy-tailed noise), but no negative result, no experiments, and presentation issues. The paper under review is meaningfully stronger: cleaner story, both negative + positive results, empirical validation.
- *To Clip or not to Clip* (jmN1zXMq0O, scores 6/6/6/8, accepted): High-dim analysis of clipping dynamics in linear regression. Comparable impact level; that paper had more precise mechanics in its specific setting but narrower scope. The paper under review is at roughly comparable quality.
- *Provable Benefit of Adaptivity in Adam* (qx02elXpf9, scores 3/8/5/3, rejected): Similar territory (proving something novel about Adam theoretically), but its negative result was contested on assumption validity. The paper under review is on stronger ground.
- *Convergence of Adaptive Gradient Methods under Refined Assumptions* (UmMKbG2Ubr, scores 5/5/6/6, rejected): Comparable contribution level but narrower novelty gap over prior work than the paper under review.

**Assessment:** The paper's theoretical contributions—a genuine negative result and first polylogarithmic-δ high-probability bounds for Adam/AdaGrad under heavy-tailed noise without sub-Gaussian assumptions—are solid and fill a real gap. The weaknesses (experiment-theory clipping gap, negative result limited to α=2, β₂ restriction) are real but non-fatal. This paper is clearly above the 5/5/5 rejected threshold and in the range of the accepted clipping paper at 6–7. The major clipping-type disconnect in experiments and the α=2-only negative result keep it from the higher end. I place it at **6.5**.

**Novelty:** High — first results of this kind for Adam/AdaGrad under heavy-tailed noise
**Technical soundness:** High — theorems are well-structured, assumptions clearly stated, limitations acknowledged
**Empirical support:** Moderate — illustrative but the experiment-theory gap is a genuine concern
**Significance:** High for the optimization theory community
**Clarity:** Good — the narrative is coherent and the assumptions are carefully contextualized

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>