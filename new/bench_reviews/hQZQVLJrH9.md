Now I have enough calibration data. Let me synthesize my meta-review.

## Calibration summary:
- **Activation Transport (AcT)** - accepted spotlight (scores 8,8,6,8): Novel OT-based framework for steering, strong experiments across LLMs and diffusion models, modality-agnostic, well-written.
- **Conceptor steering** - withdrawn/rejected (6,5,6,3): Theory-heavy steering approach, limited experiments, outdated models, no error bars, concerns about practical significance.
- **Confident Directions** - rejected (5,5,6,6): Limited baselines, unclear theoretical connection, weak empirical scope.
- **Do Influence Functions Work on LLMs?** - withdrawn/rejected (5,3,3,5): Negative empirical result paper about influence function fragility.
- **Versatile Influence Function (VIF)** - rejected (5,6,6,5): Theory paper extending IF to non-decomposable losses, but concerns about strong assumptions, limited baselines, small-scale experiments.
- **Loss Landscape via Convex Duality** - accepted oral (8,8,8,8,8): Strong theoretical framework with complete proof structure, rigorous results.
- **Dual RL** - accepted spotlight (10,6,8,6): Unification framework, strong theoretical contribution plus practical algorithm.
- **Weak Correlations / Linearization** - rejected (1,3,3): Ambitious theory, unclear presentation, no experiments.

Now let me carefully evaluate each criticism of this paper against the actual content:

**Key criticisms to verify:**

1. **Overstated equivalence claim** - The paper claims "equivalence" in the abstract but all theorems are first-order/local. Need to check: does the paper acknowledge this? Yes, the abstract says "to first order," and Section 1 mentions "small-edit regime." The theorems are explicitly first-order. The no-free-lunch theorem (6.2) also qualifies the scope. But the language in places like "any steering vector can be represented as an influence weighting" and the conclusion's "integrated workflow" language does overstate what's formally proven. This is a valid but somewhat moderated concern.

2. **Influence function fragility** - Valid concern. The paper uses classical IF (Koh & Liang 2017) and acknowledges damping but doesn't engage with Basu et al. 2021's fragility findings. However, the paper isn't trying to claim IF works well; it's saying *if* you accept IF's first-order approximation, then it's dual to steering. The duality claim is mathematical, not empirical. Still, the paper's practical workflow claims depend on IF being useful.

3. **No data attribution experiment** - This is the most damaging criticism. The paper's abstract promises "a constructive algorithm for mapping undesired behaviors back to causal training examples" but never demonstrates this. There is no experiment computing ρ_s, identifying top training examples, and verifying causality. This is a significant empirical gap.

4. **Slope 1.50 in Figure 1** - The paper shows cosine similarity of 0.978 but slope of 1.50, meaning first-order predictions systematically overestimate by 50%. This is a real discrepancy that undermines the practical validity of the first-order approximation for the magnitudes used.

5. **GPT-2 Medium only for LM experiments** - Valid. The paper claims "billion-parameter models" relevance but only tests on 355M params.

6. **Limited baselines** - Only compares with CAA. Valid concern but somewhat mitigated since the paper's main claim is theoretical (the duality itself), not that IAS is better at detoxification.

7. **Assumptions not verified empirically** - The feasibility, Lipschitz, and affine independence assumptions are not validated. Valid but somewhat standard for theory papers.

8. **No error bars** - Valid but somewhat standard for single-run evaluations in this area.

Let me now write the final review.</think>Now I have sufficient calibration data. Let me write the final consolidated review.

## Summary

This paper establishes a first-order duality between activation steering (adding a vector to intermediate activations) and influence functions (tracing model outputs to training examples via second-order perturbation analysis). The authors prove that any steering perturbation can be represented as a signed influence weighting over training data and vice versa, to first order. They introduce Influence-Aligned Steering (IAS) as a constructive bridge, provide alignment diagnostics (ω) characterizing when steering can or cannot match influence, derive spectral optimality results for choosing steering directions, and prove generalization bounds for low-rank interventions. Experiments on GPT-2 Medium and ResNet-50 test the first-order fidelity and spectral optimality claims.

## Strengths

- **Novel conceptual unification.** Connecting activation steering and influence functions—two previously disjoint interpretability toolkits—under a single mathematical framework is a genuine contribution. The primal–dual formulation (Section 3) cleanly explains why and when one can substitute for the other, and the no-free-lunch impossibility result (Theorem 6.2) honestly demarcates when steering *cannot* suffice.

- **Constructive and complete theoretical framework.** The paper provides existence (Theorem 5.2), error bounds dependent on alignment (Theorem 5.1 with the √(1−γ²) bound), impossibility (Theorem 6.2), spectral optimality (Theorem 5.3), layer composability (Lemma 5.4), and generalization guarantees (Theorem 6.1). This gives a fairly thorough theoretical picture.

- **Principled diagnostic (ω).** The alignment cosine ω(x) between Im(J_{θ→y}) and Im(J_{h→y}) provides an actionable, cheaply computable check for whether steering is feasible before committing to it. The observation that ω increases with layer depth (Figure 2) is intuitive and useful for layer selection guidance.

- **Computationally tractable constructions.** The IAS formula Δh* = J†_{h→y} J_{θ→y} Δθ requires only two JVP/VJP passes and a rank-d pseudoinverse, making it implementable without full Hessian inversion for the steering computation itself.

## Weaknesses

### Major:

- **The central "equivalence" claim is significantly overclaimed relative to the formal results.** The abstract states that activation steering and influence functions are "equivalent" and that "any steering vector can be represented as an influence weighting over training data and vice versa." But Theorem 4.2 holds only to O(α²), Theorem 5.1 bounds the relative approximation error by √(1−ω²), and Theorem 6.2 proves that low ω makes exact matching *impossible*. The feasibility assumption Im(J_{θ→y}) ⊆ Im(J_{h→y}) is required for uniqueness. These are not global equivalence results—they are local, conditional, first-order approximations with explicit impossibility regimes. The paper does include qualifying language ("to first order," "small-edit regime") in several places, but the headline framing and the conclusion's practical workflow claims ("steer first, trace provenance, edit weights only when the geometry demands it") read as unqualified operational guidance. While the mathematical content is correct, the narrative wraps local results as a general duality.

- **No empirical validation of the data-attribution direction—a core advertised contribution.** The abstract promises "a constructive algorithm for mapping undesired behaviors back to causal training examples," and Section 4.1/Corollary 1 claims ρ_s "pinpoints the fewest training examples to relabel/remove/examine." Yet no experiment in Section 7 computes ρ_s for any behavior, identifies top-weighted training examples, or verifies that removing/relabeling those examples produces the predicted change. This is the paper's most distinctive practical promise, and it is entirely untested. Without this, the "constructive algorithm for mapping... back to causal training examples" remains a theoretical construct with no demonstrated empirical validity.

- **The first-order approximation shows a notable systematic deviation.** Figure 1 reports a slope of 1.50 (not 1.0) between predicted and actual logit shifts, meaning the first-order prediction systematically overestimates the true shift by 50%. Combined with a cosine of 0.978, this establishes directionality but not magnitude fidelity. The paper does not discuss this discrepancy or characterize the range of steering magnitudes α over which the first-order approximation remains reliable (e.g., within 10% error). This matters because practical detoxification (Table 1) uses non-infinitesimal steering magnitudes where the O(α²) errors may be substantial.

- **Influence function fragility is assumed away.** The framework builds on classical Koh & Liang–style influence functions, but well-known work (including the cited Basu et al., 2021) documents that these are fragile in deep networks—sensitive to Hessian conditioning, damping, and non-convexity. The paper treats damping as a Tikhonov regularizer but does not analyze how approximate H⁻¹ affects ρ_s, the spectral recipe (Theorem 5.3), or the claimed equivalence. Since the "causal data attribution" interpretation of ρ_s depends entirely on influence function correctness, this gap undermines the practical significance of the steer→data direction.

### Minor:

- **Experiments are limited in scale and scope.** Only GPT-2 Medium (∼355M) is tested for the LM detoxification task, with a single ResNet-50 experiment for spectral optimality. The paper claims the framework scales to "billion-parameter models" without evidence beyond 355M parameters. This is within scope for a primarily theoretical paper, but the gap between claim and evidence is notable.

- **The Corollary 1 ℓ₁-minimality claim relies on an unverified assumption.** Corollary 1 requires affine independence of {I(z→x)}_{z∈Z}, which is strong and likely violated in overparameterized settings. The paper notes this in passing ("otherwise the ℓ₁-minimal solution need not be unique") but does not discuss practical implications or provide a workaround.

- **The Rademacher-complexity generalization bound (Theorem 6.1) is disconnected from the experiments.** It is a direct application of Pinto et al. (2024) plus the observation that IAS is a rank-k correction. The "practical guidance" bullets following it (prefer low rank, skip steering when ω < 0.5) are plausible but entirely heuristic, with no empirical study confirming that they lead to better performance.

### Trivial:

- No error bars or variance estimates are reported for the numbers in Table 1 or for Figure 1.

## Nice-to-Haves

- Demonstrating the full data-attribution pipeline (computing ρ_s, identifying top training examples, and verifying their causal role by removal/relabeling) would dramatically strengthen the paper's most distinctive claim.
- Testing on a model in the 7B+ parameter range to substantiate scaling claims.
- Comparing IAS against methods from both literatures it unifies—e.g., representation engineering (Zou et al., 2023) or TracIn on the influence side—rather than only CAA.
- An ablation over steering magnitude α measuring where the first-order approximation breaks down, which would clarify the practical operating regime.

## Removed Points

These points are flagged to be removed, treated with caution:

- **"The theoretical results are largely standard linear algebra."** While some results (pseudoinverse solution, subspace angle bounds) employ standard tools, the *framing and synthesis*—connecting the optimization dual, the data-level duality, and the spectral recipe—is the paper's core novelty. Applying standard tools to a novel problem is a legitimate contribution; calling it "standard" undercounts the insight.

- **"No comparison against ground-truth influence functions."** The paper's claim is a *mathematical* equivalence between steering and influence at first order, not that its ρ_s outperforms standalone IF computations. Verifying that IAS reconstructions correlate with independently computed IF scores would be valuable, but this is a nice-to-have rather than a core flaw.

- **"Fisher–influence matrix Σ is expensive to compute at scale."** The paper acknowledges this (Section 9: "computing exact pseudoinverses is challenging for deep stacks; exploring Krylov or randomized SVD methods is an open engineering problem"). The paper also provides a power-iteration recipe for Σ's top eigenvector. Scalability concerns are real but partially addressed.

- **"The paper is dense and hard to follow."** Formatting and presentation concerns are excluded per the rules.

## Novel Insights

The most insightful observation that emerges from the reviews is that the paper contains two distinct contributions of unequal maturity: (1) a clean, formally rigorous geometric framework linking parameter-space and activation-space perturbations (Theorems 5.1–5.3, 6.2), and (2) a promised but untested practical pipeline for mapping steering behaviors back to causal training data via ρ_s. Contribution (1) stands on its own even without (2), but the paper is marketed primarily around (2), creating a mismatch between the promise and the delivery. The ω diagnostic and no-free-lunch result are the strongest elements: they provide honest, quantitative limits on when steering can work, which is genuinely useful to practitioners regardless of the data-attribution claims.

## Suggestions

1. **Downscale the data-attribution claims** or add a concrete experiment: compute ρ_s for a toxic steering vector on GPT-2, retrieve the top-10 training examples, and show that removing them reduces toxicity. Even a small-scale demonstration would validate the paper's most distinctive claim.
2. **Analyze the 1.50 slope**: report how the slope between predicted and actual logit shifts varies with α, characterize the regime where the first-order approximation is accurate, and discuss the second-order correction explicitly.
3. **Tighten the abstract/introduction framing**: replace "equivalent" and "any steering vector can be represented as" with qualified language like "equivalent to first order" and "any infinitesimal steering perturbation admits a signed influence measure reproducing the same first-order logit shift." The no-free-lunch result already provides the natural caveats—use them prominently.

## Evaluation on Key Axes

- **Originality**: High. The duality between steering and influence is genuinely new and non-obvious. While the mathematical tools are standard, the synthesis connecting two previously disjoint areas with constructive results is novel.
- **Importance of research question**: High. Unifying steering and influence addresses a real practical dichotomy faced by practitioners.
- **Claims well supported**: Partially. The theoretical claims are proven correctly (within their stated assumptions), but the practical equivalence and data-attribution claims exceed what is formally shown or empirically validated.
- **Soundness of experiments**: Moderate. Figure 1 supports first-order directionality, Figure 2 supports ω-increases-with-depth, and the spectral experiment on ImageNet is a helpful sanity check. But the core workflow (steer→data) is untested, and the only LM experiment is a single model on a single task.
- **Clarity**: The paper is dense but technically coherent. The section ordering (primal/dual → data-level → guarantees → experiments) could be improved, but the material is present.
- **Value to community**: Moderate-to-high. The ω diagnostic and impossibility result are actionable; the IAS construction is implementable. But the untested data-attribution pipeline limits immediate practical value.

## Calibration

Papers compared:
- **Activation Transport (AcT)** (scores 8,8,6,8, accepted spotlight): Novel framework with strong experiments across modalities. This paper has similar ambition but weaker experiments and no test of the core pipeline.
- **Conceptor steering** (scores 6,5,6,3, withdrawn/rejected): Theory-heavy steering paper with limited experiments and outdated models. Similar concerns about practical significance and limited baselines, but our paper's theoretical contribution is more substantial (duality, impossibility).
- **Dual RL** (scores 10,6,8,6, accepted spotlight): Unification framework with theoretical and practical contributions. Our paper has a similar "unification" angle but without the empirical demonstration of the practical workflow.
- **VIF (Versatile Influence Function)** (scores 5,6,6,5, rejected): Extension of IF theory with strong assumptions and limited empirical validation. Similar pattern of theoretical contribution with empirical gaps.

This paper sits between the rejected theory-papers-with-weak-experiments (VIF, conceptor) and the accepted unification papers (Dual RL, AcT). Its theoretical contribution—particularly the alignment diagnostic and impossibility result—is genuine and novel, but the mismatch between the advertised data-attribution workflow and the actual experiments is a significant weakness, as is the 1.50 slope indicating the first-order regime is already stretched at the magnitudes used.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>