## Summary

The paper develops a constraint-based causal discovery framework for stochastic processes modeled by systems of SDEs. It establishes a global Markov property for coordinate processes on path segments with respect to a "lifted" dependence graph, proves soundness and completeness of causal discovery algorithms (for both fully and partially observed data) that uniquely recover the full DAG by exploiting time directionality, and proposes a consistent signature kernel-based conditional independence (CI) test on path space that makes the framework practically applicable. Empirical results demonstrate strong performance against existing methods across various SDE settings.

## Strengths

1. **Novel theoretical framework.** The lifted graph construction and the global Markov property (Proposition 3.1) connecting SDE structure to CI constraints on path segments is a non-trivial and elegant idea. Proving that the resulting CI relation ⊥⁺_{s,h} satisfies d-separation in the lifted graph, and then using this to achieve sound and complete causal discovery (Theorem 3.2), is a genuine theoretical contribution that goes beyond what standard constraint-based methods achieve (which only recover Markov equivalence classes).

2. **Consistent CI test on path space without density assumptions.** Adapting kernel-based CI testing (KCIPT/SDCIT) to path-valued random variables via signature kernels, and proving consistency without relying on densities (Appendix A.14), fills a genuine gap. This contribution has independent interest beyond causal discovery.

3. **Comprehensive handling of continuous-time challenges.** The framework simultaneously addresses irregularly sampled observations, path-dependent drifts/diffusions, partial observations with latent confounders, and diffusion-dependent causal effects—settings where existing methods (PCMCI, Granger, etc.) fundamentally fail or are inapplicable.

4. **Strong empirical performance in most settings.** Tables 1 and 2 show clear improvements over PCMCI, Granger, and CCM, and generally competitive or superior performance against SCOTCH across drift, path-dependent, and nonlinear settings. The hyperparameter sensitivity of SCOTCH is a legitimate practical concern that the paper correctly identifies.

## Weaknesses

### Fatal
None.

### Major

1. **Overclaimed identifiability in the partially observed setting.** The abstract states the algorithm is capable of "uniquely recovering the underlying or induced ancestral graph" for both fully and partially observed data. For the partially observed case, the claim that time directionality reduces the FCI output (a PAG representing an equivalence class) to a single uniquely identifiable ancestral graph is too strong. In general, even with temporal constraints, there exist non-Markov-equivalent ADMGs consistent with the same observable CI patterns (e.g., certain latent common cause structures). The paper acknowledges in Section 3.1 that the method "reduces" the PAG to a single graph, but the word "uniquely" in the abstract and conclusion implies uniqueness that is not rigorously established—especially given the claim of handling "arbitrary unobserved processes." The partially observed results are deferred to Appendix D with only a single illustrative example (Figure 4) and no systematic evaluation. This matters because identifying latent confounding structure is a primary motivation for constraint-based methods, and practitioners could be misled about what is recoverable.

2. **Weak performance on diffusion dependence—the setting most central to SDE modeling.** Table 1 shows SigKer achieves SHD of 72±6 (n=200) and 63±5 (n=400) on diffusion dependence, while SCOTCH achieves 25±13 and 9±8 with specific hyperparameters. The paper argues this is mitigated because SCOTCH's performance is hyperparameter-sensitive and tuning is impossible without ground truth. This is a fair practical argument, but it does not change the fact that one of the four key SDE settings—the one that uniquely motivates the SDE model over simpler alternatives—shows a genuine performance gap. The diffusion dependence case should be acknowledged more explicitly as a current limitation rather than explained away entirely.

3. **Limited evaluation of scalability and partial observation.** Table 2 shows SHD growing rapidly with dimension (e.g., 4946 ± 133 at d=50), and the only partial-observation experiment is a single 4-node example (Figure 4) without systematic evaluation over varying graph structures, latent variable counts, or dimensionality. For a paper claiming contributions in the partially observed setting, this insufficient empirical grounding weakens confidence in the practical viability of this aspect of the work.

### Minor

1. **Sensitivity to split-point parameters s and h.** The h-local CI test ⊥⁺_{s,h} requires choosing a time-split point s and interval length h. The paper uses s = 0.1·T and reports this "performed best" (Table 5 in appendix), but no theoretical guidance or systematic sensitivity analysis is provided. The Markov property and algorithm correctness hold for any s, h, but power will vary substantially, and practitioners lack principled criteria for selection.

2. **Self-described as "hyperparameter-free" is not fully accurate.** While the method avoids SCOTCH-style sparsity parameters, the signature kernel CI test involves choices: RBF kernel bandwidth (via median heuristic), permutation count, and the KCIPT vs. SDCIT selection. The paper selects these via preliminary experiments (Appendix B.2) but does not report sensitivity. The median heuristic in particular can significantly affect kernel test power on path-valued data.

3. **Incomplete articulation of faithfulness assumptions in the main text.** The algorithm relies on "parent faithfulness" (discussed in Appendix A.8), a variant of the standard faithfulness assumption. The relationship between this and standard faithfulness, and its practical implications (e.g., potential violations in SDE models where path cancellation effects can induce near-unfaithfulness), are deferred to the appendix and should be discussed in the main text given their importance for practical reliability.

4. **Type I error calibration not reported.** The experiments focus on test power and SHD but do not report size control (Type I error rates) for the CI test, especially as conditioning set sizes increase. Since kernel-based CI tests are known to miscalibrate with large conditioning sets (Shah & Peters, 2020; Lundborg et al., 2022), this omission is concerning for the constraint-based discovery pipeline, where errors compound.

### Trivial

1. The paper states "SigKer—free of any hyperparameter choices" (Section 4), which is an overstatement given the kernel bandwidth and test design choices noted above.

## Nice-to-Haves

- Systematic evaluation of the partially observed setting with varying graph sizes, latent variable counts, and observation patterns.
- Wall-clock time comparisons across methods for different dimensions, since the combinatorial growth of CI tests and the cost of permutation-based testing are the primary practical bottleneck.
- Analysis of robustness to near-faithfulness violations (weak causal effects) and time-varying SDE coefficients.
- A more explicit acknowledgment of the diffusion-dependence limitation and discussion of whether a diffusion-aware variant might close this gap.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Harsh Critic Point 1 (partial overclaiming in fully observed case):** The harsh critic questioned whether the identifiability result for the fully observed case is valid, asking whether "parent faithfulness suffices for full orientation of all edges, including self-loops, purely from CI." The paper provides Theorem 3.2 with proof in Appendix A.7 that establishes this under parent faithfulness—a standard and clearly stated assumption. The result is similar in nature to soundness/completeness results for the PC algorithm under faithfulness, which are well-accepted. The claim is not overstated for the fully observed case.

- **Harsh Critic Point 5 (pairs trading interpretation):** The harsh critic argues the real-world example cannot substantiate causal discovery claims. However, the paper presents it explicitly as a "proof-of-concept study" (Section 4, "Real-world pairs trading example") and a "concise summary," not as validation of causal correctness. The framing is appropriate.

- **Spark's suggestion about additional continuous-time baselines beyond SCOTCH:** The paper compares against the most relevant available baselines: CCM, Granger, PCMCI (representing the discrete-time paradigm the paper argues against), Laumann et al. (the closest functional-data CI test), and SCOTCH (the only existing SDE-based discovery method). There are no other continuous-time causal discovery baselines that handle the same setting (irregular sampling, path-dependence, diffusion, partial observation).

- **Spark's suggestion about near-unfaithfulness experiments:** This is a standard concern for all faithfulness-assuming causal discovery methods and not specific to this paper's methodology. Testing near-faithfulness is a "nice-to-have" rather than a core weakness.

- **Demand for theoretical guidance on s and h selection as a fatal flaw:** The Markov property holds for any valid s and h by Proposition 3.1, so correctness does not depend on their choice. Optimal selection for power is a legitimate but minor practical concern.

## Novel Insights

The lifted graph construction—splitting each node into "past" and "future" copies and using the arrow of time to create additional d-separation constraints that go beyond what symmetric CI can provide—is a genuinely novel idea in continuous-time causal discovery. It transforms the problem of identifying edge direction from one of comparing Markov equivalence classes to one where temporal asymmetry provides identifying information that is provably sufficient (under faithfulness) for full graph recovery. This insight could potentially transfer to other temporal settings beyond SDEs, such as Hawkes processes or other continuous-time point processes.

## Suggestions

- Moderate the identifiability claim for the partially observed case: state clearly that the method recovers *a* maximally informative graph consistent with the observed CIs and temporal constraints, not necessarily *the unique* graph, unless additional assumptions about the latent structure are imposed.
- Report Type I error rates for the signature kernel CI test, at least for the conditioning set sizes used in the causal discovery experiments (d=3,5,10).
- Include a brief systematic evaluation of the partially observed setting (even 3-5 different latent variable configurations) to provide empirical grounding for this claimed contribution.
- Acknowledge the diffusion-dependence gap more prominently and discuss whether it stems from the CI test or the algorithmic framework.

## Score and Decision

**Calibration:**
- SCOTCH (V1GM9xDvIY): scores 6,6,8,8 (avg ~7), accepted as poster — similar topic (SDE causal discovery), weaker theoretical contribution but strong empirical method
- IDOL (2efNHgYRvM): scores 8,8,8, accepted as oral — stronger theoretical identifiability results, cleaner presentation
- Kernel CI tests (GPcSYm89wK): scores 5,5,3,5 (avg ~4.5), rejected — incremental contribution to CI testing, limited evaluation
- PIT (orD5t7blqV): scores 3,6,5,3 (avg ~4.25), rejected — incremental algorithm, weak evaluation

This paper has substantially stronger theoretical contributions than SCOTCH (Markov property, soundness/completeness, CI test consistency), addresses a broader set of continuous-time challenges, and has competitive empirical results. It is weaker than IDOL in terms of polish and completeness of evaluation (especially partial observation). The overclaiming on partial observation identifiability and the diffusion-dependence performance gap are substantive but not fatal weaknesses. The paper is above the acceptance threshold but has clear room for improvement.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>