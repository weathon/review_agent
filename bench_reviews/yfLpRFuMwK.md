## Summary

The paper proposes Non-Replacement Function Space Sampling (NRFS), a novel Bayesian optimization acquisition strategy that estimates the Probability of Optimality (PO)—the probability that a candidate point is the global optimizer—by sampling functions from a Truncated Gaussian Process and assigning them to "buckets" based on their optimizers. The method removes sampled functions from consideration after each acquisition (conceptually), iteratively identifying candidates that maximize the fraction of remaining function space for which they are the optimizer. Empirical evaluation across synthetic benchmarks and real-world materials discovery tasks demonstrates competitive or superior performance compared to EI, PES, UCB, and other baselines.

## Strengths

- **Principled reformulation of BO acquisition:** The paper provides a compelling conceptual framing—maximizing the probability that a candidate is the true optimizer—rather than relying on heuristic exploration-exploitation trade-offs or information-theoretic rewards. This directly targets the fundamental BO objective and avoids hyperparameter tuning inherent to UCB ($\beta$), $\epsilon$-EI ($\epsilon$), or adaptive schedules.

- **Strong empirical performance on challenging multimodal objectives:** NRFS consistently outperforms baselines on benchmarks requiring escape from local optima (GM, Modified Rosenbrock, Shekel). The real-world SFE materials discovery task (Figure 6) provides convincing evidence that NRFS successfully identifies global optima where EI gets trapped in local minima and PES fails to exploit promising regions.

- **Robustness to noise:** Appendix A.5 demonstrates stable performance across SNR levels (4, 16, 64), with NRFS maintaining advantages even when noise causes the TGP threshold to shift. The analysis that "small noise can improve performance" by broadening the function space (Figures 12b, 13b) is an interesting observation.

## Weaknesses

- **Overclaiming in abstract:** The abstract states NRFS achieves "consistently improving optimization performance in all settings," but Appendix A.4 documents that EI outperforms NRFS on Branin because NRFS "acquires all global optima rather than focusing on a single one." This is a legitimate trade-off, not consistent superiority—Branin has three equivalent global optima where finding any one suffices.

- **Unsupported theoretical claims:** Section 3.3 claims the OSLA variant "has the potential to achieve the maximum convergence rate" without formal definition or proof. Similarly, Eq. (13)'s product formulation ($R_T = 1 - \prod_{t=1}^{T}(1 - P(\cdot))$) implicitly assumes independence across sequential acquisitions, which does not hold—acquisitions are correlated through shared GP posteriors. The independence assumption is never acknowledged.

- **Convergence guarantee relies on unrealistic assumption:** The paper states "as long as the surrogate contains the true objective, non-replacement sampling guarantees convergence." This is a strong surrogate correctness assumption that is rarely satisfied in practice. No analysis of robustness under GP misspecification is provided.

- **Non-replacement mechanism implementation ambiguity:** The conceptual framework describes "removing" functions from the pool, but the implementation samples fresh functions ($M=1000$) from the updated TGP each iteration. The relationship between the conceptual "removal" and the practical TGP threshold mechanism is never clearly bridged—readers cannot determine what state, if any, is actually carried between iterations.

- **Multi-optimizer handling lacks analysis:** For functions with multiple global optima, the paper assigns them "randomly" to valid buckets to enforce one-to-one mappings. The impact of this arbitrary assignment on PO estimation accuracy is not analyzed.

- **Continuous domain theoretical gap:** In continuous design spaces, the probability that two distinct GP sample functions share exactly the same optimizer is zero. The paper uses Parzen estimation to approximate a discrete optimizer distribution but never justifies why the mode of this estimate corresponds to maximizing $|F_{\mathbf{x}}^D|$ in the continuous limit.

- **Missing directly relevant baselines:** BORE (Bayesian Optimization by Density-Ratio Estimation, Tiao et al., ICML 2021) similarly uses truncation on the current best to define "good" vs "bad" regions—conceptually close to NRFS's TGP conditioning. Max-value Entropy Search (MES) also conditions on the maximum value, making it directly relevant. Neither is included in comparisons.

- **No ablation on sampling budget $M$:** All experiments use $M=1000$ sampled functions. The method's sensitivity to this parameter—whether performance degrades with fewer samples or improves with more—remains unexamined despite being central to the Monte Carlo approximation of PO.

- **Figure 10 caption error:** The caption states "current best of 20 trials for Branin and BraninRcos2" but the figure panels show Mopta08, Lasso-DNA, and Rover benchmarks. This is either a copy-paste error or mislabeling.

## Nice-to-Haves

- Comparison with modern high-dimensional BO methods (TuRBO, HEBO, SaasBO) for the 50D experiments would strengthen scalability claims, though the current scope is reasonable.

- Theoretical analysis of regret bounds or formal convergence rates would strengthen the methodology contribution.

- Explicit comparison with Thompson Sampling variants to disentangle the contribution of "non-replacement" versus truncation alone.

## Removed Points

*These points are flagged to be removed, treat them with caution*

- **Demand for statistical significance tests:** Formal tests like Wilcoxon rank-sum are not standard practice in BO methodology papers; mean/std curves with 20 trials are typical and acceptable for this venue.

- **Oracle tuning complaint for $\epsilon$-EI:** The harsh critic suggests $\epsilon$-EI was unfairly advantaged. The paper explicitly notes the tuning burden ("requires sweeping $\epsilon$ from 0.1 to 0.9"), framing this as a practical disadvantage of baselines—not an unfair comparison. The paper's position that NRFS requires no tuning is a legitimate methodological contribution.

- **High-dimensional real-world benchmarks in main text:** The harsh critic wanted Mopta08 and Lasso-DNA moved from appendix. Figure 5 already includes 5D-50D scaling analysis in the main paper. Placement of additional benchmarks in appendix is an organizational choice, not a flaw.

- **Comparison against noise-aware acquisition functions (EI-Noisy):** The noise experiments in Appendix A.5 evaluate all methods under the same noisy conditions, which is fair for assessing relative robustness. Specialized noise-aware methods would require separate tuning, complicating comparison.

## Novel Insights

The key conceptual insight is that traditional acquisition functions optimize surrogate quantities (expected improvement, entropy reduction) rather than directly targeting the probability of finding the optimizer. NRFS reframes the problem: each sampled function "votes" for its optimizer, and the method selects the candidate with the most votes from functions that could plausibly be the true objective (i.e., those whose minima improve on the current best). The truncation mechanism elegantly filters out functions that cannot be the true objective—a function whose minimum is worse than the current best observation cannot represent the true objective. This provides a principled, parameter-free mechanism that implicitly balances exploration (regions with high uncertainty have broader optimizer distributions) and exploitation (regions with low predicted values accumulate votes). The non-replacement framing reveals that standard BO methods may repeatedly sample from the same subset of plausible functions, wasting evaluations on regions that cannot contain the global optimum.

## Suggestions

1. **Add pseudocode:** A clear algorithm box showing what state is carried between iterations (if any) versus what is recomputed would resolve ambiguity about the "non-replacement" implementation.

2. **Clarify the convergence claim:** Either formalize the convergence guarantee under explicit assumptions, or reframe the claim to acknowledge dependence on surrogate correctness.

3. **Include BORE and MES as baselines:** Both are directly relevant to the truncation mechanism and would strengthen empirical positioning.

4. **Add ablation on $M$:** Demonstrate how performance scales with fewer samples (e.g., $M \in \{100, 500, 1000\}$) to assess computational-performance trade-offs.

5. **Correct Figure 10 caption:** Fix the caption-text mismatch to improve experimental presentation integrity.