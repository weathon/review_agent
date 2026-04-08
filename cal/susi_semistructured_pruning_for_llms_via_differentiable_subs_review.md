=== CALIBRATION EXAMPLE 14 ===

# Final Consolidated Review
## Summary

SUSI proposes a semi-structured pruning method for LLMs that reformulates N:M sparsity mask learning using Weighted Reservoir Sampling (WRS) combined with the Gumbel-Top-K trick for differentiable subset sampling. This reduces the trainable parameter complexity from $O(\binom{M}{N})$ per weight group (as in MaskLLM) to $O(M)$, yielding up to 1.5× parameter reduction for 2:4 sparsity and 3.5× for 2:8. Experiments on OPT models (125M–1.3B) show SUSI achieves lower perplexity than MaskLLM and competitive zero-shot accuracy, while remaining tractable at 4:8 sparsity where MaskLLM fails.

## Strengths

- **Core parameter efficiency contribution is genuine and well-motivated.** The reformulation from a categorical distribution over $\binom{M}{N}$ mask configurations to a WRS parameterization requiring only $M$ parameters per group is a clean, non-trivial insight. This is not merely an engineering tweak—the mathematical reduction is exponential in $M$ and directly enables Table 7's 4:8 results where MaskLLM is infeasible.

- **Strongest empirical evidence lies in aggressive sparsity regimes.** Table 7 (4:8 sparsity) is the paper's most compelling result: MaskLLM literally cannot execute due to $\binom{8}{4} = 70$ parameters per group, while SUSI completes training and achieves strong performance (41.84% average on OPT-125M). This validates the core motivation in a way that marginal 2:4 improvements cannot.

- **Thorough ablations and robustness analysis.** The ablation in Figure 4 cleanly isolates the contributions of the power term $p$ and temperature annealing (removing annealing → infinite PPL; $p=1$ → PPL 998.33 vs. $p=3$ → PPL 28.05). The cross-seed mask overlap analysis (Figure 5, overlap ≥ 0.83) addresses a common concern in stochastic pruning methods.

- **Appendix extends beyond OPT.** Table 8 evaluates on Qwen2.5-0.5B and Llama3.2-1B, demonstrating SUSI remains tractable on modern architectures, even though performance gaps to dense models are larger than in the OPT family.

## Weaknesses

- **Gap between theoretical guarantee and implemented algorithm.** Theorem 1 proves equivalence of the expected loss under the exact WRS distribution (Equation 1). However, the actual training procedure modifies the Gumbel-Top-K update with a power term $p > 1$ in Equation 11: $\boldsymbol{\alpha}_i^{(k)} := \boldsymbol{\alpha}_i^{(k-1)} - |\log(1 - \boldsymbol{\mu}_i^{(k-1)})|^p$. This deviates from the standard Gumbel-Top-K update (Equation 3), which corresponds to exact WRS sampling. While the ablation shows $p=3$ is critical for convergence, the paper provides no theoretical justification for why this modified sampling distribution still preserves the variational bound properties of Theorem 1, or whether it should be viewed as a biased but effective estimator. This disconnect between the clean theory and the practical algorithm needs honest acknowledgment.

- **"Minimal computational cost" framing in the abstract is misleading.** The abstract claims SUSI learns masks "with minimal computational cost," referring specifically to mask parameter memory. However, SUSI requires training on 1B tokens (Section 4.1: 2,000 steps, batch 256, seq 2048). Compared to one-shot methods like Wanda or SparseGPT (which use <1024 samples), the *data and FLOP cost* is orders of magnitude higher. The efficiency gain is real but narrow (mask parameters only); the abstract's framing obscures this critical trade-off. Practitioners choosing between "1B tokens + slightly better perplexity" versus "1K samples + slightly worse perplexity" need honest accounting of both sides.

- **Marginal improvements on the standard 2:4 pattern.** On the practically most important sparsity setting (2:4, supported by NVIDIA hardware), SUSI's gains over MaskLLM are small: +0.41% average accuracy on OPT-125M, +0.18% on OPT-350M, +0.61% on OPT-1.3B. Perplexity improvements are similarly modest (50.24 vs. 50.91 on OPT-125M). On individual tasks, SparseGPT frequently outperforms SUSI (ARC-C at 125M and 1.3B). The practical significance of these differences is questionable for the dominant deployment case. SUSI's clear advantage only manifests at higher-M patterns (2:8, 4:8) where hardware support is currently limited.

- **Evaluation limited to small model scales.** The largest model tested is OPT-1.3B (with Qwen2.5-0.5B and Llama3.2-1B in the appendix). Semi-structured pruning is most impactful for 7B–70B models where memory and compute constraints are severe. The paper's core claim—parameter efficiency enabling scalable pruning—would be far more convincing with at least one 7B+ experiment showing SUSI remains practical and effective where MaskLLM becomes prohibitively expensive.

- **No measured training time or GPU memory comparison with MaskLLM.** The efficiency argument rests entirely on trainable parameter counts (Figure 3a). A reduction from 1.5× parameters does not directly translate to proportional training time or peak GPU memory savings, since forward/backward passes through the frozen LLM backbone dominate both costs. Without wall-clock time or measured peak memory, it is unclear whether SUSI's parameter reduction yields meaningful practical efficiency gains or merely represents a theoretical improvement.

- **Missing comparison with AST (Huang et al., 2025).** The paper identifies AST as a closely related learnable mask method for semi-structured LLM pruning in the related work (Section 4.4), noting computational overhead as a shared challenge. Yet AST is absent from all experimental comparisons. Since AST also targets N:M sparsity with a different approach to mask learning, its exclusion leaves a gap in understanding SUSI's relative position among learnable methods.

## Nice-to-Haves

- Measured inference latency/throughput on hardware with native 2:4 sparsity support (Ampere/Hopper GPUs). While 2:4 sparsity speedups are well-established in prior work, directly measuring them would strengthen the deployment narrative.
- Calibration data sensitivity analysis varying the token budget below 1B. Figure 3(b) partially addresses this but only for OPT-350M; showing how performance degrades at very small calibration sizes (e.g., 10M–100M tokens) would inform practitioners with limited compute budgets.
- Analysis isolating whether WRS formulation itself or merely fewer parameters drives SUSI's improvements—e.g., a controlled experiment where MaskLLM is artificially parameter-limited.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"No attempt to evaluate on LLaMA/Mistral"** (Harsh Critic): Factually incorrect. Appendix A.8 explicitly evaluates on Qwen2.5-0.5B and Llama3.2-1B.
- **"4:8 results suspicious/need verification"** (Spark Finder): The paper clearly states MaskLLM "could not be executed on our infrastructure in this setting due to the excessive number of trainable parameters"—this is precisely the paper's predicted outcome given $\binom{8}{4}=70$ parameters per group. No grounds for suspicion.
- **"Missing pruning+quantization combined experiments"** (Spark Finder): Outside the paper's stated scope of semi-structured pruning alone.
- **"Mask transfer across tasks"** (Spark Finder): Interesting but outside the paper's scope.
- **Formatting/equation rendering complaints** (Harsh Critic): The critic themselves note these are parser artifacts.
- **Missing related works**: Per rules, cannot confirm existence of uncited works.
- **Reproducibility concerns about hyperparameters**: The paper provides full hyperparameters in Table 4 and Appendix A.3, plus an anonymous repository.

## Novel Insights

The paper reveals an underappreciated tension in semi-structured pruning: the parameter overhead of learnable mask methods grows combinatorially with group size $M$, meaning the very sparsity patterns where learnable masks could add the most value (higher-$M$ patterns with more mask flexibility) are precisely where they become most intractable. SUSI's WRS formulation breaks this curse, but the catch is that current hardware predominantly supports only 2:4—exactly the regime where SUSI's advantage is smallest. This suggests the practical impact of this line of work depends heavily on future hardware support for broader N:M patterns, a dependency the authors honestly acknowledge in their limitations section.

## Suggestions

- Add an honest paragraph in the introduction or experiments explicitly quantifying the trade-off between SUSI's 1B-token training cost versus one-shot methods' near-zero cost, so practitioners can make informed decisions.
- Acknowledge the theory-practice gap from the power term $p$ explicitly—frame it as a biased but effective modification rather than leaving the impression that Theorem 1 fully covers the implemented algorithm.
- Include at least one wall-clock training time comparison (SUSI vs. MaskLLM) even on the smallest model, to validate that parameter count reduction translates to real efficiency gains.
- Add AST to the experimental baselines; if implementation is unavailable, discuss the expected comparison based on AST's reported results.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
