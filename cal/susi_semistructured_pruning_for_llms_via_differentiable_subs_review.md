=== CALIBRATION EXAMPLE 4 ===

# Final Consolidated Review
## Summary

SUSI proposes a semi-structured pruning method for LLMs that reformulates N:M sparsity mask learning as differentiable weighted reservoir sampling (WRS). By modeling the mask selection process as sampling N items from M without replacement—rather than maintaining a categorical distribution over all $\binom{M}{N}$ feasible masks as in MaskLLM—SUSI reduces trainable parameter complexity from $O(\binom{M}{N})$ to $O(M)$ per group, yielding up to 3.5× parameter reduction for 2:8 sparsity. Experiments on OPT-125M/350M/1.3B with 2:4 sparsity show SUSI achieving the best perplexity and competitive or best zero-shot accuracy against SparseGPT, Wanda, and MaskLLM.

## Strengths

- **Principled parameter complexity reduction with theoretical backing.** The core insight—reformulating mask learning from categorical distributions over $\binom{M}{N}$ configurations to sequential sampling without replacement from M items—is both novel and well-grounded. Theorem 1 (Appendix A.1) formally proves that the WRS-based variational objective yields equivalent expected loss to the exact distribution, ensuring no approximation gap in the formulation itself. The parameter savings grow dramatically with M (1.5× for 2:4, 3.5× for 2:8, 8.75× for 4:8), making previously intractable sparsity patterns feasible (Table 7 shows MaskLLM cannot run 4:8, while SUSI can).

- **Strong perplexity results and consistent zero-shot performance on 2:4 sparsity.** SUSI achieves the lowest perplexity across all three OPT models on WikiText-2 (Table 2: 50.24 vs. 50.91 for MaskLLM on OPT-125M; 54.14 vs. 55.86 on OPT-350M; 28.05 vs. 28.56 on OPT-1.3B). Average zero-shot accuracy is also highest for 2:4 sparsity (Table 1), with SUSI winning or matching on most individual benchmarks.

- **Robustness evidence and ablation rigor.** Figure 5 shows high mask overlap across seeds (0.83–0.94) with <0.5% accuracy variance, and Figure 4 provides clear ablation isolating both the power term p and temperature annealing, demonstrating that removing annealing causes divergence (infinite PPL) and p=1 yields dramatically worse results (PPL 998.33→28.05 on OPT-350M).

## Weaknesses

### Major:

- **Evaluation limited to models ≤1.3B, with appendix extensions only reaching 1B.** All main experiments use OPT-125M/350M/1.3B. Appendix A.8 adds Qwen2.5-0.5B and Llama3.2-1B—still sub-2B. For ICLR 2026, the community standard for LLM pruning research has moved to 7B+ models (e.g., Llama-3-8B). The paper's core selling point—parameter efficiency enabling learnable mask methods at scale—remains unevidenced at the scale where it matters most. The 4:8 result showing MaskLLM failing on OPT-1.3B is suggestive but does not prove SUSI scales to 7B+. Without at least one 7B-class experiment, the significance of the contribution is substantially undermined.

- **Computational efficiency claims are supported only by parameter counts, not training time or FLOPs.** The paper repeatedly claims "minimal computational cost" (abstract), "efficient deployment" (conclusion), and "substantial computational and memory savings" (Section 4.3.1). However, the sequential Gumbel-Top-K procedure applies N softmax operations per group per forward-backward pass (Equations 10–12), yielding $O(N \cdot M)$ sampling complexity per group per step. This is potentially more expensive per step than MaskLLM's single Gumbel-Softmax pass, even though SUSI has fewer parameters. The absence of wall-clock training time, GPU memory peak, or FLOP comparisons makes the efficiency claims unverifiable. Given that the paper's primary motivation is efficiency over MaskLLM, this is a critical gap.

- **Abstract and main text overclaim accuracy superiority across all sparsity patterns.** The abstract states SUSI "outperforms baselines such as SparseGPT, Wanda, and MaskLLM in perplexity while maintaining competitive zero-shot accuracy." For 2:4 sparsity, this is well-supported. However, Table 6 (2:8 sparsity) shows MaskLLM achieving higher average zero-shot accuracy than SUSI on both OPT-125M (37.27% vs. 37.22%) and OPT-350M (35.91% vs. 35.22%). While the differences are small, the blanket claim of outperforming baselines is inaccurate. The paper should qualify that accuracy gains are strongest for 2:4 sparsity and that a parameter-accuracy trade-off emerges at higher sparsity ratios.

### Minor:

- **The power term p is critical yet only empirically motivated.** Figure 4 shows that p=1.0 yields catastrophically worse results (PPL 998.33 on OPT-350M) and p=3.0 is selected from {1,2,3}. The ablation is only on OPT-350M; no sensitivity analysis across model sizes or sparsity patterns is provided. Given the method's instability without this term, the lack of justification for p=3.0 as a general choice is concerning.

- **No inference latency or throughput measurements on sparse-accelerated hardware.** The paper motivates N:M sparsity by hardware compatibility (NVIDIA Ampere/Hopper), but does not measure actual inference speedup. While this is partly a deployment concern, the paper's framing around "efficient deployment" and "accelerating inference" (Section 1) makes the absence notable.

### Trivial:

- The 4:8 sparsity results (Table 7) lack a MaskLLM comparison entirely (noted as infeasible), making it hard to assess SUSI's relative standing at that sparsity level. This is understandable given the infrastructure constraint but limits the completeness of the evaluation.

## Nice-to-Haves

- Evaluation on a 7B+ model (e.g., Llama-3-8B) to validate that parameter efficiency translates to practical scalability gains.
- Wall-clock training time comparison with MaskLLM and total GPU-hour comparison with one-shot methods (Wanda, SparseGPT) to contextualize the cost-benefit trade-off.
- Correlation analysis between learned mask probabilities and Hessian-based importance scores to verify SUSI identifies theoretically important weights rather than fitting calibration noise.
- Fine-tuning recovery experiments on the pruned model to assess whether the learned masks preserve adaptability.
- Joint quantization experiments (SUSI + 4-bit quantization), since practical deployment rarely uses sparsity alone.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: Questioning whether MaskLLM's parameterization is accurately characterized.** The harsh critic suggests MaskLLM might use O(W) parameters instead of $O(\binom{M}{N})$. However, MaskLLM (Fang et al., 2024) explicitly models categorical distributions over all C(M,N) feasible masks per group, and the paper's characterization is consistent with the cited work. Per the rules, cited works are treated as accurately described.

- **Weakness: Potential overfitting to calibration data.** The paper uses 1B tokens from C4 for training and evaluates on 6 diverse zero-shot benchmarks with competitive results. There is no evidence of overfitting, and this criticism is speculative.

- **Weakness: Missing downstream application validation beyond zero-shot tasks.** Zero-shot evaluation on standard NLP benchmarks is the established norm for LLM pruning papers. Demanding additional fine-tuning or domain-specific evaluation is scope creep.

- **Weakness: Missing related works.** Per rules, I cannot confirm the existence of unspecified related works.

- **Weakness: Reproducibility concerns about undisclosed hyperparameters.** All hyperparameters are listed in Table 4 and Appendix A.3.

- **Weakness: Theoretical justification demanded for the power term.** Requiring theoretical proofs for an empirical design choice in a systems/empirical paper goes beyond field standards.

- **Weakness: SUSI could benefit from joint weight updates like SparseGPT.** This is an asymmetry that favors the baseline. SUSI achieves competitive results without weight updates; if it also used them, it would likely perform even better.

## Novel Insights

The connection between Weighted Reservoir Sampling and N:M sparsity mask learning reveals a deeper structural insight: the combinatorial explosion in mask space ($\binom{M}{N}$ configurations) is an artifact of the *representation* (categorical distributions over masks) rather than the *problem itself*. Since the pruning objective depends only on which weights are selected—never on the order of selection—the WRS decomposition exploits this permutation invariance to collapse the parameter space from factorial to linear in M. This suggests that other combinatorial optimization problems in neural network compression (e.g., mixed-precision assignment, structured pruning of attention heads) might similarly benefit from order-invariant reparameterizations that avoid enumerating combinatorial spaces.

## Suggestions

- Add at least one experiment on a 7B+ model (e.g., Llama-3-8B with 2:4 sparsity) to the main paper. Even a single data point would substantially strengthen the scalability argument and differentiate from prior work evaluated only on small models.
- Report wall-clock training time and peak GPU memory for SUSI vs. MaskLLM on the same hardware. This directly tests whether parameter reduction translates to practical efficiency gains given the sequential sampling overhead.
- Qualify the abstract and Section 4.2 claims to note that accuracy advantages are most pronounced for 2:4 sparsity, and that MaskLLM retains a slight edge on zero-shot accuracy at 2:8 sparsity in exchange for higher parameter cost.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
