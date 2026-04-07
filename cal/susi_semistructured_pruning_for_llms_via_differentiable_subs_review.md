=== CALIBRATION EXAMPLE 22 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title accurately reflects the contribution. The abstract claims SUSI "outperforms baselines such as SparseGPT, Wanda, and MaskLLM in perplexity while maintaining competitive zero-shot accuracy." However, the gains over MaskLLM (the closest and most relevant baseline) are marginal — e.g., PPL 50.24 vs. 50.91 on OPT-125M — and the abstract's phrasing somewhat overstates the magnitude of the improvement. The abstract also implies broad LLM applicability, but the primary evaluation is limited to OPT models up to 1.3B, which the abstract does not acknowledge upfront.

---

### Introduction & Motivation

The motivation is clear: MaskLLM's parameter complexity scales as C(M,N) × (W/M) parameters, which for 2:4 is 1.5× the original model and for 4:8 is ~8.75×. Reducing this overhead is a legitimate problem. However, there is a subtle conflation throughout the introduction: the paper presents reduced *trainable parameters* as equivalent to reduced *training cost*, yet no wall-clock training time comparison is ever provided. The actual computational bottleneck during mask training is the forward/backward pass through the (frozen) LLM, not the mask parameters themselves. For 2:4 sparsity, the 1.5× parameter reduction is the paper's flagship claim, but it is unclear whether this translates to meaningful end-to-end speedup.

The introduction also omits a proper discussion of AST (Huang et al., 2025), which is cited in the related work section but never included as a baseline in any experiment — a significant gap given that AST is a concurrent semi-structured mask-learning approach also addressing MaskLLM's overhead.

---

### Preliminaries (Section 2)

Sections 2.1 and 2.2 are clearly written. However, there is a subtle but important issue: the paper frames WRS as the conceptual backbone of the approach, but the actual method relies on the Gumbel-Top-K trick (Section 2.2), which was independently known to be equivalent to sampling without replacement under a Softmax distribution (Xie & Ermon, 2019). The connection to WRS is theoretically valid (Gumbel-Max is a monotonic transformation of WRS), but it is somewhat misleading to brand the method as "Weighted Reservoir Sampling" when the operational mechanism is simply Gumbel-Top-K, a technique already used in ML. The novelty lies in the *parameterization* choice (M logits per group instead of C(M,N)), not in the sampling mechanism itself.

---

### Methodology (Section 3)

**Theorem 1 and the approximation gap.** Theorem 1 (Appendix A.1) establishes that the expected loss under the exact distribution P(**m**|**ϕ**) equals the expected loss under the WRS-restricted parameterization. This is a clean and necessary result. However, the proof as presented (Appendix A.1) is largely notational — the key step reduces to: the sum over all ordered subsets summing to a given mask **m** equals P_WRS(S_m|**ϕ**), which is asserted but not fully elaborated. The argument is plausible but not fully self-contained.

**The power term modification (Equation 11).** This is the most problematic aspect of the paper. The standard Gumbel-Top-K update rule (Equation 3) is:
> **α**^(k) = **α**^(k−1) + log(1 − **µ**^(k−1))

SUSI modifies this to:
> **α**^(k) = **α**^(k−1) − |log(1 − **µ**^(k−1))|^p

For p = 1, since log(1−µ) ≤ 0 for µ ∈ [0,1], both expressions subtract |log(1−µ)|, so they are equivalent. For p > 1, however, the update is no longer equivalent to the standard Gumbel-Top-K procedure. This means the theoretical justification of Theorem 1 — which is derived under the standard WRS/Gumbel-Top-K framework — does not directly apply to the actual SUSI algorithm with p = 3. The paper never acknowledges this gap, which undermines the theoretical foundation. The power term is motivated purely empirically ("improves stability during training"), and the ablation (Figure 4) shows it matters a lot (p=1 gives PPL 998.33, p=3 gives PPL 28.05). That the method essentially fails without this theoretically unjustified modification raises a red flag about whether the Gumbel-Top-K relaxation is actually the right framework here.

**Algorithm 1 inconsistency.** Line 9 of Algorithm 1 uses the vanilla update `α^(k+1) ← α^(k) + log(1 − µ^(k))`, which *contradicts* Equation 11 (which includes the absolute value and power term). It is unclear whether the theoretical analysis or the empirical results use the modified update. This inconsistency must be resolved.

**Two temperatures.** The method introduces two temperature hyperparameters (τ and λ) plus a power term p, each with its own annealing schedule. This is a non-trivial hyperparameter burden. No sensitivity analysis for τ_init, τ_end, λ_init, λ_end is provided — the ablation only varies p and whether annealing exists at all.

---

### Experiments & Results (Section 4)

**Scale of evaluation.** The primary experiments cover only OPT-125M, OPT-350M, and OPT-1.3B. By ICLR 2026 standards, this is severely inadequate. MaskLLM (NeurIPS 2024, the main baseline) was evaluated on LLaMA-2-7B and larger models. The paper includes a brief appendix (A.8) with Qwen2.5-0.5B and Llama3.2-1B, but provides no comparison against MaskLLM on these models. There is no evaluation on any model ≥7B parameters, which is where N:M sparsity is most practically relevant. The claim of being a "robust and practical solution" for LLM deployment is hard to accept without validation on modern, production-scale models.

**Marginal gains over MaskLLM.** The improvements over MaskLLM are consistently small:
- Table 2 PPL: 50.24 vs. 50.91 (OPT-125M), 54.14 vs. 55.86 (OPT-350M), 28.05 vs. 28.56 (OPT-1.3B)
- Table 1 avg. accuracy: 41.06 vs. 40.65, 39.94 vs. 39.76, 46.48 vs. 45.87

No statistical significance testing is reported. Given the standard variance in LM evaluation and that each number comes from a single run (with robustness across seeds shown only for one model/configuration), it is impossible to determine whether these differences are meaningful.

**2:8 results (Table 6, Appendix).** Under 2:8 sparsity, SUSI *underperforms* MaskLLM in average zero-shot accuracy on both OPT-125M (37.22 vs. 37.27) and OPT-350M (35.22 vs. 35.91). This directly contradicts the framing that SUSI consistently outperforms MaskLLM. While the perplexity values are not shown clearly in Table 6, this result weakens the performance story considerably.

**No wall-clock time comparison.** Despite the paper's emphasis on parameter efficiency as an efficiency gain, there is no measurement of actual training time (GPU-hours). For the dominant 2:4 case, the 1.5× parameter reduction may have negligible impact on total training time since the bottleneck is the LLM forward pass, not the mask parameters. Without this measurement, the practical efficiency claim is unsubstantiated.

**Missing baseline.** AST (Huang et al., 2025) is explicitly cited in Section 4.4 as a relevant concurrent method, yet it is never compared against in any table. This omission is notable.

**Perplexity drop for modern architectures (Appendix A.8).** The results on Qwen2.5-0.5B show a drop from 55.33% to 43.75% average accuracy (~12% absolute) and PPL increasing from 22 to 46. This is a significant degradation compared to OPT, and no comparison to MaskLLM or other baselines is provided on these architectures. The authors acknowledge this in the text but do not investigate why modern architectures appear more sensitive, nor do they compare competing methods.

**Discussion numbering error.** Section 4.2 discusses points (i) and (iii) for Table 2 but skips point (ii), suggesting that text was inadvertently removed during editing.

---

### Ablation Study (Section 4.3.2)

The ablation is focused on just two binary decisions (power term on/off, annealing on/off) plus the value of p ∈ {1, 2, 3}. This is useful but incomplete. Missing ablations include:
- Sensitivity to the number of Gumbel samples (Monte Carlo variance in gradient estimation)
- The effect of initialization distribution (only N(0, 0.01) is used)
- Comparison at matched compute budgets (e.g., SUSI trained for fewer tokens due to parameter efficiency vs. MaskLLM trained with same compute)

---

### Robustness Analysis (Section 4.3.3)

The mask overlap analysis across seeds is a useful robustness check. However, overlap is measured only for the first transformer block on OPT-350M. It would be more convincing to show this across all layers or at least across a few depths, since sparsity patterns in early and late layers are known to differ in behavior.

---

### Limitations (Appendix A.9)

The limitations section is honest about hardware dependency and the English-centric evaluation. However, it omits two important points: (1) the theoretical inconsistency introduced by the power-term modification and the resulting gap between Theorem 1 and the actual algorithm, and (2) the fact that the method shows considerably larger accuracy degradation on modern architectures (Qwen, LLaMA) compared to OPT, which is relevant for practical adoption.

---

### Overall Assessment

SUSI addresses a genuine and practically motivated inefficiency in MaskLLM's parameterization. The core idea — reformulating the N:M sparsity mask learning problem with M logits per group instead of C(M,N) — is elegant and well-motivated. For settings with large M (e.g., 4:8 sparsity where MaskLLM becomes infeasible), SUSI offers a meaningful advantage. However, the paper has several significant weaknesses for ICLR 2026: (1) The primary experiments are limited to tiny OPT models (up to 1.3B), with no evaluation on the ≥7B models where semi-structured pruning is most relevant; (2) the critical power-term modification in Equation 11 breaks the theoretical correspondence with WRS/Gumbel-Top-K that underlies Theorem 1, yet this is never discussed; (3) Algorithm 1 is inconsistent with the stated method; (4) gains over MaskLLM are marginal and lack significance testing, and SUSI actually underperforms MaskLLM under 2:8 sparsity; (5) no wall-clock training time comparison is provided despite efficiency being the paper's central claim; and (6) the comparison omits AST, a directly relevant concurrent baseline. In its current form, the contribution falls below the ICLR acceptance bar: the theoretical claims are undermined by the ad-hoc power term, and the empirical case for superiority over MaskLLM is not convincingly made at relevant scale.

# Neutral Reviewer
## Balanced Review

### Summary
The paper introduces SUSI (Semi-structured prUning via Subset samplIng), a method for post-training pruning of Large Language Models (LLMs) that enforces N:M sparsity using differentiable subset sampling based on Weighted Reservoir Sampling (WRS). Unlike prior learnable mask approaches like MaskLLM that parameterize the full combinatorial distribution of sparsity patterns, SUSI models mask selection as sequential sampling without replacement, reducing the trainable parameter complexity from combinatorial $O(\binom{M}{N})$ to linear $O(M)$ per weight group. The method is evaluated on OPT models across multiple architectures and sparsity patterns, demonstrating superior perplexity and competitive zero-shot accuracy compared to baselines like SparseGPT and MaskLLM, while remaining feasible for aggressive sparsity ratios (e.g., 4:8) where competitors fail due to parameter explosion.

### Strengths
1.  **Parameter Efficiency in Learnable Masking:** The core methodological contribution addresses a specific limitation of recent learnable pruning methods like MaskLLM, which incur prohibitive trainable parameter overhead. By employing Gumbel-Top-K sampling over $M$ logits rather than $\binom{M}{N}$ mask configurations, SUSI achieves a significant reduction in optimization parameters (e.g., 1.5x reduction for 2:4 sparsity, 8.75x for 4:8). This makes the training of semi-structured masks more tractable for large-scale models.
2.  **Comprehensive Empirical Validation:** The evaluation is robust, covering three model sizes (125M to 1.3B), multiple benchmarks (Wikitext-2, ARC, HellaSwag, etc.), and various sparsity ratios (2:4, 2:8, 4:8). The results consistently show SUSI outperforming training-free baselines (Magnitude, Wanda, SparseGPT) in perplexity and maintaining competitive accuracy. The extension to newer models (Qwen, Llama) and aggressive sparsity patterns further demonstrates scalability.
3.  **Detailed Ablation and Robustness Analysis:** The paper provides a thorough ablation study on key hyperparameters (power term $p$ and temperature annealing), showing their critical role in convergence stability. Additionally, the robustness analysis across different random seeds (measuring mask overlap) confirms that the learned pruning patterns are stable and not sensitive to initialization, which is crucial for reliability.
4.  **Clear Reproducibility Statement:** The authors provide an anonymous repository link and detailed documentation of datasets and hyperparameters, aligning with ICLR's expectations for reproducibility. The use of standard benchmarks and publicly available models facilitates verification.

### Weaknesses
1.  **Computational Overhead vs. One-Shot Baselines:** While SUSI is more efficient than MaskLLM, it still requires iterative training (2,000 steps) to optimize the mask parameters. The paper emphasizes the reduction in *mask trainable parameters*, but the overall calibration training cost (GPU hours) compared to training-free methods like SparseGPT or Wanda is not explicitly discussed or quantified. This makes it unclear if the performance gains justify the added computational cost compared to simpler, one-shot baselines.
2.  **Hardware Dependency Limitations:** Semi-structured sparsity (N:M) relies on specific hardware acceleration (e.g., NVIDIA Ampere/Hopper) to realize inference speedups. The paper acknowledges this in the Limitations but the "Efficient Deployment" claim in the abstract might be slightly overstated if the target hardware lacks native N:M support. The practical benefit is primarily in memory footprint reduction, not necessarily raw latency, on general-purpose hardware.
3.  **Generalization Gap on Newer Architectures:** While performance on OPT models is strong, the results on Qwen2.5-0.5B and Llama3 show a more significant accuracy drop (e.g., 55% to 43% on PPL 46 for Qwen) compared to OPT. This suggests that the learned masks might be more sensitive to specific architectural choices or that the method scales differently than implied by the OPT results, warranting further discussion on transferability.
4.  **Theoretical Justification:** The method largely combines existing techniques (WRS, Gumbel-Top-K, Variational Optimization) in a new context. While applied effectively, the theoretical proof (Theorem 1) could be more deeply connected to why this distribution approximates the optimal mask distribution better than a uniform or other initialization, rather than just demonstrating it has fewer parameters.

### Novelty & Significance
**Novelty:** The novelty is **moderate**. The techniques used (Weighted Reservoir Sampling, Gumbel-Top-K reparameterization) are established in combinatorial optimization and sampling literature (e.g., Xie & Ermon, 2019; Efraimidis & Spirakis, 2006). The paper's contribution lies in the specific reformulation of semi-structured LLM pruning as a differentiable subset sampling problem to bypass the combinatorial explosion of learnable masks found in MaskLLM. It is a methodological refinement rather than a fundamental theoretical breakthrough, but it addresses a clear practical bottleneck in existing learnable pruning literature.
**Significance:** The significance is **high** for the specific niche of learnable semi-structured pruning. As N:M sparsity becomes the standard for hardware acceleration, methods that can learn optimal masks without exceeding memory constraints (like MaskLLM) are increasingly necessary. SUSI makes high-ratio sparsity (4:8) viable for pruning where MaskLLM fails, offering a practical path to compressing LLMs while maintaining hardware compatibility.

### Suggestions for Improvement
1.  **Quantify Training Costs:** Explicitly compare the wall-clock training time or FLOPs required for SUSI calibration against SparseGPT and Wanda. Since these one-shot methods are widely used for their speed, quantifying SUSI's overhead would help practitioners decide if the accuracy gains are worth the extra compute.
2.  **Clarify Deployment Benefits:** Elaborate on the distinction between memory reduction (achievable with any sparsity) and inference speedup (requiring specific hardware). Discuss strategies or future work for CPU or non-Ampere GPU deployment where kernel support might be lacking.
3.  **Deepen Analysis of Architecture Sensitivity:** Provide more insight into why SUSI performs less robustly on Qwen/Llama compared to OPT. Is it related to layer normalization differences, context length, or the nature of pre-training data? A brief discussion would strengthen the claim of generalizability.
4.  **Strengthen Theoretical Intuition:** The proof of equivalence (Theorem 1) shows distributional equivalence but could be supplemented with an explanation of *inductive bias*. Why does the sequential sampling approach lead to better convergence or generalization than the full categorical distribution in the context of LLM weights, beyond just parameter efficiency?

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Evaluate on 7B+ Models:** Test on Llama-2-7B or similar scales. 1.3B models do not reflect pruning challenges in modern LLMs, making scalability claims unconvincing for ICLR.
2. **Report Wall-Clock Latency:** Measure inference throughput on supported hardware (e.g., A100). Parameter reduction does not guarantee speedup without verified kernel implementation.
3. **Calibration Data Ablation:** Vary calibration data from 128 samples to 1B tokens. High data requirements undermine efficiency claims against one-shot baselines like Wanda.

### Deeper Analysis Needed (top 3-5 only)
1. **Impact of Frozen Weights:** Analyze why SUSI does not update weights while SparseGPT does (Table 1). Allowing weight updates for SUSI is necessary for a fair performance comparison.
2. **Gradient Variance Quantification:** Measure the variance of the WRS gradient estimator during training. High variance may necessitate the strict annealing schedules shown in Figure 4.
3. **Hyperparameter Robustness:** Test sensitivity to annealing schedules and power term $p$. Performance collapse without them (Figure 4) suggests the method is fragile.

### Visualizations & Case Studies
1. **Mask Entropy Curves:** Plot distribution entropy over training steps. Demonstrates whether the relaxation successfully converges to discrete binary masks.
2. **Layer-wise Sparsity Heatmaps:** Visualize learned sparsity patterns across Attention vs. MLP layers. Reveals if the method learns meaningful structural importance or uniform patterns.
3. **Efficiency Pareto Frontiers:** Plot Accuracy vs. Trainable Parameters and Accuracy vs. Latency. Visually demonstrates the trade-off advantages over MaskLLM beyond tabular data.

### Obvious Next Steps
1. **Post-Pruning Fine-Tuning:** Include results with lightweight fine-tuning (e.g., LoRA). Freezing weights limits recovery potential compared to standards in the field.
2. **Combined Quantization Evaluation:** Evaluate SUSI alongside 4-bit/8-bit quantization. Practical deployment typically requires combined compression techniques.
3. **Comparison with Structured Pruning:** Benchmark against methods like SliceGPT. Related work mentions structured pruning but excludes it from empirical comparison.

# Final Consolidated Review
## Summary

SUSI proposes a semi-structured pruning method for LLMs that reduces the trainable parameter complexity of learnable N:M sparsity masks from O(C(M,N)) to O(M) per weight group by reformulating mask selection as differentiable subset sampling via Gumbel-Top-K. The key insight is that instead of learning a categorical distribution over all C(M,N) possible sparsity patterns (as in MaskLLM), SUSI learns importance scores over M individual weights and samples N weights sequentially without replacement. Experiments on OPT models (125M to 1.3B) demonstrate competitive perplexity and zero-shot accuracy compared to SparseGPT, Wanda, and MaskLLM while enabling pruning at aggressive sparsity ratios where MaskLLM becomes infeasible.

## Strengths

- **Genuine parameter efficiency contribution:** The core technical insight — reformulating N:M mask learning as sequential sampling without replacement rather than maintaining a full categorical distribution — provides exponential reduction in trainable parameters from O(C(M,N)) to O(M). This is mathematically sound (Theorem 1) and practically meaningful; for 2:4 sparsity, SUSI uses 1.5× fewer parameters than MaskLLM, and for 4:8, MaskLLM becomes completely infeasible while SUSI remains tractable (Table 7).

- **Enables aggressive sparsity patterns:** SUSI successfully trains masks for 4:8 sparsity where MaskLLM cannot run due to parameter explosion. The paper notes that MaskLLM would require ~8.75× the original model parameters for 4:8, making it impractical. SUSI's linear scaling in M makes previously inaccessible sparsity ratios feasible.

- **Robustness across random seeds:** The mask overlap analysis (Figure 5) shows high consistency across seeds (88%, 83%, 94% overlap for different layer types), suggesting the learned masks are stable and not artifacts of particular initializations.

## Weaknesses

- **Critical power-term modification breaks theoretical correspondence:** Equation 11 modifies the standard Gumbel-Top-K update rule to α^(k) = α^(k−1) − |log(1 − μ^(k−1))|^p with p=3. For p=1, this matches standard Gumbel-Top-K; for p>1, it deviates significantly. The ablation (Figure 4) reveals the method essentially fails without this modification (PPL 998.33 with p=1 vs. 28.05 with p=3). However, Theorem 1 is derived under standard Gumbel-Top-K sampling and does not account for the power term. This creates a gap between the theoretical justification and the empirical algorithm that the paper never addresses. The method works in practice but the theory does not explain why.

- **Algorithm 1 contradicts Equation 11:** Line 9 of Algorithm 1 (Appendix A.4) implements the vanilla update `α^(k+1) ← α^(k) + log(1 − μ^(k))`, which lacks both the absolute value and power term of Equation 11. This inconsistency leaves unclear whether the theoretical analysis, the algorithm, or the empirical implementation corresponds to the actual method.

- **Marginal and inconsistent gains over MaskLLM:** The improvements over MaskLLM on 2:4 sparsity are consistently small (PPL 50.24 vs. 50.91, 54.14 vs. 55.86, 28.05 vs. 28.56). More concerningly, under 2:8 sparsity (Table 6), SUSI underperforms MaskLLM in average accuracy on both OPT-125M (37.22 vs. 37.27) and OPT-350M (35.22 vs. 35.91), directly contradicting claims of consistent improvement. No significance testing is reported.

- **No wall-clock training time comparison despite efficiency claims:** The paper's central claim is parameter efficiency, yet no GPU-hours or training wall-clock times are measured. For 2:4 sparsity, the dominant computational cost is the frozen LLM forward/backward pass, not the mask parameters. The actual speedup from 1.5× parameter reduction may be negligible in practice.

- **Evaluation limited to small models:** The primary experiments cover only OPT-125M, 350M, and 1.3B. Appendix A.8 includes Qwen2.5-0.5B and Llama3.2-1B but shows substantially larger degradation than OPT (e.g., Qwen accuracy drops from 55% to 44%). No evaluation on ≥7B models where semi-structured pruning is most practically relevant, and no comparison to MaskLLM on modern architectures.

- **Missing directly relevant baseline:** AST (Huang et al., 2025) is cited as a concurrent semi-structured pruning approach that also addresses MaskLLM's overhead, but it is never included in experiments. This omission is notable given the explicit acknowledgment in related work.

## Nice-to-Haves

- Evaluation on 7B+ models to demonstrate scalability to production-scale LLMs
- Wall-clock training time and inference latency measurements to substantiate efficiency claims
- Sensitivity analysis for the two temperature hyperparameters (τ and λ) beyond the binary on/off ablation
- Quantification of gradient variance during training to understand why the power term is necessary
- Combined evaluation with quantization (e.g., INT4/INT8) as practical deployment typically requires stacked compression

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh critic's discussion numbering error (missing point ii):** The numbering inconsistency is a minor editorial issue that does not affect technical validity.

- **Harsh critic's claim that "no Monte Carlo samples ablation" is missing:** The Gumbel-Top-K relaxation is deterministic given the temperature; Monte Carlo sampling is not a hyperparameter in this framework.

- **Spark finder's calibration data ablation request:** The paper already shows results across different token budgets (Figure 3b), and 1B tokens is a reasonable calibration size for this methodology. This is not a core flaw.

- **Spark finder's request to compare with structured pruning methods like SliceGPT:** Structured pruning (removing entire layers/heads) is a fundamentally different problem from semi-structured N:M sparsity. The paper correctly scopes its contribution to semi-structured pruning.

- **Harsh critic's demand for significance testing on marginal improvements:** While valid in principle, single-run evaluation without variance estimates is standard practice in this area (e.g., Wanda, SparseGPT, MaskLLM all report single runs). The burden should not be higher for this paper.

## Novel Insights

The mask overlap analysis (Figure 5, Appendix A.6) reveals an interesting finding: SUSI's learned masks achieve substantially higher cross-seed similarity (82-94%) compared to one-shot methods like Wanda (66%) and SparseGPT (75%). This suggests that learnable mask methods converge toward a consistent subset of important weights, while importance-score heuristics produce more variable patterns. The correlation between mask stability and downstream performance (higher overlap → better PPL/accuracy) is a valuable observation that could inform future pruning research: stable mask convergence may be a diagnostic for mask quality, not just an implementation detail.

## Suggestions

- Resolve the Algorithm 1 / Equation 11 inconsistency explicitly. Either update Algorithm 1 to match Equation 11, or clarify if two variants exist and which corresponds to the reported results.

- Add a brief theoretical discussion of why the power term p>1 is necessary despite the theoretical framework assuming p=1. This could be framed as an empirical finding that the standard Gumbel-Top-K relaxation is insufficiently sharp for this application.

- Report at least one experiment on a 7B-scale model to establish that the approach scales to relevant deployment sizes.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
