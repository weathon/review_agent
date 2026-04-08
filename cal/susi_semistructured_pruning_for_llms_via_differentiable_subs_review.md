=== CALIBRATION EXAMPLE 11 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title accurately conveys the core technical contribution. The abstract makes several well-grounded claims: parameter reduction of up to 1.5× for 2:4 sparsity, outperformance on WikiText-2 perplexity, and competitive zero-shot accuracy. However, the phrase "minimal computational cost" is vague — the paper reports parameter counts but never measures actual training wall-clock time or GPU memory footprint during training, making this claim impossible to verify from the experimental evidence. The stated 1.5× parameter reduction for the headline 2:4 case (the one most relevant to deployed hardware) is modest and does not obviously justify a new method over MaskLLM.

---

### Introduction & Motivation

The motivation is clear and focused. The central claim — that MaskLLM requires C(M, N) parameters per group while SUSI requires only M — is correct: for 2:4, that is 6 vs. 4 parameters (1.5×); for 4:8 it is 70 vs. 8 (8.75×). The paper correctly notes that the advantage grows significantly with M and when N ≈ M/2.

One concern: the related work mentions AST (Huang et al., 2025), a recent semi-structured adaptive sparse training method, but it is never included as a baseline in any experiment table. For a paper submitted to ICLR in 2026, omitting a directly competing 2025 method is a noticeable gap.

---

### Method (Section 3)

**Theoretical grounding of WRS parameterization (Theorem 1):** The equivalence result is sound at a high level — parameterizing an N-hot mask as a sum of elements drawn from the WRS distribution yields the same expected loss as sampling from the exact categorical distribution over all C(M,N) configurations. This is a meaningful theoretical underpinning for the parameter reduction.

**The power term modification (Eq. 11):** This is the most technically concerning aspect of the paper. The standard Gumbel-Top-K update rule (Eq. 3 in Section 2.2) is:
$$\alpha^{(k)} := \alpha^{(k-1)} + \log(1 - \mu^{(k-1)})$$
SUSI modifies this to:
$$\alpha^{(k)}_i := \alpha^{(k-1)}_i - |\log(1 - \mu^{(k-1)}_i)|^p$$

This modification breaks the mathematical correspondence between Gumbel-Top-K and WRS sampling. The Gumbel-Top-K trick is an exact reparameterization of WRS only when the update follows the standard log-sum form. By inserting an absolute value and a power term p > 1, the authors introduce a heuristic departure from the theoretically justified procedure. Theorem 1 is derived for the unmodified WRS distribution, but the actual training algorithm uses the power-term variant. **The paper never proves or claims that the modified update rule still corresponds to any valid distribution**, which creates a gap between the theory and the practice. The ablation validates the empirical benefit of p=3 over p=1, but does not address the theoretical inconsistency.

**Temperature annealing (Eq. 13):** Two separate annealing schedules are used (τ and λ), both linearly decaying. The ablation demonstrates annealing is essential (removing it causes divergence), but there is no analysis of sensitivity to the end values, schedules (e.g., cosine vs. linear), or the interaction between the two temperatures. With p, τ, τ_init, τ_end, λ, λ_init, λ_end as hyperparameters, the method has considerable tuning complexity that the paper does not discuss.

---

### Experiments & Results

**Model scope (major weakness):** All primary experiments are on OPT models (125M, 350M, 1.3B). These are substantially smaller and older than what ICLR readers expect for LLM compression work in 2026. MaskLLM (the closest prior work, published at NeurIPS 2024) was evaluated on LLaMA-2-7B and larger. The authors include a brief appendix (A.8) extending to Qwen2.5-0.5B and Llama3.2-1B, but Table 8 appears incomplete (the actual numeric values are missing from the extracted text), and more importantly, these are still sub-2B models. The absence of ≥7B model evaluation is a critical gap: it is precisely at large scales that parameter overhead becomes prohibitive and where SUSI's advantage should be most visible.

**Magnitude of improvements (2:4 sparsity):** The perplexity gains over MaskLLM are marginal — 50.24 vs. 50.91 on OPT-125M, 54.14 vs. 55.86 on OPT-350M, 28.05 vs. 28.56 on OPT-1.3B. The zero-shot accuracy differences are similarly small (e.g., 41.06% vs. 40.65% average for OPT-125M). These are within 1% or less on almost every metric. While the claim is primarily about parameter efficiency rather than accuracy, the paper leads with accuracy claims throughout.

**2:8 sparsity results (Table 6):** Under this setting, SUSI is actually slightly *worse* than MaskLLM on both OPT-125M (37.22% vs. 37.27% avg. accuracy) and OPT-350M (35.22% vs. 35.91%). The paper does not discuss this reversal.

**4:8 sparsity (Table 7, most compelling result):** Here MaskLLM cannot run at all due to excessive parameters, while SUSI runs successfully and clearly outperforms all baselines. This is genuinely the strongest argument for the method, but it is buried in the appendix.

**Missing training efficiency measurements:** The central claim is that SUSI reduces computational overhead, but the paper never reports: (1) actual GPU memory usage during training, (2) training wall-clock time, or (3) peak memory vs. MaskLLM. The parameter count comparison is a proxy, not a direct measurement. For example, 1.5× fewer parameters does not automatically mean 1.5× less memory if activations, gradients, and optimizer states dominate.

**No comparison to AST:** Huang et al. (2025) is cited as a related method but never compared against, even in Table 1 or Table 2. This is an unexplained omission.

**Ablation scope:** The ablation varies p ∈ {1, 2, 3} and annealing on/off, tested only on OPT-350M. No ablation is provided on the choice of M parameters (batch size, sequence length sensitivity), the annealing schedule shape, or the learning rate schedule. The effect of varying τ and λ endpoints is not explored.

**No statistical significance:** Results are reported as point estimates. While the robustness section (4.3.3) shows seed stability for masks, it does not report variance of the final perplexity or zero-shot accuracy across seeds.

---

### Writing & Clarity

The items ii and iii in Section 4.2 are both labeled "(iii)" — item "(ii)" is missing from the main text (it appears as a floating fragment in the table area due to parsing, but suggests a layout problem in the original paper). The Related Work section is placed after the experiments (Section 4.4), which is unusual and may confuse readers expecting it before the method. These are organization choices that impede flow.

---

### Limitations & Broader Impact

The limitations section (A.9) is candid and reasonable: hardware dependency for inference speedup, focus on English-centric OPT models, limited benchmark diversity. However, it does not acknowledge: (1) the theoretical gap introduced by the power-term modification; (2) the failure to scale evaluations to models >1.3B; (3) the sensitivity to the many temperature hyperparameters; or (4) that the method does not jointly update weights (frozen model), which may systematically limit quality compared to methods that do weight updates (e.g., SparseGPT).

---

### Overall Assessment

SUSI presents a technically interesting idea — using Weighted Reservoir Sampling and Gumbel-Top-K to reduce the parameter overhead of learnable N:M sparsity masks from O(C(M,N)) to O(M). The theoretical foundation (Theorem 1) is sound for the unmodified WRS parameterization, and the 4:8 sparsity results, where MaskLLM cannot run at all, represent the method's most compelling practical contribution. However, the paper has several significant weaknesses that collectively fall short of ICLR's standard for LLM compression work. First and most critically, all primary experiments use OPT models up to 1.3B, far below what the field expects in 2026 (≥7B); the parameter efficiency advantage is precisely most valuable at scale, yet this is where the method is not evaluated. Second, the power-term modification to Gumbel-Top-K (Eq. 11) breaks the mathematical correspondence to WRS without any compensating theoretical analysis, creating an inconsistency between the stated theory and the actual algorithm. Third, the headline 2:4 accuracy improvements over MaskLLM are marginal (<1%) and no training-time memory measurements are provided to substantiate the efficiency claim. Fourth, the directly relevant AST baseline (Huang et al., 2025) is absent from all comparisons. The paper addresses a real problem and the core idea is sound, but the experimental validation is insufficient for acceptance in its current form.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces SUSI, a semi-structured pruning framework for LLMs that learns N:M sparsity masks via differentiable subset sampling using Weighted Reservoir Sampling (WRS) and Gumbel-Top-K relaxation. By formulating mask selection as a sequential sampling-without-replacement process rather than a full categorical distribution over all valid combinations, SUSI reduces the number of trainable mask parameters from a combinatorial scale to $O(M)$ per group. Empirical evaluation on OPT models (125M–1.3B) demonstrates that SUSI achieves competitive perplexity and zero-shot accuracy compared to importance-based and learnable baselines, while offering substantially improved parameter and data efficiency.

### Strengths
1. **Meaningful Parameter Efficiency for Learnable Masks:** The core contribution effectively reduces trainable parameters for N:M mask learning. As shown in Figure 3(a) and Section 3.2, SUSI requires $M$ parameters per group instead of the combinatorial $\binom{M}{N}$ required by methods like MaskLLM. This enables tractable optimization for aggressive patterns (e.g., 2:8, 4:8) where baselines become infeasible due to memory constraints (Table 7).
2. **Well-Justified Methodological Components & Ablations:** The integration of the power term $p$ and linear temperature annealing is rigorously validated. The ablation study (Figure 4) clearly demonstrates that both mechanisms are necessary: removing annealing causes training divergence, while increasing $p$ significantly accelerates convergence and lowers final perplexity.
3. **Comprehensive Evaluation & Strong Reproducibility Package:** The paper evaluates multiple dimensions of performance (perplexity, 5 zero-shot tasks, cross-seed robustness in Figure 5) and extends to newer architectures (Qwen2.5, Llama3.2 in Appendix A.8). The reproducibility commitment is strong, with detailed hyperparameter tables (Appendix A.3), algorithm pseudocode (Appendix A.4), and a public code repository containing training and evaluation scripts.

### Weaknesses
1. **Limited Model Scale for Contemporary Relevance:** The primary experiments are confined to the OPT family up to 1.3B parameters. While Appendix A.8 includes 0.5B and 1B Qwen/Llama variants, ICLR typically expects validation on 7B+ scale to demonstrate that the method addresses real-world compression bottlenecks. The performance gap to dense models also widens noticeably on these newer architectures (e.g., Qwen2.5 drops from 55.33% to 43.75% accuracy).
2. **Marginal Quality Gains Over Strong Baselines:** SUSI’s primary advantage is memory efficiency, not model quality. Performance improvements over MaskLLM are often marginal (e.g., OPT-137M accuracy: 41.06% vs 40.65%; OPT-1.3B PPL: 28.05 vs 28.56 in Table 1/2). The paper should more explicitly frame its contribution as an efficiency enabler rather than a quality booster.
3. **Lack of Deployment & Inference Efficiency Metrics:** Despite emphasizing hardware-compatible N:M sparsity, the paper reports only training parameter counts and accuracy/perplexity. There is no measurement of actual inference latency, throughput, or VRAM reduction on supported accelerators (e.g., A100 with native 2:4 kernels), leaving the "practical solution" claim partially unverified.
4. **Incremental Theoretical Novelty over Existing Relaxations:** The method relies heavily on standard Gumbel-Top-K reparameterization with temperature annealing. The connection to WRS (Theorem 1) proves mathematical equivalence between the exact categorical mask distribution and sequential WRS sampling, which, while correct, closely mirrors the underlying mechanics of existing differentiable subset sampling literature.

### Novelty & Significance
**Novelty:** Moderate. The paper adapts well-established Gumbel-Top-K differentiable relaxation to the specific problem of semi-structured mask learning. The WRS framing is mathematically sound but primarily serves to justify the $O(M)$ parameter reduction rather than introducing a fundamentally new optimization paradigm.  
**Clarity:** High. The paper is logically structured, with clear preliminaries, a concise problem formulation, and effective use of figures to contrast memory overheads and ablation dynamics. Notation is consistent, and the algorithmic flow is easy to follow.  
**Reproducibility:** High. Standard datasets (C4, WikiText-2, LM-Eval harness), fully documented hyperparameters, pseudocode, and a released anonymous codebase meet and exceed ICLR reproducibility expectations.  
**Significance:** Medium. The reduction in learnable parameters addresses a genuine bottleneck in pruning-at-scale research, making aggressive sparsity patterns computationally feasible. However, the significance is tempered by the lack of 7B+ experiments, marginal accuracy gains, and absent deployment benchmarks. For ICLR acceptance, stronger evidence of scalability and hardware-level efficiency is typically required.

### Suggestions for Improvement
1. **Scale to Contemporary LLM Sizes:** Include at least one 7B model (e.g., Llama-3-8B or Qwen2.5-7B) in the main text or a dedicated extended experiment section to prove that the $O(M)$ memory savings translate to tractable fine-tuning on current hardware.
2. **Report Inference Deployment Metrics:** Measure and report actual inference throughput (tokens/sec), latency, and peak VRAM usage on a GPU with native 2:4 sparsity support (e.g., NVIDIA A100/H100 using TensorRT-LLM or vLLM sparse kernels). This will validate the practical deployment claims.
3. **Analyze Computational Trade-offs vs. One-Shot Methods:** Provide a clear FLOP and training-time comparison between SUSI (which requires forward/backward passes for mask optimization) and one-shot methods like Wanda/SparseGPT. Clarify when the accuracy trade-off justifies the additional optimization cost.
4. **Provide Layer-Wise or Module-Wise Sparsity Insights:** Analyze *where* the model prunes (e.g., attention vs. MLP, layer depth distribution). Correlating learned mask patterns with layer importance could offer deeper mechanistic insights and strengthen the paper beyond parameter counts.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Equalized Data Budget Comparison:** Compare SUSI against Wanda/SparseGPT using identical calibration data sizes (e.g., 128 samples vs. 1B tokens); the current 1B token advantage likely skews performance results unfairly, undermining the claim that SUSI outperforms one-shot methods due to method superiority rather than data volume.
2. **Scaling to 7B+ Models:** Evaluate on at least one 7B parameter model (e.g., Llama-2-7B or OPT-6.7B); OPT-1.3B is insufficient to validate scalability claims for modern LLMs at ICLR standards, where 1B-scale results are often considered toy experiments.
3. **Inference Latency Measurements:** Add end-to-end inference latency benchmarks on NVIDIA Ampere GPUs using sparse kernels; without actual speedup metrics, the claim of "efficient deployment on hardware optimized for sparse computation" is unverified and theoretical.
4. **Wall-Clock Training Time:** Report actual GPU hours required for mask learning compared to MaskLLM; fewer parameters do not guarantee faster training if the differentiable sampling operation incurs significant computational overhead per step.

### Deeper Analysis Needed (top 3-5 only)
1. **Isolating WRS Contribution:** Ablate the Weighted Reservoir Sampling component specifically against standard Gumbel-Top-K; it is unclear if the performance gain comes from the novel WRS formulation or merely the existing relaxation mechanism.
2. **Peak VRAM Usage Analysis:** Report peak memory consumption during training, not just parameter counts; differentiable sampling often incurs high activation memory overhead that parameter counts miss, directly contradicting the "memory efficient" claim if VRAM spikes.
3. **Hyperparameter Sensitivity:** Quantify sensitivity to the annealing schedule and power term $p$ across different model sizes; the noted divergence without annealing suggests high fragility that needs robustness bounds to be trustworthy.
4. **Mask Importance Correlation:** Correlate learned masks with Hessian-based importance scores from SparseGPT; this validates whether SUSI learns meaningful sparsity patterns or simply fits calibration noise during the 1B token training phase.

### Visualizations & Case Studies
1. **Speedup vs. Sparsity Curve:** Plot actual inference throughput gain against sparsity ratios on supported hardware to visualize practical utility rather than theoretical sparsity.
2. **Cost-Performance Pareto Frontier:** Plot Perplexity vs. Training GPU Hours to show if SUSI is truly more efficient than MaskLLM or just parameter-light.
3. **Layer-wise Mask Heatmaps:** Visualize learned sparsity patterns across layers compared to magnitude pruning to reveal structural differences in what is being preserved versus pruned.

### Obvious Next Steps
1. **Integrate Sparse Kernels:** Integrate with cuSPARSE or FlashDecoding to measure real-world speedups rather than relying on theoretical N:M sparsity support.
2. **Data Efficiency Sweep:** Conduct a study varying calibration data from 128 samples to 1B tokens to identify the minimum data required for competitive performance against one-shot methods.
3. **Modern Architecture Validation:** Extend experiments to Llama-3 8B or Qwen-2.5 7B to ensure compatibility with modern RoPE/SwiGLU architectures beyond the outdated OPT family.

# Final Consolidated Review
## Summary

SUSI proposes a semi-structured pruning method for LLMs that learns N:M sparsity masks via differentiable subset sampling using Weighted Reservoir Sampling and Gumbel-Top-K relaxation. The key innovation is reducing trainable mask parameters from O(C(M,N)) to O(M) per weight group, enabling efficient mask learning for aggressive sparsity patterns where prior methods become computationally infeasible.

## Strengths

- **Principled parameter efficiency for learnable masks:** The paper correctly identifies that MaskLLM requires C(M,N) parameters per weight group, which grows combinatorially. SUSI's reformulation using sequential sampling without replacement reduces this to M parameters per group. For 4:8 sparsity, this yields an 8.75× reduction (70 vs. 8 parameters), which is verified experimentally (Figure 3a, Table 7).

- **4:8 sparsity results demonstrate practical feasibility:** The most compelling contribution appears in Table 7, where MaskLLM fails to execute due to excessive trainable parameters, while SUSI completes training and outperforms one-shot baselines. This validates the core efficiency claim where it matters most—aggressive sparsity patterns that were previously inaccessible to learnable mask methods.

- **Theoretical grounding via WRS equivalence (Theorem 1):** The paper proves that parameterizing masks as sums of elements from WRS-sampled subsets yields equivalent expected loss to the full categorical distribution over all valid configurations, providing formal justification for the parameter reduction.

- **Rigorous ablation validates design choices:** Figure 4 demonstrates that the power term (p=3) and temperature annealing are both necessary—removing annealing causes divergence, while smaller p values yield substantially worse perplexity (998.33 vs. 28.05 for p=1 vs. p=3 on OPT-350M).

- **Strong reproducibility:** The paper provides detailed hyperparameters (Appendix A.3), algorithm pseudocode (Appendix A.4), anonymous code repository, and evaluates on standard benchmarks with LM-Evaluation-Harness.

## Weaknesses

- **Limited model scale below contemporary standards:** Primary experiments use OPT-125M, 350M, and 1.3B, with extensions to Qwen2.5-0.5B and Llama3.2-1B in the appendix. The parameter efficiency advantage is most valuable at scale, yet SUSI is not evaluated on ≥7B models where MaskLLM's overhead would be prohibitive. This limits confidence in real-world applicability.

- **Marginal quality improvements over strong baselines:** On 2:4 sparsity, SUSI's perplexity improvements over MaskLLM are small (50.24 vs. 50.91 on OPT-125M; 54.14 vs. 55.86 on OPT-350M; 28.05 vs. 28.56 on OPT-1.3B). Zero-shot accuracy differences are similarly within ~1%. The paper should explicitly frame its contribution as efficiency-driven rather than quality-driven.

- **No actual efficiency measurements:** Despite claiming "minimal computational cost" and "efficient deployment," the paper reports only parameter counts—not GPU memory consumption during training, wall-clock time, or inference throughput on sparse kernels. The parameter reduction proxy does not account for activation memory, optimizer states, or computational overhead of the differentiable sampling procedure.

- **Missing hardware inference validation:** The paper emphasizes hardware-compatible N:M sparsity but provides no measurements of actual inference latency or throughput on GPUs with sparse kernel support (e.g., A100 2:4 sparsity). Without speedup benchmarks, the "efficient deployment" claim remains theoretical.

- **Underperformance on 2:8 sparsity:** Table 6 shows SUSI marginally underperforms MaskLLM on 2:8 sparsity (37.22% vs. 37.27% on OPT-125M; 35.22% vs. 35.91% on OPT-350M). The paper should discuss why SUSI's efficiency advantage doesn't translate to consistent quality gains across all sparsity patterns.

- **Power term modification lacks theoretical justification:** The modified update rule (Eq. 11) with power term p > 1 departs from the standard Gumbel-Top-K formulation. While ablation shows p=3 improves results, there is no theoretical analysis of why this modification preserves or enhances the WRS correspondence established in Theorem 1, creating a gap between theory and practice.

- **Missing relevant baseline:** AST (Huang et al., 2025) is cited in related work as a recent semi-structured adaptive sparse training method but is not included in experimental comparisons, despite being directly relevant to the paper's contribution space.

## Nice-to-Haves

- Inference throughput benchmarks on NVIDIA Ampere/Hopper GPUs with native 2:4 sparse kernels to validate real-world deployment benefits.

- Analysis of how calibration data volume affects SUSI compared to one-shot methods, since the 1B token budget significantly exceeds typical one-shot calibration requirements.

- Layer-wise mask visualization to reveal structural patterns in learned sparsity and provide mechanistic insight into what SUSI preserves versus prunes.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Formatting nitpick about missing item (ii):** The extracted text shows "(ii) Scalability across Model Sizes" appearing correctly in the paper content. The apparent duplication of "(iii)" labels is a PDF extraction artifact, not a paper error.

- **Demand for 7B+ models as acceptance threshold:** While scaling experiments would strengthen the paper, the current evaluation on 3 model sizes plus 2 additional architectures (Qwen/Llama) provides sufficient evidence for the core efficiency claim. The reviewer conflates the paper's scope with reviewer preferences for certain model scales.

- **Request for layer-wise importance correlation with SparseGPT:** This would be interesting but is not required for the paper's stated contribution of efficient mask learning. The method optimizes masks end-to-end via calibration loss, not by matching heuristic importance scores.

- **Claim that "minimal computational cost" is completely unsubstantiated:** Parameter reduction is meaningful evidence for efficiency. While actual timing measurements would strengthen the claim, the parameter count comparison is not "impossible to verify"—it is directly observable.

## Novel Insights

The reformulation of N:M mask learning via Weighted Reservoir Sampling reveals an overlooked opportunity in the differentiable subset sampling literature: rather than maintaining full categorical distributions over combinatorial mask spaces, one can sample indices sequentially without replacement. This insight—that mask selection admits a natural "sampling without replacement" formulation—could potentially extend beyond N:M sparsity to other combinatorial optimization problems in neural network compression where the search space explodes. The empirical finding that a modified power term improves convergence while diverging from the theoretical formulation suggests that the optimal training dynamics for differentiable pruning may not perfectly align with the theoretically derived sampling distributions, warranting further investigation into what objective the power term implicitly optimizes.

## Suggestions

- Add wall-clock training time and peak GPU memory measurements comparing SUSI against MaskLLM to substantiate efficiency claims with actual resource consumption.

- Evaluate on at least one ≥3B model to demonstrate scalability; consider Llama-3-8B or Qwen2.5-7B with RoPE/SwiGLU architectures to validate beyond the older OPT family.

- Discuss the 2:8 underperformance explicitly and provide hypothesis (e.g., whether the relative sparsity level affects the quality of the gradient signal for differentiable sampling).

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
