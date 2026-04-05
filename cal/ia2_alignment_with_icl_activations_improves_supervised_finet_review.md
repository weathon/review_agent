=== CALIBRATION EXAMPLE 77 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Title accuracy:** The title accurately reflects the contribution: aligning SFT with ICL activations to improve performance.
- **Abstract clarity:** The abstract clearly outlines the problem (SFT's data hunger and miscalibration vs. ICL's quality), the method (IA2 priming step to mimic ICL internal states), and the key results (consistent gains in accuracy and calibration across 12 benchmarks). 
- **Claims:** The claims are well-supported by the body of the paper. One minor caveat: the abstract mentions "calibrated responses" broadly, but the authors correctly restrict calibration claims (ECE) to single-token tasks in the main text due to the difficulty of measuring confidence in open-ended generation. This is an appropriate scope limitation, not an unsupported claim.

### Introduction & Motivation
- **Motivation & Gap:** The motivation is well-grounded. The contrast between parameter-efficient SFT (cheap inference, but prone to shortcut learning and poor calibration in low-data regimes) vs. ICL (sample-efficient, calibrated, but compute-heavy at inference) is clearly articulated. The identification of the "functional mechanism" gap via activation divergence (Section 3) effectively motivates the core research question.
- **Contributions:** The contributions are clearly stated and accurate. The shift from output-only distillation to internal activation alignment as a *priming* step is a clear conceptual distinction.
- **Claims check:** The introduction does not appear to over-claim. It carefully frames the hypothesis that ICL activations encode "generalizable patterns" and treats the method as a way to transfer this behavior. The claim of a "conceptual window into inner mechanics" is justified by the activation similarity and subspace overlap analyses in later sections.

### Method / Approach
- **Clarity & Reproducibility:** The two-stage pipeline (IA2 convergence followed by standard SFT) is straightforward and reproducible. Equations 3 and 4 clearly define the objective. The use of self-generated ICL responses (`Y^`) ensures no external teacher is needed.
- **Key Assumptions & Logical Gaps:** 
  - **Contextual Mismatch in Attention:** A crucial technical detail requires clarification. Equation 1 shows that `SA(T)` depends on `K` and `V`, which are context-dependent. In ICL, the attention heads for response tokens attend to in-context demonstrations (`I`); in the SFT model, those keys/values do not exist. How are activations at "output token positions" aligned given this fundamental difference in attention masks and history? Simply matching `A_ICL[response tokens]` to `A_SFT[response tokens]` implicitly asks the SFT model to hallucinate ICL-like internal states without the context. The paper should clarify if this is intended as distilling the *outcome* of context integration, and whether positional embeddings or causal masking differences affect the MSE calculation.
  - **Activation Scaling/Normalization:** MSE loss on raw hidden states is highly sensitive to layer-wise activation magnitudes, which vary significantly across transformer layers and models. Did the authors apply layer-wise normalization or weighting? The `β` scaling for joint loss (Section 6) hints at magnitude mismatches, but pure IA2 training (Eq 4) assumes raw MSE is stable across layers. This should be stated for reproducibility.
  - **Self-Distillation Target Quality:** The method relies on `Y^`, the model's own ICL-generated response. If ICL hallucinates or makes an error on a sample, `IA2` actively aligns the SFT model to these potentially faulty functional pathways. The paper mentions ICL overfitting (Figure 6), but does not discuss how `IA2` is affected by noisy self-generated targets compared to alignment against ground-truth activations (if available).
- **Edge Cases:** The method fixes `G=200` for activation collection in multi-token tasks. For samples with shorter/longer ICL responses, how are truncation or padding handled in the activation tensor? Variable-length sequence alignment introduces boundary effects in `L_IA2`.

### Experiments & Results
- **Claim Testing:** The experiments rigorously test the primary claims. The few-shot regime (`N` up to 64/128 for single-token, up to 16 for multi-token) directly addresses the data-scarce setting where SFT typically struggles.
- **Baselines & Fairness:** Baselines are appropriate and comprehensive: ICL (upper bound), SFT only, IA2 only, IA2+SFT (joint optimization), and a soft-label KD baseline. Using the exact same data splits and `N-1` samples for ICL construction ensures fairness.
- **Missing Ablations:** 
  - **Layer Selection:** Figure 2 shows that ICL/SFT activations naturally align in initial/final layers but diverge in middle layers. Did the authors test *selective* alignment (only middle layers) to save compute or avoid disrupting stable embeddings? The discussion mentions this as future work (Section 6), but an ablation here would strengthen the claim about *where* ICL-like behavior resides.
  - **Module Targeting:** Table 7 ablates LoRA ranks/targets, but all experiments use LoRA. Does `IA2` transfer to full fine-tuning or adapter-only methods beyond `(IA)³`? The authors note resource constraints, which is acceptable for ICLR, but limits the generality of the claim.
- **Statistical Significance:** The inclusion of 5 random seeds, standard deviations, and paired t-tests (Table 8) is excellent practice. The results are robust.
- **Datasets/Metrics:** The mix of single-token (classification) and multi-token (generation/math) is appropriate. Restricting ECE to single-token tasks is methodologically sound. Parsing rules for math/QA are clearly defined in Appendix A.
- **Cherry-picking:** The paper consistently reports `IA2 → SFT` outperforms SFT across diverse tasks, with statistical backing. Aggregate plots (Figure 1) and detailed tables (Appendix C) show the trend holds broadly, not just in hand-picked cases.

### Writing & Clarity
- **Clarity:** The paper is well-structured and generally clear. The distinction between functional alignment (activations) and output alignment (tokens) is maintained throughout.
- **Figures/Tables:** 
  - Figure 2 effectively visualizes the activation similarity gap and how `IA2` bridges it.
  - Figure 4 (Subspace Overlap) is particularly insightful, showing that `IA2 → SFT` occupies a weight subspace orthogonal to `SFT only`, which strongly supports the "different functional mechanisms" claim.
  - *Note:* The provided text contains parser artifacts (e.g., duplicated Figure 2 captions, split table cells in Table 1). I am ignoring these as per instructions, but advise the authors to verify table layout in the final PDF, as Table 1 is difficult to parse in the raw text.

### Limitations & Broader Impact
- **Acknowledged Limitations:** The authors honestly discuss computational overhead (one-time ICL inference), ICL overfitting risks, LoRA rank limitations, and the gap remaining between `IA2 → SFT` and pure ICL in highly sample-efficient models.
- **Missed Limitations:** 
  - **Catastrophic Forgetting:** The introduction mentions specialized applications (e.g., news classification). While the paper focuses on few-shot adaptation, it does not measure if `IA2` priming harms the base model's performance on other unrelated tasks. Activation alignment might over-constrain the residual stream, potentially reducing the model's versatility. This is mentioned only as future work in the conclusion.
  - **SFT-only Catch-up:** Table 5 shows the gap between ICL and `IA2 → SFT` closes rapidly as `N` increases. At what point does `SFT only` match `IA2 → SFT`? If the benefit vanishes beyond `N=32` or `N=64`, the "practical utility" claim is restricted strictly to the extreme low-data regime.
- **Societal Impact:** Not explicitly discussed. However, as a general training improvement for LLMs, there are no immediate negative safety impacts identified. Efficiency gains could lower fine-tuning costs, which is a positive impact.

### Overall Assessment
This is a strong empirical paper that addresses a practical bottleneck in LLM adaptation: improving SFT quality in low-data regimes by leveraging the internal mechanics of ICL. The core insight—that ICL and SFT diverge in activation space despite potentially similar outputs, and that aligning these spaces improves SFT—is well-supported by extensive experiments across 12 datasets, multiple model families, and rigorous statistical testing. The methodology is simple, requiring no extra parameters or external data, making it highly practical for the community. The most significant concern is the technical clarification needed regarding how activation alignment reconciles the contextual mismatch in attention masks (ICL sees demos, SFT does not) and the lack of normalization details for the MSE loss. Additionally, while the subspace analysis provides some mechanistic insight, a deeper investigation into *what* specific features are being aligned (e.g., attention patterns vs. value representations) would elevate the paper's conceptual contribution. Despite these points, the results are robust and the contribution is clear and reproducible. **Recommendation: Weak Accept / Accept.**

# Neutral Reviewer
## Balanced Review

### Summary
The paper investigates the internal behavioral divergence between In-Context Learning (ICL) and Supervised Fine-Tuning (SFT), demonstrating that they follow distinct activation trajectories despite producing similar surface-level outputs. To bridge this gap, the authors propose IA2, a two-stage adaptation pipeline that first aligns a model's self-attention activations with those produced during ICL (via an MSE loss), followed by standard cross-entropy fine-tuning. Extensive experiments across 12 benchmarks and multiple model families show that IA2 priming consistently improves accuracy, calibration, and out-of-distribution generalization in low-data regimes.

### Strengths
1. **Extensive Empirical Validation & Rigor:** The authors train and evaluate over 13,000 models across 12 diverse tasks (single and multi-token), two model families (Qwen, Llama), and varying data regimes. Results are systematically reported across different $N$ values, accompanied by statistical significance testing (e.g., paired t-tests in Table 8), which aligns well with ICLR's emphasis on robust empirical claims.
2. **Mechanistic Diagnostic Analysis:** The paper moves beyond black-box metrics by analyzing activation similarity (`asim`) and LoRA weight subspace overlap. Figure 2 convincingly illustrates ICL/SFT misalignment, while Figure 4 shows that IA2→SFT models occupy a weight subspace (~39% overlap with IA2-only) largely unreachable by standard SFT. This provides a clear, interpretable rationale for why the method works.
3. **Simple, Practical, and Well-Integrated Methodology:** IA2 is straightforward to implement, requires no external teachers or synthetic data generation, and maintains identical inference costs to vanilla SFT. The sequential pipeline respects practical constraints, and the release of complete code, hyperparameters, and random seeds ensures high accessibility.
4. **Comprehensive Ablations & Baseline Comparisons:** The authors thoroughly compare IA2 against SFT-only, IA2-only, joint IA2+SFT training, and knowledge distillation on ICL soft labels. The consistent superiority of IA2→SFT in both accuracy and calibration, even when trained on identical data, strongly supports the central hypothesis that ICL activations contain a unique training signal.

### Weaknesses
1. **Theoretical & Mechanistic Grounding of Activation Matching:** The core premise assumes that minimizing MSE on self-attention outputs ($SA(I \circ X) \approx SA(X)$) transfers ICL's "functional behavior." However, attention outputs are highly non-linear and context-dependent. Matching them token-wise may enforce superficial representational alignment rather than true inductive bias transfer. The paper lacks deeper circuit-level or causal tracing analysis to verify *which* functional mechanisms are actually being captured.
2. **Under-Specified Computational Overhead:** While framed as a "one-time cost," collecting ICL activations requires a full forward pass for every training sample with $N$ demonstrations prepended, scaling noticeably with context length and $N$. The paper does not quantify this overhead (e.g., FLOPs, wall-clock time, or memory footprint relative to SFT), making it difficult for practitioners to assess the trade-off in resource-constrained settings.
3. **Multi-Token & Capacity Limitations Not Fully Explored:** On complex reasoning tasks (Table 5), IA2 sometimes underperforms ICL. The authors attribute this to suboptimal LoRA capacity and mid-training biases, but only report rank 8 ablations. Higher ranks or alternative PEFT methods are deferred to future work, leaving the claim that IA2 shifts models into an "inaccessible subspace" partially unverified for generation-heavy tasks. Additionally, joint training (IA2+SFT) is dismissed due to ICL/GT response length mismatches, which feels like an engineering workaround rather than a fundamental limitation.
4. **Baseline Coverage Relative to Recent Literature:** The related work cites several context distillation and activation steering papers, but direct empirical comparisons under identical data/compute constraints are missing. Relying primarily on self-designed baselines (soft-label KD, IA2+SFT joint) may overstate the relative novelty of the improvements.

### Novelty & Significance
**Novelty:** Moderate-to-High. Activation matching and representation alignment are established concepts, but applying sequential activation priming to transfer ICL's inductive biases into SFT weights is a fresh and well-motivated formulation. The specific use of attention outputs as a distillation signal for functional alignment, combined with the two-stage pipeline, distinguishes it from standard output-only distillation.
**Clarity:** High. The paper is logically structured, the method is clearly defined (Fig 1, Eq 4), and the progression from diagnostic analysis (why ICL $\neq$ SFT internally) to method proposal and empirical validation follows a coherent narrative. Minor notational inconsistencies in equations (likely parser artifacts) do not impede understanding.
**Reproducibility:** Strong. The authors provide a public code repository, explicit training loops, dataset construction details, hyperparameter ranges, LoRA configurations, and statistical testing procedures. The use of fixed random seeds and early-stopping criteria further ensures reliable reproduction.
**Significance:** High for ICLR. The work bridges mechanistic interpretability (activation subspace analysis, calibration behavior) with practical model adaptation. Given the widespread reliance on SFT and the computational cost of ICL at scale, a lightweight priming step that consistently improves low-data adaptation aligns well with ICLR's focus on principled, scalable learning methods.

### Suggestions for Improvement
1. **Quantify and Discuss Compute Trade-offs:** Add a subsection or table reporting the activation collection overhead (e.g., additional GPU hours/FLOPs relative to SFT-only) across different $N$ values and context lengths. This will help practitioners decide when IA2 is cost-effective.
2. **Deepen Mechanistic Verification:** Complement the `asim` and subspace overlap analyses with targeted diagnostic experiments, such as layer-wise ablation of IA2 or causal tracing/path patching, to demonstrate that IA2 actually restores ICL-like computation pathways (e.g., improved attention routing or gradient-like update emulation) rather than just matching vector magnitudes.
3. **Expand Capacity & Joint Training Experiments:** Test higher LoRA ranks (e.g., 16, 32) or alternative PEFT methods on the multi-token/math tasks to verify whether the ICL performance gap is strictly capacity-bound. Additionally, explore a curriculum-based or token-masked joint objective for IA2+SFT instead of fully abandoning joint training; this could yield a more unified optimization landscape.
4. **Strengthen External Baseline Comparisons:** Run side-by-side comparisons against recent state-of-the-art context distillation or activation-guided fine-tuning methods (e.g., those that compress demonstrations into continuous embeddings or learn adapter weights on-the-fly) under identical few-shot settings. This will better position IA2's contributions within the current literature and prevent claims from appearing inflated due to narrower baselines.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add an SFT-only baseline trained for the same total number of epochs/gradient steps as IA2→SFT because without controlling for optimization budget, the reported gains may stem from longer training rather than activation alignment.
2. Include a sequential "SFT on ICL tokens → SFT on GT tokens" baseline. Since IA2 requires autoregressively generating ICL responses to collect targets, this control isolates whether the MSE activation loss or simply intermediate self-distillation on the model’s own outputs drives the improvement.
3. Evaluate on models ≥7B parameters. Results on 1B–4B models risk attributing improvements to activation matching when they may reflect instability or limited capacity in smaller models; ICLR expects evidence of validity on standard-scale LLMs.

### Deeper Analysis Needed (top 3-5 only)
1. Provide a causal probe linking the identified weight subspace (Figure 4) to performance. High subspace overlap is correlational; demonstrate that surgically swapping IA2 LoRA weights into the SFT model transfers the gains to prove the subspace causally drives calibration/accuracy improvements.
2. Justify the exclusive focus on Self-Attention activations. Transformer layers process information via FFNs and residual connections as well; ablate by matching FFN outputs to prove attention alignment is the unique driver, ensuring the "functional mechanism" claim isn't just a proxy for generic representation matching.
3. Quantify the impact of ICL demonstration order on activation variance. ICL is highly order-sensitive; measure how permutation variance propagates to $A_{ICL}$ targets and whether IA2 suppresses or amplifies this noise to validate the claim that IA2 extracts a stable, rich signal.

### Visualizations & Case Studies
1. Provide qualitative case studies for instances where IA2→SFT underperforms SFT-only (e.g., Table 1 FinS accuracy drop). Show specific activation divergences and token probabilities in failure cases to demonstrate when and why the method introduces degradation.
2. Plot reliability diagrams for ICL, SFT, and IA2→SFT across single-token tasks. Scalar ECE hides calibration pathologies; curves will reveal if IA2 reduces specific overconfident modes or merely shifts global bias, which is essential for trusting the calibration claims.
3. Visualize representation shifts using PCA/t-SNE on the activation tensors before and after IA2. Showing that IA2 moves SFT representations toward a more structured or linearly separable manifold would concretely verify the "functional alignment" hypothesis rather than relying on layer-wise cosine similarity alone.

### Obvious Next Steps
1. Measure and report calibration metrics for multi-token reasoning tasks. Stating "confidence is hard to measure" undermines the paper’s core narrative on improved calibration; implement token-level entropy or verbalized probabilities to assess IA2’s impact on long-form reliability.
2. Conduct a sensitivity analysis on the IA2-to-SFT switching point. Since early vs. late switching likely affects the trade-off between representation alignment and target fitting, map performance across varying epoch ratios to establish robust practical guidelines.
3. Test IA2 on full fine-tuning beyond LoRA. As LoRA imposes low-rank constraints, results might not generalize to full-rank updates; validating on full fine-tuning determines if IA2 is a universal SFT improvement or a regularizer specific to parameter-efficient bottlenecks.

# Final Consolidated Review
## Summary
This paper investigates the internal representational divergence between In-Context Learning (ICL) and Supervised Fine-Tuning (SFT), demonstrating that the two adaptation paradigms follow distinct activation trajectories despite often producing similar surface-level outputs. To bridge this gap, the authors propose IA2, a sequential self-distillation pipeline that first aligns a model's self-attention activations with those produced during ICL, followed by standard cross-entropy fine-tuning. Extensive empirical evaluation across 12 benchmarks and multiple model families confirms that IA2 priming consistently improves accuracy, calibration, and robustness in low-data regimes, offering a practical, zero-inference-overhead enhancement to standard SFT.

## Strengths
- **Rigorous, large-scale empirical validation:** The study trains and evaluates over 13,000 models across diverse single- and multi-token tasks, varying shot counts, and multiple model families. Results are supported by standard deviations over multiple seeds, paired statistical significance testing (Table 8), and consistent out-of-distribution improvements, establishing a robust empirical foundation that exceeds typical single-dataset adaptation claims.
- **Interpretable mechanistic diagnostics:** Rather than treating the improvement as a black box, the paper bridges representation-level and weight-level analyses. The use of activation similarity (`asim`) and LoRA weight subspace overlap reveals that IA2→SFT occupies a functionally distinct parameter region (~39% overlap with IA2-only, nearly orthogonal to SFT-only in Figure 4). This provides clear, interpretable evidence that the priming step accesses a training signal fundamentally unreachable by standard SFT.
- **Practical, scalable, and well-baselined methodology:** IA2 requires no external teachers, synthetic data, or architectural modifications, maintaining identical inference costs to vanilla SFT. The authors systematically compare against strong alternatives (ICL upper bounds, joint IA2+SFT optimization, soft-label KD, and alternative PEFT methods), demonstrating that isolated activation priming yields more stable accuracy and calibration gains than joint objectives or output-only matching.

## Weaknesses
- **Optimization budget vs. alignment signal confounding:** The two-stage IA2→SFT pipeline inherently involves more training iterations than single-stage SFT-only, as convergence is reached independently for each phase. While early stopping is applied and gains are consistent across data regimes, the paper lacks a step- or FLOP-matched SFT baseline. In few-shot settings, extended training can act as an implicit regularizer or help SFT escape sharp local minima; a direct control matching total gradient steps would more definitively isolate the contribution of the activation alignment signal from extended optimization dynamics.
- **Surface-level matching vs. verified inductive bias transfer:** The core hypothesis assumes that minimizing MSE on self-attention outputs transfers ICL's functional computation. However, the validation relies on correlational metrics (aggregate cosine similarity and subspace overlap) rather than causal or circuit-level interventions. Without targeted ablations—such as layer-wise masking, FFN vs. SA alignment isolation, or probing specific attention routing patterns—it remains partially unclear whether IA2 truly captures ICL's gradient-descent-like reasoning pathways or simply imposes a strong representational prior that regularizes SFT against shortcut learning.

## Nice-to-Haves
- Quantify the computational overhead of the activation collection phase (e.g., additional GPU hours, FLOPs, or memory footprint relative to SFT-only) to help practitioners assess the cost-benefit trade-off in deployment scenarios.
- Provide reliability diagrams or calibration curves alongside scalar ECE metrics to reveal whether IA2 uniformly improves confidence or selectively corrects extreme overconfidence modes in single-token tasks.
- Explore layer-wise ablation or selective alignment (e.g., targeting only middle layers where ICL/SFT divergence peaks in Figure 2) to reduce compute and test whether activation matching in specific circuit depths drives the bulk of the gains.

## Novel Insights
The synthesis of the reviews suggests a broader implication that extends beyond the paper's immediate pipeline: activation alignment may function as an optimization navigational prior rather than a mere distillation objective. The finding that SFT-only and ICL weight updates occupy nearly orthogonal subspaces suggests that standard cross-entropy fine-tuning in low-data regimes is highly susceptible to converging to brittle, calibration-breaking local minima. Sequential activation priming appears to steer the model's initial weights into a more robust, context-aware basin before the final output alignment occurs. This reframes IA2 not just as a method to copy ICL, but as a mechanism to inject the model's own context-activated knowledge back into its parameters, effectively using inference-time compute to regularize parameter updates and prevent SFT from over-specializing on shallow shortcuts.

## Suggestions
- Implement a step-matched or FLOP-matched SFT-only baseline (e.g., by artificially extending SFT training epochs or adding a dummy alignment step) to explicitly control for total optimization budget and isolate the priming effect from extended training.
- Quantify and report the exact computational cost of the ICL activation collection stage alongside SFT training times, providing a clear ROI metric for adoption in data-scarce vs. data-rich scenarios.
- Conduct a targeted diagnostic ablation comparing activation alignment on SA layers vs. FFN layers, or restrict matching to the middle transformer layers, to verify whether the performance gains stem from transferring context-processing circuits or from general representation smoothing.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 6.0]
Average score: 6.0
Binary outcome: Accept
