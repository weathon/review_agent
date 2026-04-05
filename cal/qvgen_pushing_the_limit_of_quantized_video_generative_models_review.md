=== CALIBRATION EXAMPLE 38 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately reflects the paper's focus on low-bit quantization for video generative models. The abstract clearly states the problem, proposes the auxiliary module $\Phi$ and rank-decay strategy, and highlights the key empirical claim: "the first to reach full-precision comparable quality under 4-bit settings." This claim is largely supported by Table 1, though it would be more precise to note that "comparable" is metric-dependent (e.g., Scene Consistency occasionally shows non-negligible drops versus BF16). The abstract is well-scoped and correctly positions the contribution.

### Introduction & Motivation
The motivation is strong: video DiTs are computationally prohibitive, and existing quantization methods (mostly designed for image DMs or CNNs) fail at $\leq 4$ bits for video. The contributions are clearly enumerated. 
**Concern:** The introduction identifies *that* video DMs are harder to quantize but doesn't adequately explain *why* until Section H. Video generation introduces temporal error accumulation, frame-to-frame distribution shifts, and timestep-dependent activation outliers. Briefly previewing these domain-specific quantization challenges in the Introduction would strengthen the motivation and clarify why naive QAT adaptation fails.

### Method / Approach
**3.1 Convergence & Auxiliary Modules $\Phi$:** The paper correctly identifies that reducing the gradient norm $\|\mathbf{g}_t\|_2$ stabilizes QAT. However, Theorem 3.1 is a standard regret bound from convex online optimization, and the non-convex analysis (Appendix C) simply rederives the standard stationary-point convergence bound. Mathematically sound, but not novel. The actual contribution is empirical: introducing $\Phi$ to absorb quantization residuals and smooth the loss landscape. The connection between the theorem and the practical mechanism should be framed as an *inspiration* rather than a strict theoretical guarantee, which the authors partially do, but the distinction could be sharper to avoid overclaiming theoretical novelty.

**3.2 Rank-Decay Strategy:** The scheduled decomposition and pruning of $\Phi$ is a clever engineering solution to eliminate inference overhead. The observation that singular values of $\mathbf{W}_\Phi$ concentrate near zero during training (Fig. 4) justifies low-rank approximation.
**Concerns:**
1. **Computational Overhead of Repeated SVD:** The method applies SVD iteratively across decay phases. For large linear layers (e.g., FFN projections in 5B/14B models), full or top-$r$ SVD is non-trivial. The paper sets $r \ll d$ but does not clarify whether truncated/randomized SVD is used, nor does it report the wall-clock overhead of decomposition versus the training cost of a decay phase. This impacts reproducibility and deployment feasibility during training.
2. **Connection to Low-Rank Adaptation (LoRA/DoRA):** $\Phi$ is structurally equivalent to a linear adapter applied to quantized activations. The method progressively prunes this adapter along singular directions. This closely resembles scheduled low-rank adaptation or magnitude-based pruning of LoRA modules. The Related Work and Method sections should explicitly position QVGen against LoRA-based QAT or dynamic low-rank compensation methods to clarify conceptual novelty versus implementation details (cosine schedule + SVD truncation).

### Experiments & Results
**Setup & Baselines:** The evaluation covers 4 SOTA video DMs (1.3B to 14B) and adapts strong QAT/PTQ baselines. This is appropriate given the claimed novelty. 
**Concerns:**
1. **Statistical Rigor & Variance:** **This is a critical weakness for ICLR.** VBench scores for generative models exhibit high run-to-run variance due to stochastic sampling seeds and the sensitivity of VLM-based evaluators. All primary results (Tables 1, 2, 4, 5, 6) report single-run point estimates with no standard deviations, confidence intervals, or multi-seed averaging. Without variance reporting, claims of "comparable to full-precision" or "large margin improvements" lack statistical grounding. ICLR expects at least 3 random seeds or explicit variance quantification for generative benchmarks.
2. **Dataset Scale for QAT:** The training uses only 16K videos from OpenVidHQ-0.4M across 8-16 epochs. While calibration datasets for PTQ are often small, QAT typically requires larger/more diverse data to prevent overfitting to the quantization error surface. The strong generalization to VBench suggests this works, but a discussion on dataset sensitivity (e.g., ablation with 4K vs 32K samples) or analysis of overfitting to the training distribution would strengthen robustness claims.
3. **Evaluation Metrics:** The paper relies entirely on VBench/VBench-2.0. While standard, these metrics are known to correlate imperfectly with human perception, especially for temporal coherence and scene consistency. Including Frame-wise Video Distance (FVD) or a small-scale human preference study would significantly bolster the claim of visual fidelity, particularly for the 3-bit setting where drops are noted.

### Ablations & Analysis
The ablations (Tables 3-6, Appendix L) are thorough and systematically validate each component.
**Concerns:**
1. **Choice of Initial Rank $r$:** Table 5 shows performance peaks at $r=32$ and degrades at $r=64$. The explanation ("increasing $r$ to $2r$ introduces an additional decay phase, which shortens the training time allocated to each phase") conflates schedule length with model capacity. The degradation is more likely due to over-parameterization of the auxiliary module destabilizing the QAT objective or interfering with gradient flow in the base weights. A clearer optimization-based analysis is needed.
2. **Decay Schedule Hyperparameters:** Table 4 tests $\lambda \in \{1/4, 1/2, 3/4, 1\}$ with fixed iterations per phase. The optimal $\lambda=1/2$ is sensible, but the interaction between $\lambda$, total decay phases, and total training budget isn't fully decoupled. A principled recommendation for scaling $\lambda$ with model size or bit-width would improve reproducibility.

### Writing & Clarity
The paper is well-structured and generally clear. Mathematical notation is consistent, and figures effectively illustrate key observations (e.g., singular value evolution in Fig. 4, training dynamics in Fig. 3). The algorithmic breakdown in Appendix A is helpful. No major clarity issues impede understanding. Minor note: The transition from theoretical analysis to practical heuristic ($\Phi$ initialization and rank-decay) could be smoother, but this is acceptable for an applied methods paper.

### Limitations & Broader Impact
The conclusion acknowledges the focus on video generation and suggests NLP generalization. 
**Concerns:**
1. **Compute & Environmental Impact:** Evaluating Wan 14B on VBench requires 128 $\times$ H100 GPUs and training uses up to 32 H100s for 16 epochs. The paper does not discuss the substantial carbon footprint or financial cost, which is increasingly expected for large-scale generative modeling submissions at ICLR.
2. **Failure Modes:** While performance drops are noted for 3-bit models, the paper doesn't thoroughly analyze *where* the model fails catastrophically (e.g., specific prompt categories, extreme motion scenes, or rare object compositions). A qualitative error analysis would provide actionable insights for future work.

---

### Overall Assessment
QVGen presents a well-motivated and empirically strong framework for ultra-low-bit quantization of video diffusion models. The core idea—using a learnable residual module to stabilize QAT and then systematically pruning it via scheduled low-rank decomposition—is elegant and effectively eliminates inference overhead while preserving quality. The extensive evaluation across multiple model scales (up to 14B) and bit-widths demonstrates practical relevance.

However, the paper has notable shortcomings relative to ICLR standards: **(1)** a lack of statistical rigor (no multi-seed variance reporting on highly stochastic generative metrics), **(2)** modest theoretical novelty (the convergence bounds are standard, and the empirical mechanism is the true contribution), **(3)** missing explicit connections to low-rank adaptation literature (LoRA/DoRA for QAT), and **(4)** insufficient analysis of training computation overhead (SVD cost) and environmental/resource impact. The dataset scale for QAT (16K samples) is also unusually small and warrants sensitivity analysis.

With the addition of variance reporting, a clearer positioning against low-rank QAT baselines, and a discussion of computational overhead, this paper would be a strong candidate for acceptance. In its current form, the empirical results are promising but require greater statistical and methodological rigor to firmly meet the ICLR bar.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces QVGen, the first quantization-aware training (QAT) framework specifically designed for ultra-low-bit (≤4-bit) video diffusion models (DMs). Motivated by theoretical and empirical analysis linking QAT convergence stability to gradient norm minimization, the authors introduce auxiliary modules ($\Phi$) to mitigate quantization-induced errors. To ensure zero inference overhead, a novel SVD-based "rank-decay" schedule progressively shrinks and eliminates $\Phi$ during training. Experiments on state-of-the-art video DMs (1.3B–14B parameters) demonstrate that QVGen achieves performance comparable to full precision at W4A4 while significantly outperforming existing PTQ and QAT baselines.

### Strengths
1. **Strong Theoretical Foundation Backed by Empirical Evidence:** The paper provides a clear convergence analysis (Theorem 3.1 and Appendix C) demonstrating that minimizing the average gradient norm $\|g_t\|_2$ is critical for QAT stability, extending both convex regret bounds and standard non-convex smoothness assumptions. This theory is tightly coupled with empirical results: Fig. 3 shows QVGen maintains consistently lower gradient norms and smoother training loss curves than baselines, directly validating the core hypothesis.
2. **Innovative Zero-Overhead Auxiliary Module Design:** The rank-decay strategy (Sec. 3.2) effectively solves the classic dilemma of auxiliary modules in quantization: improving training while preserving inference efficiency. By iteratively applying SVD and decaying low-impact components via rank-based regularization $\gamma$, the method completely removes $\Phi$ at inference. Table 3 confirms <0.6% performance drop post-removal, and Table 6 demonstrates clear superiority over alternative decay strategies (Sparse, Res. Q., Linear) in both quality and training time.
3. **Comprehensive Evaluation & Reproducibility:** The framework is rigorously tested across 4 SOTA open-source video DMs (CogVideoX-2B/5B, Wan-1.3B/14B). Tables 1 & 2 show consistent state-of-the-art results in W4A4 and highly competitive W3A3 performance. The authors provide detailed experimental setups (dataset size, batch sizes, optimizers, GPU counts, training epochs), release code/models, and include thorough efficiency analysis (Table 7 for training costs, Fig. 6 & Appendix N-O for latency breakdowns), meeting high reproducibility standards.

### Weaknesses
1. **Theoretical Link to $\Phi$ Lacks Formal Guarantee:** While Theorem 3.1 and Appendix C establish that reducing $\|g_t\|_2$ improves convergence, the paper does not formally prove that the proposed auxiliary module $\Phi$ inherently minimizes this norm under quantization constraints. The connection remains empirical: $\Phi$ is initialized with weight quantization error and shown to smooth training, but a lemma or perturbation analysis explicitly linking $\Phi$'s structure/update to the gradient bound would strengthen the theoretical rigor expected at ICLR.
2. **Evaluation Relies Solely on Automated Metrics:** Performance is measured exclusively using VBench and VBench-2.0. While these are standard, automated benchmarks for video generation are known to have blind spots regarding temporal coherence, physical plausibility, and semantic faithfulness. The absence of human preference studies, expert evaluations, or frame-level optical flow consistency metrics limits confidence in how well the scores translate to actual perceptual quality.
3. **Unquantified Training Overhead from Iterative SVD:** The authors claim negligible training overhead (~1.02x GPU days vs. naive QAT in Table 7), but do not explicitly report the FLOPs or wall-clock time consumed by performing layer-wise SVD and low-rank truncation across all decay phases. For models scaling beyond 14B, the cost of repeated SVD on large weight matrices across distributed setups could become non-trivial, and this scalability bottleneck is not discussed.

### Novelty & Significance
**Novelty:** High. This is the first dedicated QAT method for video DMs, introducing two key innovations: (1) framing low-bit QAT instability through the lens of gradient norm minimization, and (2) the SVD-based rank-decay mechanism that enables auxiliary modules to vanish at inference without quality loss. The approach is architecture-agnostic and orthogonal to existing PTQ/QAT works (Appendix I shows compatibility with SVDQuant).
**Significance:** High. Quantization is critical for deploying large video DMs on resource-constrained hardware. Achieving full-precision comparable quality at W4A4 with zero inference overhead addresses a major practical bottleneck. The strong results on widely used open-source models, combined with code release, make this highly valuable for the community. The paper meets ICLR's acceptance bar in technical soundness, empirical rigor, and impact, pending minor clarifications.

### Suggestions for Improvement
1. **Strengthen the Theory-Empirical Bridge:** Add a brief perturbation analysis or lemma showing how $\Phi$ (initialized as $W - Q_b(W)$ and trained jointly) bounds the quantization-induced gradient perturbation term. This would formally connect the auxiliary module design to the $\|g_t\|_2$ reduction established in Theorem 3.1/Appendix C.
2. **Incorporate Human Evaluation or Temporal Metrics:** Supplement VBench scores with a small-scale human preference study or established temporal consistency metrics (e.g., frame-to-frame optical flow variance, Flicker Index). This would address potential biases in automated video benchmarks and better validate the claim of "full-precision comparable quality."
3. **Explicitly Profile SVD/Decay Training Cost:** Provide a breakdown of the wall-clock time and memory footprint attributable specifically to the iterative SVD and rank-truncation steps during QAT. Discuss potential distributed SVD approximations or scheduling strategies for models >20B parameters to clarify scalability.
4. **Clarify Real-World Speedup Potential:** Since the reported 1.2–1.4x inference speedup is constrained by unfused kernels, provide a concrete benchmark or projection after applying standard CUDA fusion (as projected in Appendix N). Readers need a clear expectation of deployment gains on commodity hardware vs. research prototypes.
5. **Discuss Activation Quantization Strategies for Video:** Section L.6 and Fig. B correctly note that activations are harder to quantize than weights. Consider a dedicated discussion or ablation on video-specific activation dynamics (e.g., per-timestep vs. per-token quantization ranges, caching strategies for temporal tokens) to highlight how QAT for video fundamentally differs from image DMs.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Run all VBench evaluations across multiple random seeds and report mean±std, as single-run generative metrics are too noisy to credibly claim "full-precision comparable" performance at ICLR standards.
2. Benchmark against a compute-matched naive QAT baseline (trained for equivalent or longer epochs/data) to rigorously prove that Φ enables genuine convergence gains rather than simply acting as an implicit regularizer or warm-start trick.
3. Report real end-to-end generation latency and throughput using a fully fused W4A4 CUDA kernel, since current speedup claims rely on isolated operator timing or projected fusion gains that overstate practical deployment benefits.

### Deeper Analysis Needed (top 3-5 only)
1. Replace the standard online/nonconvex gradient norm bounds with a quantization-specific analysis that directly links STE gradient mismatch, activation outlier clipping, and low-bit discretization error to the convergence bottleneck.
2. Provide a layer-wise and timestep-wise sensitivity analysis to identify whether spatial/temporal attention blocks or specific denoising steps dominate quantization error, rather than assuming uniform Φ effectiveness across the entire DiT.
3. Quantify the impact of the 16K-video training corpus size on generalization, as ultra-low-bit generative QAT is highly prone to distribution collapse and VBench overfitting on small captioned datasets.

### Visualizations & Case Studies
1. Plot weight and activation distribution histograms before, during, and after rank-decay to demonstrate that Φ genuinely aligns quantized and full-precision feature distributions rather than masking quantization artifacts.
2. Visualize frame-to-frame optical flow divergence or temporal LPIPS curves for challenging motion prompts to expose whether the method preserves motion dynamics or fails on high-frequency temporal transitions.
3. Show layer-wise gradient norm trajectories during the decay phases to directly validate the core claim that shrinking Φ maintains stable optimization rather than causing sudden gradient spikes or loss-of-gradient.

### Obvious Next Steps
1. Integrate QVGen with state-of-the-art video acceleration techniques (e.g., sparse attention, feature caching) and report combined end-to-end speedups, since linear layer quantization alone ignores the 50%+ attention compute bottleneck.
2. Release a minimal fused kernel implementation or detailed integration pipeline, as claiming "zero inference overhead" while deferring kernel fusion entirely to future work leaves the primary deployment claim unverified.
3. Include human preference studies (e.g., pairwise win rates against BF16) alongside VBench, since automated video metrics notoriously correlate poorly with perceptual quality, temporal coherence, and text alignment in quantized generative models.

# Final Consolidated Review
## Summary
QVGen introduces a quantization-aware training (QAT) framework specifically designed for ultra-low-bit (≤4-bit) video diffusion models. It identifies gradient norm reduction as critical for QAT convergence stability and introduces a learnable auxiliary module ($\Phi$) to absorb quantization residuals. To ensure zero inference overhead, the authors propose a novel SVD-based rank-decay schedule that progressively shrinks and completely removes $\Phi$ during training. Evaluated on four SOTA video DMs (1.3B–14B parameters), QVGen achieves W4A4 performance comparable to full-precision models and substantially outperforms existing QAT/PTQ baselines adapted from the image domain.

## Strengths
- **First effective low-bit QAT for video DMs.** The framework successfully pushes quantization to 4-bit and 3-bit across multiple architectures (CogVideoX, Wan) while maintaining high VBench scores. The empirical gains are substantial, consistent, and address a clear deployment bottleneck.
- **Elegant resolution of the training-inference tension.** The $\Phi$ module + rank-decay mechanism cleanly sidesteps the permanent overhead typical of auxiliary quantization compensation. Thorough ablations (Tables 3–6) validate the design, showing <0.6% quality drop upon complete removal of $\Phi$ and clear superiority over linear, sparse, and residual decay alternatives.
- **Strong empirical-convergence linkage.** Gradient norm tracking (Fig. 3) convincingly demonstrates that $\Phi$ stabilizes QAT optimization. The analysis correctly identifies why naive QAT fails for video models (temporal error accumulation amplifies quantization noise) and validates that controlling $\|g_t\|_2$ correlates directly with improved generation quality.

## Weaknesses
- **Lack of statistical rigor in generative evaluation.** All primary VBench/VBench-2.0 results are reported as single-run point estimates without multi-seed averaging, standard deviations, or confidence intervals. Given the well-documented high variance of VLM-based video evaluators and stochastic sampling pipelines, claims of "comparable to full-precision" quality cannot be statistically verified. This is a substantive gap for ICLR standards.
- **Theoretical framing overstates novelty.** Theorem 3.1 and Appendix C reproduce standard regret bounds and non-convex descent inequalities. The paper does not formally prove that $\Phi$ minimizes the derived gradient norm bound under straight-through estimator (STE) quantization constraints. The theoretical contribution is standard optimization theory used as motivation, not a new guarantee, yet it is presented with language suggesting a tighter theoretical foundation.
- **Insufficient conceptual distinction from low-rank adaptation.** The $\Phi$ module is structurally analogous to low-rank compensation techniques (e.g., LoRA-style updates applied to quantized activations). The paper does not explicitly clarify how its optimization objective, initialization strategy, and scheduled decay differ fundamentally from existing low-rank fine-tuning or dynamic compensation methods for QAT, making it difficult to isolate whether the novelty lies in the rank-decay schedule itself or a broader algorithmic insight.
- **Exclusive reliance on automated metrics.** VBench/VBench-2.0 are standard but correlate imperfectly with temporal coherence and physical plausibility. Without supplementary temporal consistency metrics (e.g., FVD, frame-wise optical flow variance) or human preference evaluation, the preservation of complex motion dynamics remains partially unverified, particularly for the 3-bit setting where scene consistency shows noticeable drops.

## Nice-to-Haves
- Provide a layer-wise or timestep-wise sensitivity analysis to determine whether spatial/temporal attention blocks or specific denoising steps dominate quantization error and drive the gradient norm inflation.
- Include a brief ablation on calibration dataset scale (e.g., 4K vs. 16K videos) to verify that performance gains do not stem from overfitting to a narrow distribution.
- Release a minimal fused kernel implementation or concrete end-to-end latency benchmark, as current speedup projections assume future engineering that is not yet realized.

## Novel Insights
The paper's core insight is recognizing that video diffusion models are uniquely sensitive to gradient norm inflation during QAT due to temporal error accumulation, and that a progressively collapsing auxiliary low-rank module can stabilize this without permanent overhead. The empirical observation that $\mathbf{W}_\Phi$ naturally concentrates near zero during training (Fig. 4), enabling safe, scheduled rank truncation, shifts the paradigm from static error compensation to dynamic, self-eliminating regularization. This bridges training stability and deployment efficiency in a manner that could plausibly extend to other temporal or sequential generative architectures facing similar quantization-induced instability.

## Suggestions
- Re-evaluate primary VBench results across ≥3 random sampling seeds and report mean±std or confidence intervals to establish statistical significance for the "full-precision comparable" claim.
- Clarify in the introduction and related work that the convergence analysis serves as an empirical motivation for gradient stabilization, and explicitly discuss how $\Phi$'s optimization and rank-decay differ conceptually from standard low-rank adaptation or dynamic compensation methods.
- Supplement VBench scores with temporal consistency metrics (e.g., FVD, optical flow divergence, or temporal LPIPS) to better validate motion preservation, particularly for complex 3-bit generation scenarios.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0, 8.0, 6.0]
Average score: 6.8
Binary outcome: Accept
