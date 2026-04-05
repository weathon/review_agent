=== CALIBRATION EXAMPLE 1 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**
The title accurately reflects the core technical contribution (convex relaxation via Frank-Wolfe for mask selection). The abstract claims that rounding the relaxed solution yields an approximate solution to the original combinatorial problem, which aligns with the theoretical section. However, the abstract implies that the FW algorithm alone successfully recovers performance, without hinting at the critical heuristic introduced later (fixing 90% of weights based on a baseline). For ICLR, where practical efficacy is heavily weighted, the abstract slightly overstates the standalone capability of FW and should foreshadow the necessity of the saliency-guided fix to match empirical claims.

**Introduction & Motivation**
The problem is well-motivated: layer-wise pruning avoids expensive retraining, and existing methods rely on greedy heuristics that ignore weight interactions. The transition from Eq. `(MASK SELECTION)` to the convex relaxation is clear and mathematically sound. The contributions are crisply listed. My main concern is that the introduction positions SparseFW as a principled, global optimization alternative to greedy methods, but does not acknowledge the severe practical limitation that pure FW ($\alpha=0$) underperforms baselines. This framing mismatch sets up expectations that Section 2.3 later contradicts. A brief acknowledgement of the local-global objective mismatch in the introduction would better calibrate the reader.

**Methodology (Section 2)**
- *2.1 Preliminaries & Greedy Methods:* The derivation connecting Wanda and RIA to single-step greedy pruning is insightful and effectively grounds the motivation for interaction-aware optimization. The row-wise decomposition is correctly noted.
- *2.2 Frank-Wolfe & LMO:* The formulation of the constraint set $\mathcal{C}_k$ and the derivation of the LMO (Eq. 12) are correct and well-explained. The projection-free nature of FW is appropriately leveraged.
- *2.3 SparseFW & The $\alpha$-Fixing Heuristic:* This is the most critical section. The authors note a "caveat": vanilla FW yields worse perplexity despite lower local error, and performance is rescued by fixing a fraction $\alpha$ (optimally 0.9) of high-saliency weights from a warmstart mask. This heuristic dominates the practical success of the method but is relegated to a brief paragraph. From an ICLR perspective, this requires deeper analysis: Why does pure FW fail? Is it due to ill-conditioning of the Hessian $G=XX^\top$, slow convergence in high-dimensional polytopes, or the known mismatch between local reconstruction error and global perplexity? Treating this as an implementation detail obscures the actual mechanism driving the results. This caveat effectively turns SparseFW into a *local refinement* of Wanda rather than a standalone mask optimizer, which should be explicitly acknowledged and analyzed.

**Experiments & Results (Section 3)**
- *Baseline Selection:* The paper compares only to Wanda and RIA, explicitly excluding reconstruction-based methods like SparseGPT. While this maintains methodological consistency (mask selection only), it limits the practical relevance. State-of-the-art post-training pruning often relies on lightweight reconstruction to recover accuracy. Comparing only to 2023 baselines (Wanda/RIA) may undersell the true competitive landscape for ICLR 2026. Including at least one reconstruction baseline or explicitly positioning SparseFW as a pre-reconstruction step would strengthen the claim.
- *Statistical Rigor:* Table 1 explicitly omits standard deviations "for legibility." Given the reliance on small calibration sets (256 samples) and stochastic sampling from C4, performance variance can be substantial. ICLR expects statistical reporting, especially when claiming "consistent gains." Standard deviations or confidence intervals should be included, even in supplementary material.
- *Compute & Memory Profiling:* The method claims memory efficiency, but precomputing $G=XX^\top$ and $H=WG$ stores $O(d_{in}^2)$ and $O(d_{out}d_{in})$ matrices. For modern 70B+ models with $d_{in}=16384$, this becomes a significant bottleneck. Moreover, running 2000 FW iterations per layer across ~30+ transformer blocks incurs substantial wall-clock time. Without FLOP, VRAM, or runtime profiling, it's difficult to assess whether the perplexity gains justify the overhead, especially when deployed at scale.
- *$\alpha$ Ablation Placement:* The ablation over $\alpha$ (Appendix Table 2) is arguably more important than several main-text figures. Given that $\alpha=0.9$ is the default, this analysis should be in the main text to transparently show how sensitive the method is to the warmstart heuristic.

**Theoretical Results (Section 4)**
Lemma 1 provides a convergence bound for FW on the relaxed problem plus a thresholding error. While mathematically standard for Frank-Wolfe, two aspects limit its relevance to the paper's empirical contributions:
1. *Vacuous Constants:* The bound scales with $\lambda_{\max}(Q)$ where $Q = \text{Diag}(w)(XX^\top)\text{Diag}(w)$. In LLMs, activation outliers and weight super-magnitudes cause $\lambda_{\max}(Q)$ to be enormous, and the dimension terms ($\sqrt{d_{in}d_{out}k}$) are massive. Without empirical estimates or condition number analysis, the bound risks being vacuous in practice.
2. *Theory-Practice Mismatch:* The theory justifies the full relaxed problem ($\alpha=0$), but Section 2.3 and Appendix Table 2 show that $\alpha=0$ consistently underperforms baselines. The bound does not account for the $\alpha$-fixing strategy, which is what actually yields the reported perplexity gains. This disconnect is a notable weakness for ICLR's expectations on theoretical grounding. The authors should either (a) extend the analysis to the constrained subspace induced by $\alpha$-fixing, or (b) explicitly frame the theory as applying to the pure relaxation while acknowledging that practical performance requires heuristic stabilization.

**Limitations & Broader Impact**
The authors correctly identify the local-global objective mismatch and the necessity of preserving high-saliency weights. This is honest and aligns with broader observations in post-training compression. However, the discussion misses two important limitations: (1) The reliance on a pre-computed warmstart mask ties SparseFW's ceiling to the baseline it refines; it cannot discover masks outside the warmstart's structural prior. (2) Scalability to multi-GPU setups or pipeline-parallel models is not discussed. LMO operations require global top-k selection over potentially distributed weight matrices, introducing communication overhead that is unaddressed.

### Overall Assessment
This paper introduces a principled convex relaxation approach to layer-wise LLM pruning using Frank-Wolfe, with clear mathematical derivations and consistent empirical improvements over Wanda and RIA. However, the central practical contribution hinges on a heuristic that fixes 90% of the mask based on a baseline saliency score, which is currently under-analyzed and buried in the text/caveats. This creates a theory-practice gap: the theoretical guarantees apply to the full relaxation ($\alpha=0$), which empirically underperforms, while the successful variant ($\alpha=0.9$) lacks theoretical backing. Additionally, the omission of standard deviations, lack of compute/memory profiling, and narrow baseline selection limit the paper's readiness for ICLR's high empirical bar. If the authors elevate the $\alpha$-fixing heuristic to a core contribution, provide runtime/VRAM profiling, include statistical variance, and better align the theoretical narrative with the empirical reality, this work could be highly impactful.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes SparseFW, a post-training, layerwise pruning method for LLMs that formulates mask selection as a convex program over the relaxed constraint polytope and solves it using the Frank-Wolfe (FW) algorithm. By replacing greedy saliency heuristics with a principled constrained optimization approach, the method explicitly models weight interactions and provides theoretical approximation guarantees for the rounded binary mask. Empirically, SparseFW substantially reduces per-layer reconstruction error and matches or outperforms strong baselines (Wanda, RIA) across multiple modern architectures and sparsity regimes.

### Strengths
1. **Principled Optimization Formulation:** The paper successfully reframes the combinatorial mask selection problem into a tractable convex program, moving beyond the myopic single-weight removal used by greedy baselines. The derivation of the efficient Linear Minimization Oracle (LMO) and the projection-free FW update (Section 2.2, Algorithm 1) is mathematically sound and directly addresses weight interactions.
2. **Comprehensive Empirical Evaluation:** The experiments cover 5 contemporary LLMs (LLaMA-3.1, Gemma-2, Yi-1.5, Qwen2.5, DeepSeek) across 50%, 60%, and 2:4 sparsity levels. Table 1 and zero-shot accuracy results consistently show performance parity or gains, particularly in high-sparsity regimes. Figure 3's calibration ablation correctly identifies that SparseFW benefits from more samples, unlike Wanda, highlighting a distinct advantage in utilizing calibration data more effectively.
3. **Theoretical Justification:** Lemma 2 (formalized in Appendix E) provides a clear error guarantee that decomposes the final mask quality into optimization error (scaling with `1/T`) and thresholding error. This bridges a notable gap in the post-training literature, where greedy methods rarely come with convergence or approximation guarantees.
4. **Scalability Engineering:** The precomputation of `G = XX^T` and `H = WG` (Section 2.3) elegantly decouples per-iteration cost from batch size `N` and sequence length `L`, making the algorithm memory-efficient and practical for modern context lengths without sacrificing gradient accuracy.

### Weaknesses
1. **Critical Dependence on the Wanda Prior (`α=0.9`):** Appendix C reveals that running FW without fixing high-saliency weights (`α=0.0`) consistently yields worse perplexity than baselines, and the sweet spot is `α=0.9`. This means FW only optimizes 10% of the mask, raising questions about whether the gains stem from FW's convex optimization or simply the strong Wanda inductive bias. The local-global objective mismatch (Section 5) is acknowledged but left unresolved.
2. **Missing Computational Cost Analysis:** While the paper claims efficiency and sequence-length independence, it omits concrete wall-clock pruning times, FLOP counts, or peak VRAM usage compared to `O(1)` baselines. For ICLR, a clear cost-benefit tradeoff (e.g., "X% longer pruning time yields Y perplexity reduction") is necessary to assess deployment viability.
3. **Narrow Baseline Scope:** Comparisons are limited to Wanda, RIA, and SparseGPT. The field has advanced toward gradient-aware and light-reconstruction methods (e.g., OWL, Wanda-S, or calibration-based magnitude variants) that also avoid full retraining. Excluding these makes it difficult to position SparseFW against the true state-of-the-art.
4. **Theoretical Bound Practicality:** The approximation guarantee in Lemma 2 depends on `λ_max(Q)` and dimensionality terms. These quantities can be large or ill-conditioned for typical LLM layers, potentially rendering the bound loose. The paper does not empirically measure the tightness of this bound or discuss its computational implications, limiting its practical interpretability.

### Novelty & Significance
The application of Frank-Wolfe and convex relaxation to LLM mask selection is a novel and well-motivated departure from dominant greedy paradigms. The theoretical rounding guarantee adds rigor often missing in heuristic pruning, aligning well with ICLR's appreciation for principled optimization perspectives. However, the novelty is partially tempered by the heavy reliance on fixing 90% of weights via a Wanda prior; in practice, SparseFW acts more as a localized refinement of Wanda rather than a standalone replacement. Significance is moderate-to-high: while perplexity improvements are incremental, the framework establishes a promising new direction for constrained optimization in post-training compression, provided the computational overhead is justified and the local-global alignment issue is addressed.

### Suggestions for Improvement
1. **Deepen the α=0.9 Analysis:** Investigate *why* unconstrained FW fails to generalize to perplexity despite lowering the local objective. Consider analyzing the Hessian conditioning or gradient alignment with downstream loss gradients across layers. If feasible, propose an adaptive or regularized scheme that reduces reliance on a fixed `α` while preserving perplexity.
2. **Add Comprehensive Efficiency Metrics:** Report wall-clock pruning time, FLOPs, and peak memory usage for SparseFW vs. Wanda/RIA across model scales. A structured efficiency-vs-accuracy analysis (e.g., Pareto frontier) will greatly strengthen the practical impact claims.
3. **Expand Baseline Comparisons:** Include at least one recent state-of-the-art calibration-aware or light-adaptation method (e.g., OWL, Wanda-S, or gradient-based saliency variants) to contextualize SparseFW's standing. If reconstruction is intentionally excluded, explicitly justify this design choice in the experimental setup.
4. **Empirically Ground Theoretical Guarantees:** Compute `λ_max(Q)` for representative layers and compare the theoretical error bound to the actual observed optimality gap (e.g., using a small MILP baseline on toy layers or tight heuristic bounds). This will clarify whether the bound is practically informative or purely asymptotic.
5. **Clarify Algorithmic Flow & Notation:** Section 2.1's transition from Eq. 1 to the Wanda/RIA derivations is dense. Adding a brief intuitive narrative or pseudocode step that contrasts greedy vs. FW update trajectories would improve accessibility for a broad machine learning audience.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add wall-clock time, peak memory, and FLOP measurements against Wanda/RIA across all architectures, because the paper claims "efficiency and scalability" but only provides qualitative assertions.
2. Evaluate on modern reasoning and instruction-following benchmarks (MMLU, GSM8K, IFEval), because WikiText perplexity and generic zero-shot tasks no longer suffice to claim meaningful LLM improvement at ICLR, especially when perplexity gains rarely translate to downstream capabilities.
3. Include SparseGPT as a baseline, because it targets the exact same layerwise reconstruction objective and sets a recognized performance ceiling for post-training pruning; excluding it leaves the "outperforms SOTA" claim unverified.
4. Ablate the α=0.9 fixed-weight ratio across varying sparsity levels and model scales, because tuning this single heuristic to maximize Wanda's scores risks masking the true capability of the FW relaxation and undermines the generality of the method.

### Deeper Analysis Needed (top 3-5 only)
1. Explain why vanilla FW (α=0.0) empirically fails Wanda despite proven convergence to the continuous optimum, specifically by quantifying how the local layerwise objective misaligns with the true global cross-entropy loss.
2. Analyze the eigenvalue spectrum and condition number of the Hessian Q per layer, because your theoretical thresholding error bound scales with λ_max(Q) and this metric directly dictates why naive top-k rounding degrades performance in practice.
3. Investigate whether the 10% of weights optimized by FW consistently fall in structurally critical regions (e.g., specific attention heads or output channels), as this determines if FW is learning meaningful corrections or simply optimizing noise.

### Visualizations & Case Studies
1. Overlay the final pruned masks of Wanda and SparseFW to show exactly which weight subspaces FW modifies, revealing whether the method preserves architectural coherence or scatters changes arbitrarily.
2. Plot the correlation between per-layer pruning error reduction and actual perplexity/accuracy deltas across architectures, exposing whether local objective minimization genuinely drives global performance or if the α=0.9 heuristic artificially decouples them.
3. Visualize the convergence trajectory of continuous vs. thresholded masks on attention vs. MLP layers separately, because thresholding degradation may be architecture-specific and requires targeted mitigation.

### Obvious Next Steps
1. Implement sequential layer pruning with recalibrated activations X after each layer, directly addressing the acknowledged local-to-global objective mismatch that forces the α=0.9 workaround.
2. Develop a principled rounding or thresholding scheme aligned with your theoretical bounds (e.g., constrained optimization or warm-started binary projection), because naive top-k rounding is empirically shown to destroy early optimization gains.
3. Integrate SparseFW into a lightweight calibration fine-tuning or sparse recovery step to test if the method's performance bottleneck is inherent to the one-shot setting rather than the FW framework itself.

# Final Consolidated Review
## Summary
This paper introduces SparseFW, a post-training pruning method for LLMs that reformulates layerwise mask selection as a convex optimization problem solved via the Frank-Wolfe algorithm. By relaxing binary constraints to their convex hull and employing an efficient linear minimization oracle, the approach explicitly accounts for weight interactions and provides theoretical approximation guarantees upon rounding. The method achieves consistent reconstruction error reductions and matches or improves perplexity and zero-shot accuracy over greedy saliency baselines across multiple modern architectures and sparsity regimes.

## Strengths
- **Principled optimization formulation:** The paper successfully transforms a combinatorial mask selection problem into a tractable convex program, directly addressing weight interactions ignored by greedy baselines. The derivation of the efficient Linear Minimization Oracle (LMO) and the projection-free update rule are mathematically sound and clearly presented (Section 2.2).
- **Scalability-focused engineering:** Decoupling the per-iteration gradient cost from calibration batch size ($N$) and sequence length ($L$) by precomputing $G=XX^\top$ and $H=WG$ (Section 2.3) is a practical and well-motivated design choice that makes the algorithm memory-viable for long contexts without relying on stochastic approximations.
- **Consistent empirical gains across architectures:** Evaluation on five contemporary models (7B–14B) across 50%, 60%, and 2:4 sparsity patterns shows reliable reductions in per-layer reconstruction error (up to 80%) and consistent zero-shot accuracy improvements, particularly at higher sparsity levels where greedy methods degrade faster (Table 1, Figure 2).

## Weaknesses
- **Theory-practice gap driven by the $\alpha$-fixing heuristic:** The theoretical convergence and thresholding guarantees (Lemma 2) apply to the fully relaxed problem, yet the paper explicitly states that vanilla FW ($\alpha=0$) consistently underperforms baselines on perplexity. Empirical success hinges entirely on fixing $\alpha=0.9$ of high-saliency weights from a warmstart mask, meaning FW only refines the remaining 10%. The paper lacks a principled justification for this ratio and does not explain *why* the unconstrained convex relaxation diverges from global performance, making SparseFW function more as a localized corrector of Wanda than a standalone optimizer.
- **Unquantified computational overhead:** While the authors acknowledge SparseFW is "more compute-intensive than Wanda" and argue the tradeoff is worthwhile, the main text omits concrete wall-clock pruning times, FLOP counts, or peak VRAM profiling relative to baselines. Without a quantitative cost-benefit analysis, it is difficult to assess deployment viability or verify the "memory-efficient" and "scales to large models" claims under realistic production constraints.
- **Limited practical interpretability of the theoretical bound:** The error guarantee scales with $\lambda_{\max}(Q)$, where $Q = \text{Diag}(w)(XX^\top)\text{Diag}(w)$. Given the heavy-tailed weight distributions and activation outliers characteristic of modern LLMs, $\lambda_{\max}(Q)$ can be extremely large, potentially rendering the bound loose in practice. The paper does not empirically measure the condition number or bound tightness on representative layers, leaving the guarantee primarily asymptotic rather than diagnostically useful.

## Nice-to-Haves
- Report mean ± standard deviation across multiple random calibration seeds (even if in the appendix) to confirm that perplexity/accuracy gains are robust to calibration data variance.
- Move the $\alpha$-ratio ablation (Appendix Table 2) to the main text or provide a deeper analysis of how $\alpha$ interacts with sparsity levels and layer types, as it is the most empirically critical hyperparameter.
- Visualize the final mask differences between Wanda and SparseFW, or plot the correlation between local reconstruction error reduction and downstream perplexity deltas across layers, to better illustrate whether FW targets structurally important weights or merely corrects noise.
- Investigate Hessian conditioning or gradient alignment across layers to explain the local-to-global objective mismatch that necessitates the $\alpha$-fixing workaround.

## Novel Insights
The paper inadvertently reveals a fundamental decoupling in LLM pruning: greedy saliency scores macro-captures structural weight importance (explaining why fixing the top 90% is optimal), while convex optimization acts as a high-precision local corrector for the remaining ambiguous weights. This suggests that layerwise convex relaxation alone cannot overcome the misalignment between local reconstruction error and global cross-entropy loss, and that post-training compression at scale may fundamentally require hybrid strategies where global priors constrain local optimization rather than purely mathematical relaxation driving global performance.

## Suggestions
- Explicitly frame the contribution as a *constrained local refinement* paradigm rather than a full mask replacement, and dedicate a short subsection to analyzing the necessity and limitations of the $\alpha=0.9$ heuristic.
- Add a table or paragraph reporting pruning wall-clock time, peak memory footprint, and FLOPs for one representative model (e.g., LLaMA-3.1-8B) compared to Wanda, so reviewers can independently weigh the accuracy gains against computational costs.
- Briefly acknowledge in the limitations section that the theoretical bound may be loose for ill-conditioned layers due to activation outliers, and discuss whether preconditioning or layer-normalization-aware formulations could tighten it in practice.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 0.0]
Average score: 0.0
Binary outcome: Reject
