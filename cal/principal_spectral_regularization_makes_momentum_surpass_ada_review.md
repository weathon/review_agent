=== CALIBRATION EXAMPLE 30 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title claims "momentum surpass[es] Adam for LLM training," but the paper's own experimental results tell a more nuanced story. In the short-training (10K steps) regime, SGD-M-PSR does surpass AdamW. However, in the most practically relevant experiment—LLaMA-1.3B trained for 36B tokens—the paper openly concedes that PSR is "strictly worse than running Muon" and only marginally above AdamW (Table 4: 46.56 vs. 46.32 avg. for 0-shot). The abstract's framing ("can surprisingly surpass Adam") is technically accurate but omits this important qualification. A reader expecting robust, regime-independent superiority over AdamW will be misled. The abstract also asserts that PSR is more computationally efficient than Newton-Schulz, but Table 3 shows PSR is *slower* than Newton-Schulz for models up to 3B—the very scales at which the main LLM experiments are conducted.

---

### Introduction & Motivation (Section 1)

The problem setup is well-framed and the motivation to find an interpolation point between SGD-M and Muon is sensible. However, the contributions as stated have an internal tension: the third bullet claims "SGD-M-PSR surpasses AdamW in pretraining LLMs," while Section 5 later reveals this is heavily dependent on training length. This should be qualified upfront. The framing of Muon's Newton-Schulz overhead as the primary bottleneck is partially misleading—at the 350M–3B scales where the paper's training comparisons live, PSR is empirically slower.

---

### Insights Section: Spectral Visualizations (Section 3.1)

The spectral visualization analysis in Fig. 1 is the paper's strongest empirical contribution. The *spiked-head-heavy-tail* characterization is interesting and the contrast between attention and MLP layer spectral decay rates is a genuinely useful observation. One concern: Fig. 1 is produced at **1000 training steps** on a 350M model—a very early, warm-up stage. It is not established whether this spectral structure persists across training or is model-scale dependent. A heatmap at, say, step 5000 or 10000 would substantially strengthen the claim.

---

### Styblinski-Tang Toy Experiment (Section 3.2)

This section has a fundamental conceptual problem that is not addressed. The Styblinski-Tang function is *fully separable*: f(x) = Σᵢ (xᵢ⁴ - 16xᵢ² + 5xᵢ). Each dimension is independent—there are no cross-dimensional interactions. This means the gradient is an element-wise function of x, and the Hessian is **diagonal**. Critically, PSR is designed to operate on **weight matrices** with genuine spectral cross-structure (off-diagonal singular value interactions between input and output dimensions). Applying PSR to a 1D vector of size 1024 by treating it as a flat gradient (not a 2D matrix) does not exercise the matrix spectral structure the algorithm is designed to exploit. The toy experiment therefore demonstrates a different phenomenon than what PSR actually does in LLM training. It motivates "partial regularization of dominant gradient directions" abstractly, but not the specific Lanczos bidiagonalization + deflation design. The link between this toy experiment and the actual algorithm design is asserted rather than demonstrated.

Furthermore, the improvement in Table 1 over Adam with p=5%, d=64 is visually modest given the large standard deviations (implied by the range of values across configurations), and the "optimal" configuration conveniently matches the one transferred to LLM experiments—raising concerns about post-hoc selection.

---

### Method: PSR Algorithm (Section 4)

**Algorithm 1 notation issues:** The parameter η in Algorithm 1 is described as a "regularization factor" controlling attenuation (set to 0.95 in practice, meaning dominant directions are shrunk to 5%). However, in Algorithm 2, step 7, η appears again with a completely different meaning (the learning rate, set to 0.5). This dual use of η is confusing and indicative of notation not cleaned up carefully.

**Magic constant 0.18:** The update RMS rescaling factor of 0.18 (Alg. 2, step 7) is determined from only 2 time points (steps 1000 and 2000) on a single LLaMA-350M model (Table 6). No ablation on the sensitivity of results to this constant is provided, and no justification for why this value should generalize across architectures and scales (350M to 7B). This is a fragile design choice for a method presented as a general optimizer.

**Hyperparameter transfer:** The paper states that PSR hyperparameters (η=0.95, K=2, r=m/32) are "configured according to the optimal setup in the Styblinski-Tang function." However, the Styblinski-Tang experiment (Table 1) uses a vector (n=1024) not a matrix, a different η formulation (proportional scaling, not deflation), and different dimension coverage. The hyperparameter connection is tenuous. In practice, these values appear tuned on the LLM experiments (confirmed by the ablation in Table 9), with the toy experiment used post-hoc as motivation.

**PSR is applied to the Nesterov lookahead gradient, not the momentum.** This is a significant design choice (stated in the text: "we perform PSR on the lookahead gradient"). No ablation compares applying PSR to the momentum vs. the lookahead. The motivation for this choice—"according to empirical verifications of Muon"—is not self-contained.

**Complexity claim vs. practice:** Theorem 4.1 claims PSR overhead < ½m²n, roughly 2% of Muon's 30m²n. However, Table 3 directly shows that PSR is **more than 2× slower** than Newton-Schulz for LLaMA-1.3B attention layers (4.85ms vs 2.01ms) and for LLaMA-3B (5.54ms vs 4.09ms). The crossover only occurs at 7B. The paper mentions this discrepancy but attributes it to "sequential execution" and "lack of half-precision support," and defers to future kernelization. Since all the main LLM training experiments are at 350M–3B, the claimed efficiency advantage is not realized in practice for any of the experiments run. This is a significant credibility issue for the efficiency argument.

---

### Experiments & Results (Section 5)

**Positive aspects:** The experiments span four model scales (350M–7B), use a standard setup (LLaMA + C4/en, following Zhao et al. 2024a), include an extended 36B-token run, and cover nine downstream benchmarks. The SOAP comparison (Appendix D.3) is a valuable addition.

**Concerns:**

1. **The strongest experiment contradicts the title claim.** In the 36B-token LLaMA-1.3B run (Fig. 4a), Muon overtakes PSR in the later stages of training. The paper admits PSR is "strictly worse than running Muon." The regime where PSR outperforms AdamW but is surpassed by Muon is only moderately interesting; the paper's contribution is better described as "a cheaper alternative to Muon that mostly outperforms AdamW in early-to-mid training." This is a meaningful but more modest contribution than the title suggests.

2. **LLaMA-7B is undertrained.** The 7B model is trained for only 10,000 steps with local batch size 1 (gradient accumulation 64), totaling roughly 5.2B tokens. This is far below Chinchilla-optimal for a 7B model (~140B tokens). Comparisons at this scale are in the warm-up phase and may not reflect steady-state optimizer behavior. The claim of "substantial speed-up over AdamW" at 7B is therefore premature.

3. **Statistical significance.** Downstream benchmark comparisons in Tables 4 and 8 show differences of 0.1–1.0 percentage points on individual tasks. No standard deviations or confidence intervals are reported across runs with different seeds. Many comparisons (e.g., WinoGrande: 52.64 PSR vs. 52.25 Muon at 0-shot) are within what would be expected random variation.

4. **Missing baselines:** SWAN (Ma et al., 2024) is directly relevant—it also extends SGD with whitening/normalization—but is not compared. AdaMS (Zhang et al., 2025) is cited but excluded. The rationale for including SOAP but not SWAN is unexplained.

5. **All experiments on C4/en only.** No validation on other datasets (e.g., The Pile, SlimPajama, or FineWeb). Given that optimizer behavior can vary with data distribution and domain, this limits the generalizability claims.

6. **Fig. 3 and Fig. 4 are referenced out of order** in the text (Section 5.2 says "Fig. 3 presents the training dynamics" but then refers to "Fig. 4" for scaled experiments while the 1.3B/7B scaling figure appears to be Fig. 4). This ordering creates readability confusion.

---

### Ablation Studies (Appendix D)

Table 9 provides useful ablation of (η, m/r). However, the ablation only covers 2B-token training. At 10K steps with batch 512, this is still relatively early-stage. The trend that "larger rank improves performance but costs more" is intuitive but the optimal m/r=32 for LLaMA-1.3B requires 18.30 perplexity vs. Muon at 18.36—the gap is 0.06 in perplexity, which may not be significant. Reporting uncertainty estimates would help.

---

### Computational Complexity Analysis (Appendix E)

The proof is detailed and follows standard FLOP counting. One concern: the theorem statement says the bound holds when "16 ≤ m ≤ n," but the proof derives the tighter condition "n ≥ m ≥ 160" (the text says "m ≥ 16 > 15.6" but the derivation uses r = m/32 and requires 7mnr ≤ (1/2)m²n, which gives 7/32 ≤ 1/2 after simplification—this part is fine). However, the bound is loose enough that it says nothing about constant factors relevant for practical comparison. The wall-clock results are far more informative and, as noted, show PSR is worse than Newton-Schulz for ≤3B.

---

### Limitations & Broader Impact

The paper is honest about PSR not matching Muon in long-run training (Section 5.2), which is commendable. However, several additional limitations are unacknowledged:
- Sensitivity of the 0.18 rescaling constant across architectures
- The PSR is only applied to matrix-shaped parameters (like Muon); embedding tables and biases still use AdamW—no discussion of this hybrid approach
- No analysis of memory overhead beyond the per-layer comparisons in Table 3
- The code is not released (deferred to "upon publication"), limiting reproducibility

---

## Overall Assessment

This paper makes a genuine and interesting observation: selectively attenuating dominant spectral directions in SGD momentum is more effective than full orthogonalization (Muon) in some training regimes while being cheaper. The spectral visualization analysis is illuminating, and the efficiency gains at 7B+ scale are real. However, the contribution is weaker than presented. The key performance claims depend heavily on training length—PSR's advantage over AdamW shrinks with longer training and PSR remains strictly below Muon in the only long-run experiment. The motivating toy experiment (Styblinski-Tang) is conceptually flawed for the problem at hand since it's a separable function that cannot exhibit the matrix spectral structure PSR is designed to exploit. The method contains a hardcoded rescaling constant (0.18) derived from a single model at two time steps—a fragile design choice with no sensitivity analysis. In practice, PSR is slower than Newton-Schulz for all models below 7B, contradicting the main efficiency selling point for the scales where experiments are actually run. For ICLR, the bar for optimizer papers is high: they typically require either strong theoretical guarantees, consistent empirical dominance across a wide range of settings, or compelling computational advantages. This paper achieves none of these robustly, and revisions addressing the Styblinski-Tang conceptual issue, the efficiency-at-scale gap, the 0.18 constant fragility, and the long-run performance gap would be needed to make a convincing case.

# Neutral Reviewer
## Balanced Review

### Summary
This paper challenges the necessity of full momentum orthogonalization in LLM training by proposing Principal Spectral Regularization (PSR), which selectively penalizes dominant spectral components using Lanczos bidiagonalization rather than computing the full matrix inverse via Newton-Schulz iteration. The authors argue that the "spiked-head-heavy-tail" structure of LLM momentum makes full orthogonalization computationally expensive and potentially suboptimal. Through experiments on LLaMA models (350M to 7B), they demonstrate that PSR allows SGD with Momentum to consistently outperform AdamW, while achieving competitive performance with Muon at lower theoretical computational overhead.

### Strengths
1.  **Motivation from Spectral Structure:** The paper provides a clear and insightful motivation by visualizing the spectral distribution of momentum in LLMs (Fig 1). The observation that gradients exhibit a "spiked-head-heavy-tail" structure effectively sets the stage for questioning why existing methods like Muon enforce uniform orthogonalization across all directions.
2.  **Scalability Analysis:** The theoretical complexity analysis (Theorem 4.1) and runtime comparison (Table 3) offer valuable practical insights. The paper demonstrates that for large dimensions ($m \ge 16$), the computational overhead of PSR ($O(m^2n)$) is significantly lower than the full matrix orthogonalization required by Muon ($30m^2n$), claiming a reduction to approximately 2% extra FLOPs compared to standard SGD.
3.  **Comprehensive Empirical Validation:** The authors conduct extensive experiments across multiple model scales (350M, 1.3B, 3B, 7B) and benchmarks (Table 4, Table 8, Table 9). This breadth provides strong evidence regarding the optimizer's robustness and helps identify regimes where PSR excels (early training, smaller models) versus where it trails Muon (later stages, larger loss minimization).

### Weaknesses
1.  **Performance Discrepancy vs. Claims:** While the title and abstract claim PSR enables Momentum to "Surpass Muon," the results consistently show Muon achieving better convergence and downstream performance in the long term (Section 5.2, Appendix D). The paper explicitly admits that PSR is "strictly worse than running Muon" in the steady training stage. This creates a disconnect between the ambitious framing (Surpassing the new SOTA) and the empirical reality (Better than Adam, slightly worse than Muon).
2.  **Practical Runtime Trade-offs:** Although Theorem 4.1 claims 2% overhead, Table 3 reveals that in practice, PSR can be slower than Muon in wall-clock time for certain matrix dimensions (e.g., LLaMA-7B Attention: 9.63ms vs 14.83ms is not a clear win for PSR relative to overhead, and for 1.3B Attention PSR is 4.85ms vs 2.01ms which is slower). The sequential nature of the Lanczos/deflation steps in PyTorch negates the theoretical FLOP advantages for smaller models, a nuance that should be discussed more deeply as a limitation.
3.  **Weak Theoretical Justification for "Partial":** The connection between the Styblinski-Tang function analysis (Fig 2) and LLM pretraining is acknowledged by the authors as "vague" (Appendix D). The paper lacks a rigorous mathematical proof or geometric intuition explaining *why* regularizing only the head of the spectrum is theoretically optimal for general LLM training landscapes, relying instead on empirical observation.

### Novelty & Significance
**Novelty:** Moderate. The concept of spectral regularization is well-established (Shampoo, SOAP, Muon). The specific contribution of PSR—using block Lanczos for partial head-regularization to approximate full orthogonalization less expensively—is a novel engineering and algorithmic approach. However, the theoretical novelty regarding the optimality of partial spectral directions without full preconditioning remains an open empirical question.

**Significance:** High, given the current focus on reducing LLM pretraining costs. If PSR can match Muon's sample efficiency at significantly lower memory/compute overhead, it would be a practical optimization for industry. However, the gap between theoretical FLOPs and practical wall-clock time must be bridged for this significance to materialize.

### Suggestions for Improvement
1.  **Align Claims with Results:** Adjust the title and abstract to reflect that PSR surpasses AdamW and is competitive with (but not necessarily superior to) Muon in all settings. Avoid the "Surpass Muon" narrative when the data shows Muon converges to lower loss in long-term training.
2.  **Clarify Runtime Overhead:** Provide a detailed discussion on the sequential bottlenecks of the Lanczos procedure. The theoretical FLOP savings are clear, but the practical latency (due to lack of kernelization) is the real bottleneck in LLM training. Compare against AdamW on the *same* hardware implementation details more closely.
3.  **Strengthen Theoretical Analysis:** Include a deeper discussion or lemma explaining the spectral trade-off. Why does preserving the heavy tail (via PSR) outperform flattening it (via Muon) in some regimes? A connection to generalization bounds or loss landscape geometry would strengthen the theoretical contribution beyond empirical observation.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Train to Chinchilla-optimal token counts** — Current experiments stop at 36B tokens or 10K steps, which is insufficient to claim LLM pretraining superiority; ICLR expects training curves that demonstrate sustained performance, not just warm-up advantages.
2. **Add SOAP and GaLore as direct baselines** — Both are low-rank spectral methods closely related to PSR; without comparison, the claim that PSR offers a unique efficiency-accuracy tradeoff is unsupported.
3. **Report wall-clock time per training step, not just theoretical FLOPs** — Table 3 shows PSR can be slower than Newton-Schulz for smaller models due to kernel inefficiency; the 2% FLOP claim is misleading without actual runtime measurements at scale.
4. **Test on multiple architectures beyond LLaMA** — Claims about LLM pretraining generalization require validation on at least one non-LLaMA architecture (e.g., GPT-2, PaLM-style) to rule out architecture-specific effects.

### Deeper Analysis Needed (top 3-5 only)
1. **Explain why PSR underperforms Muon in later training stages** — The paper admits PSR is "strictly worse than running Muon" in steady-state training but offers no mechanistic analysis; without this, the core insight about partial vs. full orthogonalization remains speculative.
2. **Quantify the stability-variance tradeoff across training phases** — Fig. 4 shows PSR converges faster during warm-up but plateaus; analysis of gradient variance or effective learning rate across phases would clarify when PSR helps vs. hurts.
3. **Connect Styblinski-Tang results to LLM spectra** — The synthetic function experiments use arbitrary weight distributions; without showing that the optimal K=64 or p=5% matches actual LLM momentum spectra, the motivation appears post-hoc.

### Visualizations & Case Studies
1. **Plot which spectral directions are regularized over training time** — A heatmap showing which singular values are suppressed at step 1K vs. 10K would reveal whether PSR adapts or uses fixed directions, directly testing the "principal components" claim.
2. **Show failure cases where PSR diverges or becomes unstable** — Table 9 hints that η=0.975+ causes instability; visualizing the loss curves or gradient norms in these regimes would expose the method's operational boundaries.

### Obvious Next Steps
1. **Kernelize the Lanczos/SVD operations for GPU efficiency** — The paper admits sequential PyTorch operations negate theoretical speedups; a custom CUDA kernel or torch.compile integration should be included to validate the efficiency claim.
2. **Extend experiments to 7B+ models at meaningful token counts** — The 7B results are limited to 10K steps; ICLR expects evidence that gains persist at scales where optimizer choice actually matters for compute budgets.

# Final Consolidated Review
## Summary

This paper proposes Principal Spectral Regularization (PSR), which selectively attenuates dominant spectral components in SGD momentum using Lanczos bidiagonalization, rather than performing full matrix orthogonalization like Muon. The method is motivated by the observation that momentum in LLM training exhibits a "spiked-head-heavy-tail" spectral structure, where a few dominant directions capture most variance. The authors demonstrate that SGD-M-PSR can outperform AdamW on LLaMA models (350M–7B) during early-to-mid training, though Muon remains superior in long-run training scenarios.

## Strengths

- **Spectral visualization provides genuine insight**: Figure 1's characterization of the "spiked-head-heavy-tail" structure in momentum spectra, and the contrast between attention layers (sharper decay) and MLP layers (more uniform), offers a concrete empirical foundation for questioning why full orthogonalization treats all directions equally. This observation is original and useful for understanding optimizer behavior.

- **Theoretical complexity analysis is sound**: Theorem 4.1 correctly establishes that PSR's overhead is bounded by O(½m²n), approximately 2% of Muon's 30m²n FLOPs. The proof in Appendix E is detailed and follows standard FLOP counting conventions.

- **Comprehensive empirical validation across scales**: Experiments span four model sizes (350M–7B), include an extended 36B-token run, and cover nine downstream benchmarks. The SOAP comparison in Appendix D.3 provides additional context for low-rank spectral methods.

- **Honest about limitations**: The paper explicitly states that PSR is "strictly worse than running Muon" in later training stages (Section 5.2), acknowledging rather than hiding this regime dependency.

## Weaknesses

- **Empirical efficiency advantage emerges only above 7B parameters**: Table 3 shows that PSR is slower than Newton-Schulz in wall-clock time for models ≤3B (e.g., LLaMA-1.3B attention: 4.85ms vs. 2.01ms). Since all main experiments are conducted at 350M–3B, the claimed efficiency advantage does not materialize in practice for the reported results. The paper attributes this to "sequential execution" in naive PyTorch but defers kernelization to future work.

- **Training experiments are insufficiently long**: The 7B model is trained for only 10,000 steps (~5.2B tokens), far below Chinchilla-optimal scale (~140B tokens). The 36B-token LLaMA-1.3B experiment shows Muon overtaking PSR in later stages. Stronger claims about optimizer superiority require training curves that demonstrate sustained performance, not just warm-up advantages.

- **The 0.18 rescaling constant lacks justification for generalization**: This value is derived from only two time points (steps 1000 and 2000) on a single LLaMA-350M model (Table 6). No sensitivity analysis or ablation is provided to show this constant works across architectures, scales, or training stages.

- **Styblinski-Tang experiment has conceptual mismatch with PSR's matrix design**: The function f(x) = Σᵢ(xᵢ⁴ - 16xᵢ² + 5xᵢ) is fully separable with a diagonal Hessian. Applying PSR to a 1D vector treats it as a flat gradient rather than a matrix with genuine off-diagonal spectral structure. The experiment motivates "partial gradient regularization" abstractly but does not validate the specific Lanczos bidiagonalization + matrix deflation design used for LLM weight matrices.

- **Notation inconsistency confuses implementation details**: In Algorithm 1, η is the regularization factor (attenuation strength), but in Algorithm 2 line 7, η appears as the learning rate with a different meaning. This dual usage makes reproduction more difficult.

- **Limited architectural diversity**: All experiments use LLaMA architectures on C4/en. No validation on other transformer variants (e.g., GPT-2, decoder-only with different attention patterns) or datasets (The Pile, FineWeb) is provided, limiting generalization claims.

## Nice-to-Haves

- Kernelized Lanczos/SVD operations to validate efficiency claims at practical scales
- Experiments on at least one non-LLaMA architecture to test generalization
- Training runs approaching Chinchilla-optimal token counts for the larger models
- Ablation on the 0.18 rescaling constant across architectures and training stages
- Analysis of which spectral directions PSR regularizes over training time (do they change? are they stable?)

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Title claims surpassing Muon"** (Harsh Critic, Balanced Reviewer): The title actually says "Momentum Surpass Adam for LLM Training," not Muon. This is a misreading of the paper.

- **"Missing SOAP baseline"** (Spark Finder): The paper includes SOAP comparison in Appendix D.3 and Figure 5, showing SOAP outperforms AdamW but trails SGD-M-PSR and Muon.

- **"No standard deviations reported"** (Harsh Critic): Table 7 in Appendix reports mean ± std across 1000 random initializations for Styblinski-Tang experiments. For LLM experiments, the computational cost of multiple full runs is prohibitive, which is standard practice in this research area.

- **"Figures referenced out of order"** (Harsh Critic): The paper's figure ordering follows the text flow correctly—Fig. 3 shows training dynamics for 350M/1.3B/3B models, Fig. 4 shows scaled training scenarios. This is standard narrative structure.

- **"Citing unreleased methods"** (Harsh Critic): All cited works (Muon, SOAP, SWAN, AdaMS) are referenced with arXiv preprints and follow ICLR's acceptance of preprint citations.

## Novel Insights

The paper's key insight—that full matrix orthogonalization is unnecessary because LLM momentum spectra exhibit "spiked-head-heavy-tail" structure with only a few dominant directions—is genuinely novel. This challenges Muon's design assumption that uniform amplification of all spectral directions is beneficial. The observation that attention layers exhibit sharper spectral decay than MLP layers suggests different layer types may benefit from different spectral treatment, opening a direction for layer-adaptive regularization. However, the finding that Muon overtakes PSR in later training suggests the heavy-tail directions (which Muon amplifies uniformly but PSR preserves unchanged) become more important as training progresses—a hypothesis the paper acknowledges but does not mechanistically explain.

## Suggestions

- Develop GPU kernels for Lanczos bidiagonalization to realize the theoretical efficiency gains at practical model scales. The 2× slowdown at ≤3B models undermines the efficiency argument for the scales where experiments are actually run.

- Add an ablation comparing PSR applied to momentum vs. the Nesterov lookahead gradient. The current design applies PSR to the lookahead gradient without justification or comparison.

- Extend the 7B experiments to meaningful token counts (at least 20–50B tokens) to validate whether early-stage advantages persist or diminish.

- Provide a mechanistic explanation for why PSR underperforms Muon in later training. Is it because the preserved heavy-tail directions become noise? Or because Muon's uniform amplification helps escape local minima? Gradient variance or effective learning rate analysis across training phases would clarify this.

- Consider dynamic selection of K (number of regularized components) based on spectral decay rate, rather than a fixed m/32 proportion, to adapt to layer-specific spectral structure.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 4.0, 4.0, 2.0]
Average score: 3.2
Binary outcome: Reject
