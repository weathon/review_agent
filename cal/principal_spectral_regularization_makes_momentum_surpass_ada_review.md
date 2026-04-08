=== CALIBRATION EXAMPLE 16 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title ("Principal Spectral Regularization Makes Momentum Surpass Adam for LLM Training") is technically supported but somewhat misleading. The paper's actual results show that SGD-M-PSR **falls short of Muon** in extended training (36B tokens), which is the more relevant comparison since Muon is the primary motivation. The abstract's claim that "Momentum with marginal spectral regularization on very few dimensions can surprisingly surpass Adam" is accurate but positions this as a breakthrough when it is more of a middle ground between plain SGD-M and Muon. The abstract frames full orthogonalization as "suboptimal in some cases" — the paper only demonstrates this on a scalar toy function and under early-stage training dynamics, which is a significant gap between claim and evidence.

---

### Introduction & Motivation (Section 1)

The motivation — that full matrix orthogonalization (Newton-Schulz in Muon) may be unnecessary or suboptimal — is interesting and worth investigating. However, the introduction conflates two separate arguments: (1) theoretical suboptimality of full orthogonalization, and (2) computational efficiency of a partial alternative. These motivations are somewhat in tension: the Styblinski-Tang example is used to argue that full orthogonalization is *suboptimal for convergence*, but in the LLM results, PSR never actually improves over Muon in long-run training — so the efficiency argument is the stronger (and more honest) case. The introduction should be more careful about which claim it is making.

The framing of Adam's second moment as a form of "spectral regularization" is an interesting perspective but is stated as if obvious when it requires more justification — Adam does not penalize dominant directions in the way PSR does; it *normalizes* coordinate-wise variance.

---

### Related Work (Section 2)

Reasonably thorough. The authors correctly identify the gap between spectral norm regularization (top-1 only) and full matrix preconditioning (Muon/Shampoo), positioning PSR as a middle ground. However, SWAN (Ma et al., 2024) is mentioned in related work but never compared experimentally in the main paper — it has conceptual similarity (normalization of gradients) and should appear in the comparison.

---

### Section 3: Insights on Spectral Regularization

**3.1 – Spectral visualization of LLM training:** The "spiked-head-heavy-tail" observation is well-presented and is the strongest empirical motivation for the work. The differential behavior between attention and MLP layers (attention spectra decay more sharply) is a genuine and useful insight. However, Fig. 1 only shows results at step 1000 of a 350M model — it is unclear whether this spectral structure is consistent across model scales and throughout training. The claim about Muon "amplifying noisy directions" is stated without empirical verification: the paper does not show that tail directions in the Muon momentum are actually noisy rather than useful.

**3.2 – Styblinski-Tang experiments:** This is the conceptually weakest part of the paper. The authors themselves acknowledge in Appendix D.1 that "the connection between mathematical function optimization and LLM pretraining is relatively vague." Several issues:

- The Styblinski-Tang function is a separable function, meaning there is no coupling between dimensions. The advantage of partial orthogonalization in this separable setting may not generalize to highly coupled parameter matrices in transformers.
- The power-law weight is introduced to mimic heavy-tailed spectra, but it is unclear why the *optimization problem* should have this structure (as opposed to the *gradient covariance*).
- Table 1 shows differences in final loss like 5.5561 vs. 5.6698 (SGD-M). With no statistical error bars reported (standard deviations or confidence intervals), these differences are hard to assess. The appendix (Tab. 7) provides SD but only in a reduced setting with n=256 and 2000 steps.
- The optimal hyperparameters (K=2, r=m/32, η=0.95, p=5%) are selected on this toy problem and then directly transferred to LLM pretraining. There is no ablation *on LLMs* to justify this transfer, only an after-the-fact Tab. 9.

---

### Section 4: Principal Spectral Regularization Method

**Algorithm 1 (PSR):** The Lanczos bidiagonalization approach is well-motivated and technically sound in its basic construction. However:

- There is a **notation inconsistency**: in Algorithm 1, the regularization factor is called η (with value 0.95 in the prose), but in Algorithm 2 (SGD-M-PSR), the PSR call uses η=0.5. This is confusing and unexplained. Is η=0.5 in the actual LLM experiments, or 0.95? The ablation table (Tab. 9) uses 1−η = 5%, implying η=0.95, but Algorithm 2 states η=0.5.

- **The 0.18 rescaling constant (Algorithm 2)** is empirically calibrated on a 350M model and then used universally. This is a "magic constant" with no principled justification. Table 6 shows RMS measurements only at steps 1000 and 2000 of a 350M model — the claim that this holds across all scales and throughout training is not verified.

- The QR-Orthogonal subroutine (lines 2-6 of BIDIAGONAL) performs classical Gram-Schmidt, which is numerically less stable than modified Gram-Schmidt. For half-precision BFloat16 training, this numerical concern should be acknowledged.

**Theorem 4.1 / Complexity Analysis:** There is a discrepancy in the paper: the theorem states the condition as "16 ≤ m ≤ n" but the subsequent discussion says "160 ≤ m ≤ n holds for all LLMs." These are different conditions, and the discussion should match the theorem. The 2% figure relative to Muon's 30m²n overhead is correct under the K=2, r=m/32 parameterization, but the critical issue is:

- **Wall-clock time** (Table 3) tells a different story: PSR is *slower* than Newton-Schulz for Attention and MLP layers at both 1.3B and 3B parameter scales (the scales where all the main experiments are run). The paper attributes this to sequential QR/SVD execution and lack of kernel optimization, and suggests improvements at 7B+, but the **main experiments are at 350M–3B**. There is a significant disconnect between the theoretical efficiency claim and practical runtime for the scales actually tested.

**Table 2 (matrix property comparison):** The "spectral fidelity" D_spec metric (lowest for PSR) is described as desirable because PSR preserves the tail, but this metric is presented without enough context. The claim that PSR "achieves comparable subspace distance" to Newton-Schulz and QR is for a single random matrix — no distribution over matrices is reported.

---

### Section 5: Experiments

**Setup:** The experimental setup follows Zhao et al. (2024a) and is standard. The C4/en dataset is appropriate. However:

- All experiments use the *same learning rate* (3×10⁻⁴) for all optimizers across all scales. No per-optimizer LR tuning is performed. This is a common issue in optimizer comparisons that can favor certain methods; AdamW's absolute level of tuning may differ from SGD-M-PSR's optimal LR.
- The 10K-step evaluation corresponds to ~5.2B tokens (batch 512 × seqlen 1024 × 10K steps). For a 7B parameter model, this is a very early-stage snapshot (below Chinchilla-optimal) and results may not be representative of final capability ordering.

**Results (Fig. 3 and 4, Tables 4 and 8):**

- The core finding — SGD-M-PSR beats AdamW — is substantiated across 350M, 1.3B, 3B, and 7B scales in perplexity and downstream tasks. This is a genuine positive result.
- However, the **honest bottom line** from the 36B-token experiment (Fig. 4a, Table 4) is that **Muon outperforms PSR** in both perplexity and downstream evaluation after extended training. PSR's early-stage convergence advantage disappears. The paper acknowledges this but may not emphasize it sufficiently given the framing in the title and abstract.
- For the LLaMA-7B experiment (Fig. 4b), only 10K steps (about 5B tokens) are shown, which is far from convergence. The small gap between PSR and Muon at 10K steps may or may not persist; the authors acknowledge needing more compute but this leaves the 7B conclusion open.
- **No error bars** are reported for any LLM experiments (single seeds). For perplexity differences that are sometimes small (e.g., 22.49 vs. 22.54 in Tab. 9 at 350M), this is important.

**Missing ablations / comparisons:**
- No comparison to SWAN (conceptually related: normalizes SGD gradients), only mentioned in related work.
- No study of how PSR interacts with learning rate schedules differently than Adam or Muon.
- No embedding or unembedding layer treatment — Muon is typically applied only to hidden-layer weight matrices. Is PSR applied to all layers including embedding?
- Tab. 9 ablates η and m/r but only for 350M and 1.3B, and only at 2B tokens — not the full 36B run.

---

### Writing & Clarity

A few structural issues that impede understanding: the Algorithm 2 box appears mid-paper (between the complexity theorem and its discussion), interrupting the flow. The η=0.5 in Algorithm 2 vs. η=0.95 in the prose is never reconciled. The sentence in Section 5.2 "SGD-M outperforms AdamW with PSR across most benchmarks" (Tables 4 and 8 captions) contradicts itself grammatically — it should read "SGD-M with PSR outperforms AdamW."

---

### Limitations & Broader Impact

The authors are relatively honest in the conclusion about PSR not matching Muon in downstream performance or long-run training, which is commendable. However, the paper does not adequately address:

1. **Lack of convergence theory:** No formal convergence guarantees are provided for PSR, even in simple settings. The method is purely empirically motivated.
2. **Hyperparameter sensitivity:** The "constant" hyperparameters (0.18 rescaling, η=0.95, r=m/32, K=2) are transferred from a toy problem. If these require per-problem tuning, the practical advantage diminishes.
3. **Half-precision numerical stability** of the Lanczos procedure is not discussed.
4. **Scalability beyond 7B:** The wall-clock advantage is shown for 7B+ layers in a microbenchmark, but no actual training runs at 7B+ are presented.

---

### Overall Assessment

This paper makes a genuine and interesting observation — that selectively penalizing dominant spectral directions in momentum can partially replicate the benefits of full matrix orthogonalization at lower computational cost — and demonstrates this concretely for LLM pretraining up to 1.3B tokens (36B tokens). The empirical finding that SGD-M-PSR beats AdamW is robust across scales. However, the contribution falls short of its framing in several respects. The primary theoretical motivation (full orthogonalization is *suboptimal*, not just *expensive*) is supported only by a separable toy function where the analogy to LLM training is acknowledged to be weak. The method itself relies on multiple empirically-tuned constants (0.18 scaling, η=0.95, r=m/32) with limited ablation and no theoretical grounding. The claimed computational advantage does not hold in wall-clock time at the scales actually tested (1.3B–3B). Most critically, PSR is strictly worse than Muon in extended training, making it not a replacement but a lower-fidelity approximation. For ICLR, this work sits below the acceptance threshold in its current form: the insights are genuine but the evidence for the key claims is insufficient, the method has unexplained inconsistencies (notably the η discrepancy), and the honest contribution — a cheaper but weaker alternative to Muon — is obscured by overreaching framing. A significant revision strengthening the theoretical underpinning, resolving the algorithmic inconsistencies, and recalibrating the claims relative to the empirical evidence would substantially improve the paper.

# Neutral Reviewer
## Balanced Review

### Summary
This paper investigates whether full momentum orthogonalization (e.g., Muon's Newton-Schulz iteration) is necessary or optimal for LLM pretraining, observing instead that training dynamics exhibit a "spiked-head-heavy-tail" spectral structure where only a few directions dominate updates. The authors propose Principal Spectral Regularization (PSR), an optimization update transformation that uses block Lanczos bidiagonalization to identify and selectively penalize the top-$K$ dominant spectral components of the momentum matrix while preserving the heavy-tailed structure. Empirical results show that PSR-enhanced SGD with momentum consistently outperforms AdamW and achieves competitive validation perplexity and downstream scores with Muon across LLaMA-350M to 7B scales, while theoretically requiring only ~2% of Muon's orthogonalization FLOPs.

### Strengths
1. **Strong empirical motivation and clear spectral analysis:** Figure 1 convincingly demonstrates the "spiked-head-heavy-tail" structure in LLaMA attention and MLP layers across training steps. This visualization directly motivates the hypothesis that full orthogonalization may waste compute on low-variance directions and grounds the PSR design in observable training dynamics.
2. **Practical algorithmic design with theoretical bounds:** Algorithm 1 provides a clear block-Lanczos procedure that bridges randomized SVD and momentum deflation. Theorem 4.1 in Appendix E rigorously bounds PSR's overhead at $O_{\text{overhead}} < \frac{1}{2}m^2n$, offering a concrete theoretical advantage over the $30m^2n$ bound of Muon's 5-step Newton-Schulz iteration. This bridges the gap between low-rank projection and full-matrix preconditioning.
3. **Comprehensive empirical evaluation and ablations:** The paper evaluates multiple model scales (350M to 7B) and includes extended training (36B tokens for 1.3B), downstream benchmarks (Table 4, 8), wall-clock/memory profiling (Table 3), and hyperparameter ablations on regularization strength and rank ratios (Table 9). This multi-angle validation aligns well with ICLR's expectation for thorough optimizer benchmarking.

### Weaknesses
1. **Runtime inefficiency contradicts theoretical FLOP gains for practical scales:** Table 3 reveals that PSR is significantly slower than Newton-Schulz for matrix dimensions up to 2560×2560, primarily due to sequential PyTorch QR/SVD calls. PSR only becomes faster in wall-clock time at 70B-scale dimensions. While the authors acknowledge this, the practical efficiency claim remains weak for the majority of current academic and mid-scale industry training setups without custom CUDA kernels.
2. **Inconsistent narrative and performance claims:** The title and abstract emphasize "surpass Adam," which is supported by the data. However, Section 5.2 explicitly acknowledges that SGD-M-PSR is "strictly worse than running Muon" during the loss-steady stage, and downstream averages (Tables 4, 8) show Muon still frequently leads. The framing occasionally overstates PSR's dominance relative to Muon, creating a slight mismatch between claims and results.
3. **Notation and parameter discrepancies:** Algorithm 2 uses $\eta$ to denote the learning rate (Line 7) while the PSR function call in Line 6 also sets a regularization factor $\eta=0.5$, despite the main text (Section 4) stating the optimal regularization factor is $\eta=0.95$. This symbol collision and value inconsistency reduce clarity and hinder reproducibility without code inspection.
4. **Limited training scale and architecture diversity for LLM optimizer claims:** Experiments cap at 7B parameters and 10,000 steps for the largest models (except the 1.3B/36B token run). Given ICLR's high bar for LLM optimization papers, the absence of convergence behavior on ≥7B models trained to ~100k+ steps or across different architectures (e.g., Mixtral, Llama-3 variants) limits the certainty of generalization claims. The comparison also relies on a fixed 5-step Muon schedule without exploring Muon's own learning rate or iteration sensitivity.

### Novelty & Significance
**Novelty:** Moderate-High. Applying truncated Lanczos bidiagonalization to selectively deflate dominant momentum directions is a novel interpolation between spectral regularization and full-matrix orthogonalization. While partial SVD and Lanczos are classical numerical linear algebra tools, their targeted application to momentum spectrum shaping for LLMs is timely and distinct from existing low-rank projection optimizers (e.g., GaLore, SOAP) that modify the gradient space rather than deflating the momentum spectrum directly.
**Clarity:** Generally high. The paper is well-structured, with clear algorithms, illustrative figures, and a logical progression from spectral observation to toy problem analysis to LLM experiments. Minor notation issues and some dense complexity derivations slightly hinder readability.
**Reproducibility:** Strong. The authors provide detailed experimental setups, hyperparameter tables (Appendix B), dataset references, and a reproducibility statement promising code release. The algorithmic steps are explicit, though the PyTorch implementation inefficiencies noted by the authors must be resolved for fair runtime replication.
**Significance:** High. The work challenges the prevailing trend toward full orthogonalization or heavy second-order approximation by demonstrating that marginal spectral conditioning can recover most benefits at a fraction of the theoretical cost. If kernelized and validated at larger scales, PSR could offer a highly efficient alternative to memory-heavy optimizers in resource-constrained LLM training regimes.

### Suggestions for Improvement
1. **Resolve notation and hyperparameter inconsistencies:** Use distinct symbols for the learning rate (e.g., $\alpha$ or $\lambda$) and the PSR regularization factor $\eta$. Update Algorithm 2 to match the empirically optimal $\eta=0.95$ reported in Section 4 and Table 9, or provide a principled justification for the 0.5 value used in the algorithm pseudocode.
2. **Strengthen practical efficiency claims:** Provide estimates or profiling of a lightweight CUDA implementation for the Lanczos/QR steps, or explicitly frame the current runtime results as a "naïve implementation upper bound." Showing the crossover point where PSR strictly dominates in wall-clock time (e.g., specific dimension thresholds) would better anchor the efficiency narrative.
3. **Align claims with empirical trade-offs:** Soften or clarify statements comparing to Muon. A dedicated "Limitations / Trade-off Analysis" subsection summarizing when PSR excels (early training, memory-constrained settings, moderate scales) vs. when Muon retains an advantage (long-horizon steady-state convergence) would improve scientific rigor and better reflect the results in Section 5.2.
4. **Expand empirical scope or explicitly frame limitations:** If training larger models is computationally infeasible, explicitly discuss this limitation and analyze the scaling trends from Table 9 (rank proportion vs. perplexity) to forecast behavior at 13B/70B scales. Additionally, consider adding a brief sensitivity analysis for learning rate scheduling, as momentum-based optimizers often interact strongly with warmup/decay profiles, which could solidify the robustness claims for ICLR acceptance.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Wall-clock Convergence Curves:** Plot validation loss against wall-clock time, not just training steps, because Table 3 shows PSR is slower than Muon on smaller models despite lower theoretical FLOPs. Without this, the claim of computational efficiency is unverified.
2. **Memory Comparison vs. AdamW:** Provide explicit peak memory measurements comparing PSR directly against AdamW, not just Muon, since AdamW is the primary baseline being challenged. The paper claims efficiency but fails to quantify the memory saving over the standard optimizer.
3. **Statistical Significance Testing:** Repeat LLM pretraining runs with at least 3 different random seeds to include error bars on training curves and downstream tables. Single-run results are insufficient for ICLR given the high variance in LLM training.
4. **Hyperparameter Robustness Sweep:** Conduct a systematic sweep of the regularization factor $\eta$ and rank $r$ across different model sizes (350M to 7B) to prove the method is not overfitted. Table 9 suggests sensitivity, but the main claims rely on tuned "optimal" settings.
5. **Scaling Beyond 7B Parameters:** Validate the method on models larger than 7B (e.g., 13B or 30B) to support the claim of suitability for "LLM Training." Performance trends at 7B do not guarantee scalability to state-of-the-art sizes.

### Deeper Analysis Needed (top 3-5 only)
1. **FLOPs vs. Runtime Discrepancy:** Explain why the theoretical 2% FLOP overhead translates to slower wall-clock times than Muon for models under 7B in Table 3. This contradiction undermines the core efficiency argument and requires a hardware-aware analysis.
2. **Late-Training Performance Degradation:** Analyze why PSR is surpassed by Muon in later training stages (Section 5.2) and whether the "heavy-tail" preservation becomes detrimental over time. The paper admits this weakness but does not explain the mechanistic cause.
3. **Memory Overhead Quantification:** Explicitly calculate the auxiliary memory cost of PSR buffers versus AdamW's second-moment storage to validate the memory efficiency claim. The current text discusses FLOPs but lacks a concrete memory budget breakdown.
4. **Toy Function to LLM Generalization:** Provide a stronger analytical justification linking the Styblinski-Tang function results to actual LLM loss landscapes. The current connection is asserted via visual similarity rather than mathematical grounding.
5. **Layer-wise Mechanism Validation:** Investigate whether PSR benefits Attention and MLP layers differently, as suggested by Figure 1, rather than applying uniform regularization. Understanding this could explain why uniform orthogonalization (Muon) behaves differently.

### Visualizations & Case Studies
1. **Loss vs. Wall-clock Time:** Plot training loss against actual training time to reveal if the reduced FLOP count translates to real-world speedups. This is the only way to verify the efficiency claim given the runtime data in Table 3.
2. **Spectral Distribution Evolution:** Visualize the momentum spectrum at early, middle, and late training stages to show if PSR maintains the desired structure over time. This would expose whether the regularization effect degrades as training progresses.
3. **Memory Scaling Plot:** Graph peak GPU memory usage against model size for AdamW, Muon, and PSR to visually demonstrate the memory efficiency trajectory. This directly supports the contribution regarding resource-intensive pretraining.
4. **Hyperparameter Sensitivity Heatmap:** Display performance metrics across a grid of $\eta$ and $K$ values to visualize the stability of the method. This reveals whether the method is practical or requires exhaustive tuning.
5. **Layer-wise Perplexity Contribution:** Breakdown downstream performance gains by layer type (Attention vs. MLP) to validate the spectral hypothesis. This confirms whether the method works for the reasons stated in the introduction.

### Obvious Next Steps
1. **Fused CUDA Kernel Implementation:** Develop and benchmark a custom CUDA kernel for the Lanczos and deflation steps, as the paper admits the naive PyTorch implementation is a bottleneck. Without this, the efficiency claims remain theoretical.
2. **Convergence Theory for Non-Convex Settings:** Provide theoretical convergence guarantees for the deflation method in non-convex optimization to meet ICLR standards. The current paper relies entirely on empirical evidence without theoretical backing.
3. **Comparison with Adafactor and Shampoo:** Include direct comparisons against Adafactor and Shampoo, which are the true memory-efficient spectral baselines, not just Muon. Omitting these weakens the positioning within the optimizer landscape.
4. **Fine-tuning and RLHF Stability:** Evaluate the optimizer on supervised fine-tuning (SFT) and RLHF tasks, as pretraining performance does not guarantee downstream utility. This is critical for claims regarding "LLM Training" broadly.
5. **Validation on >10B Parameter Models:** Extend experiments to models exceeding 10 billion parameters to verify scaling laws. Claims about LLM pretraining are not credible without evidence at modern scales.

# Final Consolidated Review
## Summary

This paper investigates whether full momentum orthogonalization (as in Muon's Newton-Schulz iteration) is necessary for LLM pretraining, observing that training dynamics exhibit a "spiked-head-heavy-tail" spectral structure where only a few directions dominate updates. The authors propose Principal Spectral Regularization (PSR), which uses block Lanczos bidiagonalization to selectively penalize dominant spectral components while preserving the heavy-tailed structure. Empirical results show PSR-enhanced SGD with momentum consistently outperforms AdamW across LLaMA scales from 350M to 7B parameters, while requiring only ~2% of Muon's theoretical orthogonalization FLOPs.

## Strengths

- **Strong empirical motivation from spectral analysis:** Figure 1 convincingly demonstrates the "spiked-head-heavy-tail" structure in LLaMA attention and MLP layers at step 1000. The observation that attention spectra decay more sharply than MLP spectra provides genuine insight into why uniform treatment across layers may be suboptimal. This visualization directly grounds the PSR design in observable training dynamics.

- **Theoretical efficiency advantage with rigorous bounds:** Theorem 4.1 establishes that PSR's overhead is bounded by O(m²n/2), compared to Muon's 30m²n for 5-step Newton-Schulz iteration—approximately 2% of the cost. The complete proof in Appendix E is technically sound and correctly derives the conditions under which this bound holds.

- **Empirical validation across multiple scales and metrics:** The paper evaluates perplexity on C4/en across LLaMA-350M, 1.3B, 3B, and 7B models, includes a 36B-token extended training run, and reports downstream benchmark results (ARC, BoolQ, HellaSwag, MMLU, etc.) in Tables 4 and 8. The finding that SGD-M-PSR consistently outperforms AdamW is robust across these settings.

- **Transparent acknowledgment of limitations:** The paper explicitly states that "SGD-M with PSR is still strictly worse than running Muon" in extended training (Section 5.2) and provides ablation studies on regularization factor η and rank proportion m/r (Table 9), showing how performance varies with these choices.

## Weaknesses

- **Inconsistent notation and hyperparameter values:** Algorithm 2 uses η=0.5 for the PSR regularization factor, while Section 4 states the optimal value is η=0.95, and Table 9's header "1-η" implies η=0.95. This discrepancy reduces reproducibility and creates confusion about what value was actually used in experiments. Additionally, the 0.18 rescaling constant in Algorithm 2 is calibrated on a 350M model at steps 1000–2000 (Table 6) but applied universally across scales without verification of its stability.

- **Practical wall-clock time contradicts theoretical efficiency for tested scales:** Table 3 reveals that PSR is slower than Newton-Schulz for Attention and MLP layers at 1.3B and 3B scales (the scales used in main experiments). The theoretical 2% FLOP advantage only translates to wall-clock gains at 7B+ dimensions. The efficiency claim is thus primarily theoretical for the scales actually tested, though the paper transparently acknowledges this limitation and identifies kernel optimization as future work.

- **No statistical significance testing or error bars:** All LLM experiments report single-run results (Tables 4, 8, 9). Given that perplexity differences can be small (e.g., 18.36 vs. 18.30 in Table 9), and given the known variance in LLM training, this limits confidence in the robustness of reported improvements.

- **Fixed learning rate across optimizers without tuning:** All experiments use LR=3×10⁻⁴ uniformly (Table 5). Optimizers often have different optimal learning rates—AdamW and SGD-M typically require different scales. The paper does not establish whether SGD-M-PSR's advantage persists under per-optimizer LR tuning, which is standard practice in optimizer benchmarking.

- **Limited scope for the strongest claims about Muon:** The extended 36B-token experiment (Figure 4a, Table 4) shows Muon outperforms PSR in both perplexity and downstream tasks. While PSR maintains early-training advantages, the title and abstract framing that emphasizes "surpass Adam" while positioning Muon as the motivating baseline creates a slight mismatch: PSR is better characterized as a computationally cheaper approximation to Muon that outperforms AdamW but remains inferior to Muon in long-run convergence.

## Nice-to-Haves

- A principled analysis of why PSR's early-training advantage diminishes in later stages would strengthen the understanding of spectral dynamics throughout training. The current discussion in Section 5.2 speculates about "heavy-tail preservation" but could be more mechanistic.

- Wall-clock convergence curves (loss vs. actual training time) would clarify whether PSR's theoretical efficiency can be realized in practice, given the sequential PyTorch implementation currently used.

- Comparison with Adafactor or Shampoo, which also target memory efficiency through spectral methods, would better contextualize PSR within the broader optimizer landscape.

## Removed Points

These points are flagged to be removed, treat them with caution:

- *Criticisms demanding theoretical convergence guarantees:* This is an empirically motivated paper with computational analysis. Demanding non-convex convergence proofs is not standard for optimizer design papers at ICLR and would be scope creep.

- *Criticisms demanding experiments on 10B+ parameter models:* The paper tests up to 7B parameters, which is within the range of typical academic ICLR submissions. Larger-scale validation would be valuable but is not a requirement given computational constraints.

- *Criticisms about missing SWAN comparison:* SWAN is mentioned in related work and noted as conceptually similar but distinct (normalizes gradients vs. momentum). The paper's contribution is positioned relative to Muon, which is the direct inspiration. Demanding additional baselines is scope creep.

- *Criticisms about single learning rate being unfair to AdamW:* While per-optimizer tuning is ideal, the paper uses the same LR for all methods, which is conservative—the comparison could favor AdamW if SGD-M-PSR needs different tuning, making this a weakness but not a fatal flaw.

- *Demands for CUDA kernel implementation:* This is an algorithm design paper. The engineering effort of custom kernels is substantial and not reasonably expected for initial work. The paper transparently identifies this as future work.

- *Criticisms about Styblinski-Tang function being "separable and not representative":* The authors themselves acknowledge this limitation in Appendix D.1. The toy problem serves as initial motivation, not as definitive proof, and is appropriately framed.

## Novel Insights

The observation that attention and MLP layers exhibit qualitatively different spectral decay patterns (Figure 1c,d)—with attention spectra decaying sharply toward y=x while MLP spectra show greater uniformity—suggests that uniform orthogonalization across all layers may be suboptimal. This raises an intriguing hypothesis: PSR's partial regularization may be better matched to the heterogeneous spectral structure across layer types than Muon's uniform approach. If the heavy tail contains genuinely important directions that Muon's full orthogonalization over-amplifies (noise) while PSR preserves them selectively, this could partially explain why PSR excels in early training but Muon gains ground later—early optimization benefits from head suppression, while long-run convergence may require more nuanced tail treatment. This layer-heterogeneity angle could guide future research into layer-adaptive spectral methods.

## Suggestions

- Resolve the η notation inconsistency by using distinct symbols for learning rate (e.g., α) and regularization factor (η), and ensure Algorithm 2 matches the empirically optimal η=0.95 used in experiments.

- Add wall-clock time plots (loss vs. actual training time) to directly assess whether the theoretical FLOP advantage translates to practical speedups at tested scales, or explicitly frame efficiency as a theoretical property requiring kernel implementation.

- Report results from at least 3 random seeds with standard deviations for perplexity and downstream metrics, even if full re-runs are computationally prohibitive—this significantly strengthens empirical claims.

- Provide a brief analysis of η and m/r sensitivity across model sizes (not just 350M/1.3B in Table 9) to establish whether the "optimal" constants are robust or require per-scale tuning.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 4.0, 4.0, 2.0]
Average score: 3.2
Binary outcome: Reject
