=== CALIBRATION EXAMPLE 21 ===

# Final Consolidated Review
## Summary

This paper proposes Principal Spectral Regularization (PSR), a method that selectively penalizes the dominant spectral components of the momentum matrix rather than fully orthogonalizing it (as Muon does via Newton-Schulz iterations). Motivated by the observation of a "spiked-head-heavy-tail" spectral structure in LLM training momentum, PSR uses block Lanczos bidiagonalization and deflation to regularize only the top few singular directions, preserving the heavy tail. Experiments on LLaMA models (350M–7B) show SGD-M-PSR surpasses AdamW in validation perplexity and downstream benchmarks while consuming ~2% of Muon's FLOP overhead, though it does not match Muon in long-horizon or large-scale settings.

## Strengths

- **Identifies a genuine and under-explored trade-off in spectral optimization.** The observation that momentum spectra exhibit a spiked-head-heavy-tail structure (Fig. 1), and that Muon's full orthogonalization flattens this entirely, raises a legitimate question about whether targeting all spectral directions is necessary. The conceptual shift from "orthogonalize everything" to "regularize only the dominant head" is a meaningful algorithmic contribution that could inform future optimizer design.

- **Strong computational efficiency gains in theory and at large scale.** Theorem 4.1 establishes that PSR requires at most ½m²n FLOPs versus Muon's 30m²n, a ~60× reduction. Table 3 confirms this translates to real wall-clock and memory savings at 7B+ scale (e.g., LLaMA-70B MLP: 35.85ms/568MB for PSR vs. 184.33ms/1536MB for Newton-Schulz). This is a concrete and significant efficiency result for the large-scale regime.

- **Provides spectral visualizations that offer genuine mechanistic insight.** Figure 1's heatmaps across attention and MLP layers, and the sequential comparison of gradient → momentum → AdamW → Muon spectra, give a clear and novel picture of how different optimizers transform the spectral structure. This visualization alone is a useful contribution for understanding optimizer behavior.

## Weaknesses

- **PSR does not match Muon in the regime where optimizer choice matters most—long-horizon training.** The paper explicitly acknowledges (Section 5.2) that SGD-M-PSR is "strictly worse than running Muon" in the loss-steady stage and that Muon surpasses it as training progresses (Fig. 4a). For any practitioner doing serious pretraining, Muon would remain the better choice. The paper's title ("Makes Momentum Surpass Adam") sets a low bar: surpassing AdamW is necessary but insufficient if a strictly better alternative (Muon) exists at modest additional cost. The practical niche for PSR is therefore unclear—it is cheaper than Muon but worse, and better than AdamW but with narrow margins and added complexity.

- **The primary empirical results rely on very short training horizons for 3B and 7B models.** The LLaMA-3B and 7B results (Fig. 3, Table 8) are based on only 10,000 steps (~2B tokens), which is a negligible fraction of a real pretraining run. The paper itself shows that the PSR advantage is most pronounced during warmup (Fig. 4a) and diminishes later. The only long-horizon experiment is LLaMA-1.3B at 36B tokens, where PSR is already surpassed by Muon. Without evidence that the AdamW-surpassing advantage persists at convergence for models ≥3B, the core claim is under-supported.

- **Performance margins over AdamW are narrow and lack statistical validation.** In Table 4 (1.3B, 36B tokens), SGD-M-PSR averages 46.56 vs. AdamW's 46.32—a 0.24-point difference across 9 benchmarks. No standard deviations or multi-seed confidence intervals are reported for any LLM experiment. Given the known variance in LLM downstream evaluations, it is unclear whether this gap is statistically meaningful. This is especially concerning because the "surpass Adam" claim is the paper's central contribution.

- **The wall-clock efficiency advantage does not materialize at the scales where the accuracy results are demonstrated.** Table 3 shows PSR is *slower* than Newton-Schulz for LLaMA-1.3B and 3B (e.g., 1.3B Attention: 4.85ms vs. 2.01ms). The efficiency gains only appear at 7B+. This creates a mismatch: the perplexity improvements are shown at 350M–3B where PSR is not faster, and the speed improvements are shown at 70B where no accuracy results exist. The paper attributes this to a "naive PyTorch implementation" and promises future kernelization, but the practical efficiency claim is currently unvalidated for the common 1B–7B regime.

- **The Styblinski-Tang motivation provides intuition but the connection to LLM optimization is not established.** The paper acknowledges this (Appendix D: "the connection between mathematical function optimization and LLM pretraining is relatively vague"). A separable, near-convex synthetic function with hand-crafted power-law weights cannot capture the non-convex, saddle-rich landscape of transformer pretraining. The toy result (optimal d=64, p=5%) is used to set LLM hyperparameters, but there is no analysis showing why this particular configuration should transfer. This weakens the principled motivation for PSR.

## Nice-to-Haves

- **Theoretical convergence analysis for SGD-M-PSR.** The method modifies the update direction via deflation and re-normalization, so standard SGD convergence results do not directly apply. A convergence guarantee—even under restrictive assumptions—would strengthen the paper's claim that PSR is a principled method rather than an engineering trick.

- **Experiments beyond C4/en and LLaMA.** Testing on another architecture (e.g., GPT-2 style) or corpus (e.g., The Pile) would help assess robustness.

- **Layer-specific PSR application.** The paper notes different spectral structures for attention vs. MLP layers (Fig. 1) but applies PSR uniformly. Ablating whether layer-adaptive regularization further improves performance would be informative.

- **Comparison with Lion, Adam-mini, or SWAN.** These single-momentum or memory-efficient optimizers are cited as related work but not benchmarked. While not strictly necessary given the focus on the spectral regularization perspective, including at least one would contextualize PSR's place among lightweight alternatives to AdamW.

## Removed Points

*These points were flagged for removal—treat them with caution.*

- **"Code not released during review / reproducibility concern about implementation availability."** The paper states code will be released upon publication. Per hard rules, reproducibility concerns about unreleased artifacts are not valid criticisms at the submission stage. Removed.

- **"Missing comparison with Shampoo/Kron as baselines."** The paper does compare with SOAP (Appendix D.3), which is the most directly comparable spectral preconditioning method. Demanding every cited optimizer be benchmarked is scope creep. Partially addressed by SOAP comparison. Removed.

- **"Theoretical convergence guarantees should be provided."** This is a valid suggestion but not standard for empirical optimizer papers at ICLR. Moved to Nice-to-Have rather than a core weakness.

- **"The title overstates the contribution."** While the margins over AdamW are narrow, the paper does show consistent improvements across scales and benchmarks. The title is bold but not factually wrong given the data presented. The concern about statistical significance is kept separately; the title claim itself is not removed but the narrow margin is noted in Weaknesses.

- **"PSR's memory savings are small relative to total training memory (activations, weights)."** This is speculative and may be incorrect; optimizer states are often a significant fraction of memory for large models. No evidence is provided that the savings are negligible in practice. Removed.

## Novel Insights

The spectral visualization comparing Momentum → AdamW → Muon as a progressive flattening of the spiked-head-heavy-tail structure is genuinely insightful. It reframes AdamW and Muon not as fundamentally different paradigms but as points on a spectrum of spectral regularization intensity: AdamW partially attenuates the head, Muon fully flattens it, and PSR selectively shrinks the head while preserving the tail. This unified perspective suggests that the optimal degree of spectral regularization may be architecture- and training-stage-dependent—a hypothesis supported by PSR's advantage during warmup and Muon's advantage at convergence. This framing could guide future work on adaptive spectral regularization that interpolates between these regimes during training.

## Suggestions

- **Train at least one model (3B or 7B) to 50B+ tokens** to demonstrate whether the AdamW-surpassing advantage persists beyond the warmup phase. This is the single most impactful experiment for validating the core claim.

- **Report multi-seed variance for downstream benchmarks.** Given the 0.24-point average gap over AdamW, even 3 seeds with standard deviations would substantially strengthen or appropriately weaken the "surpass" claim.

- **Provide wall-clock training time for the full pretraining runs** (not just per-operation timing in Table 3). This would resolve the ambiguity about whether PSR's FLOP advantage translates to actual training speedup at the scales where accuracy results are reported.

- **Consider a decaying or adaptive η schedule** that increases regularization strength over training (moving from PSR toward full orthogonalization), motivated by the observation that Muon's full flattening becomes beneficial in later stages. This could combine PSR's warmup advantage with Muon's convergence advantage.

---

**Quality Assessment:**

- **Novelty:** Moderate-to-strong. The partial spectral regularization idea is novel and well-motivated, distinct from both spectral norm regularization and full-matrix preconditioning.

- **Technical soundness:** Moderate. The algorithm is clearly described and the complexity analysis is rigorous, but the empirical validation has significant gaps (short training, no variance, efficiency-accuracy mismatch).

- **Empirical support:** Moderate-to-weak for the central "surpass Adam" claim. The results are consistent but margins are thin without significance testing, and the advantage fades in longer training.

- **Significance:** Moderate. The spectral perspective is valuable, but PSR's practical niche—cheaper than Muon but worse, better than AdamW but narrowly—is limited.

- **Clarity:** Good. The spectral visualizations and algorithm description are clear, though the jump from toy function to LLM could be better motivated.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 4.0, 4.0, 2.0]
Average score: 3.2
Binary outcome: Reject
