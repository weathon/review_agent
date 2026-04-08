=== CALIBRATION EXAMPLE 51 ===

# Final Consolidated Review
## Summary

This paper proposes Principal Spectral Regularization (PSR), a method that selectively penalizes the dominant spectral components ("spiked head") of the momentum matrix in SGD, motivated by the observation that momentum spectra in LLM training exhibit a spiked-head-heavy-tail structure. Using block Lanczos bidiagonalization with deflation, PSR regularizes only the top ~1/16 of spectral directions at roughly 2% of Muon's FLOP overhead, enabling SGD with momentum to surpass AdamW in LLM pretraining across LLaMA models from 350M to 7B parameters—though it remains strictly worse than Muon in long-horizon training and downstream benchmarks.

## Strengths

- **Principled spectral insight motivating partial orthogonalization:** The observation that momentum spectra in LLM training follow a "spiked-head-heavy-tail" structure (Fig. 1), and that full orthogonalization (Muon) may be unnecessary or even suboptimal, is a genuine contribution. The Styblinski-Tang experiments (Fig. 2, Tab. 1) provide concrete evidence that partial regularization can outperform both full orthogonalization and no regularization, validating the core hypothesis.

- **Efficient algorithmic design with theoretical grounding:** The use of block Lanczos bidiagonalization (K=2 iterations) to identify dominant directions, followed by deflation, is computationally elegant. Theorem 4.1 provides a concrete complexity bound of < ½m²n, and the wall-clock comparisons (Tab. 3) confirm substantial speedups at 7B+ scales (e.g., 29.93ms vs. 110.79ms for LLaMA-70B attention), supporting the claim of ~2% FLOP overhead relative to Muon.

- **Honest self-assessment of limitations:** The paper explicitly acknowledges that "SGD-M with PSR is still strictly worse than running Muon" in extended training (Section 5.2) and provides the hypothesis for why (Muon amplifies low-magnitude directions more effectively). This candor strengthens the paper's scientific contribution.

## Weaknesses

### Major:

- **Insufficient training horizons for the largest models to support strong claims about LLM pretraining:** The primary experiments for LLaMA-3B and LLaMA-7B run for only 10,000 steps (~2B tokens). For 7B-parameter models, Chinchilla-optimal training requires orders of magnitude more compute. The paper's own 1.3B/36B-token experiment reveals that PSR's early advantage over Muon diminishes and reverses as training progresses (Fig. 4a), demonstrating that short-horizon results are not predictive of long-horizon behavior. Without at least one longer training run at 3B+ scale, the central claim that PSR enables Momentum to "surpass Adam for LLM Training" rests on incomplete evidence for the regime that matters most.

- **No statistical significance testing for LLM results, where improvements over AdamW are marginal:** The downstream benchmark improvements over AdamW are small (e.g., 46.56 vs. 46.32 average on LLaMA-1.3B 0-shot, Tab. 4). No error bars, standard deviations, or multiple seeds are reported for any LLM experiment. Given the known stochasticity of LLM pretraining, these differences could easily be within noise. The Styblinski-Tang experiments in the appendix (Tab. 7) do report standard deviations, making the omission for the more important LLM experiments conspicuous.

- **PSR consistently underperforms Muon, undermining the practical significance:** While the title focuses on surpassing AdamW, Muon is the more relevant frontier. Tab. 4 shows Muon winning the downstream average (47.04 vs. 46.56 for 0-shot), and the 36B-token training (Fig. 4a) shows Muon overtaking PSR. The paper positions PSR as a "promising direction" for efficiency, but if Muon's 5-step Newton-Schulz iteration can be reduced to 3 steps (which the paper does not ablate), the efficiency gap may narrow substantially while Muon retains its performance advantage.

### Minor:

- **Algorithm 1, Line 4 contains ambiguous notation:** The deflation step $M \leftarrow M - \eta u (u^\top M v^\top) v$ is dimensionally inconsistent as written. Given $u \in \mathbb{R}^{m \times 1}$ and $v \in \mathbb{R}^{n \times 1}$, the term $u^\top M v^\top$ does not yield a valid scalar. The intended operation is presumably $M \leftarrow M - \eta (u^\top M v) u v^\top$. This should be corrected for clarity and reproducibility.

- **Wall-clock inefficiency at small scales contradicts the broad efficiency framing:** Tab. 3 shows PSR is 2.4× slower than Newton-Schulz for LLaMA-1.3B attention layers (4.85ms vs. 2.01ms) and only matches/breaks even at 7B+. The introduction frames PSR as "computationally efficient" without qualifying that this benefit is scale-dependent, which could mislead readers working at the 1B–3B scale that dominates academic LLM research.

- **Limited baseline comparisons:** The paper compares primarily against AdamW and Muon, with SOAP only in the appendix (Fig. 5). Other recent optimizers mentioned in related work—Lion, MARS, Adam-mini—are not empirically evaluated. While the spectral perspective justifies focusing on Muon as the primary point of comparison, showing where PSR falls relative to a wider optimizer zoo would strengthen the practical positioning.

- **The Styblinski-Tang function is a weak proxy for LLM optimization landscapes:** The paper acknowledges "the connection between mathematical function optimization and LLM pretraining is relatively vague" (Appendix D.1). The power-law weighting and Gaussian noise injection are reasonable first steps but do not capture the non-convex, stochastic, high-dimensional dynamics of transformer pretraining. The toy experiment provides intuition but should not be overweighted as evidence.

### Trivial:

- The term "regularization" in PSR is mildly misleading—PSR modifies the update direction like a preconditioner rather than adding a penalty to the loss function as in standard spectral norm regularization. This is a terminological choice, not a technical flaw.

## Nice-to-Haves

- Training at least one model (e.g., 3B) to Chinchilla-optimal token counts to validate long-horizon behavior.
- Error bars or multiple seeds for the LLaMA pretraining experiments.
- Ablation on Muon's Newton-Schulz iteration count T (e.g., T=3 vs. T=5) to test whether the efficiency advantage is robust.
- Validation on a non-LLaMA architecture (e.g., GPT-2 or an encoder-decoder) to demonstrate generality.
- Optimized CUDA kernels for the Lanczos/deflation steps rather than relying on naive PyTorch, which would make the efficiency claims empirically observable at all scales.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: "Theorem 4.1 proof is garbled/unverifiable"** — This appears to be an OCR artifact from the review process, not an actual error in the paper's mathematical derivation. The proof structure in Appendix E is intact and the complexity bound is consistent with the empirical wall-clock results. Removed as factually incorrect about the paper.

- **Weakness: "PSR memory usage is measured only during orthogonalization, not full training footprint"** — Table 3 explicitly reports peak GPU memory for the orthogonalization step, which is the only component that differs between methods. The rest of training memory is identical. This criticism misrepresents what the table measures and why.

- **Weakness: "Missing comparison against GaLore"** — GaLore is a gradient low-rank projection method for memory-efficient training, not a different optimizer per se. It operates in a fundamentally different regime (projecting gradients into a low-rank subspace vs. regularizing spectral components in the original space). The paper correctly positions itself relative to GaLore in Appendix D.3. This is a demand for comparison outside the paper's scope.

- **Weakness: "No user study / no evaluation on non-NLP tasks"** — The paper is explicitly about LLM pretraining. Demanding evaluation outside this scope is scope creep.

## Novel Insights

The most interesting emergent observation from the reviews is the **perplexity-vs-downstream disconnect**: at 10k steps on LLaMA-1.3B, SGD-M-PSR achieves slightly lower test perplexity than Muon (18.30 vs. 18.36 in Tab. 9), yet Muon wins on downstream benchmarks (47.04 vs. 46.56 in Tab. 4). This suggests that lower training perplexity does not automatically translate to better generalization when the optimization trajectory differs in spectral structure. PSR's partial regularization may converge to regions with lower training loss but less favorable generalization properties compared to Muon's more uniform spectral exploration. This finding—that *which* spectral components are regularized matters for generalization, not just *how many*—deserves further investigation and could inform a richer theory of optimizer design beyond simple loss comparisons.

## Suggestions

- Run LLaMA-3B for at least 50k steps (or to ~100B tokens) with at least 2 seeds for both SGD-M-PSR and AdamW. This single experiment would address the two most significant weaknesses (short horizons + no statistical validation) and dramatically strengthen the paper.

- Add a clear upfront statement in the abstract/introduction positioning PSR relative to Muon: "PSR achieves 98% of Muon's FLOP reduction while matching or exceeding AdamW, though it does not match Muon's long-horizon performance." This would be more scientifically honest than the current framing and still compelling.

- Fix Algorithm 1, Line 4 to read $M \leftarrow M - \eta (u^\top M v) u v^\top$ for dimensional consistency.

- Add a brief ablation on Muon's Newton-Schulz iteration count (T=3 vs T=5) to show whether the efficiency advantage is robust or fragile to this hyperparameter choice.

## Axis Evaluations

- **Novelty:** Moderate. The spiked-head-heavy-tail observation is interesting and the partial regularization idea is natural but well-motivated. The Lanczos-based implementation is a sensible engineering choice. The paper's primary novelty is the insight that full orthogonalization is unnecessary, not the algorithm itself.

- **Technical soundness:** Moderate. The method is well-described and the complexity analysis is rigorous. However, the K=2 Lanczos choice lacks theoretical justification beyond empirical validation, the Algorithm 1 notation has an error, and the absence of error bars for LLM experiments weakens the empirical claims.

- **Empirical support:** Moderate to weak. The trends are consistent across scales and the paper is honest about limitations, but the key experiments are too short (3B/7B at 10k steps), lack statistical validation, and the method does not beat the most relevant baseline (Muon) in the regime that matters most (long-horizon training). The 1.3B/36B-token experiment is the strongest result but is a single run.

- **Significance:** Moderate. PSR carves out an interesting niche between AdamW and Muon, and the spectral analysis provides genuine insight into optimizer design. However, since Muon already exists and outperforms PSR, and the efficiency advantage is scale-dependent, the practical impact is limited. The conceptual contribution (partial orthogonalization suffices to beat AdamW) may prove more influential than the method itself.

- **Clarity:** Good. The paper is well-organized and the spectral visualization (Fig. 1) effectively communicates the key insight. The notation issue in Algorithm 1 is a blemish on an otherwise clearly written paper.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 4.0, 4.0, 2.0]
Average score: 3.2
Binary outcome: Reject
