Now I have a thorough understanding of the paper. Let me write the final review.

## Summary

This paper provides theoretical and empirical evidence that Multi-Grade Deep Learning (MGDL) — which incrementally trains shallow networks on the residuals of previous grades — outperforms standard end-to-end training (SGDL). Theoretical contributions include GD convergence guarantees for both frameworks, a convex reformulation for single-layer ReLU grades (Theorem 3), and eigenvalue analysis of the iteration matrix linking spectral properties to training stability. Empirically, MGDL is benchmarked against SGDL on image regression, denoising, deblurring, CIFAR-10/100 classification, and time series forecasting with transformers.

## Strengths

- **Learning rate robustness study (Section 6, Figure 2)** is genuinely informative: for the high-frequency synthetic regression setting, SGDL converges only at η ≈ 0.005 while MGDL remains stable for η ∈ [0.08, 0.3], demonstrating a qualitatively different robustness profile that is practically useful.

- **Eigenvalue monitoring across tasks (Section 7, Figures 4–6)** provides concrete, visualizable evidence linking training dynamics to spectral properties: SGDL's smallest eigenvalues of I − ηH_F drop below −1 (correlating with oscillatory loss) while MGDL's remain within (−1, 1) across all grades. This gives a mechanistic — if not entirely novel — explanation for the observed stability differences.

- **Broad empirical coverage**: The paper evaluates across image regression (6 images, Table 1), denoising (6 noise levels, Table 2), deblurring (3 blur levels, Table 3), CIFAR-100 classification (Figure 3), CIFAR-10 classification (Figure 6), and time series with transformers (Tables 4–5), showing the phenomenon is consistent across tasks and architectures.

- **Matched depth and parameter budgets**: The architectures are designed with matched total hidden-layer depth (e.g., SGDL: 8 hidden layers vs. MGDL: 4 grades × 2 hidden layers), yielding comparable parameter counts (~117K each for the image regression setting), which makes the comparison fair in terms of model capacity.

## Weaknesses

### Fatal

None.

### Major

- **Confounded comparison: architectural form vs. training mechanism.** The MGDL output is $\bar{g}_L = \sum_{l=1}^L g_l$, a *sum of shallow paths through progressively transformed frozen features* — structurally similar to a network with skip connections at every grade boundary. SGDL produces a *single deep composition* with no skip connections. While depth and parameter budgets are matched, these are different function classes. The paper attributes performance gains to the multi-grade training procedure, but without a critical ablation — training the MGDL architecture (the sum-of-residual-paths structure with all parameters unfrozen) end-to-end — it is impossible to determine whether the gains come from the architecture's skip-connection structure or from the incremental training mechanism. This affects all experimental claims in Sections 5–8 and is the central gap in the paper's argument.

- **The convexity result (Theorem 3) requires $m_l \geq P_l$, which is never satisfied in practice.** By Cover's counting theorem (cited as Cover (2006)), $P_l$ — the number of distinct sign patterns — for $N$ data points in $\mathbb{R}^{d}$ can be as large as $2\sum_{i=0}^{d-1}\binom{N-1}{i}$, which is exponential in $d$. For the paper's own architectures (hidden dimensions of 128, datasets with thousands of points), $P_l$ is astronomically large. The paper presents the convexity equivalence as a key theoretical advantage ("extends convexification from shallow to deep architectures," line 220) without acknowledging that the condition $m_l \geq P_l$ is never satisfied in any of its experiments or in any realistic setting. The abstract's claim that MGDL "reduces to a sequence of convex optimization subproblems" omits this crucial requirement. This is not a mathematical error, but the presentation is misleading about practical relevance.

### Minor

- **Convergence guarantees rely on an unverified bounded-iterates assumption and deliver only stationary-point convergence.** Theorems 1, 2, and 5 all assume GD iterates remain in a compact convex set $\mathcal{W}$, which is the hardest part of convergence analysis to establish. Even granting all assumptions, the conclusion is convergence to a stationary point ($\nabla \mathcal{L} = 0$), which is a standard and relatively weak result. The paper's language ("rigorous convergence guarantees," "greater robustness") somewhat overstates what these conditional results deliver. This is a common practice in the optimization literature, so the issue is primarily one of framing.

- **CIFAR classification uses MSE loss instead of cross-entropy, which is suboptimal and unconventional.** MSE loss for classification is known to be less effective than cross-entropy, and may hurt SGDL disproportionately since cross-entropy specifically interacts well with standard deep network training. This makes it difficult to assess whether the nearly two-orders-of-magnitude loss gap on CIFAR-100 (Figure 3) reflects a genuine MGDL advantage or an unfavorable SGDL configuration.

- **No comparison with standard domain baselines for denoising/deblurring.** While the MGDL-vs-SGDL comparison is the paper's focus, the absence of any comparison with established methods (e.g., BM3D for denoising) means we cannot assess whether either method achieves competitive performance, only that MGDL > SGDL.

- **No comparison with other incremental/greedy training methods.** The paper cites Bengio et al. (2006)'s greedy layer-wise training as motivation but never benchmarks against it. A comparison under matched parameter budgets would strengthen the claim that MGDL's specific formulation (residual training with frozen features) provides advantages beyond general incremental training.

### Trivial

- The Transformer section (Section 8) claims MGT "requires only 33% of the training time," but this may simply reflect training 1 block per grade rather than a genuine efficiency advantage of the multi-grade mechanism. The claim is not wrong but could be more precisely stated.

## Nice-to-Haves

- A single plot comparing: (a) MGDL training, (b) end-to-end training of the same architecture with all parameters unfrozen, (c) standard SGDL, would immediately clarify whether the gain is from training or architecture. This is the single most impactful addition the authors could make.

- Computing or estimating $P_l$ for the actual experimental settings to quantify the gap between the convexity condition and practical architectures.

- Comparison with cross-entropy loss for classification tasks.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"Missing appendix/proofs" complaints**: The parser strips appendices; the original submission contains full proofs in Appendix A and architecture details referenced as equations 26–29.

- **"Unfair comparison favors the author's method"**: The harsh critic states the comparison is unfair because MGDL uses a different architecture. However, the architectures have *matched total depth and comparable parameter budgets* — this is not an asymmetric comparison that favors MGDL by giving it more capacity. The real issue is confounding, not unfairness.

- **"SGDL shown oscillating badly may reflect poor learning rate tuning"**: Section 6 explicitly studies learning rate effects, showing SGDL fails across a wide range of learning rates while MGDL is robust. The oscillation is not simply a tuning artifact.

- **"Theorem 4's condition τ < 1 is unverifiable"**: While technically true, this is a standard type of sufficient condition in optimization. The paper uses it correctly as a theoretical framework to interpret the empirical eigenvalue observations, not as a standalone practical claim.

- **"Key claimed insight that α_l ≪ α is obvious"**: While the fact that shallower networks have smaller Hessian spectral norms is well-known, the paper's contribution is quantifying this in the context of the multi-grade framework and connecting it to the admissible learning rate range (η_l ∈ (0, 2/α_l)). This is a useful formalization even if not surprising.

- **"MGT training time advantage may simply reflect fewer blocks per stage"**: While true, this is actually a legitimate advantage of the method — you genuinely need less compute per grade. The claim is not misleading, just could be more precisely framed.

## Novel Insights

The paper's most valuable insight is the concrete eigenvalue monitoring that links training oscillations to spectral properties of the iteration matrix. While "shallower networks have smaller Hessians" is not novel, the systematic demonstration across multiple tasks that MGDL's I − ηH_F eigenvalues stay within (−1, 1) while SGDL's escape this range — directly explaining the oscillation-vs-stability dichotomy — provides a useful diagnostic framework. However, this observation alone does not isolate the training mechanism from the architectural form as the cause.

## Suggestions

- Add the critical ablation: train the MGDL architecture (sum-of-residual-paths with all parameters unfrozen) end-to-end using standard GD/Adam. Even a single experiment on one task would dramatically clarify whether the gains are architectural or procedural.
- Acknowledge the practical limitations of the convexity result explicitly, ideally with an estimate of $P_l$ for the experimental settings.
- Replace "rigorous convergence guarantees" with more precise language like "convergence guarantees under bounded-iterates assumptions."

## Calibration

**Anchors retrieved:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Layer-wise UAT (68DwQWtdwr) | /home/wg25r/review_agent/human_reviews_2026/68DwQWtdwr.md | 4.0 | Very similar topic (greedy layer-wise training for ResNets with convergence claims). Rejected for conflating approximation and optimization, weak technical grounding. Current paper has broader experiments but similar core weakness of confounded claims. |
| Convex reformulation two-layer (444mACDffR) | /home/wg25r/review_agent/human_reviews_2026/444mACDffR.md | 4.5 | Convex reformulation of shallow networks. Current paper extends this to deep via MGDL but with the same practical vacuity issue. |
| VAE architecture study (2hTLJEgCbv) | /home/wg25r/review_agent/human_reviews_2026/2hTLJEgCbv.md | 1.0 | Confounded architecture comparison, no real contribution. Current paper is clearly stronger with real experiments and theory. |
| Prot2RNA (BPNK5HDEMh) | /home/wg25r/review_agent/human_reviews_2026/BPNK5HDEMh.md | 2.5 | Confounding factors in gains attribution. Current paper has a similar but less severe confounding issue, with more substance overall. |
| Cautious Weight Decay (Gwe6gbGng5) | /home/wg25r/review_agent/human_reviews_2026/Gwe6gbGng5.md | 7.0 | Conditional convergence assumptions but with strong, clean practical results and simple drop-in modification. Current paper is weaker: confounded comparison, less impactful method, overclaimed theory. |
| Polar Express (yRtgZ1K8hO) | /home/wg25r/review_agent/human_reviews_2026/yRtgZ1K8hO.md | 8.0 | Optimal theoretical results with clear practical impact on GPT training. Current paper is far from this level of contribution. |

The current paper sits between the low-scoring confounded-comparison papers (1–2.5) and the medium-tier theory-with-practical-gaps papers (4–5). It has more substance than the low anchors but the confounded comparison undermines its core claim. It's comparable to the layer-wise UAT paper (4.0) but with broader experiments, placing it slightly above. Relative to the high anchors (7–8), the current paper's contribution is clearly weaker due to the confounded empirical design and overclaimed theory.

**Evaluation across axes:**
- **Originality**: Moderate. The MGDL framework itself (Xu 2025) is prior work; this paper provides analysis and extensions. The convex reformulation adapts Pilanci & Ergen (2020) to the multi-grade setting. The eigenvalue analysis formalizes a well-known observation.
- **Importance of research question**: High. Understanding why incremental training works better is important.
- **Claim support**: Weak. The core claim is confounded; the convexity result is practically vacuous.
- **Experiment soundness**: Moderate. Broad coverage but missing the critical ablation and standard baselines.
- **Clarity**: Moderate. Well-organized but overclaims theoretical contributions.
- **Community value**: Moderate. The learning rate robustness finding and eigenvalue diagnostic are useful, but the confounded comparison limits confidence in the broader claims.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>