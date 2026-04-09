## SummaryThis paper challenges the prevailing "diversity-first" paradigm in multi-task LoRA, showing that a simplified multi-head architecture (M-LoRA) with high inter-head similarity outperforms complex diversity-seeking variants like R-LoRA, and that a standard single-adapter LoRA with increased rank matches multi-component architectures. Based on these findings, the authors hypothesize that learning task-shared representations is more effective than architectural isolation, and propose Align-LoRA, which adds an explicit alignment loss (KL divergence or MK-MMD) on the shared down-projection matrix **A** to encourage task representations to converge in the low-rank space.

## Strengths

- **Provocative empirical challenge to the prevailing paradigm.** The finding that M-LoRA (router removed, heads summed) outperforms R-LoRA despite exhibiting much higher inter-head similarity (medians >0.85 vs. R-LoRA's low similarity in Figure 2) directly contradicts the core design principle of diversity-focused methods, making this a substantive conceptual contribution.

- **Zero inference latency through weight merging.** Unlike multi-component architectures with dynamic routing that cannot be pre-merged, Align-LoRA retains LoRA's key practical advantage — the trained adapter can be merged into the base model (Appendix C, Eq. 7), eliminating inference overhead entirely. This is a significant deployment advantage that the multi-component literature has largely accepted as a necessary trade-off.

- **Clear narrative progression from observation to method.** The paper moves systematically: (1) M-LoRA's paradoxical success → (2) high-rank single LoRA matches multi-component → (3) hypothesis that shared knowledge matters more than isolation → (4) Align-LoRA as a principled operationalization. This makes the motivation for Align-LoRA well-grounded rather than arbitrary.

- **Consistent empirical improvements with fewer parameters.** In Table 4, A-LoRA-K achieves the best BBH scores across all three base models while using fewer trainable parameters (0.20%) than HydraLoRA and R-LoRA (0.25%). Table 5 shows per-task wins on 7 of 8 tasks for Qwen2.5-7B.

## Weaknesses

### Major:

- **No ablation against a simple auxiliary regularization baseline.** The paper attributes performance gains specifically to "representation alignment," but never tests whether adding *any* auxiliary loss on the **A** matrix (e.g., L2 regularization, variance minimization) would produce similar improvements. Without this control, it is impossible to determine whether the alignment mechanism itself — as opposed to the regularizing effect of an additional training objective — drives the observed gains. This is critical because the core claim of the paper is that *alignment* specifically (not generic regularization) is the key to better multi-task LoRA.

- **Missing statistical significance reporting.** Tables 1, 4, and 5 report single-run results with no error bars or standard deviations across seeds. Some of the claimed improvements are modest (e.g., Table 1: M-LoRA at 82.52 vs. R-LoRA at 82.03 on QNLI; Table 4: A-LoRA-K at 48.84 vs. M-LoRA at 45.35 on LLaMA3-8B — though this is a larger gap). For a paper whose central contribution rests on empirical findings that challenge a paradigm, the absence of variance estimates weakens confidence that these differences are reproducible and not artifacts of run-to-run variation.

- **Inference from weight similarity to "shared knowledge" is not directly verified.** Figure 2 measures cosine similarity between flattened **B**_i weight vectors, not between the *representations* they produce. The paper infers that high weight similarity implies the heads learn "shared knowledge" (Section 3.3), but this causal chain is incomplete: weights with high cosine similarity can still produce divergent outputs depending on input distributions. A direct measurement of representation-level similarity (e.g., CKA or mutual information between head outputs on shared inputs) would substantially strengthen the claim that the heads are genuinely learning shared representations rather than merely converging in parameter space.

### Minor:

- **Strong distributional assumption without justification.** The alignment loss models batch-wise representations as multivariate Gaussians with *diagonal* covariance (Section 5.1). For LLM hidden states, which exhibit strong feature correlations, the diagonal assumption may be a poor fit. The paper provides no empirical or theoretical justification for why full covariance is unnecessary or why the diagonal approximation suffices.

- **The theoretical contribution is incremental.** The generalization bound in Appendix F follows standard domain adaptation theory (Ben-David et al., 2006) applied to the LoRA MTL setting. The derivation does not leverage any specific properties of low-rank decomposition (e.g., the relationship between rank _r_ and the bound's tightness), making the theory generic rather than specifically informative about why Align-LoRA works.

- **No evaluation on genuinely conflicting task pairs.** Appendix H.2 tests "highly dissimilar" tasks (e.g., Translation vs. QA), but dissimilarity is not the same as conflict. Tasks with opposing optimal representations (e.g., sentiment on different domains where class distributions flip) could cause alignment to induce negative transfer. The paper does not address this failure mode, which limits the claimed generality of the "shared representations are superior" hypothesis.

### Trivial:

- The claim in the abstract that Align-LoRA "significantly surpasses baselines" could be more precisely calibrated given the modesty of some improvements (e.g., +1.49 average over M-LoRA on Qwen2.5-7B in Table 5).

## Nice-to-Haves

- Wall-clock inference latency measurements comparing merged Align-LoRA vs. unmerged multi-component methods under realistic batch sizes and sequence lengths, to quantitatively substantiate the practical deployment advantage.
- Layer-wise analysis identifying which transformer layers benefit most from alignment; if only certain layers matter, the method could be made even more efficient.
- A deeper mechanistic analysis (e.g., gradient trajectory or norm comparison) of why summation-based aggregation outperforms dynamic routing, beyond the intuitive "collaborative vs. competitive" framing.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Abstract should be more precise about alignment mechanism"** — Formatting/style nitpick.
- **"Brief acknowledgment of why merging fails for dynamic routers"** — The paper already explains this: input-dependent routing weights prevent pre-merging (Section 2.2, Eq. 8; Appendix C).
- **"Transition from M-LoRA to single-adapter is abrupt"** — Writing style preference, not a substantive weakness.
- **"Discuss cross-stitch/sluice networks in Related Work"** — Missing related works request; cannot verify existence per rules.
- **"Table 3 evaluation benchmark unclear"** — Section 4 text explicitly states evaluation on BBH; reviewer misread.
- **"Method less effective on smaller models"** — On Qwen2.5-3B, A-LoRA-K gains +1.55 over M-LoRA; on 7B, +1.49. The gaps are comparable; the claim is factually incorrect.
- **"Training compute details missing (gradient accumulation, batch size, devices)"** — Nitpick about trivial implementation details.
- **"Compare against non-LoRA MTL methods"** — Missing related works concern; scope creep beyond the paper's stated focus on LoRA.
- **"Per-task performance breakdown missing"** — Table 5 already provides per-task scores for all 8 tasks; reviewer missed this.
- **"t-SNE visualization missing/corrupted"** — Figure 5 in Appendix I.1 provides t-SNE visualizations; this is factually wrong.
- **"Larger model scale (70B+) validation"** — Generic "add more scale" weakness; the current model zoo (3B–14B) is adequate.
- **"Parameter count bias in M-LoRA comparisons"** — Tables report %Param columns showing M-LoRA uses *fewer* parameters (0.41–0.42%) than HydraLoRA/R-LoRA (0.45%), so the concern is directionally wrong.
- **"Unfair comparison when single-adapter rank is increased"** — The asymmetry *favors* the baseline multi-component methods (they get their preferred architecture), making the comparison conservative for the author's method. Per hard rules, this is not a valid criticism.

## Novel Insights

The paper reveals an interesting paradox at the heart of multi-task LoRA design: the architectural mechanisms specifically introduced to promote diversity (randomized initialization, dropout, dynamic routing) may actually *interfere* with what helps most — the emergence of shared representations. The key insight is that removing the router while *keeping* dropout transforms multiple heads from competing specialists (where the router picks winners) into a collaborative ensemble (where dropout provides stochastic input perturbation and summation forces consensus). This suggests that the research community's focus on routing architecture design may have been optimizing the wrong variable — the important factor is not *which* expert handles which input, but whether the adapter space encourages tasks to share a common representation subspace. The finding that alignment on the **A** matrix (which prior work identifies as task-general) specifically helps, while prior work focused on diversifying **B** (task-specific), suggests the field may have been looking in the wrong place within the LoRA factorization itself.

## Suggestions

- **Add an L2 regularization ablation on the A matrix** as a control. Train standard LoRA with an auxiliary L2 loss on **A**'s output of comparable magnitude to the alignment loss. If Align-LoRA still wins, the alignment mechanism is specifically responsible; if not, the gains come from generic regularization. This single experiment would dramatically strengthen the paper's core claim.
- **Report results over 3 seeds with standard deviations** for at least the main comparison tables (Tables 4 and 5). Even partial variance reporting on one model family would address the statistical significance concern substantially.
- **Add a representation-level similarity metric** (e.g., CKA between head outputs on a shared input batch) alongside the weight-level cosine similarity in Figure 2, to directly verify that weight convergence implies representation convergence.

---
**Axis Evaluation:**
- **Novelty:** Moderate-high. The empirical findings are counter-intuitive and the application of alignment losses to LoRA's bottleneck for multi-task learning is new, though the individual loss functions (KL, MMD) are borrowed from domain adaptation.
- **Technical soundness:** Moderate. The method is clearly described and empirically validated, but the diagonal covariance assumption is unjustified, the theoretical contribution is incremental, and the missing regularization ablation leaves the core causal claim under-supported.
- **Empirical support:** Moderate. Results are consistent across models and scales, but the absence of error bars and the missing ablation against simple regularization are notable gaps for a paper that rests its case on empirical evidence.
- **Significance:** High. If the findings hold under the above scrutiny, this paper could redirect multi-task LoRA research away from complex routing architectures toward simpler, more deployable alignment-based methods.
- **Clarity:** Good. The paper presents a clear, well-structured narrative from observation to hypothesis to method, with thorough appendices.