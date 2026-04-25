Now let me search for calibration anchors.Now I have enough to write the final review. Let me compile everything.

---

## Summary
The paper proposes ARM (Autoregressive Reward Model), a low-rank parametrization of reward-augmented decoding (RAD) that reduces decoding cost from O(k) to O(1) reward-model calls per token. The key insight is that RAD's learned reward matrix is empirically low-rank (rank ~10², far below |V|=50,257), motivating a dueling-network-style parametrization that leverages output embeddings to score all vocabulary candidates in a single forward pass. The paper reframes RAD's training objective as matrix completion, derives a rank bound for ARM, and demonstrates efficiency-quality parity on detoxification and sentiment control tasks with GPT-2 and LLaMa-2 variants.

## Strengths
- **Matrix completion framing (Section 3.1.1, Eq. 5):** The reinterpretation of RAD's training objective as incomplete reward matrix completion is a genuine analytical contribution that connects controlled generation to a well-studied mathematical framework, providing a principled lens for the subsequent analysis.
- **Empirical low-rank discovery with verification (Figure 1, Section 3.1.2, Appendix C.1):** The paper measures the rank of RAD's learned reward matrix across both detoxification and sentiment tasks and shows rank ~10² vs. the theoretical maximum |V|=50,257 or d=768. Crucially, Appendix C.1 also verifies that RAD *can* represent high-rank matrices, ruling out the trivial explanation that RAD is architecturally incapable of high-rank outputs.
- **Principled ARM parametrization with hard rank bound (Eq. 7–8):** The dueling-network factorization R̂_ARM = HA yields rank(R̂_ARM) ≤ min(rank(H), rank(A)) ≤ d by construction. This is a concrete, analytically derived upper bound directly encoding the low-rank inductive bias.
- **Regularization ablation and mechanistic insight (Figure 5):** Removing regularization increases rank(R̂_ARM) from ~10–20 to ~40–60 and degrades fluency. This provides causal evidence linking regularization's effect to rank reduction, not just a surface-level performance comparison.
- **Concrete efficiency gains (Table 1, Figure 6):** ARM maintains ~0.001 s/token flat across all top-k values, while RAD rises linearly to ~0.010 s/token at k=80, yielding a 10× speedup at typical settings. The O(1) vs. O(k) complexity argument is correct and the wall-clock measurement is real.
- **Multi-scale evaluation (Section 5.1):** Results span GPT-2-Large/Small and LLaMa-2-7B/13B + TinyLLaMa, demonstrating that the efficiency-quality trade-off generalizes across model scales.

## Weaknesses

### Fatal
None.

### Major
- **ARM's best result (distill) requires running RAD at distillation time, partially undermining the efficiency story.** The headline claim that ARM "performs on par with RAD at 1/k the cost" is demonstrated most convincingly by the ARM distill variant (Eq. 10), which requires a frozen RAD teacher to generate training targets. But for a practitioner who wants to avoid RAD's cost, they cannot afford to run RAD for distillation either. The genuinely self-contained variant—ARM resp. only—"slightly lags behind" (Figures 3–4, Section 5.4) in fluency at matched toxicity levels. The paper does not quantify this gap (no numerical comparison, no variance estimates), so it is impossible to determine whether "slightly" is negligible or meaningful. A clearer quantification of ARM resp. only vs. RAD (and potentially a total compute accounting for distillation cost) would be needed to fully establish the efficiency claim in the deployable scenario.

- **The explicit disagreement with Han et al. (2024) is acknowledged but unaddressed.** Section 4 states: "they observe that value function parametrization outperforms Q-function parametrization, which *disagrees with our work*," and leaves it at that. Han et al.'s Q-function parametrization is directly analogous to ARM, while their value function is analogous to RAD. If a contemporaneous paper studying the same architectural trade-off finds the opposite conclusion, the discrepancy must be explained—different tasks, model sizes, training data, or evaluation metrics could account for it. As written, the paper's core empirical finding is left undefended against directly contradictory evidence from the closest related work.

### Minor
- **No variance or significance testing on the central trade-off curves.** Figures 3 and 4 show single-run Pareto curves. The claim that ARM "closely matches" RAD is asserted visually with no error bars, confidence intervals, or statistical test. For the distill variant the gap appears small and likely robust; for ARM resp. only the gap is less clear. Even a brief mention of variance across multiple seeds for representative β values would substantially strengthen the empirical argument.

- **Efficiency measurement is reward-model-only, not end-to-end.** Figure 6 measures only reward model time in isolation. Since the reward model (GPT-2-Small/TinyLLaMa) is much smaller than the base model (GPT-2-Large/LLaMa-2-7B), the absolute end-to-end speedup would be smaller. The framing in Section 5.6 slightly overstates the practical impact; reporting end-to-end time would be more informative.

- **The low-rank observation could partly be an architectural artifact.** Section 3.1.3 argues that training data incompleteness explains the low-rank finding, but an alternative is not ruled out: in a causal Transformer, the final hidden state h_L([x′, v]) differs from h_L(x′) only by the influence of v at position L, which can be small for long prefixes. This could make R̂_RAD approximately low-rank by construction, independent of the task. The paper empirically verifies that RAD can achieve high rank (Appendix C.1), but does not test whether the rank stays low for short prefixes (where the causal masking effect would be weaker). This is a theoretical concern worth acknowledging more explicitly.

### Trivial
- The paper evaluates on only two binary classification-style tasks (toxicity, sentiment). This limits the generalization of the claim that low-rank parametrization suffices. These tasks may be structurally simpler than, say, factual grounding or multi-attribute control.

## Nice-to-Haves
- A controlled experiment testing ARM resp. only with an explicit low-rank constraint on W (e.g., W = UV^T with r ≪ d) would cleanly separate the rank inductive bias from other design choices and provide a tighter efficiency-expressiveness story.
- A total compute accounting for ARM distill including distillation data generation cost, enabling a fair apples-to-apples comparison against simply using RAD at inference time.
- A third, harder attribute-control task (e.g., formality, factual grounding) would provide stronger evidence that the "low-rank reward matrix" finding generalizes beyond binary classifiers.
- An explanation of the Han et al. (2024) discrepancy in terms of concrete experimental differences (task type, model scale, evaluation metric) would significantly strengthen the paper's positioning.

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **"ARM resp. only's gap is attributable to ARM having structurally lower expressiveness" (Harsh Critic).** The paper clearly acknowledges the ARM resp. only gap and attributes it to distillation providing cleaner averaged targets (Section 5.4). This is a reasonable explanation—not a structural expressiveness deficit—and the paper demonstrates ARM resp. only is still competitive with GeDi/DExperts. The claim that the gap "is at least equally consistent with ARM having structurally lower expressiveness" is speculative and not grounded in the paper's evidence.
- **"Eq. 5 conditions not stated in main text" (Harsh Critic).** The conditions for the MSE compression to a weighted average are deferred to Appendix A, which exists in the original submission. Per the rules, criticisms about absent appendices are removed.
- **"DExperts comparison unclear" (Harsh Critic).** The paper does explicitly contrast ARM vs. DExperts in Section 4: ARM adds the W matrix and the baseline w, providing the rank bound and abstention regularization. This is noted.
- **Generic strength: "compatibility with black-box API access" (Strength Finder).** This feature—requiring only top-k logits—is shared with DExperts, GeDi, and RAD, so it is not a distinguishing contribution of ARM specifically.

## Novel Insights
The most genuinely novel observation in the paper is not just the efficiency gain but the *mechanistic link between regularization, rank, and fluency*: the ablation (Figure 5) shows that regularization toward the baseline lowers rank(R̂_ARM) and improves fluency, with the extreme case (very strong regularization) producing rank-1 output that leaves the base model's distribution unchanged. This establishes regularization as a rank-control mechanism with interpretable semantic meaning (abstention), offering a principled design principle for reward models beyond this paper's specific setting. The matrix completion framing may also be useful for future work studying reward models with different coverage patterns in training data.

## Calibration Anchors

| Path | Avg Score | Notes |
|---|---|---|
| `shgx0eqdw6.md` (ARGS: Alignment as Reward-Guided Search) | 7.00, Accept | Closely related topic (reward-guided decoding); accepted with concerns about novelty and baselines. Paper under review has stronger analytical contribution but narrower evaluation. |
| `jY5oml9fe9.md` (LLMs as Self-Detoxifiers, SASA) | 6.00, Accept | Same detoxification task and Pareto-curve evaluation methodology; accepted with all-6 scores. Comparable scope. |
| `488A64eOf6.md` (Decoding as Metrics Optimization) | 6.25, Accept | Related decoding-time control; accepted. Has broader metric coverage than this paper. |
| `9WbNpRuFuS.md` (Approximately Aligned Decoding) | 5.75, Reject | Most comparable paper in efficiency-quality trade-off framing; rejected for marginal novelty and incremental contribution. Paper under review has stronger analytical content. |
| `gql60q5W4z.md` (RL with Fine-grained Reward) | 4.00, Reject/Withdrawn | Similar topic (controllable generation), weaker execution. Low anchor. |
| `xfw92pDy2u.md` (Distilled Diffusion LMs) | 3.50, Reject | Efficiency-via-distillation paper; rejected for not meeting quality threshold. Weaker than this paper. |

**Positioning:** The paper is stronger than the rejected anchors (≤4.0) by virtue of its clean theoretical analysis and genuine efficiency contributions. It is comparable to the borderline-accept group (~5.75–6.25): it has stronger analytical framing than "Approximately Aligned Decoding" but narrower evaluation scope than the accepted papers. The two major weaknesses (distillation-dependency caveat + unresolved Han et al. contradiction) are real but do not invalidate the core contribution. I position the paper at **5.5**—borderline, with sufficient contributions to warrant serious consideration but needing the Han et al. contradiction addressed and better characterization of the ARM resp. only gap to be fully convincing.

## Score and Decision

**Originality:** Moderate-to-good. The matrix completion framing and rank bound are novel contributions. The ARM parametrization is incremental over DExperts but is meaningfully differentiated and well-motivated.

**Importance of research question:** Good. Decoding efficiency for controlled generation is practically important, especially as k grows large.

**Whether claims are well-supported:** Partially. The efficiency claim is solid. The quality-parity claim is convincing for the distillation variant but inadequately characterized for the non-distillation variant. The Han et al. contradiction is a real gap.

**Soundness of experiments:** Adequate. Two tasks, multiple model sizes, ablation included. Missing variance estimates and end-to-end timing.

**Clarity of writing:** Good. The paper is clearly structured and the contributions are well-delineated.

**Value to the research community:** Moderate-to-good. The matrix completion framing and rank-regularization insight are likely to inform future reward model design.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>