Now I have all the information needed to write the consolidated review.

## Summary

HMoRA proposes combining Mixture of Experts with LoRA for multi-task fine-tuning via three mechanisms: (1) hierarchical hybrid routing that shifts from token-level routing in shallow layers to task-level routing in deeper layers (controlled by a layer-dependent mixing coefficient α^(l)), (2) a Constrained Generalized Jensen-Shannon (CGJS) auxiliary loss that jointly promotes routing certainty and expert balance, and (3) a task encoder that produces task representations for routing without task labels. Experiments on Flan v2 training with 7 NLP benchmarks show consistent improvements over MoE-LoRA baselines and full fine-tuning.

## Strengths

- **CGJS auxiliary loss is a principled contribution.** The formulation (Eq. 11) elegantly decomposes the routing objective into promoting certainty (minimizing individual gate entropy) and balance (maximizing average gate entropy), with tunable constraints γ_b and γ_c to prevent over-regularization. Table 1 shows CGJS outperforms standard load balancing loss with top-k routing (63.72 vs 63.19 avg), and Figure 3 visually confirms that CGJS maintains balance while improving certainty, whereas standard load balancing degrades certainty.

- **Compelling unsupervised task differentiation result.** The task router with CGJS differentiates 42/57 (73.68%) unseen MMLU sub-tasks without any task labels during training, compared to 0/57 without any auxiliary loss and only 7/57 (12.28%) with standard load balancing (Section 4.3, Appendix E.8). This is a striking result that demonstrates the auxiliary loss creates meaningful task separation in routing space. Figure 4's t-SNE visualizations provide clear qualitative support.

- **Consistent empirical improvements across all benchmarks.** HMoRA w/o LW outperforms all baselines on all 7 benchmarks (Table 2, avg 64.16 vs. MoLoRA 63.02, Full FT 63.15), and even the lightweight variant (3.9% parameters) beats Full FT on 5/7 benchmarks with minimal gaps on the remaining two.

- **Lightweight design options with explicit tradeoff analysis.** Figure 2(c) quantifies the parameter/time tradeoff: lightweight designs reduce parameters from 6.31% to 3.90% and training time by ~37% (1618s to 1018s per 1k steps), with only a 0.28-point average accuracy drop (Table 2).

## Weaknesses

### Fatal
None.

### Major

- **The hierarchical routing contribution is insufficiently isolated from other components.** The paper's namesake contribution is hierarchical hybrid routing—varying the token/task routing mix across layers (Eq. 7–8). While Appendix E.5 does ablate ε and μ (the paper states "setting ε > 0 generally leads to better performance," implying ε=0 was tested), the main results in Table 2 conflate hierarchical routing with the CGJS auxiliary loss and the task encoder. Missing comparisons with token-only routing and task-only routing make it impossible to assess whether the hierarchical mixing specifically helps, or whether simply having both routing types (even with uniform α) would achieve similar gains. This matters because the paper's title and primary framing center on the hierarchical mechanism.

- **The "generalization to unseen tasks" claim is partially unsupported.** The paper shows the task router *differentiates* unseen MMLU tasks (73.68% in Appendix E.8) and forms distinct clusters (Figure 4), but does not demonstrate that this differentiation *improves downstream performance* on those tasks. Table 3 shows removing L_aux from the task router drops performance by ~1 point, but this conflates the differentiation ability with the certainty/balance effects of the loss. Additionally, MMLU tasks (QA, NLI, commonsense reasoning) substantially overlap in domain with Flan v2's 1,836 tasks, making the "unseen" claim weaker than presented. The leap from "forms distinct clusters on similar task types" to "generalizes to unseen tasks" needs either downstream performance evidence on genuinely held-out task families or more careful qualification.

### Minor

- **The abstract's framing of the 3.9% result is misleading.** The abstract states HMoRA "outperforms full fine-tuning across multiple NLP benchmarks, while fine-tuning only 3.9% of the parameters," but the 3.9% variant (HMoRA w/ LW) outperforms Full FT on only 5/7 benchmarks. The variant that outperforms on all 7 benchmarks uses 6.31% of parameters—more than LoRA r=64 (4.78%) and MoLoRA (3.82%). The abstract should either qualify the claim or report the 6.31% figure.

- **No standard deviations reported despite 5 repeated runs.** The improvements over baselines are modest (e.g., ~1.14 avg points over MoLoRA for HMoRA w/o LW), and the gains from L_aux in Table 1 are ~0.7–0.9 avg points. Without variance information, it is unclear whether these differences are statistically significant. Reporting standard deviations would strengthen the claims.

- **Evaluation is limited to multiple-choice benchmarks.** All seven benchmarks (MMLU, MMLU-Pro, ARC, OpenBookQA, SWAG, CommonsenseQA) are multiple-choice tasks. No generation, summarization, translation, or structured output tasks are evaluated, which limits the generality of the "multi-task" effectiveness claims.

### Trivial
None.

## Nice-to-Haves

- A direct comparison of hierarchical routing (varying α) vs. token-only routing vs. task-only routing as an ablation in the main text would significantly strengthen the paper's central architectural claim.
- Testing on generation tasks (e.g., summarization, translation) rather than only multiple-choice benchmarks would broaden the scope of the multi-task claims.
- Quantifying the computational overhead (FLOPs, latency, memory) of the task encoder would help practitioners assess practical viability.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Full fine-tuning is undertrained with 10,000 steps.** The paper uses early stopping (evaluate every 200 steps, stop if no improvement for 10 consecutive evaluations), and the same training budget applies to all methods. The early stopping mechanism ensures methods converge before the budget is exhausted, making this a fair comparison. Removed because the early stopping protocol addresses this concern.

- **Task encoder is underspecified / overhead not quantified.** While the task encoder description ("can be a single or multi-layer Transformer encoder") is vague in the main text, this is a practical detail rather than a methodological flaw. The overhead is relatively minor since the task representation is computed once per input sequence. Moved to nice-to-have.

- **α(l) parameterization is ad hoc / lacks theoretical motivation.** The sigmoid form is a reasonable engineering choice with clear intuitive motivation (gradual shift from token to task routing), and the hyperparameters ε and μ are ablated. This is standard practice for schedule design in deep learning. Removed as it demands theoretical justification where empirical validation suffices.

- **CGJS clustering claim deferred to Appendix D.** While it would be preferable to have a summary in the main text, the core CGJS formulation and its effect (certainty + balance) ARE presented in the main body. The clustering argument is supporting evidence, not the primary claim. Removed as minor presentation preference.

- **γ_b not ablated (only γ_c in Appendix E.4).** This is a minor gap in ablation coverage. Since γ_b controls balance and the paper sets γ_b = 1 (maximum balance), this is the most natural default. Removed as minor.

- **Missing related works.** Removed per hard rules—cannot verify existence of suggested references.

## Novel Insights

The CGJS auxiliary loss's dual objective—simultaneously maximizing average entropy (balance) and minimizing individual entropy (certainty)—provides a clean theoretical decomposition that standard load-balancing losses lack. The empirical finding that this decomposition also creates a clustering effect on task router gate values (enabling unsupervised task differentiation) is an unexpected and potentially useful property that goes beyond the loss's stated design goal. This suggests that routing certainty and task separation are linked in MoE models, which could inform future work on routing design even outside the LoRA context.

## Suggestions

- Add a main-text table directly comparing hierarchical routing (ε > 0) vs. uniform mixing (ε = 0) vs. token-only routing vs. task-only routing, isolating the hierarchical contribution from the CGJS loss and task encoder.
- Report standard deviations across the 5 runs for all tables; this is especially important for the modest improvements in Tables 1 and 2.
- Qualify the "generalization to unseen tasks" claim to distinguish between task differentiation (which is well-supported) and improved downstream performance on genuinely novel tasks (which is not yet established).

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| MoRE (MoE+LoRA, marginal gains, missing ablations) | `/home/wg25r/review_agent/human_reviews/LWvgajBmNH.md` | 4.0 (Reject) | HMoRA is stronger: CGJS loss is more principled, task differentiation is compelling, improvements are consistent across all benchmarks |
| MoIN (LoRA MoE, unfair comparisons, overclaimed) | `/home/wg25r/review_agent/human_reviews/L0PciKdHsP.md` | 4.5 (Reject) | HMoRA is stronger: better experimental methodology, more complete baseline comparisons, clearer contribution |
| MLAE (masked LoRA, overclaimed SOTA, missing ablations) | `/home/wg25r/review_agent/human_reviews/ZEO9ibXr46.md` | 5.33 (Reject) | HMoRA is comparable: both have missing ablations and modest improvements, but HMoRA's CGJS loss is more principled |
| ELREA (LoRA MoE ensemble, incomplete baselines) | `/home/wg25r/review_agent/human_reviews/l0gZS0sAlf.md` | 5.80 (Accept Poster) | HMoRA is comparable: both have novel routing ideas and some missing ablations; HMoRA's CGJS loss and task differentiation are more principled |
| MeteoRA (embedded LoRA MoE, missing ablations) | `/home/wg25r/review_agent/human_reviews/yOOJwR15xg.md` | 6.20 (Accept Poster) | HMoRA is slightly weaker: MeteoRA has a more complete system evaluation; HMoRA's core claim (hierarchical routing) is less well-supported |
| DynMoE (dynamic MoE gating, strong results) | `/home/wg25r/review_agent/human_reviews/T26f9z2rEe.md` | 7.0 (Accept Poster) | HMoRA is clearly weaker: DynMoE has a more novel and well-evaluated contribution |
| Sampling Info Sets (fundamentally flawed) | `/home/wg25r/review_agent/human_reviews/XWfjugkXzN.md` | 1.67 (Reject) | HMoRA is far stronger on every dimension |

HMoRA sits between the rejected MoE+LoRA papers (4.0–5.33) and the borderline-accepted ones (5.80–6.20). The CGJS auxiliary loss and the task differentiation result are genuine contributions, but the hierarchical routing—the paper's namesake—is insufficiently isolated, and the "generalization to unseen tasks" claim is overstated. The paper is comparable to MLAE (5.33) but slightly stronger due to the CGJS loss and more consistent improvements, and comparable to but slightly weaker than ELREA (5.80) due to the missing token-only/task-only routing baselines.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>