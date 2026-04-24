## Summary

This paper reformulates Reward Augmented Decoding (RAD) as an incomplete reward matrix completion problem, observes empirically that RAD learns low-rank reward matrices, and proposes Autoregressive Reward Model (ARM)—a low-rank, single-forward-pass alternative that reduces per-step decoding complexity from O(k) to O(1). On detoxification and sentiment control, ARM closely matches RAD’s trade-off between control and fluency while being substantially faster.

## Strengths

- **Useful conceptual reframing.** The paper casts RAD’s training objective as matrix completion (§3.1.1), which provides a clean lens for understanding the efficiency–expressivity trade-off and motivates the low-rank ARM design.
- **Real and clearly demonstrated efficiency gains.** Table 1 and Figure 6 show that ARM requires only one forward pass per decoding step versus k passes for RAD, with measured wall-clock time scaling flatly with top-k while RAD scales linearly.
- **Empirical parity on standard benchmarks.** On detoxification and sentiment control, the distilled ARM student closely tracks RAD (Figures 3 and 4), and the response-only ARM remains competitive with other guided-decoding baselines.
- **Careful differentiation from prior work.** The paper clearly positions ARM against GeDi/DExperts (single-pass but less expressive) and RAD (expressive but expensive), and evaluates with up-to-date classifiers.

## Weaknesses

### Fatal
None.

### Major
- **Low-rank motivation is confounded by data sparsity.** The paper’s central conceptual argument—that RAD “does not use its full flexibility” (Abstract, §3.1.2)—is built on observing that RAD learns matrices of rank ~10². However, §3.1.3 explicitly notes that the extreme sparsity of P_Ω(R) means a rank-1 matrix can be compatible with the observed entries when prefixes are unique. The paper acknowledges that “the presence of a low-rank solution compatible with Ω does not imply that the true reward … is necessarily low rank,” yet it still claims “the data has low minimal rank” without showing that the *full empirical* matrix (rather than the sparse observed subset) is actually low-rank. This weakens the analytical foundation: the observation may be an artifact of matrix completion rather than an intrinsic property of the reward function. The empirical results partially rescue the practical claim, but the conceptual justification remains circular.

### Minor
- **Narrow evaluation scope in the main text.** The primary experiments are limited to GPT-2-scale models on two tasks (detoxification and sentiment control). LLaMa-2-(7b/13b) results are mentioned briefly and relegated to Appendix F.1.1, which limits the main text’s support for broad claims about efficiency–expressivity trade-offs in modern controlled generation.
- **Lack of statistical variance estimates.** Figures 3 and 4 present single-point trade-off curves without confidence intervals, standard errors, or significance tests. Metrics like Perspective API toxicity and perplexity are noisy across seeds and prompt subsets; variance estimates would strengthen the “on par” claim.
- **Slight overclaim in the abstract.** The abstract states ARM “performs on par” with RAD, but the response-only ARM visibly lags RAD in fluency on detoxification (Figure 3, §5.4). The distillation result matches or exceeds RAD, but that is a less direct comparison because the student is explicitly trained to imitate the teacher.

### Trivial
- **Mismatch in access assumptions.** §2.1 frames the setup as black-box access to top-k logits, but ARM’s parametrization (§3.2) and experiments (§5.1) require access to the base model’s output embeddings, which are frozen in the reward model. This is a slightly stronger assumption that could be stated more explicitly up front.

## Nice-to-Haves
- Move the LLaMa-2 experiments from the appendix into the main text to strengthen scalability claims.
- Report confidence intervals or multi-seed variance for the trade-off curves in Figures 3–4.
- A controlled rank ablation (e.g., SVD-truncating RAD’s learned matrix or varying ARM’s inner dimension) would directly test how much rank reduction is possible before performance degrades.
- Disentangle the effect of the regularizer (Eq. 11) from rank reduction by fixing rank while varying regularization strength.

## Removed Points
These points are flagged to be removed, treat them with caution.

- *Criticism about missing appendix proofs/details.* The original submission contains appendices; the parser strips them. Claims deferred to appendix (e.g., Appendix B.2 on minimal rank) should not be penalized as missing.
- *“The distillation comparison is circular.”* While expected, the paper is transparent about presenting both distillation and response-only results. This is a presentation preference, not a factual error.
- *Missing related works / obvious next steps.* Demanding multi-attribute control or OOD-prefix testing is useful feedback but falls outside the paper’s stated scope and standard expectations for a decoding-method paper.
- *Formatting / style nitpicks.* None were raised, but any such issues would be parser artifacts rather than author errors.

## Novel Insights

The matrix-completion reformulation of RAD is genuinely novel and clarifying: it lets the authors talk precisely about rank, observed entries, and the softmax bottleneck in a unified framework. The observation that regularization in ARM drives both lower empirical rank and better fluency (Figure 5) hints at an interesting connection between rank, conservatism, and generation quality that could be explored further in future work.

## Suggestions
- Reframe the conceptual story to emphasize the empirical finding that low-rank ARM works well in practice, rather than leaning heavily on the claim that RAD’s learned low rank proves expressivity is unnecessary. This would make the paper more honest without weakening its practical contribution.
- Add error bars or shading to Figures 3 and 4, even if only across a few random seeds or bootstrap resamples of the prompt set.
- Explicitly state in the introduction that ARM requires the base model’s output embedding matrix (or a compatible learned embedding), not just top-k logits.

## Score and Decision

**Calibration anchors used:**
- `/home/wg25r/review_agent/human_reviews/xoXn62FzD0.md` (avg 8.0, Accept Oral) — SMC-controlled generation, 4 diverse tasks, strong theory and open-source system. The current paper is below this in scope and depth.
- `/home/wg25r/review_agent/human_reviews/shgx0eqdw6.md` (avg 7.0, Accept Poster) — ARGS reward-guided search, diverse alignment tasks, GPT-4 eval. The current paper is comparable in topic but narrower in evaluation and weaker in theoretical grounding.
- `/home/wg25r/review_agent/human_reviews/jY5oml9fe9.md` (avg 6.0, Accept Poster) — SASA self-detoxification, multiple models and benchmarks. The current paper has stronger conceptual framing but narrower main-text evaluation.
- `/home/wg25r/review_agent/human_reviews/0er6aOyXUD.md` (avg 5.4, Reject) — RewardMATH benchmark, incremental contribution, limited scope. The current paper has clearer methodological novelty and stronger empirical results.
- `/home/wg25r/review_agent/human_reviews/Ys1ZbGBzHJ.md` (avg 4.0, Withdrawn) — ACD contrastive decoding, methodological rigor concerns. The current paper is well above this in soundness and clarity.

The paper under review has real contributions—a neat conceptual framework, convincing efficiency gains, and solid empirical results on two standard tasks. Its main flaws are a confounded analytical motivation and a narrow main-text evaluation. Relative to the anchors, it sits between the 6.0 poster-level work (similar empirical scope but better conceptual framing than SASA) and the 5.4 reject-level work (clearly stronger than RewardMATH). A score of **6.0** reflects a useful, above-median contribution that would benefit from revision but is already solid enough for poster acceptance.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>