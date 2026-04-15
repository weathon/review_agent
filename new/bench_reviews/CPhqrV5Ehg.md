Let me read some calibration papers.Now let me write the final consolidated review.

---

## Summary

This paper analyzes Reward Augmented Decoding (RAD) by reformulating its training objective as an incomplete reward matrix completion problem. The authors empirically show that despite RAD's capacity to represent high-rank reward matrices, the learned matrices are low-rank in practice. Motivated by this, they propose ARM (Autoregressive Reward Model), a low-rank parameterization that predicts rewards for all next-token candidates in a single forward pass (vs. *k* passes for RAD), achieving comparable performance on detoxification and sentiment control tasks with substantially lower decoding cost.

---

## Strengths

- **Novel matrix completion reformulation:** The reformulation of RAD's training objective as low-rank matrix factorization (§3.1) is the paper's clearest conceptual contribution. The connection to the softmax bottleneck is elegant, and the derivation in §3.1.1 is well-executed and provides genuine insight into why RAD's expressivity may be unnecessary for these tasks.

- **Substantial and well-quantified efficiency gain:** Table 1 and Figure 6 convincingly demonstrate that ARM achieves O(1) forward passes per decoding step while RAD requires O(k). At top-k=80, this translates to roughly 10× speedup in reward model time per token on an RTX A6000 GPU. This is the paper's most solidly supported contribution.

- **Competitive empirical performance:** On both detoxification (Fig. 3) and sentiment control (Fig. 4), ARM (especially the distilled variant) closely matches or slightly exceeds RAD on the toxicity/fluency and sentiment/fluency trade-off curves. Results also extend to LLaMA-2-7b/13b (Fig. 14, Appendix).

- **Principled parameterization with ablation support:** The decomposition into baseline + marginal reward (Eq. 6–7), with regularization toward zero marginal reward for unrelated tokens (Eq. 11), is a clean and well-motivated design. Figure 5 provides directionally useful ablation evidence that both components contribute to fluency and rank reduction.

- **Empirical verification of RAD's high-rank capacity:** Appendix C.1 explicitly verifies that RAD *can* learn high-rank matrices when the target is high-rank—confirming that the low-rank behavior observed in practice is not a limitation of RAD's architecture but reflects the nature of the task. This is an important control experiment.

---

## Weaknesses

### Fatal
*(None. The core empirical claims are supported; the paper does not make unsupportable broad theoretical claims as its primary contribution.)*

### Major

- **Distillation confounds the central efficiency claim.** The paper's headline contribution is a low-rank parameterization that matches RAD's expressivity with lower cost. However, the strongest results (Figs. 3–4) come from *distilled ARM*, which requires training a RAD teacher first. ARM trained on responses only—the standalone variant that would validate the low-rank claim in isolation—shows "slightly worse fluency" (§5.4, Fig. 3) and lags behind on sentiment (Fig. 4). The paper's conjecture in §5.4 (distillation provides a smoothed target vs. noisy raw responses) is reasonable but untested. The practical implication is significant: users who want the best performance must first incur the full cost of RAD training, then additionally train ARM. This substantially weakens the "no need for RAD's expressivity" narrative.

- **The low-rank explanatory claim is empirical on a narrow base and somewhat overstated in the abstract/introduction.** The abstract states ARM demonstrates "RAD does not use its full flexibility" as a general principle, but this observation comes from Figure 1 (rank measured on up to 4,000 sampled training prefixes for two binary control tasks) and §3.1.3 explicitly acknowledges that the true, fully-observed reward matrix need not be low-rank. The paper correctly scopes this in the conclusion and limitations ("further qualitative research is needed"), but the introduction and abstract do not communicate this limitation clearly. The paper's real claim—*that on these two tasks, RAD empirically learns low-rank matrices and a low-rank model suffices*—is well-supported. The stronger mechanistic claim is not.

### Minor

- **Limited task diversity.** Both evaluation tasks (detoxification, sentiment) are binary coarse-grained attribute control problems. The paper acknowledges this in Limitations, and the low-rank argument is task-dependent by the authors' own admission. At minimum, a failure-case analysis—identifying conditions under which ARM's rank constraint hurts relative to RAD—would sharpen the paper's boundary conditions.

- **No uncertainty quantification on trade-off curves.** Figures 3 and 4 compare trade-off curves across models using stochastic decoding over thousands of prompts, yet no confidence intervals or variance estimates are reported. Several claims ("ARM closely follows RAD," "ARM slightly outperforms RAD") rely on visual curve comparisons that could be affected by sampling noise.

- **The theoretical explanation of why low-rank structure persists is incomplete.** Section 3.1.3 motivates the low-rank phenomenon through the extreme case where each prefix appears once (giving a rank-1 compatible matrix). The paper acknowledges this is a simplification and defers to Appendix B.2. While the appendix provides additional support, the main text argument is thin for the weight placed on this claim.

### Trivial

- The efficiency comparison in Figure 6 is measured for GPT-2-Small scale; efficiency numbers for LLaMA-2 (where the absolute savings would be larger and more practically meaningful) are not reported.

---

## Nice-to-Haves

- **Explicit rank-controlled ablation:** Directly varying the effective rank of ARM (e.g., via low-rank adapters with tunable r ≪ d) would map the efficiency–quality frontier and identify the minimal sufficient rank for each task. This would convert the empirical low-rank observation into a quantified design parameter.

- **Addressing the softmax bottleneck empirically:** The paper notes that ARM inherits the softmax bottleneck (rank ≤ d) and that mitigations exist (Yang et al., 2018; Ganea et al., 2019), but does not explore whether these would help or whether the bottleneck is ever binding in practice.

- **Analysis of whether distillation benefit is due to low-rank structure or teacher smoothness:** The ARM student trained via distillation outperforms training on responses; the paper conjectures this is because RAD provides smoothed targets. Disentangling whether the benefit is structural (low-rank) or statistical (smooth labels) would clarify the mechanism.

- **Qualitative generation examples:** Side-by-side text samples comparing ARM vs. RAD would help determine whether "on par" quantitative metrics reflect genuine equivalence or mask systematic quality differences.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**R1 (Harsh Critic — Evaluation metrics not aligned with "distribution preservation"):** The paper uses MAUVE (against unguided generations) and perplexity (under GPT-2-XL/OLMo) as fluency proxies. The harsh critic argues these do not directly measure "preserving the base LM distribution." However, these are the exact standard metrics used throughout the controlled generation literature (including in RAD, DExperts, GeDi), and MAUVE against unguided outputs is specifically designed to measure distributional similarity. The paper frames its evaluation correctly and cannot be faulted for following community standards.

**R2 (Spark — Unfair comparison with GeDi/DExperts using old numbers):** The paper explicitly states: *"We rerun the evaluation for RAD, GeDi and DExperts with an up-to-date Perspective API classifier"* (Fig. 3 caption, §5.4). The concern is directly addressed by the paper.

**R3 (Spark — Scaling to larger models not tested):** The paper reports LLaMA-2-7b and LLaMA-2-13b results (Fig. 14, Appendix F.1.1). This concern is addressed.

**R4 (Human Finder — Limited comparison to recent methods):** Omitted per the DO NOT MENTION MISSING RELATED WORKS rule. Cannot confirm what recent methods exist without external sources.

**R5 (Human Finder — Reliance on automatic metrics without human evaluation):** Human evaluation is not standard for this specific task and community; prior work (RAD, DExperts, GeDi) all use the same automatic metrics. Moving to Nice-to-Haves would be appropriate, but given the reviewers cite papers at similar quality levels that also lacked human evaluation and were accepted (e.g., J0qTpmbSbh), this is not a meaningful differentiator.

**R6 (Harsh Critic — Missing decoding hyperparameter fairness in curve comparisons):** While a reasonable methodological concern, the paper follows the exact evaluation protocol of RAD (Deng & Raffel, 2023) by sweeping β for both models. The curves are directly comparable within this protocol.

---

## Novel Insights

The most genuinely novel observation in this paper is the matrix completion lens on RAD, which precisely characterizes why RAD's expressivity is architecturally higher than empirically necessary. The connection between the low-rank ARM parameterization and the classic softmax bottleneck (Yang et al., 2018) is a useful bridge between the controlled generation and language modeling literature. The further observation that regularization toward zero marginal reward effectively reduces the estimated rank of the output matrix (Fig. 5a) is a clean empirical insight linking the regularizer's design to the paper's central analysis. These together constitute a coherent (if empirically narrow) analytical story about efficiency-expressivity tradeoffs in reward modeling.

---

## Suggestions

1. **Test ARM in the responses-only (standalone) regime more extensively.** Add training ablations (regularization schedules, data augmentation, curriculum over prefix lengths) to understand whether the gap with distilled ARM can be closed without a RAD teacher.

2. **Add failure case analysis.** Construct tasks or prompts where the low-rank constraint is expected to hurt. Even negative results here would sharpen the paper's scope claims substantially.

3. **Clarify the abstract/introduction.** The abstract currently implies a general principle ("RAD does not use its full flexibility") without communicating the empirical and task-specific nature of this claim. One sentence acknowledging this limitation upfront would prevent misreading.

4. **Report efficiency numbers for LLaMA-2 settings.** This would contextualize practical savings for modern deployment scenarios, which are the primary motivation for this work.

---

## Score and Decision

**Calibration:**

- **J0qTpmbSbh (GenARM, Accept, scores 8/6/6/6):** Most similar topic. GenARM had theoretical guarantees (provably guides frozen LLMs toward any RM-achievable distribution), three different evaluation settings (preference alignment, weak-to-strong, multi-objective), and stronger empirical scope. This paper under review lacks theoretical guarantees and covers only 2 binary tasks—it is clearly below GenARM's quality level.

- **UAA2nWUtVl (CARDS, Reject, scores 6/5/6/6):** Decoding-time alignment paper with similar practical scope but criticized for limited baselines and unverified claims. This paper is comparably scoped but has cleaner methodology and a more principled analytical contribution.

- **TOveLu4O51 (DETOXIGEN, Reject, scores 5/5/8/5):** Detoxification paper with limited evaluation scope and weaker theoretical motivation. This paper is stronger than DETOXIGEN due to the matrix completion reformulation and more rigorous experiments.

- **KMWGzQi7Qy (tokenwise RGTG, Reject, scores 5/8/6/3):** Similar focus on partial-sequence reward models, mixed scores. This paper is stronger in empirical execution.

**Assessment:** The paper sits between UAA2nWUtVl (rejected, ~5.75 average) and GenARM (accepted, ~6.5 average). The efficiency contribution is real and well-documented, the conceptual contribution (matrix completion view) is genuinely insightful, and the empirical results are competitive. However, the scope is narrow (2 tasks), the strongest results depend on distillation from the expensive model being replaced, and the core analytical claim is somewhat overstated in the abstract. This is a solid incremental paper that advances practical efficiency for reward-guided decoding with a principled conceptual framework, but it falls short of the theoretical depth and evaluation breadth that characterize stronger papers in this space.

**Final Score: 5.5**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>