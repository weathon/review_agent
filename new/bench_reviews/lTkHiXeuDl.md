Now let me read some of the referenced human review files for calibration.## Summary
HMoRA is a parameter-efficient fine-tuning method for LLMs that integrates Mixture-of-Experts with LoRA in a hierarchical manner. Its two main technical contributions are: (1) a **hierarchical hybrid routing** scheme that smoothly interpolates between token-level and task-level routing as a function of layer depth (shallow layers favor token routing, deep layers favor task routing), and (2) a **Constrained Generalized Jensen-Shannon (CGJS) auxiliary loss** that simultaneously promotes routing certainty (low per-sample entropy) and load balance (high mean-distribution entropy). Trained on Flan v2 and evaluated on seven multiple-choice NLP benchmarks using Qwen2 1.5B, the method is claimed to outperform full fine-tuning with only 3.9% of the trainable parameters.

---

## Strengths

- **Well-motivated hierarchical routing.** The design decision to increase the weight of task-level routing with layer depth is grounded in the empirically well-established finding that shallow LLM layers encode syntactic/token-level features while deeper layers encode semantic/task-level features (Geva et al., 2021). This is a principled and non-trivial instantiation of that insight into a routing schedule.

- **Novel CGJS auxiliary loss.** The CGJS formulation is technically sound: it separates the "balance" objective (maximizing entropy of the mean distribution) from the "certainty" objective (minimizing mean individual entropy) using clamped constraints, avoiding the destructive interference that straight GJS optimization causes. Figure 3 provides direct empirical evidence that standard load-balancing loss (L_bc) reduces certainty while CGJS maintains balance *and* increases certainty. Table 1 confirms +0.82 to +0.85 average accuracy gains from applying CGJS.

- **System-level results are competitive.** Table 2 shows HMoRA w/o LW achieves 64.16 average accuracy vs. 63.15 for full fine-tuning and 63.02 for MoLoRA, while using only 6.31% and 3.82% trainable parameters respectively. Compared to all LoRA-MoE baselines tested, HMoRA is consistently best.

- **Practical lightweight variants.** The router-sharing and hydra-LoRA designs (Figure 2c) reduce both parameters (1.61% → 1.21%) and training time (~37%), providing practical flexibility for deployment.

- **Unseen-task differentiation is partially demonstrated.** Figure 4's t-SNE comparison clearly shows that without CGJS, MMLU tasks form an undifferentiated blob, while with CGJS they separate into meaningful clusters. The quantitative metric (42/57 subtasks differentiated vs. 0 without any loss and 7 with L_bc) supports the claim that the auxiliary loss induces task-separable routing behavior.

---

## Weaknesses

### Fatal
*(None identified — the paper makes real contributions and the core system-level results are not invalidated by the weaknesses below.)*

### Major

- **Missing ablation isolating the hierarchical routing mechanism.** The central claimed novelty is the *hierarchical* α^(l) schedule — that shallow layers should emphasize token routing and deep layers should emphasize task routing. Yet the ablation study never compares **fixed-α hybrid routing** (uniform mixing across all layers) versus the hierarchical schedule of Eq. 8. The paper only reports in Appendix E.5 that "increasing α^(l) with l generally leads to better performance," but this does not quantify the gain from hierarchical scheduling relative to simply using a non-zero fixed α. Without a direct comparison of (a) token-only, (b) task-only, (c) uniform-α hybrid, and (d) hierarchical-α hybrid, the paper cannot establish that the *hierarchy* — not merely the combination of the two routing types — is responsible for the gains. This is the paper's headline architectural contribution and it lacks the ablation to support it.

- **Unseen-task generalization claim is overstated relative to the evidence.** The paper repeatedly claims (Abstract, Section 1, Section 3.3, Section 4.3) that the task router "generalizes to unseen tasks." However, the only evaluation of unseen-task routing is on MMLU subtasks — while MMLU itself is not in Flan v2, Flan v2 explicitly covers "natural language inference, question answering, translation, and sentiment analysis" (Section 4), highly overlapping with MMLU's subject areas. The t-SNE and differentiation statistics show that CGJS induces cleaner clustering on *similar-distribution* held-out tasks, not genuinely novel domains. The paper offers no evaluation on tasks from categorically different domains or formats (e.g., code generation, structured prediction, dialogue) where the "generalization" claim would be non-trivial. Appendix D (theoretical clustering derivation) is not included in the reviewed text, so the mechanistic claim rests on empirical visualization alone.

- **No variance reported despite 5-run repetitions; margins are small.** Section 4 explicitly states "each experiment is repeated 5 times, and we report the mean." Yet Tables 1–3 report no standard deviations, confidence intervals, or significance tests. The headline advantage over full fine-tuning is ~1 point average accuracy (64.16 vs. 63.15), and several per-benchmark differences are below 0.5 points. Without variance, it is impossible to determine whether these differences are statistically meaningful. Given that the authors already ran 5 seeds, adding variance estimates is essentially zero-cost and its omission substantially weakens confidence in the claimed superiority.

### Minor

- **Evaluation restricted to multiple-choice benchmarks.** All seven evaluation benchmarks (MMLU, MMLU-Pro, ARC-E/C, OpenBookQA, SWAG, CommonsenseQA) use multiple-choice format. The model was trained on Flan v2, which covers generation, translation, summarization, and other formats. The paper claims general "multi-task" capability, but provides no evidence on generative or open-ended tasks. This limits the scope of conclusions that can be drawn.

- **Small model scale limits generalizability.** Experiments are conducted exclusively on Qwen2 1.5B and LLaMA 3.2 1B. At these scales, the relative effectiveness of token vs. task routing and of expert specialization may differ substantially from 7B+ models, where task diversity is richer and MoE dynamics are known to change. The claim that HMoRA is an effective LLM fine-tuning strategy would be substantially stronger with at least one 7B-scale experiment.

- **Task encoder computational overhead underreported.** The TaskEncoder (a Transformer encoder processing the full input sequence per batch) introduces inference overhead not captured by the "trainable parameter percentage" metric. Figure 2c shows training time for lightweight variants only; there is no wall-clock comparison of the full HMoRA system against baselines at inference time. For a PEFT paper that emphasizes efficiency, this omission is notable.

- **The soft combination of g_task and g_token lacks justification.** Equation 7 linearly mixes the two gate distributions. The two routers are trained with different loss terms (CGJS applied to both, but the task router also receives the unseen-task differentiation signal), so their output scales and calibrations may differ. The paper does not discuss whether direct linear mixing is well-calibrated, nor compare against normalizing the combined distribution.

### Trivial

- The α^(l) formula (Eq. 8) involves two hyperparameters (ε, μ) that, combined with CGJS's (γ_b, γ_c) and λ, yield five new hyperparameters. Ablation on μ shows insensitivity; a simplified linear schedule would improve usability.

---

## Nice-to-Haves

- **Expert-activation analysis by layer.** Visualizing which experts are activated for syntactic vs. semantic inputs at shallow vs. deep layers would directly verify the presumed specialization behavior that motivates the hierarchical design.
- **Hard-switch ablation.** Comparing the soft α interpolation (Eq. 7) against a binary depth-threshold switch (token-only in first L/2 layers, task-only in last L/2) would empirically justify the soft parameterization.
- **Per-task breakdown of accuracy.** Showing per-task improvements would clarify whether gains are broad-based or driven by a subset of benchmarks.
- **Validation on at least one generative task** (e.g., GSM8K or XSum) to widen the empirical scope.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **[Harsh Critic, Issue 1 — Full FT comparison unfairness]** The harsh reviewer claims that full FT under a 10k-step budget is "not a fair oracle baseline." However, evaluating PEFT methods against full fine-tuning under a fixed training budget is the standard practice in the PEFT literature (LoRA, MixLoRA, HydraLoRA all do the same). The point of PEFT is precisely that it achieves good performance under constrained budgets. The comparison is appropriate; the concern reduces to the missing variance issue, which is already captured as a Major weakness. The "structural" framing — that HMoRA's outperformance of full FT is an artifact of the budget — is speculative without evidence that full FT would surpass HMoRA under a longer budget.

- **[Harsh Critic — Gate value commensurateness]** The claim that g_task and g_token are not "commensurate or calibrated enough to be mixed directly" is a theoretical concern but is speculative without empirical evidence of miscalibration. Both are softmax outputs over the same e=8 experts, making the concern plausible but not demonstrated.

---

## Novel Insights

The CGJS auxiliary loss is a technically interesting contribution that separates two objectives typically conflated in load-balancing losses: global balance (high entropy of the mean distribution) and individual certainty (low entropy of each sample's distribution). The constrained formulation — using floor/ceiling thresholds on the two entropy terms rather than direct maximization/minimization — avoids over-regularization and preserves model flexibility. This is a generalizable technique beyond the HMoRA architecture: it could be applied to any soft MoE system where expert specialization is desired without collapsing all traffic to a small set of experts. The observation that this same loss induces unsupervised task clustering in the routing space (Figure 4) — effectively acting as a contrastive pressure even without explicit task labels — is a noteworthy emergent behavior that connects routing regularization to representation learning.

---

## Suggestions

1. **Add the non-hierarchical hybrid ablation** (fixed uniform α across all layers) to the main paper. This single experiment directly validates the hierarchical claim that is central to the paper's novelty.
2. **Report standard deviations** in Tables 1–3, since 5 runs are already done. This immediately addresses the significance concern on small margins.
3. **Add at least one generative benchmark** (e.g., GSM8K or TriviaQA open) to broaden the empirical coverage.
4. **Qualify the "unseen task generalization" claim** to "unseen tasks from similar domains" and evaluate on at least one categorically different task format to substantiate or limit the generalization claim.
5. **Report inference latency** of the task encoder relative to baselines, not only training time, to fully substantiate the efficiency claim.

---

## Score and Decision

**Calibration:**

| Paper | Topic | Decision | Avg Score |
|---|---|---|---|
| ReMoE (4D0f16Vwc3) | Differentiable MoE routing | Accept | ~6.6 |
| LRR (eWNEqdH0vk) | Layerwise recurrent routing | Accept | ~5.75 |
| Tight Clusters / AC Router (Pu3c0209cx) | Routing via clustering theory | Accept | ~7.0 |
| PERFT (PPjpGTPG5K) | MoE+LoRA PEFT | Reject | ~5.3 |
| Glider (0gVatTOgEv) | Hierarchical global+local routing | Reject | ~4.0 |

**Positioning:** HMoRA is above PERFT (which lacked external comparisons and clear motivation) and well above Glider (which had confused objectives). It is comparable to LRR, which was accepted despite small-scale experiments. However, unlike LRR (which provided clean, well-isolated empirical evidence for its core mechanism), HMoRA lacks the critical ablation isolating hierarchical routing from the rest of the system. The overstated unseen-task generalization claim and missing variance further reduce confidence. The paper lands between LRR (accepted) and PERFT (rejected) — at the borderline. The contributions are real (CGJS loss is novel and validated, system results are solid), but the core architectural claim is undersubstantiated and the evaluation is narrow. Score: **5.0** — borderline reject.

**Axis-by-axis:**
- **Originality:** Moderate. Hierarchical routing is novel in the PEFT context; CGJS loss is technically interesting.
- **Importance of research question:** High. Multi-task PEFT is practically important.
- **Claims well supported:** Weak. The headline claim about outperforming full FT lacks variance; the hierarchical routing mechanism lacks direct ablation; unseen-task generalization is overstated.
- **Soundness of experiments:** Moderate. Solid comparison against MoE-LoRA baselines; gaps in hierarchical ablation and variance.
- **Clarity:** Good overall; method is reasonably well-explained.
- **Value to research community:** Moderate. CGJS loss is a potentially reusable component; results on small models are encouraging but not conclusive.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>