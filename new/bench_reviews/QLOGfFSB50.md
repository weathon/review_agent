Now let me search for calibration anchors.Now I have enough information to write the final meta-review. Let me synthesize everything.---

## Summary

SPARC is a rehearsal-free continual learning (CL) method that replaces standard convolutions with depth-wise separable convolutions (DSCs), isolates depth-wise filters per task as "working memories," shares half of the point-wise filters across tasks as a "semantic memory" (updated via EMA after each task), and applies an IQR-based weight re-normalization to the classification layer to mitigate task-recency bias. The method achieves state-of-the-art performance on Seq-CIFAR100 and Seq-TinyImageNet while using only ~6% of the parameters of the best rehearsal-based competitor (CLS-ER), and demonstrates linear parameter growth with the number of tasks in contrast to the super-linear growth of PNN-style architectures.

---

## Strengths

- **Dramatic parameter efficiency with leading accuracy on CIFAR100 and TinyImageNet (Table 1):** SPARC uses 1.04M parameters vs. 33.69M for CLS-ER on Seq-CIFAR100, yet achieves 49.03% Class-IL accuracy vs. CLS-ER's 43.80%. On Seq-TinyImageNet, SPARC achieves 32.29% Class-IL with only ~1.9M parameters, while CLS-ER reaches 23.47% at 33.69M — directly validating the core claim.

- **Linear parameter growth confirmed at scale (Table 4):** SPARC grows from 1.04M (5 tasks) → 1.90M (10 tasks) → 3.62M (20 tasks), compared to PNNs growing from 216.7M → 2645.05M. The 20-task evaluation in Figure 2 (SPARC: 88.18% Task-IL, next best CPG: 80.89%) is a concrete long-horizon validation rarely seen in the literature.

- **Clean ablation of the semantic memory design (Table 5):** Comparing shared-all vs. partial-shared vs. fully-isolated point-wise filters clearly shows the proposed design achieves 49.13% Class-IL with 1.04M parameters, nearly matching fully-isolated filters (51.57%) at 59% fewer parameters. This is a well-executed ablation that justifies the architectural choice.

- **Strong performance on Seq-ImageNet100 (Table 3):** SPARC achieves 50.90% incremental accuracy vs. LUCIR's 41.4% (next best) without dataset-specific hyperparameter tuning, demonstrating the method generalizes to larger-scale settings.

- **Minimal per-step computational cost (Table 1):** SPARC requires only 1F and 1B per training step, compared to CLS-ER (3F, 1B), Co²L (4F, 1B), and OCDNet (2F, 1B), making it practically lighter during training.

---

## Weaknesses

### Fatal
None.

### Major

- **Backbone mismatch undermines direct comparison in Tables 1 and 3.** SPARC uses a ResNet-18-like architecture with all convolutions replaced by DSCs, while every baseline uses standard ResNet-18. DSCs have different inductive biases, fewer parameters per task, and different regularization properties. Furthermore, SPARC's parameter budget grows with tasks (1.04M → 1.90M), while all baselines operate within a fixed 11.23M model. This creates an attribution problem: it is impossible to determine from Tables 1 and 3 alone how much of the performance advantage is due to the CL algorithm (EMA semantic memory + weight re-normalization) vs. the architectural switch (DSC isolation, growing capacity, per-task BN). The paper appropriately references Appendix D.2 ("performance of competing approaches with SPARC-like backbone"), but placing this critical validity check only in the appendix weakens the main paper's causal claims considerably. This experiment should be elevated to the main paper.

- **CIFAR-10 underperformance and unexplained high variance.** On Seq-CIFAR10 Class-IL — the most canonical CL benchmark — SPARC achieves 61.22% vs. OCDNet's 73.38% (a 12-point gap), while DER++ (64.88%), TAMIL (68.84%), and Co²L (65.57%) all exceed SPARC. More troublingly, SPARC's standard deviation of ±4.81 across three runs is 5–10× larger than any other method in the same column (most have ±0.05–1.44). This training instability on CIFAR-10 is never analyzed, and the abstract's claim of "superior performance on Seq-TinyImageNet and matches rehearsal-based methods on various CL benchmarks" glosses over a substantial weakness. The bolding of the entire SPARC row in Table 1 is misleading when SPARC ranks 6th of 10 methods on CIFAR-10 Class-IL.

### Minor

- **The "model-surrogate-free" framing is slightly overstated in the title.** Equation 4's EMA update on the shared point-wise filters does store and utilize parameters from previous task training — a form of partial model state retention. The paper consistently qualifies its claim as "full model surrogates," which is technically accurate (CLS-ER maintains two full separate copies of ResNet-18, whereas SPARC retains only a shared subset of filters). However, the title ("beyond model surrogates") and some abstract language could be more precise. The contribution is better characterized as "lightweight partial surrogate" rather than the complete elimination of surrogate mechanisms. This is a framing issue, not a methodological one.

- **Table 3 (Seq-ImageNet100) omits rehearsal-based baselines.** The comparison on ImageNet100 only includes LwF, EWC, MUC, and LUCIR — all regularization-based methods. None of the rehearsal/surrogate methods from Table 1 (CLS-ER, OCDNet, TAMIL, etc.) appear in Table 3, where SPARC shows its strongest result. Given that SPARC's strongest claims are against rehearsal-based methods, omitting them from the largest-scale experiment is a notable gap. (A practical explanation — compute cost or dataset access — would be sufficient, but none is given.)

- **Figure 2's 20-task comparison shows Task-IL only.** Class-IL is the harder and more practically significant evaluation, yet Figure 2 reports only Task-IL results at 20 tasks. This is precisely the extended-sequence regime where SPARC claims advantages; Class-IL results here are deferred to the appendix.

- **No ablation of the κ = 5 constant in weight re-normalization.** Section 3.3 introduces κ as a scaling constant set to 5 "in our experiments," but unlike α (ablated in Figure 4 right), κ receives no sensitivity analysis. Since κ directly scales all FC layer weights and thereby affects Class-IL argmax decisions, it is a non-trivial hyperparameter deserving at minimum a brief ablation.

### Trivial

- The stability-plasticity comparison in Figure 4 (left) is limited to ER, DER++, and LIDER (all rehearsal-based methods). Since SPARC belongs to the parameter-isolation family, comparison against other isolation methods (PNNs, PackNet, CPG) in this figure would be more informative about where SPARC stands within its natural peer group.

---

## Nice-to-Haves

- **Analysis of task-order sensitivity:** Because the EMA semantic memory (Eq. 4) is initialized from task 1's filters and updates slowly (high α), early tasks disproportionately shape the shared representation. Experiments under different task orderings (coarse-to-fine vs. fine-to-coarse) would reveal whether SPARC's results are order-sensitive.

- **Inference cost analysis:** For Class-IL with k tasks, SPARC processes every image through all k sub-networks. Section 3.4 describes this but does not quantify the latency cost at 10 or 20 tasks. A wall-clock comparison would be valuable given SPARC's positioning as a resource-efficient method.

- **Mechanistic visualization of semantic memory:** A CKA or activation-similarity analysis showing what K̃^c actually captures across tasks would substantiate the paper's CLS-theory analogy beyond an architectural metaphor.

---

## Removed Points
*These points are flagged for removal; treat them with caution.*

- **Harsh Critic W1 (EMA = structural model surrogate, fatal framing):** Removed as overstated. The paper consistently uses the qualifier "full model surrogates" and Equation 4 operates on a shared subset of point-wise filters within the main model — not a separate model copy. This is meaningfully different from CLS-ER's two full ResNet-18 EMA models. The framing issue is real but Minor, not structural/Fatal.

- **Harsh Critic on Appendix D.2 being "stripped":** Per the rules, the appendix exists in the original submission and the backbone comparison in D.2 is available. Removed as a reproducibility/appendix criticism.

- **Strength Finder – "Clear architectural decomposition with cognitive motivation":** Kept but not elevated to a primary strength — the CLS analogy is motivational framing rather than a scientific result.

- **Strength Finder – "Stability-plasticity trade-off better than rehearsal methods" (Figure 4):** Weakened. The comparison is constrained to rehearsal-based methods with buffer sizes that limit their anti-forgetting mechanism; the result is informative but not a clean peer comparison.

---

## Novel Insights

The paper's most genuinely insightful observation is that DSCs' channel-wise independence makes them naturally suited to parameter isolation with far lower overhead than standard convolutions — a property that has been noted in efficiency literature but not previously leveraged for CL task isolation. The EMA on a shared subset of point-wise filters (rather than the full model) achieves 95% of the performance of full isolation at 37% fewer parameters (Table 5), which provides a practical design principle: sharing cross-channel mixing (point-wise) while isolating spatial filtering (depth-wise) is an efficient factorization for CL. The IQR-based re-normalization — using outlier-robust statistics from the final training epoch to normalize FC weights — is a simple, data-free, no-overhead solution to task-recency bias that could transfer to other parameter isolation methods.

---

## Suggestions

1. Promote Appendix D.2 (baselines with SPARC-like backbone) to the main paper. This is the critical experiment for validating that the CL algorithm, not just the architectural switch, is responsible for the gains in Tables 1 and 3.
2. Add an ablation of κ in weight re-normalization, even a brief 3-value sweep on one dataset.
3. Report Class-IL results alongside Task-IL in Figure 2's 20-task comparison.
4. Investigate and explain the ±4.81 variance on CIFAR-10 Class-IL — this instability should be characterized.
5. Revise Table 1 bolding to reflect actual best-per-column results rather than universally bolding the SPARC row.

---

## Score and Decision

**Calibration anchors:**

| Path | Avg Human Score | Comparison to SPARC |
|---|---|---|
| HCCkCjClO0 (Online Weight Approximation for CL) | 3.00 | Rejected: weak baselines (only one replay method), unclear methodology, poor writing. SPARC is substantially stronger. |
| tVNZj27pb3 (Parameter Isolation Question) | 3.67 | Rejected: strong title claim not backed by evidence, poor experiments on small convnets only, weak writing. SPARC is clearly above this. |
| Hf54sNeeBM (Knowledge Accumulating Contrastive Prompt) | 4.75 | Rejected: state-of-the-art comparisons but insufficient novelty/analysis. SPARC has stronger ablations and more thorough evaluation. |
| nAs4LdaP9Y (Federated Orthogonal Training CL) | 6.33 | Accepted: novel CL idea with clear experiments. Comparable to SPARC in scope and evidence quality. |
| MeB86edZ1P (Hebbian SNN for CL) | 6.50 | Accepted: novel combination, strong results, clean experiments. Comparable to SPARC — similar level of novelty, strong empirical backing, a few noted weaknesses. |
| mnLmmtW7HO (Active CL) | 6.60 | Accepted: well-motivated, backed by experiments and theory. Slightly higher due to theoretical grounding; SPARC is more empirically focused. |

SPARC is clearly above the low band (≤4): it has strong, comprehensive baselines, genuine parameter-efficiency gains, and a clean novel design. It is at the level of the high band (6.3–6.6) in terms of empirical contribution and novelty, but the backbone-mismatch concern (even if addressed in appendix) is a meaningful caveat that the anchors in the high band did not face in the same form. The instability on CIFAR-10 and the overstated abstract ("superior performance") are additional negatives. Positioning SPARC just inside the lower bound of the high-scoring band is appropriate.

**Final Score: 6.0 — Accept (Poster)**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>