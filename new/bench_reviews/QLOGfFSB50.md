Now let me search for calibration papers.Now I have enough information to write the final review. Let me compile it.

---

## Summary
SPARC proposes a rehearsal-free, surrogate-free continual learning (CL) method that uses depth-wise separable convolutions (DSCs) to create parameter-efficient, task-isolated sub-networks ("working memories"), with shared point-wise filters as a task-agnostic "semantic memory" updated via EMA across tasks. A weight re-normalization technique addresses task-recency bias in the classification layer. SPARC achieves strong Class-IL accuracy on Seq-CIFAR100 and Seq-TinyImageNet using dramatically fewer parameters (~1–3% of full-model surrogate methods), positioning it as a practical CL solution for memory-constrained settings.

---

## Strengths

- **Strong parameter efficiency with superior empirical results on harder benchmarks (Table 1):** SPARC achieves 49.03% Class-IL on Seq-CIFAR100 with only 1.04M parameters, beating the next-best method (OCDNet at 44.29%) which uses 22× more parameters. On Seq-TinyImageNet, SPARC reaches 32.29% — the best across all 15+ compared methods — while using only 6% of CLS-ER's parameters. These are the paper's most defensible and striking results.

- **Compelling parameter growth comparison (Table 4 + Figure 2):** SPARC grows from 1.04M (5 tasks) to 3.62M (20 tasks) on Seq-CIFAR100, while PNNs explode from 216.7M to 2645.05M. Figure 2 shows SPARC maintains 88.18% Task-IL accuracy at 20 tasks, outperforming all parameter isolation alternatives. This is concrete evidence of the scalability claim.

- **Clean ablation of memory sharing configurations (Table 5):** The comparison across (i) all filters shared, (ii) only point-wise shared, (iii) semantic memory consolidation (EMA), and (iv) fully separate filters cleanly shows the EMA approach achieves 49.13% at 1.04M — nearly matching full isolation (51.57%) at 1.65M (59% more parameters). This validates the core memory sharing design.

- **Width/depth ablation (Table 2):** A thorough sweep from 0.009M to 3.91M provides a useful efficiency frontier, confirming consistent behavior with prior work on network scaling, and gives users practical guidance on tuning SPARC.

- **Honest and unusually thorough limitations section (Section 5):** The paper explicitly acknowledges task-boundary requirements, static resource allocation, linear growth in long sequences, and current confinement to CNN architectures — more candid than most ICLR submissions.

---

## Weaknesses

### Fatal
None.

### Major

- **Weight re-normalization (Section 3.3) is listed as a primary contribution but has no isolated ablation.** Table 5 ablates different memory-sharing configurations but never evaluates the effect of removing the weight re-normalization while holding everything else fixed. The paper cannot currently claim that this technique improves performance — it may be neutral or even harmful relative to omitting it. Given that this is explicitly enumerated as one of three contributions in the introduction, this is a substantive gap that weakens the paper's claim structure.

- **Inference cost is not communicated in Table 1, yet is central to the paper's deployment motivation.** Section 3.4 clearly states: "each image is independently processed through all sub-networks, including their respective batch normalization layers." For Class-IL on Seq-TinyImageNet with 10 tasks, SPARC requires 10 serial forward passes at test time while all fixed-capacity baselines (ER, DER++, CLS-ER) require 1. The paper motivates SPARC for edge device deployment, but inference latency scales O(T) with task count. The F/B column in Table 1 covers training passes only — this is not wrong per se, but the inference-time overhead is conspicuously absent from both Table 1 and the limitations section. A reader evaluating SPARC for practical deployment would be materially misled. The authors should add an inference cost comparison (FLOPs or wall-clock) for T=5, 10, 20 tasks.

### Minor

- **Table 3 (Seq-ImageNet100) compares only against non-rehearsal, relatively weak baselines (LwF, EWC, MUC, LUCIR).** None of the rehearsal-based or surrogate-based methods from Table 1 appear here. The comparison is favorable but incomplete; adding at least one rehearsal baseline (e.g., ER, DER++) to Table 3 would strengthen the claim.

- **The α sensitivity analysis (Figure 4, right) only shows stability — not accuracy or plasticity — as α varies.** Stability increases monotonically to α=1.0. The paper notes that α=1 (no EMA update, effectively frozen filters from task 1 onward) can be "detrimental when tasks in a sequence are completely different," but this is not shown empirically. The figure should include overall accuracy or the full stability-plasticity tradeoff across the α range to justify the chosen α < 1 operating point.

- **High variance on Seq-CIFAR10 Class-IL (±4.81%) is unexplained.** SPARC gets 61.22% ± 4.81% vs. OCDNet's 73.38% ± 0.32% — a 12-point gap and 4–15× higher variance than all compared methods. The paper acknowledges the performance lag in one sentence but provides no analysis of what drives the instability (sensitivity to κ, α, or random initialization of shared filters). This is the paper's weakest benchmark result and warrants investigation.

- **Table 5 reports 49.13% for the semantic consolidation variant while Table 1 reports 49.03%** for the same setting. This small discrepancy should be resolved or explained (likely due to different random seeds).

### Trivial
None.

---

## Nice-to-Haves

- **Long-sequence experiments (50+ tasks):** The paper claims scalability, but experiments go only to 20 tasks (Table 4, Figure 2). Given that SPARC's advantage over fixed-capacity methods inverts at some finite task count (acknowledged in Section 5), showing the crossover point explicitly would substantially strengthen the scalability narrative.

- **Backbone-controlled comparison in the main paper:** Section D.2 (Appendix) reportedly evaluates competing methods with a SPARC-like DSC backbone. This analysis directly addresses whether gains come from the training scheme vs. the DSC substitution and should be included in the main paper.

- **Extending the stability-plasticity tradeoff analysis (Figure 4 left) to include Task-IL settings** would provide a more complete picture of SPARC's balance.

---

## Removed Points
*These points are flagged to be removed — treat them with caution.*

- **"1F, 1B misrepresentation in Table 1" (Harsh Critic, Structural Issue #1):** The table header explicitly defines F/B as training passes, and the other baselines' F/B counts (e.g., DER++ 2F, CLS-ER 3F) also refer to training. The training-only interpretation is consistent and not a fabrication. The inference cost concern is retained above as a legitimate transparency issue, but the framing of it as a "materially incorrect comparison" in Table 1 is overstated — the column is a training metric, not an inference metric.

- **"Semantic memory degeneracy at α=1" (Harsh Critic, Structural Issue #4):** The harsh critic argues that near-α=1.0 is essentially "frozen initialization from task 1," undermining the "consolidation" narrative. However, the paper acknowledges this directly: "no information aggregation can be detrimental when tasks in a sequence are completely different than the first task," justifying α < 1. This is a valid presentation concern (the full tradeoff at varying α should be shown) but is not the structural contradiction the critic implies. Retained as a minor weakness above.

- **"6% claim is a snapshot" (Harsh Critic, Structural Issue #2):** The 6% claim is numerically accurate at 10 tasks against CLS-ER's fixed size, and the paper acknowledges linear growth in the limitations section. The fact that the comparison eventually inverts at ~90+ tasks is honestly stated, making this a minor framing issue rather than a structural error. Already weakened to a nice-to-have.

- **"PackNet masks as surrogates stretch the definition" (Harsh Critic, Section 2 note):** This is a framing judgment call in the introduction, not a factual error. REMOVE as a pure terminology disagreement.

- **"CLS theory analogy is overstated" (Harsh Critic):** The biological analogy is clearly framed as inspiration, not a rigorous claim. Standard practice in empirical CL papers.

---

## Novel Insights
The paper's most interesting structural insight is that replacing standard convolutions with DSCs within a parameter-isolation framework achieves two goals simultaneously: it reduces per-task parameter footprint enough to make task-specific isolation practical even at 10–20 tasks, and the forced decomposition into spatial (depth-wise) and channel-mixing (point-wise) operations creates a natural split for task-specific vs. shared knowledge. The empirical observation that sharing only the point-wise (channel-mixing) filters — updated slowly via EMA — provides nearly the same accuracy as full separation while using 37% fewer parameters is a non-trivial finding with broader implications for efficient multi-task representation learning. The stability-plasticity tradeoff in Figure 4 showing SPARC achieving the best stability while maintaining reasonable plasticity relative to rehearsal-based methods offers a complementary perspective on why parameter isolation with small sub-networks can be competitive with rehearsal despite having no access to previous task data.

---

## Suggestions

1. **Add an isolated ablation for weight re-normalization**: Run SPARC with and without Section 3.3 on at least Seq-CIFAR100 and report the delta. If it helps, quantify by how much; if it doesn't, revise the contribution list accordingly.

2. **Add an inference cost comparison table**: Report per-sample FLOPs and wall-clock inference time at T=5, 10, 20 tasks for SPARC alongside key baselines. This is central to the edge-device deployment claim and should not be hidden or absent.

3. **Fix the α sensitivity plot (Figure 4, right)**: Add overall accuracy (or full tradeoff metric) as a function of α, not just stability, to justify the operating point and show why α=1 is suboptimal in practice.

4. **Move the backbone-controlled comparison (currently Appendix D.2) into the main paper**, or add a dedicated paragraph in Section 4 summarizing the result. This is important for establishing whether SPARC's gains are architectural or algorithmic.

5. **Analyze the Seq-CIFAR10 variance**: Identify what drives the ±4.81% standard deviation across 3 runs and add a brief explanation in Section 4.1.

---

## Score and Decision

**Calibration anchors:**

| Path | Avg Score | Comparison |
|---|---|---|
| `/home/wg25r/review_agent/human_reviews/sSyytcewxe.md` | 7.0 | Accepted poster; ensemble CL, exemplar-free, strong Class-IL results. SPARC has similarly strong empirical results but missing ablation of one stated contribution. |
| `/home/wg25r/review_agent/human_reviews/FbuyDzZTPt.md` | 6.0 | Accepted poster; rehearsal-free CIL, comparable benchmark scope. Similar level of contribution. |
| `/home/wg25r/review_agent/human_reviews/DJZDgMOLXQ.md` | 6.5 | Accepted poster; novel CIL approach, good empirical validation. |
| `/home/wg25r/review_agent/human_reviews/1nHQRsb3Ze.md` | 5.0 | Rejected; solid CL paper but limited novelty. SPARC is stronger in results and framing. |
| `/home/wg25r/review_agent/human_reviews/tVNZj27pb3.md` | 3.67 | Rejected; parameter isolation CL, weak contribution and experiments. SPARC is clearly stronger. |
| `/home/wg25r/review_agent/human_reviews/ZHTYtXijEn.md` | 2.33 | Rejected; structurally weak CL paper. SPARC is far better. |
| `/home/wg25r/review_agent/human_reviews/HCCkCjClO0.md` | 3.0 | Rejected; weak continual learning paper. SPARC is clearly stronger. |

SPARC sits clearly above the low anchors (3.0–3.67): it has genuine empirical contributions, an honest scope, and well-motivated design. It aligns most closely with the 6.0–7.0 accepted-poster tier: strong empirical results in a real problem space, a practical engineering contribution, but with a missing ablation for one of its three stated contributions and a transparency gap around inference cost. The sSyytcewxe (7.0) analog is close but SPARC's inference cost issue and missing ablation pull it slightly below that level. I anchor the score at **5.5** — a borderline weak accept — reflecting a paper with real merit but two correctable gaps (weight re-normalization ablation and inference cost disclosure) that the authors should address.

**Axis summary:**
- **Originality:** Moderate — DSC-based task isolation is a natural engineering choice, but its application to this CL setting with shared semantic memory is novel enough.
- **Importance:** Good — parameter-efficient CL without rehearsal is highly relevant for edge deployment.
- **Claims supported:** Partially — Class-IL and Task-IL results are well-supported; weight re-normalization contribution is unverified.
- **Soundness of experiments:** Good on main benchmarks; Table 3 baseline selection is weak.
- **Clarity:** Good overall; inference cost and α analysis need improvement.
- **Value to community:** Solid poster-level contribution with practical utility.

**Decision: Borderline Accept (Weak Accept)**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>