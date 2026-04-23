Now let me run calibration searches to score the paper.Now I have enough to write the final review. Let me compile it.

---

## Summary

SPARC is a rehearsal-free, surrogate-free continual learning method that decomposes a ResNet-18-like backbone built with depth-wise separable convolutions (DSCs) into per-task working memories (depth-wise filters + task-specific BN) and a task-agnostic semantic memory (shared half of the point-wise filters, updated via EMA). A simple activation-derived weight re-normalization addresses recency bias in the classification layer. On harder benchmarks (Seq-CIFAR100, Seq-TinyImageNet), SPARC achieves state-of-the-art or near-state-of-the-art Class-IL performance using dramatically fewer parameters than rehearsal-based baselines (~1M vs 23–34M), and scales gracefully to 20-task sequences.

---

## Strengths

- **Superior performance on harder benchmarks at extreme parameter efficiency (Table 1, Table 4):** SPARC achieves 49.03% Class-IL on Seq-CIFAR100 and 32.29% on Seq-TinyImageNet, exceeding all rehearsal-based baselines (best competitors: OCDNet 44.29% and CLS-ER 23.47%), while using 1.04M parameters—6% of CLS-ER's 33.69M.

- **Strong scalability to long task sequences (Figure 2, Table 4):** On Seq-CIFAR100 with 20 tasks, SPARC achieves 88.18% Task-IL accuracy vs. CPG 80.89% and PNN 77.33%, with parameter growth of 1.04M → 1.90M → 3.62M (5/10/20 tasks) versus PNNs' catastrophic 216M → 2645M.

- **Principled and validated memory decomposition (Table 5):** The ablation cleanly demonstrates that semantic consolidation (1.04M, 49.13%) nearly matches fully separate filters (1.65M, 51.57%) with 37% fewer parameters and only a 5% relative performance gap, validating the core design.

- **Effective weight re-normalization (Section 3.3):** The IQR-based re-normalization is simple, requires no extra data or parameters, and directly addresses the well-documented recency-bias problem in parameter isolation Class-IL methods.

- **Informative stability-plasticity analysis (Figure 4, left):** SPARC achieves ~50% stability vs. ~35% for DER++/LIDER on Seq-CIFAR100 with buffer 500—remarkable given SPARC uses no rehearsal at all.

---

## Weaknesses

### Fatal
None.

### Major

- **Inference cost grows linearly with number of tasks, yet is never measured or discussed.** Section 3.4 explicitly states that in Class-IL each image is processed through *all* sub-networks independently, with outputs concatenated. For Seq-TinyImageNet with T=10 tasks this means 10 forward passes per image; for 20 tasks, 20. Table 1's "1ℱ, 1𝒷" notation refers strictly to training, not inference. The paper's efficiency framing—"practical and scalable solution for CL under stringent efficiency constraints" (abstract), and Figure 3's model-size vs. performance charts—relies entirely on parameter count and training FLOPs. The dominant deployment cost for an inference-time application (multi-pass inference) is not analyzed at all. This is not a minor omission: a method requiring T inference passes is not straightforwardly more efficient than a single-forward-pass method at deployment time, and for the 10- and 20-task settings this cost is substantial.

- **The efficiency advantage over baselines is largely attributable to the DSC backbone substitution, not to SPARC's CL methodology in isolation.** All baselines use standard ResNet-18 (11.23M); SPARC uses a DSC-variant (1.04M). DSC layers independently reduce parameters by ~8–9× versus standard convolutions—a well-known architectural property of MobileNet-style networks. The paper does acknowledge this and references a comparison with SPARC-like backbones in Appendix D.2 (Section 4.1: "performance of competing approaches with SPARC-like backbone in Section D.2"), but this critical fairness check is deferred to the appendix while the main results and abstract headline the "6%" figure as SPARC's achievement. Without seeing whether CLS-ER or DER++ on the same DSC backbone reach comparable accuracy, the reader cannot attribute the performance advantage to SPARC's CL design rather than to the backbone change.

### Minor

- **The abstract overclaims generality with "matches rehearsal-based methods on various CL benchmarks."** On Seq-CIFAR10 Class-IL, SPARC achieves 61.22% ± 4.81 versus OCDNet's 73.38% ± 0.32 — a 12-percentage-point gap. The paper correctly hedges this in Section 4.1 ("In simpler scenarios like Seq-CIFAR10, SPARC's performance is competitive but lags behind most rehearsal-based approaches"), but the abstract does not reflect this caveat. Additionally, SPARC's variance on Seq-CIFAR10 Class-IL (±4.81) is 3–6× larger than comparable baselines, suggesting instability.

- **Table 3 (Seq-ImageNet100) uses selectively weaker baselines.** The methods compared—LwF, EWC, MUC, LUCIR—are weaker than those in Table 1 (CLS-ER, OCDNet, TAMIL, DER++). No rehearsal-based baseline from Table 1 appears in Table 3. While the dataset may differ in convention, the omission makes it impossible to establish SPARC's advantage over the strongest competitors in this setting.

- **The EMA momentum hyper-parameter κ = 5 in Eq. 5 is unjustified and not ablated.** The paper sets κ = 5 with the terse note "set to 5 in our experiments" and provides no sensitivity analysis. If this was tuned on test performance it represents a hyperparameter leak; if not, robustness evidence is needed.

- **Figure 4 (right) shows stability increasing monotonically with α, including α = 1 (no EMA update at all).** This raises an unexplained question about the utility of the EMA update: if freezing the shared filters after Task 1 maximizes stability, why does the EMA update add value? The paper acknowledges this trade-off ("no information aggregation can be detrimental when tasks in a sequence are completely different"), but provides no plasticity curve at varying α to confirm the trade-off; the benefit of α < 1 is stated but not shown.

### Trivial

- **Table 1 double-uses bold:** the SPARC row is bolded as the proposed method *and* individual bold values indicate per-column best results. On Seq-CIFAR10 Class-IL, SPARC's 61.22% entry is bolded but is not the best result (OCDNet 73.38% is also bolded), creating visual confusion.

---

## Nice-to-Haves

- **Move Appendix D.2 backbone comparison to the main body** (or at minimum add a summary row to Table 1 comparing the strongest baseline trained on a DSC backbone). This would definitively establish that SPARC's CL methodology—not the backbone—drives the accuracy advantage.
- **Report inference FLOPs or wall-clock latency as a function of task count** (e.g., T ∈ {5, 10, 20}) in the efficiency analysis (Figure 3 / Table 4), enabling a fair appraisal of real-world deployment cost.
- **Add a sensitivity table for κ ∈ {1, 2, 5, 10}** on Seq-CIFAR100; this would take one experiment and would establish whether the re-normalization technique is robust.
- **Plot a plasticity curve as a function of α** alongside the existing stability curve (Figure 4 right) to clarify why intermediate α < 1 is preferred over α = 1.
- **Extend comparison on Seq-ImageNet100** to include rehearsal-based baselines (CLS-ER, DER++) for a fair comparison matching the Table 1 standard.

---

## Removed Points
*These points are flagged to be removed — treat with caution.*

- **Harsh Critic's "EMA mechanism is functionally the same as model surrogates" (Section 2 critique):** While mechanistically related (both use EMA), SPARC's shared semantic memory is applied to *half of the point-wise filters* (~0.3M parameters), not a full model copy (~11–33M). The paper's distinction between "full model surrogates" and SPARC's limited EMA-based shared memory is a meaningful quantitative difference in scope even if the mechanism is similar. This is not a clear-cut weakness.

- **"Task 1 structural bias" concern (Section 3.2):** The critic notes that shared filters are initialized and trained only during Task 1, potentially biasing toward Task 1's distribution. The paper explicitly discusses this as the plasticity-stability trade-off governed by α, and provides ablation evidence in Figure 4 (right). The concern is addressed; classifying it as a weakness ignores the provided analysis.

- **Double-use of bold in Table 1 (Trivial):** Retained as Trivial rather than removed, since it creates genuine reader confusion and is actionable.

---

## Novel Insights

The stability analysis in Figure 4 (right) inadvertently reveals a potentially important structural property: SPARC's shared semantic memory is most effective when frozen (α → 1), suggesting that a *Task-1-pretrained shared extractor* plus purely task-specific DSC heads may be the dominant contributor to performance — similar in spirit to linear probing over frozen representations. If true, this would reframe SPARC not as a CL-in-the-traditional-sense algorithm but as a parameter-efficient multi-head architecture with a fixed shared backbone, which has very different implications for the generality of the approach.

---

## Suggestions

1. **Centralize the backbone fairness comparison:** Run at least one strong rehearsal-based method (CLS-ER or DER++) on the same DSC backbone and include the result in Table 1 (not the appendix). This single addition resolves the most substantive criticism.
2. **Add inference cost to the efficiency analysis:** A simple table showing FLOPs per image (or wall-clock inference time) for SPARC vs. CLS-ER as T ∈ {5, 10, 20} would make the efficiency story complete and honest.
3. **Revise the abstract:** Replace "matches rehearsal-based methods on various CL benchmarks" with a formulation that accurately reflects the Seq-CIFAR10 gap and the benchmark-dependent nature of the claim.
4. **Ablate κ** (weight renormalization constant) with 3–4 values to demonstrate robustness.
5. **Add plasticity as a function of α** in Figure 4 (right) to complete the trade-off picture.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Decision | Comparison to SPARC |
|---|---|---|---|---|
| SD-LoRA (rehearsal-free CL, LoRA decomposition) | `/human_reviews/5U1rlpX68A.md` | 7.5 | Accept Oral | Stronger: has theoretical grounding, broader backbone evaluation; SPARC is more empirical and narrower in scope |
| Continual Weighted Sparsity + Meta-Plasticity | `/human_reviews/DaUsIJe2Az.md` | 4.25 | Reject | Weaker: overlapping novelty claims, experimental methodology issues; SPARC has cleaner execution |
| "Do We Really Need Parameter-Isolation?" | `/human_reviews/tVNZj27pb3.md` | 3.67 | Reject | Weaker: poor writing, poorly motivated design, weak experiments; SPARC is clearly superior |
| Continual learning for long-tailed recognition | `/human_reviews/toWEwcbldw.md` | 4.5 | Withdrawn | Similar range: incomplete in different ways |
| Sequential Bayesian CL | `/human_reviews/6r0BOIb771.md` | 5.33 | Reject | Comparable: both have meaningful gaps in comparison fairness |
| Parameter-efficient model merging | `/human_reviews/13D1zn0mpd.md` | 5.67 | Withdrawn | Comparable: incremental contribution in a constrained setting |

**Assessment:** SPARC sits clearly above the rejected CL papers (3.67–4.25) — its empirical results on harder benchmarks are genuinely strong, its design is coherent and well-motivated, and it honestly acknowledges limitations. However, the two major weaknesses (unreported inference cost and backbone-fairness comparison buried in appendix) keep it below SD-LoRA's 7.5 tier. The paper's core claim of efficiency-with-performance is partially undermined by the inference-cost gap and the attribution question. It lands in the 5.5–6.0 range: a paper with real contributions and honest limitations that requires targeted revisions before confident acceptance.

**Originality:** Moderate — DSC-based task isolation is novel in CL context; EMA-shared filters are related to prior model-surrogate methods but applied at much smaller scale.  
**Importance:** Moderate-high — parameter-efficient CL under strict memory constraints is a real need.  
**Support for claims:** Partial — strong on performance claims, weak on efficiency claims (missing inference cost; backbone comparison in appendix).  
**Experimental soundness:** Solid on the main benchmarks; gaps in fairness of Table 3 comparison and missing ablation of κ.  
**Clarity:** Good overall; Table 1 bolding and abstract overclaim are the main clarity issues.  
**Value to community:** Genuine — the DSC + shared-EMA + weight-renorm combination is a practical and deployable approach for memory-constrained CL.

**Final Score: 5.5**

The paper is above the median for CL submissions and has genuine, replicable empirical contributions. The two major issues (inference cost and backbone attribution) are not fatal but are substantive enough to warrant revision before acceptance. The score reflects a paper that is borderline and would benefit significantly from the suggested additions, which are all feasible within a revision cycle.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>