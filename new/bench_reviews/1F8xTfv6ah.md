Now let me search for calibration papers to anchor my score.Now I have enough to write the consolidated final review.

---

## Summary

This paper introduces a post-hoc OOD detection method built on Kolmogorov-Arnold Networks (KANs). The detector computes element-wise differences between a trained KAN and an identically initialized but untrained copy on backbone latent features; in-distribution (InD) samples are expected to activate spline regions modified during training, while OOD samples activate largely unchanged regions and thus produce a smaller trained-vs-untrained discrepancy. To address the inherent limitation that a single KAN models only marginal feature distributions, the authors partition InD data into P subsets and train P separate KAN detectors, aggregating their scores via the maximum. The method is evaluated on four OpenOOD image benchmarks and three tabular medical benchmarks.

---

## Claims and Support

| Claim | Supported? | Notes |
|---|---|---|
| KAN detector outperforms SOTA across all seven benchmarks on overall AUROC | **Partially – overclaim** | Tables 1–3, 5 support image and two medical benchmarks. Table 4 (Age) shows KAN at 50.5 ± 0.5 vs. KLM 51.0 ± 0.7 and MDS 50.8 ± 1.1 — KAN is not best; all results hover near chance with overlapping confidence intervals. |
| Eq. (5) shows coefficient differences encode "where InD information is stored" and the product measures overlap | **Partial/Intuition** | Eq. (5) is a valid algebraic decomposition of the absolute activation difference in the spline basis. But the claim that coefficient changes identify where *InD information* is stored (as opposed to task-relevant discriminative changes) is asserted, not demonstrated. Presented as a working principle, not a theorem. |
| Partitioning captures the joint feature distribution | **Partially** | Table 7 shows P=1 achieves 46.08 ± 15.58 on CIFAR-10, P=10 achieves 94.12. Partitioning clearly helps, but it is an engineering workaround (mixture of marginal models), not joint distribution modeling per se. |
| Training-set size robustness compared to SOTA | **Narrowly supported** | Table 6 shows clear advantage over VIM, KNN, NAC on CIFAR-10/100. Comparison limited to 3 methods and 2 datasets. |
| Spline smoothing provides measurable benefit over binary histograms | **Supported** | Histogram baseline achieves 85.29% vs. KAN's 94.12% on CIFAR-10 — a ~9% gap confirming spline-specific benefit. |
| Robustness to training sample count | **Supported within scope** | Table 6: KAN stays at 93.21% at 0.1% training data; VIM drops to 76.38%, KNN collapses to 8.15%. |
| Method works "regardless of model architecture, training procedures, or types of OOD data" | **Overclaim** | Tested on ResNet + FT-Transformer families only. |
| "Any training task" yields a valid detector | **Overclaim** | One alternative task (regression to constant) tested; effect is benchmark-dependent. |

---

## Strengths

- **First exploitation of KAN local neuroplasticity for OOD detection.** The insight of using trained-vs-untrained spline activation differences as an InD score is architecturally original, distinct from all existing score-based, distance-based, density-based, and flow-based paradigms.

- **Strong, reproducible gains on large-scale image benchmarks.** On ImageNet-200 FS, KAN achieves 71.46 ± 0.40 overall AUROC vs. the next-best 67.18 (ASH), ~4 points. On ImageNet-1K FS, KAN achieves 78.52 vs. 76.28 (NAC). These are multi-point improvements on challenging, full-spectrum benchmarks with multiple OOD sets.

- **Training-set size robustness is a clearly demonstrated practical advantage.** Table 6 shows KAN retains 93.21% AUROC at 0.1% training data on CIFAR-10, while KNN collapses to 8.15% and VIM to 76.38%. This is practically significant for low-data or continual-learning regimes.

- **Transparent acknowledgment of the marginal-distribution limitation and a concrete ablation.** Section 2.3 explicitly identifies the joint-distribution problem, proposes partitioning, and Table 7 provides a full ablation from P=1 to P=30. Most papers would not so clearly expose their method's failure mode.

- **Histogram ablation isolates a KAN-specific benefit.** The binary histogram baseline (same framework, no smoothing) achieves 85.29% vs. KAN's 94.12% on CIFAR-10, providing evidence that spline continuity—not just the trained-vs-untrained protocol—contributes to performance.

---

## Weaknesses

### Fatal
*None that entirely invalidate the results.*

### Major

- **The base mechanism fails, but the framing credits it.** Table 7 shows P=1 achieves only 46.08 ± 15.58 AUROC on CIFAR-10—below random—while the headline results require P=10. The method that actually succeeds is a partitioned-KAN ensemble, not the simple "local neuroplasticity" detector described in Sections 2.2 and highlighted in the abstract/Figure 1. The paper under-foregrounds this: the abstract emphasizes local neuroplasticity and the comparison of two KANs as the core contribution, but the real contribution is the combination of that idea with a K-means ensemble workaround. The conceptual framing and the method that actually performs are in tension. This matters because it affects what the community learns from the paper.

- **No MLP trained-vs-untrained baseline, leaving the KAN-specific claim unsubstantiated.** The central claim is that KAN's *local neuroplasticity* drives the detection. However, nothing in the paper tests whether an MLP detector using the same trained-vs-untrained activation comparison (with k-means partitioning) would achieve similar performance. The histogram baseline is helpful but not equivalent—it replaces splines with binary bins within the KAN framework, not with a competing architecture. Without an MLP control under identical protocol, the paper cannot claim the gains are due to KAN locality rather than to the general trained-vs-untrained ensemble idea. This is the most important missing experiment.

- **Abstract and conclusion overclaim "SOTA across all seven benchmarks."** Table 4 (Age benchmark) shows KAN at 50.5 ± 0.5, KLM at 51.0 ± 0.7, MDS at 50.8 ± 1.1, SHI at 50.4 ± 0.7—all overlapping within noise, with KLM holding the highest mean. The body text correctly hedges to "consistently ranks in the top three across all tabular medical data benchmarks," but the abstract, introduction (line 33: "KAN detector outperforms current State-Of-The-Art techniques across all seven benchmarks"), and conclusion repeat the stronger universal claim. This is a factual overstatement that should be corrected.

### Minor

- **Partitioning strategy has no principled selection criterion for P.** The optimal number of partitions varies by dataset (CIFAR-10 needs P=10) with no heuristic, validation criterion, or cross-dataset analysis provided. The ablation is only on CIFAR-10, leaving unclear whether reported results on other benchmarks are sensitive to this choice.

- **Training-set robustness comparison is selective.** Table 6 compares against only three baselines (VIM, KNN, NAC), despite the paper benchmarking thirteen methods. The claim "in contrast to many other SOTA methods" is broader than the evidence. At a minimum, the results with the full set of image baselines under varied dataset sizes would be informative.

- **No computational profiling.** The method requires 2P forward passes per sample (trained and untrained KAN × P partitions). KANs are inherently slower than MLPs due to spline evaluation. No inference latency, memory, or FLOP comparison is provided against lightweight baselines like KNN or VIM. For a practical post-hoc detector, this context is important.

### Trivial

- **Eq. (5) language is stronger than necessary.** "The terms ... define the locations within the network where InD information is stored" (Sec. 2.2) is stated as fact rather than intuition. Softening to "can be interpreted as" or "serve as a proxy for" would be more accurate and less vulnerable to criticism.

---

## Nice-to-Haves

- A formal or semi-formal derivation linking $\Delta_{p,q}$ to local empirical density or likelihood ratio estimates would strengthen the theoretical grounding.
- Analysis of detector behavior under covariate shift vs. semantic shift, given that the full-spectrum (FS) benchmarks mix both.
- Correlation between high OOD scores and actual classifier errors (misclassification detection), relevant for safety-critical applications.
- Spline evolution visualizations (trained vs. untrained $\phi_{p,q}$ curves on representative features) to provide direct visual evidence of the local plasticity mechanism.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**Removed – reproducibility / implementation nitpicks (rule: hard rule against trivial implementation details):**
- Neutral reviewer criticism about histogram normalization details, KAN depth/width/spline order not specified in main text. These are standard appendix material and do not undermine the contribution.
- Spark reviewer concern about untrained KAN initialization protocol (same seed vs. fixed seed). This is a trivial implementation detail.
- Spark reviewer concern about KAN library and custom spline implementation details.

**Removed – theoretical grounding demanded for an empirical paper (soft rule: MOVE TO NICE-TO-HAVE):**
- All three reviewers' requests for formal probabilistic bounds or statistical guarantees linking $\Delta$ to density ratios. This is an empirical systems paper; such requests are not standard in this setting and have been moved to Nice-to-Haves.

**Removed – availability/existence doubts (hard rule):**
- None arose in this set.

**Removed – generic strengths:**
- "The paper is well-written and clearly structured" (neutral reviewer strength 1 framing)
- "Comprehensive benchmarking" as a standalone strength (expected of any OpenOOD paper)

**Removed – strawman on backbone interaction ablation (spark reviewer weakness 5):**
- The spark reviewer argues that regression-to-constant improving AUROC by 0.2% "suggests the KAN is merely registering samples rather than learning a meaningful task — this weakens the claim that neuroplasticity is the key mechanism." This is a misreading: the paper's own argument in Section 3.3 is that *any* task that moves coefficients near InD samples suffices. A 0.2% improvement under a different loss does not undermine the neuroplasticity claim; it corroborates the spline-coefficient-update view. This point is a strawman.

**Removed – "unfair comparison / SOTA claim" for the Age benchmark (edge case):**
- While the Age benchmark overclaim is kept as a major weakness, the harsh critic's characterization that this "contradicts" the paper is partially too strong — the body text uses the correct hedged language ("top three"), only the abstract/conclusion overclaim. The weakness is kept but scoped to the writing framing, not a contradicted empirical result.

---

## Novel Insights

The most genuinely novel methodological observation is the trained-vs-untrained KAN comparison as an OOD score, which does not require any held-out OOD data, any distributional assumption, or any density model—just a second forward pass through an untrained copy sharing the same initialization. The histogram ablation (9% gap, 85.29% → 94.12%) is a useful empirical fact that directly isolates the contribution of B-spline smoothing vs. binary binning within the same framework, something rarely done in OOD method papers. The training-set size robustness result (KNN at 0.1% collapses to 8.15%, KAN stays at 93.21%) is a practically important and crisply demonstrated finding. The core open question the paper leaves—whether an MLP under the same trained-vs-untrained protocol would match KAN—is itself a useful direction for follow-up.

---

## Suggestions

1. **Add an MLP trained-vs-untrained baseline** under identical protocol (k-means partitioning, same P, same aggregation). This is the single most important missing control and the most direct way to substantiate the "local neuroplasticity" attribution.
2. **Correct the abstract, introduction, and conclusion** to reflect that the method is best on image benchmarks and competitive (top-3) but not best on the Age tabular benchmark.
3. **Reframe the paper's narrative** to lead with the "partitioned trained-vs-untrained KAN ensemble" as the actual proposed method, with local neuroplasticity as the motivating architectural justification. Table 7's P=1 failure should be introduced in Section 2.3 proactively rather than only discovered in ablation.
4. **Provide inference cost profiling** (latency, memory, FLOPs) for KAN vs. KNN/VIM/NAC at different P values to contextualize practical overhead.
5. **Extend the training-set size robustness experiment** to more baselines (at minimum NAC, VIM, KNN, RMDS, ASH) so the "many SOTA methods" phrasing is earned.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Decision | Scores | Comparison Rationale |
|---|---|---|---|
| SCALE (RDSTjtnqCg) | Accept poster | 5,8,6,6 | Post-hoc OOD with clear mechanism analysis and strong ImageNet results. Stronger theoretical isolation than this paper. |
| HACk-OOD (xE5ZaZGqBW) | Reject | 5,6,6,5 | Post-hoc distance OOD, limited to CIFAR only; weaker empirical scope than this paper. |
| Split-Ensemble (SLA7VOqwwT) | Reject | 5,5,5,5 | Ensemble-based OOD method; similar ensemble-workaround flavor. |
| CDR post-hoc (fsEzHMqbkf) | Reject | 3,6,8,6 | Post-hoc density OOD with principled but insufficiently benchmarked approach. |
| ImageNet-OOD insights (VTYg5ykEGS) | Accept poster | 8,6,6,6 | Strong insights paper, different type. |

**Reasoning:** This paper sits above HACk-OOD and CDR in empirical scope (four image benchmarks including ImageNet-1K FS, with clear multi-point gains) and below SCALE in mechanistic rigor (SCALE proves its mechanism via a decomposition backed by both theory and ablation; this paper lacks the MLP control that would establish KAN necessity). The P=1 failure is not hidden — the paper reports it openly — but the framing mismatch between "local neuroplasticity" as the headline story and "partitioned ensemble" as the actual winning method is a real credibility gap. The Age benchmark overclaim in abstract/conclusion is correctable but not trivial. The paper's strongest asset is the ImageNet result set and the training-size robustness finding.

**Positioning:** Closer to the 5/5/5/5 rejected Split-Ensemble than to the 5/8/6/6 accepted SCALE. The paper has more novel architectural grounding than Split-Ensemble but also a more consequential mechanistic attribution gap. A 5.0 (marginally below acceptance) is appropriate.

- **Novelty:** Moderate-high — first KAN-based OOD detector with an architecturally motivated score.
- **Technical soundness:** Moderate — empirically sound on image benchmarks; mechanistic claim unsubstantiated without MLP control; overclaim in abstract.
- **Empirical support:** Moderate-high — strong image benchmarks; tabular results are competitive but not dominant; robustness finding is crisply demonstrated.
- **Significance:** Moderate — strong practical results; whether KAN is the key or ensemble is the key affects significance substantially.
- **Clarity:** Moderate — paper is readable, but the P=1 failure and its implications are buried in ablation rather than foregrounded.

**Score: 5.0**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>