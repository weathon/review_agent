## Summary

SPARC is a rehearsal-free parameter isolation approach for continual learning (CL) that replaces standard convolutional layers with depth-wise separable convolutions (DSCs). Each task gets its own DSC-based "working memory" (task-specific depth-wise + half the point-wise filters), while the other half of point-wise filters are shared across tasks as "semantic memory" updated via EMA. A weight re-normalization scheme based on the IQR of classifier activations mitigates task-recency bias. On Seq-TinyImageNet (10 tasks) and Seq-CIFAR100 (5 tasks), SPARC outperforms most rehearsal-based methods using only ~1-2M parameters versus 23-34M for the strongest baselines. On Seq-CIFAR10 it underperforms several rehearsal methods by a substantial margin.

---

## Claims and Support

| Claim | Support Status | Notes |
|---|---|---|
| Rehearsal-free & no full-model surrogates | ✅ Fully supported | Method description in Sec. 3 is clear; Eq. 6 uses only current-task cross-entropy |
| Parameter-efficient (6% of CLS-ER's params) | ✅ Supported for specific benchmark | ~5.6% on Seq-TinyImageNet 10T (1.90M vs 33.69M); abstract's general claim is slightly loose |
| Superior on Seq-TinyImageNet | ✅ Fully supported | 32.29% Class-IL vs next-best 23.47% (CLS-ER); Table 1 |
| Competitive across "various CL benchmarks" | ⚠️ Partially supported | True for Seq-CIFAR100 and Seq-TinyImageNet; false for Seq-CIFAR10 Class-IL (61.22% vs 73.38% OCDNet) |
| Semantic memory enables cross-task consolidation | ⚠️ Partially supported | Table 5 shows partial sharing beats full sharing or full isolation; EMA's isolated causal role not established |
| Weight re-normalization mitigates task-recency bias | ⚠️ Unsupported in main paper | Sec. 3.3 motivates it; Appendix E.2 cited for analysis; no main-paper ablation with/without Eq. 5 |
| Class-IL without task identity at inference | ✅ Supported | Sec. 3.4 runs all sub-networks; no oracle task ID needed |
| Scalable / practical | ⚠️ Partially supported | Parameter growth is modest (Table 4); inference cost scales with # tasks but is not reported |

---

## Strengths

- **Compelling empirical efficiency story:** On Seq-TinyImageNet (10T), SPARC reaches 32.29% Class-IL and 65.66% Task-IL using 1.90M parameters—substantially outperforming CLS-ER (23.47%/49.60% at 33.69M) and all rehearsal-based methods tested (Table 1). This is a nontrivial result.
- **Strong Seq-CIFAR100 performance:** 49.03% Class-IL and 75.52% Task-IL, beating OCDNet (44.29%/73.55%) and all other listed baselines at 1.04M vs. 22–34M parameters.
- **Honest ablation in Table 5:** The 4-row ablation (all shared → shared pointwise → SPARC → fully separate) cleanly shows the partial-sharing scheme achieves 95% of fully-separate performance at 63% of the parameter cost. This is the strongest piece of evidence for the design.
- **Candid limitations section:** The authors clearly acknowledge task-boundary dependence, static resource allocation, CNN-specificity, and linear growth with task count—unusual for a competitive paper.
- **Figure 2 / longer sequences:** SPARC leads all parameter isolation methods over 20 tasks in Task-IL on Seq-CIFAR100, providing at least moderate evidence of longer-horizon viability.
- **Seq-ImageNet100 result:** 50.90% incremental accuracy vs. LUCIR's 41.4% without dataset-specific hyperparameter tuning, suggesting generalization beyond the main benchmark suite.

---

## Weaknesses

### Fatal
*None.* The core empirical findings are real and not invalidated by any identified flaw.

---

### Major

**1. Class-IL inference cost grows with task count but is never reported.**
Section 3.4 states: *"each image is independently processed through all sub-networks, including their respective batch normalization layers. The outputs of all sub-networks are then concatenated."* This means inference FLOPs scale as O(K) in the number of tasks K—yet Table 1 reports "1F, 1B" for SPARC identically to a single fixed-backbone method. While the "F/B" column likely refers to training passes, the omission of inference-time cost (wall-clock latency, FLOPs, or even the explicit note that Class-IL requires K forward passes) is a significant transparency gap for a paper whose headline is "practical and scalable." For deployment-oriented CL on edge devices, linear inference scaling could be as limiting as linear parameter growth.

**2. Weight re-normalization—a claimed key contribution—has no main-paper ablation.**
The abstract and Sec. 1 contribution list elevate weight re-normalization (Eq. 5) as one of three core contributions. Sec. 3.3 motivates it via weight magnitude disparities. However, the main paper provides no table comparing SPARC with vs. without Eq. 5. The only evidence is a promise of Appendix E.2 ("task-recency bias"). For a claimed key mechanism, this is insufficient—particularly because the normalization constant κ=5 and the IQR-based formula appear without sensitivity analysis or comparison to simpler alternatives (e.g., logit calibration, temperature scaling).

**3. Backbone confound: DSC ResNet-18 vs. standard ResNet-18 for all baselines.**
The paper states: *"most baselines use ResNet-18 while SPARC utilizes a ResNet-18-like architecture with DSC layers."* DSCs have lower parameter counts per FLOP and different inductive biases than standard convolutions. This means comparing SPARC at 1.04M against ER/DER++ at 11.23M parameters could partly reflect architectural differences rather than CL design. Appendix D.2 is cited as providing a fix, but it is not in the main paper and the central Table 1 remains confounded. This limits how strongly one can attribute the performance gap to SPARC's CL mechanism.

---

### Minor

**4. The "competitive on various CL benchmarks" claim is overstated.**
SPARC's 61.22% Class-IL on Seq-CIFAR10 is meaningfully below OCDNet (73.38%), TAMIL (68.84%), and CLS-ER (66.19%)—a gap the authors explain as a regime where buffers are more effective. The paper itself acknowledges this ("In simpler scenarios like Seq-CIFAR10, SPARC's performance is competitive but lags"). However, the abstract and conclusion still use "various CL benchmarks" without qualification, which overstates the breadth of the advantage. The claim should be scoped to "medium-to-hard CL benchmarks with small buffer-to-class ratios."

**5. The 50% sharing fraction appears arbitrary.**
Equation 3 hard-codes a 50/50 split between task-specific and shared pointwise filters. No sensitivity analysis appears for this ratio. The reader cannot determine whether 30%, 50%, or 70% sharing is optimal, whether this ratio should vary by dataset, or whether the improvement in Table 5 is robust to the splitting fraction. This design choice is central to the method yet is entirely unmotivated in the text.

**6. EMA update (Eq. 4) is order-dependent and not analyzed.**
The shared filters are updated from the *previous task's* filters: $\tilde{K}^c = \alpha \tilde{K}^c + (1-\alpha)\tilde{K}^{t-1}$. This makes the semantic memory dependent on task ordering. Unlike a jointly-optimized shared layer, the EMA accumulates in a path-dependent way. The paper provides no analysis of task-order sensitivity, nor does it test whether EMA actually causes meaningful cross-task knowledge transfer vs. simply averaging weights.

---

### Trivial

- The paper mentions "performance under longer task sequences in Section E.3 in Appendix" but Figure 2 only shows 20 tasks in the main paper; calling this "scalability" evidence for unlimited task sequences is slightly generous.
- The Figure 4 (left) stability-plasticity comparison uses buffer size 500 for baselines while Table 1 primarily reports buffer size 200—a minor inconsistency in evaluation conditions.

---

## Nice-to-Haves

- **Forward transfer analysis:** Does the semantic EMA memory benefit new tasks when trained after related tasks? Reporting forward transfer metrics (per-task accuracy when first encountered) would strengthen the mechanistic claim about knowledge consolidation.
- **Inference-time FLOPs vs. task count:** Even a single plot showing wall-clock inference time per sample vs. number of tasks would substantially strengthen the scalability narrative.
- **Task-ordering robustness:** Running experiments with permuted task orders would reveal how sensitive performance is to the EMA's order-dependent aggregation.
- **Adaptive κ or systematic sensitivity analysis:** A 3×3 grid of (κ, sharing fraction) ablation would make the design choices principled rather than heuristic.
- **Visualization of weight magnitudes before/after re-normalization:** This would provide direct evidence for the task-recency bias claim made in Sec. 3.3.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Comparison against L2P/DualPrompt/CODA-Prompt (Spark reviewer):** These are prompt-tuning methods built on ViT backbones with pre-trained representations—a fundamentally different setup. SPARC operates in the CNN, no-pre-training regime. The comparison would mix architecture and CL mechanism, and is outside the paper's stated scope.
- **"Scalability in compute not established" as a fatal flaw (Harsh Critic):** The paper explicitly frames "scalable" relative to growing-architecture methods like PNNs (Table 4), and the limitations section openly acknowledges the linear growth issue. The inference cost concern is valid (kept as Major #1) but does not make the overall scalability claim "potentially misleading"—it's simply incomplete.
- **Claim that "6% parameter" number is too benchmark-specific to report (Harsh Critic):** The paper ties the 6% figure to a specific comparison in the text ("In Seq-TinyImageNet with 10 tasks, SPARC outperforms CLS-ER using just 6% of the parameters"), and the abstract is in line with this. This is a standard and fair way to report a representative efficiency ratio.
- **Concerns about Seq-ImageNet100 table being "too thin" (Harsh Critic):** The comparison is limited but the baselines chosen (LwF, EWC, MUC, LUCIR) include both rehearsal-free regularization and memory-based methods. The paper does not claim this table is comprehensive; removing it would have weakened the paper.

---

## Novel Insights

The central novel insight of this paper is that depth-wise and point-wise components of DSC layers have naturally different roles in a continual learning context: depth-wise filters capture local spatial patterns (inherently task-specific) while point-wise filters perform channel mixing that is amenable to cross-task sharing. This decomposition—though inspired by existing DSC literature—has not been previously applied to construct a biologically-grounded working/semantic memory architecture for CL that eliminates both rehearsal buffers and full-model surrogates. The empirical finding that sharing only the pointwise half achieves 95% of full-isolation performance at 63% of the parameter cost (Table 5) is a non-obvious and practically useful result. The weight re-normalization insight (that isolated classification heads develop magnitude disparities causing recency bias) is real but underdeveloped in its current form.

---

## Suggestions

1. **Add a single-row ablation table** (SPARC vs. SPARC w/o re-normalization) to the main paper. This is the minimum needed to substantiate the third contribution claim.
2. **Clarify the "1F, 1B" entry** in Table 1—add a footnote that Class-IL inference requires K forward passes, and report inference FLOPs or wall-clock time for at least one benchmark as tasks scale.
3. **Move the DSC-backbone fairness experiment** (Appendix D.2) into or adjacent to the main comparison table, or at minimum summarize the key finding in the main body. This is critical for the attribution of gains.
4. **Add a sensitivity study** for the 50% sharing fraction (e.g., 25%, 50%, 75%) in Table 5's framework. Even a 3-point curve would transform an apparently arbitrary hyperparameter into a principled choice.
5. **Narrow the abstract/conclusion claims**: Replace "across various CL benchmarks" with "on mid-to-large complexity CL benchmarks" to reflect the actual scope of the demonstrated advantages.

---

## Score and Decision

**Originality:** Moderate-to-good. The DSC working/semantic memory decomposition is a clean and novel application of existing components to a real CL design problem.

**Importance of Research Question:** High. Parameter-efficient, rehearsal-free CL is practically important for edge deployment.

**Claims vs. Support:** Partially supported. Two of three headline contributions are empirically backed; the weight re-normalization contribution is not adequately validated in the main paper.

**Soundness of Experiments:** Reasonable with caveats. Strong results on harder benchmarks, Table 5 ablation is solid, but the backbone confound and missing inference cost analysis are real gaps.

**Clarity of Writing:** Good. Method description is clear; figures are informative; limitations are honestly stated.

**Value to Community:** Solid. Demonstrates a meaningful and underexplored efficiency regime for CL; Table 1 and Table 4 results are reference-worthy.

**Overall:** The paper makes a genuine empirical contribution—showing that a compact, rehearsal-free isolation architecture can substantially outperform much larger rehearsal-based methods on harder benchmarks. The core idea is sound and the main empirical finding holds up under scrutiny. However, one claimed contribution (weight re-normalization) lacks main-paper validation, the backbone confound is not resolved in the main body, and the inference-cost scaling is transparently unaddressed. These gaps prevent full confidence in all stated contributions. The paper is a borderline accept that would be strengthened with targeted additions.

**Score: 5.5**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>