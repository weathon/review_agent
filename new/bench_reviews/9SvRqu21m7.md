## Summary

This paper introduces Multi-Student Distillation (MSD), a framework that partitions a diffusion teacher's conditioning space and distills each partition into a separate single-step student generator, increasing effective model capacity without additional inference cost. The method yields new state-of-the-art single-step FID scores (1.20 on ImageNet-64×64, 8.20 on zero-shot COCO) and demonstrates that TSM (Teacher Score Matching) pretraining enables successful distillation into smaller architectures. The work is well-motivated, experimentally clean in key ablations, and offers a modular drop-in upgrade to existing single-step distillation pipelines.

## Strengths

- **New SOTA single-step FID scores:** With 4 same-sized students, MSD achieves FID 1.20 on ImageNet-64×64 and 8.20 on zero-shot COCO2014 (Tables 1 and 2), directly substantiating the central claim with measurable gains over DMD2 (1.28 / 8.35).
- **Clean isolation of capacity vs. batch-size effects:** Table 3 demonstrates that 4 students at batch size 32 each (FID 2.53) outperforms 1 student at batch size 128 (FID 2.60), ruling out total-throughput as the sole driver of improvement. This is a well-designed ablation.
- **TSM pretraining enables smaller-student distillation:** Section 5.2 and Figure 2 show that without TSM, smaller students fail to converge; with TSM they achieve competitive FID 2.88 with 42% fewer parameters and 7% lower latency. The ablation confirming collapse without TSM provides actionable guidance for the community.
- **Monotonic scaling with number of students:** Table 3 shows consistent FID improvement as K increases (1→2→4→8: 2.60→2.49→2.37→2.32), confirming the capacity-allocation premise generalizes.
- **Modular, drop-in framework design:** The formulation in Section 4.1 separates partitioning strategy, distillation objective (DM/ADM), and pretraining (TSM), making MSD compatible with any conditional single-step distillation method with minimal hyperparameter changes.

## Weaknesses

### Fatal

None.

### Major

### Minor

- **Statistical uncertainty around marginal SOTA margins.** The headline improvements over DMD2 (0.08 FID on ImageNet, 0.15 on COCO) are small at this performance regime. The paper reports single-run point estimates without seed sweeps, error bars, or statistical tests. While single-run reporting is common in the diffusion distillation literature, it makes it possible that the reported gains partially reflect stochastic variance rather than genuine architectural advantage. Reporting results across multiple seeds would strengthen confidence that the multi-student advantage consistently exceeds the noise floor.

- **No single-student-with-TSM baseline for the smaller-architecture experiments.** The paper demonstrates that four 42%-smaller students achieve FID 2.88 (Table 1) but does not report results for a *single* smaller student trained with the same TSM+DM+ADM pipeline on the full dataset. The provided baseline ("Post-distillation, 4 42% smaller students," FID 11.67, Section 5.2 text) is described as lacking TSM rather than as a capacity-matched control. Without this comparison, it is unclear how much of the gain derives from MSD's multi-student capacity scaling versus the TSM pretraining alone. A single smaller student with TSM achieving similar or close FID would partially undercut the claimed MoE-style specialization benefit — and reporting it would clarify the relative contributions.

- **Routing mechanism simplicity and boundary-case robustness unanalyzed.** For ImageNet, the paper uses sequential class splitting (250 classes per student, Section 5.2), which makes routing trivially deterministic via label ID. For text-to-image (Section 5.3), the paper divides pooled text embeddings into "4 quadrants" without specifying whether this uses PCA dimensions or raw embedding coordinates. While simple partitioning is a reasonable starting point and works empirically, the paper does not analyze generation quality for conditioning inputs near partition boundaries (e.g., prompts that straddle clusters), nor does it quantify the latency overhead of any non-trivial assignment procedure. This limits claims about real-world robustness when inputs are ambiguous.

- **Training compute overhead not quantified.** The limitations section (Section 6.1) lists storage cost but does not quantify the increased wall-clock training cost from training K models sequentially with partitioned batches (which reduces per-student GPU utilization). For practitioners evaluating whether the quality gain justifies the additional training time, a compute-time comparison would be informative.

### Trivial

None.

## Nice-to-Haves

- **Per-partition FID breakdown.** Reporting each student's FID on its own partition, alongside the global aggregate, would show whether gains are uniform or concentrated in a subset of students.
- **Visualization of partition assignments.** For the COCO text embeddings, a 2D PCA projection colored by assigned student would clarify whether the "quadrant" split captures semantically meaningful clusters or arbitrary geometric cuts.
- **Ablation of TSM pretraining duration.** The paper establishes TSM's necessity but does not study how its pretraining duration or loss weighting affects final distillation quality — this would help practitioners budget compute for this additional stage.

## Removed Points

**These points are flagged to be removed; treat them with caution.**

- *"Undefined inference routing mechanism invalidates latency and 1-NFE claims" (Structural / Harsh Critic Critical Issue #3).* The paper **does** specify the routing mechanism: sequential class splitting by label ID for ImageNet (Sec. 5.2, line 240: "Each student is responsible for 250 consecutive classes in numerical order") and quadrant splitting on pooled text embeddings for COCO (Sec. 5.3, line 246). Both are simple but well-defined, and their computational overhead is truly negligible (<1ms for class lookup or threshold comparison). The claim that routing is "never formally defined" is factually incorrect. The concern about boundary-case handling is valid but minor (moved to Minor above).

- *"Capacity bleeding analysis" (Harsh Critic deeper analysis point).* The paper explicitly acknowledges and justifies reusing the full paired dataset $\mathcal{D}_{\text{paired}}$ for the regression loss while filtering condition inputs for the KL loss (Sec. 4.2, lines 141–142). The authors hypothesize that this works because "paired data from other input conditions provides effective gradient updates to the shared weights" and ablate the alternative in App. B.2. This is a deliberate design choice, not an oversight. The "capacity bleeding" framing misreads the paper.

- *"The baseline labeled 'Post-distillation' is poorly described" (Harsh Critic Section 5.2 note).* Section 5.2 does clarify: "instead of the TSM stage, we performed post output distillation on best single-step checkpoints, and observed significant drop in performance" (line 242). While the exact training pipeline details could be more precise, the description is sufficient to understand this as a comparison without TSM.

- *Ambiguous description of "Post-distillation, 4 42% smaller students" baseline as a standalone weakness.* As noted above, the paper explains this is distillation without the TSM stage. The criticism that no single-student-with-TSM control was tested above has been retained as a Minor weakness, but the claim of baseline ambiguity itself is addressed partially by the text.

- *Criticism about omitted training compute overhead.* Addressed above as a real but minor issue; the concern is downgraded to Minor since it is inherent to any MoE-style approach rather than a paper-specific failure.

- *"Claims MSD 'prevents real-time generation for computationally heavy applications' conflates distillation limits" (Harsh Critic Abstract note).* The paper's framing in the abstract (line 15) is consistent with the introduction: the speed bottleneck of single-student distilled models is the same as the teacher architecture's size. This is standard motivation and not an overclaim requiring correction.

- *Training cost dismissed.* The limitations (Sec. 6.1, limitation 4) acknowledge that training separate students is suboptimal and suggest "carefully designed weight-sharing, loss-sharing, or other interaction schemes can further enhance training efficiency," which directly addresses this concern as future work.

## Novel Insights

The core framing of this work — allocating model capacity via partitioned-condition single-step distillation rather than monolithic student training — is a clean and practically useful perspective on the speed-quality tradeoff. The most novel operational insight is the paired-data reuse strategy for the regression loss (Sec. 4.2), which avoids mode collapse from per-partition dataset shrinkage while acknowledging shared gradient benefits across partition weights. The TSM pretraining stage fills a genuine gap in the recipe for smaller-student distillation, where initialization from teacher weights is unavailable.

## Suggestions

1. **Train and report a single smaller student with the full TSM+DM+ADM pipeline** as a direct capacity-matched baseline alongside the four-student model. This would clarify the relative contributions of TSM vs. multi-student capacity scaling.
2. **Report SOTA results (1.20 FID on ImageNet, 8.20 on COCO) across at least 3 random seeds** with mean ± standard deviation. Given that the improvements over DMD2 are 0.08 and 0.15 respectively, demonstrating consistency across seeds would strengthen confidence in the SOTA claim.
3. **Add a brief analysis of boundary-case robustness** for text-to-image: sample a handful of prompts near partition boundaries and report any quality degradation relative to well-within-partition prompts.
4. **Disclose wall-clock training compute** (e.g., GPU-hours) for both single-student and multi-student setups in the appendix, so practitioners can assess the quality-compute tradeoff.

## Score and Decision

Calibration against anchor papers:
- **High-scoring anchors (7+):** OlzB6LnXcS.md (scores 8,8,8,8 — Shortcut Models, oral) sets the bar for transformative, well-validated single-step generation papers. This paper is clearly a step below that in novelty and rigor. vkOFOUDLTn.md (scores 6,6,8,8) and lS2SGfWizd.md (scores 6,5,8,6) are closer benchmarks — both present framework-level improvements to diffusion distillation with strong SOTA results but also acknowledged minor limitations (SiDA scored 6 average despite single-run FIDs).
- **Medium-scoring anchors (~5-6):** 2ySt3cdGfJ.md (scores 5,3,6,8) is comparable in scope — a distillation improvement with one notable strength and several methodological gaps. Scored ~5.5 average.
- **Low-scoring anchors (<3-4):** KBixkDNE8p.md (scores 3,3,3,3,3), iQHL76NqJT.md (all 3s) — rejected papers with missing baselines, overclaiming, or no baseline comparison. The paper under review is substantially stronger than these; it provides real empirical results, clean ablations, and genuine novelty.

This paper has verified SOTA results on standard benchmarks, a clean experimental design for isolating capacity effects, and a novel TSM pretraining contribution. Its weaknesses (marginal SOTA margins without variance, missing single-student-with-TSM baseline for smaller models, routing simplicity) are real but not disqualifying — they are consistent with weaknesses seen in human-reviewed papers that scored 5-6. Compared to SiDA (6,5,8,6), it has similar SOTA claims with similar single-run reporting but somewhat cleaner ablations. Compared to DisBack (5,3,6,8), it has fewer methodological gaps.

The missing single-student-with-TSM baseline and marginal FID improvements without statistical reporting hold this slightly below the 6-7 range, but the strong empirical results, clean capacity-vs-batch-size ablation, and useful TSM framework place it above typical borderline papers. I position it between the SiDA-level accepted papers and the DisBack-level borderline papers.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>