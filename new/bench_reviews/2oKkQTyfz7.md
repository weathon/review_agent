Now I have all the information I need. Let me compile the final review.

## Summary

This paper introduces GSA-VLN (General Scene Adaptation for Vision-and-Language Navigation), a new task requiring VLN agents to execute navigation instructions within persistent environments and adapt over time via long-term memory and unsupervised parameter updates. To evaluate this task, the authors propose GSA-R2R, a large-scale dataset expanding from 29 to 150 evaluation buildings (including non-residential OOD environments) with 90,000 instructions across 7 styles generated via a three-stage LLM-orchestrated pipeline. The paper benchmarks existing VLN methods and adaptation techniques, and proposes GR-DUET, which maintains a global topological graph across episodes with environment-specific pretraining, achieving 8–11.6% SR improvements over vanilla DUET across all evaluation splits.

## Strengths

- **Substantial and diverse evaluation dataset**: GSA-R2R provides 150 buildings across 20 types (vs. R2R's 29 scenes/6 types), 90,000 instructions in 7 styles (vs. R2R's 6,522 in 1 style), and 2,905 unseen vocabulary words (vs. 545), as shown in Table 1. This directly enables the ID/OOD evaluation framing that the paper advocates.

- **Well-motivated and clearly formalized task**: The GSA-VLN task is defined through the memory bank (Eq. 1), history-augmented policy (Eq. 2), unsupervised parameter update (Eq. 3), and environment-agnostic initialization objective (Eq. 4). This formulation cleanly distinguishes the task from standard VLN and lifelong learning, supporting both memory-based and optimization-based approaches.

- **Large, consistent improvements across all splits**: GR-DUET achieves 11.6% SR gain over DUET on Test-R-Basic (69.3 vs. 57.7, Table 4), 8.5% on Test-N-Basic (56.6 vs. 48.1), and up to ~11% on User instruction splits (Table 5), consistently outperforming all baselines across residential/non-residential environments and Basic/Scene/User instruction types.

- **Insightful benchmarking analysis of failure modes**: The paper demonstrates that TTA methods (TENT, SAR) hurt performance in sequential decision-making because error accumulation invalidates entropy as a confidence signal (Table 4: TENT drops SPL from 47.0 to 44.2), and that existing memory methods catastrophically fail with long histories (TourHAMT SR drops to 14.9%). These are non-obvious findings with practical implications.

- **Creative three-stage instruction pipeline**: The pipeline—speaker generation → VLM refinement using navigation success as a filter → LLM rephrasing with role-playing/character profiles (Figure 3)—addresses both noise in generated instructions and the lack of stylistic diversity, producing instructions validated through human evaluation (Table 2: ~80% path-instruction matching, 96.1% style distinctiveness for Scene instructions).

- **Principled ID/OOD categorization**: The residential/non-residential environment split and Basic/Scene/User instruction types provide a clean framework for analyzing generalization, with performance trends in Table 3 (residential > non-residential, Basic > User > Scene) confirming these categories capture meaningful distribution shifts.

## Weaknesses

### Fatal
None.

### Major

- **No evidence that agents actually improve across episodes within a single evaluation run**: The paper's central and repeatedly stated claim is that agents "adapt to specific environments for improved performance over time" (abstract, lines 17, 23, 37, 69). Yet all reported metrics (SR, SPL, nDTW) are aggregates averaged across all 600 episodes per environment. No experiment, plot, or analysis shows that SR in later episodes (e.g., episodes 400–600) is higher than in earlier episodes (e.g., episodes 1–200). The buffer-size ablation (Table 8) provides *indirect* evidence that accumulated history helps—α=1 yields 57.6% SR vs. α=50 yielding 69.3%—but this compares different capacity settings, not temporal improvement within a single run. For a paper whose defining contribution is adaptation *over time*, the absence of per-episode learning curves is a significant evidential gap. A simple SR-vs-episode-number plot (averaged across environments and runs) would be the most natural and important evidence.

- **The ablation does not cleanly isolate the graph mechanism's contribution from privileged pretraining**: Table 7 shows that GR-DUET without graph pretraining achieves 56.8% SR on Test-R-Basic—*worse* than vanilla DUET's 57.7%. The full model (pretrain + PREVALENT augmentation) achieves 69.3%, but the paper does not include the critical control: DUET with the same pretraining strategy (ground-truth full graphs + environment-specific fine-tuning + PREVALENT augmentation) but *without* the cross-episode graph at test time. Without this, the headline "8% improvement" and "11.6% SR increase" conflate the contribution of the proposed graph-adaptation mechanism with the contribution of the pretraining strategy and augmented data. While using ground-truth information during training and building from experience at test time is standard practice (DUET itself uses ground-truth connectivity graphs during training), the degree of privileged information here (complete topological maps) is substantial, and the paper does not disentangle its effect from the cross-episode graph mechanism itself.

### Minor

- **Memory-based baselines (TourHAMT, OVER-NAV) were not adapted for the longer-sequence setting**: These methods were designed for IVLN's 6–100 episodes per environment and catastrophically fail at 600 episodes (TourHAMT: 14.9% SR). The paper explains this as "excessively long history embeddings as input, which confuses the model" (Section 4.3.2), but no simple modifications (e.g., history truncation, sliding window) were tested. The comparison is informative about scalability but conflates architectural inadequacy with input-length limitations. The paper should more explicitly frame this as a scalability test rather than a direct comparison of memory mechanisms.

- **Disconnect between task definition and proposed method**: The formal task definition includes unsupervised parameter updates (Eq. 3), yet GR-DUET keeps parameters fixed and only updates the graph. The paper notes that optimization-based methods (TENT, SAR) are ineffective, but does not reconcile the task definition's explicit support for parameter adaptation with its proposed method's avoidance of it. A brief discussion would clarify whether Eq. 3 is aspirational (defining the full task scope) or whether parameter adaptation was found to be infeasible.

- **Small human evaluation sample**: The user study (Table 2) samples only 20 instructions from 90,000 to assess path-instruction alignment. While the results are encouraging (~80%), this sample is too small to establish reliability, particularly given that 20% of Basic instructions are estimated to be inaccurate—potentially introducing systematic noise into SR as a metric.

- **t-SNE as sole diversity evidence**: Figure 4 uses t-SNE to demonstrate OOD-ness of Scene and User instructions, but t-SNE is known to exaggerate cluster separation and is sensitive to hyperparameters. Quantitative distribution shift metrics (e.g., MMD) would strengthen the claim.

### Trivial
None.

## Nice-to-Haves

- Per-episode learning curves (SR vs. episode number) would directly validate the paper's core claim and are the single most impactful addition possible.
- A DUET+pretraining+augmentation baseline without cross-episode graph at test time would cleanly isolate the graph mechanism's contribution.
- Simple adaptations of memory baselines (e.g., history truncation to last K episodes) would make the scalability comparison more informative.
- Visualization of the global graph growing over episodes would illustrate the adaptation process.
- Analysis of instruction accuracy impact on SR (conditioning on correct vs. incorrect instructions) would address concerns about metric validity.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"GR-DUET's improvement comes primarily from privileged pretraining, not from the adaptation mechanism"** (Harsh Critic #3, original framing): The original framing overstated the issue by claiming the improvement comes "primarily" from pretraining and calling the graph mechanism alone "worse than vanilla DUET." The 56.8% figure in Table 7 is GR-DUET *without pretraining training on how to use the graph*, not the "graph mechanism alone"—a model cannot effectively use a mechanism it wasn't trained to leverage. The pretraining teaches the model to utilize graph structure, and at test time the graph is built from actual experience (not privileged information). The concern about isolating contributions is retained as a Major weakness above, but the original "primarily from privileged pretraining" framing is misleading.

- **"ScaleVLN data leakage makes Table 3 odd"**: The paper explicitly marks ScaleVLN with † and explains the leakage (line 221–222). Including it is actually informative—it shows the upper bound when training data overlaps with evaluation data. Not a weakness.

- **"Proportion baseline is not realistic"**: The Proportion method in Table 8 is a legitimate ablation that compares random ground-truth subgraph provision vs. the memory buffer mechanism. It tests whether the gradual construction process matters vs. just having partial graph information. This is a reasonable design choice, not a weakness.

- **"Conclusion contradicts findings about unsupervised methods"**: The paper shows existing unsupervised methods fail but concludes by saying future work will "explore more unsupervised learning approaches." This is a forward-looking statement about developing *better* unsupervised methods, not a contradiction. Minor at best.

- **"Pretraining description is vague for reproducibility"**: This is a nitpick about implementation detail. The paper describes the key idea (providing complete ground truth topological maps during pretraining) and the fine-tuning strategy. Standard in the field.

- **Missing related works**: Cannot verify existence of suggested references; removed per rules.

- **Formatting/style nitpicks**: Removed per rules.

## Novel Insights

The paper reveals a fundamental tension in sequential decision-making adaptation: entropy-based TTA methods (TENT, SAR) fail not because the adaptation objective is wrong per se, but because error accumulation in sequential settings corrupts entropy as a confidence signal—a finding that challenges the assumption underlying most TTA work. Similarly, the observation that Back-Translation helps for environment adaptation but not instruction adaptation (due to domain shift between authentic and styled instructions) highlights an asymmetry that future adaptation methods must address. The buffer-size ablation's inverted-U pattern (Table 8: α=50 best, α=100/150 declining) also suggests that more memory is not always better—excessively populated graphs create their own inefficiencies, an important practical consideration for persistent-environment systems.

## Suggestions

- Add per-episode learning curves (SR as a function of episode number, averaged across environments) as the single most important missing analysis. Even a simple plot grouping episodes into bins (1–100, 101–200, ..., 501–600) would directly address whether adaptation occurs over time.
- Run DUET with the same pretraining + PREVALENT augmentation but without the cross-episode graph to isolate the graph mechanism's contribution. This is the critical missing control in the ablation.

## Evaluation Assessment

**Originality**: The GSA-VLN task formulation is novel and well-motivated, addressing a genuine gap in VLN evaluation. The dataset contribution significantly advances environmental and instruction diversity over prior work. The method (GR-DUET) is a reasonable extension of DUET with a global graph mechanism, though the architectural novelty is moderate.

**Importance of research question**: High. The persistent-environment adaptation problem is practically relevant and underexplored in VLN. The finding that existing adaptation methods fail in this setting is important for the community.

**Claim support**: Partially supported. The aggregate improvements are real and consistent, but the core claim of "improved performance over time" lacks direct evidence. The ablation does not cleanly isolate the graph mechanism's contribution from the pretraining strategy.

**Experimental soundness**: The benchmarking is comprehensive (8 baselines × 5+ evaluation splits) and the results are consistent. However, the missing per-episode analysis and incomplete ablation control are notable gaps.

**Clarity**: Generally well-written with clear formalization and good use of figures/tables. The three-stage instruction pipeline and ID/OOD categorization are clearly presented.

**Community value**: High. The GSA-R2R dataset with 150 buildings and 90K instructions is a substantial resource that enables new research directions in persistent-environment VLN. The benchmarking of existing methods provides valuable baselines.

## Score and Decision

**Calibration anchors compared**:
- **High**: EQA-MX (avg 8.0, Accept Spotlight) — novel embodied QA tasks + large dataset, cleaner experiments with minimal weaknesses. GSA-VLN has comparable dataset scale but more significant methodological gaps.
- **Medium**: HAZARD (avg 6.75, Accept Poster) — novel dynamic-environment benchmark, relatively minor weaknesses. MrSteve (avg 6.5, Accept Poster) — episodic memory for Minecraft navigation, concerns about task simplicity and missing baselines. SRDF (avg 6.5, Accept Poster) — VLN data flywheel, strong results but generalization concerns. EVA-Bench (avg 5.75, Reject) — new benchmark + method but incremental method and metric issues. DA-Bench (avg 5.75, Reject) — comprehensive benchmark but outdated baselines.
- **Low**: Z91rwXnJsw (avg 2.0, Reject) — visual object navigation with poor results. N581Nje6fH (avg 1.5, Reject) — long-horizon navigation with poor validation.

This paper sits between the medium-scoring anchors. Compared to HAZARD (6.75), it has a more substantial dataset contribution and formal task definition, but more significant methodological gaps (no per-episode analysis). Compared to MrSteve (6.5), it has a larger-scale dataset but similar concerns about incomplete evaluation. Compared to EVA-Bench (5.75, Reject), it has stronger contributions (dataset + task + method + analysis) and its weaknesses are less severe (EVA-Bench had incremental method and flawed metrics). The dataset contribution alone is comparable to accepted benchmark papers, but the two Major weaknesses prevent a higher score.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>