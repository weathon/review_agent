=== CALIBRATION EXAMPLE 20 ===

# Final Consolidated Review
## Summary

SURE introduces a Prototype-Reliability Graph (PRG) for test-time adaptation of vision-language models, jointly modeling semantic affinity (from text prototypes) and class-wise temporal reliability (from sliding-window confidence statistics) to regularize predictions. The PRG dynamically gates graph edges so that unreliable classes cannot propagate noise, and a closed-loop mechanism co-evolves predictions, prototypes, and graph structure. SURE achieves consistent accuracy gains across 15 benchmarks and two CLIP backbones while maintaining competitive inference speed.

## Strengths

- **Principled integration of semantic structure with temporal reliability for TTA.** Unlike prior TTA methods that treat classes independently or rely solely on per-instance confidence, SURE explicitly models inter-class semantic dependencies and modulates them by statistical stability of predictions over time (Sec. 4.1, Eq. 4–6). This addresses a genuine failure mode—noise amplification through semantically related but unreliable classes—that existing methods do not.

- **Closed-loop co-evolution of predictions and graph structure.** The framework couples logit regularization (Sec. 4.2) with graph evolution (Sec. 4.3), so that predictions refine the graph and the graph regularizes predictions. The ablation (Tab. 4) confirms that component synergy—reliability gating correcting graph imbalances, then logit propagation consolidating reliable information—drives the gains, rather than any single component alone.

- **Competitive efficiency with strong accuracy.** Table 3 shows SURE runs at 0.067s/sample (10× faster than TPT) while achieving 66.23% average accuracy on natural shifts, exceeding both DPE and ZERO. The lightweight graph operations scale linearly with class count per sample after initial construction, making it deployment-friendly.

- **Consistent improvements across diverse evaluation settings.** Gains appear on both natural distribution shifts (Tab. 1) and cross-dataset generalization (Tab. 2), with low variance across seeds (Appendix A.3, ±0.11–0.28%) and competitive calibration (Tab. 10: ECE 7.48 vs. ProtoOnly's 11.23).

## Weaknesses

- **Cold-start vulnerability of reliability estimation.** Reliability scores are initialized as μ_j = 1.0, σ_j = 0.0, assuming all classes are perfectly reliable at t=0. For rare classes or those consistently predicted with low confidence under shift, the sliding-window buffer Q_j may remain near-empty, keeping R_j ≈ 1.0 by default. This means the PRG provides no filtering for precisely the classes most prone to error accumulation—the core failure mode SURE claims to address. The prototype counter N_i^proto = 30000 dampens prototype updates, but the reliability graph lacks an analogous warm-start dampening mechanism, creating an asymmetry that could enable early error propagation before statistics stabilize.

- **Positive feedback loop between regularized confidence and reliability updates.** Algorithm 1 (Line 12) computes confidence c(x) using the regularized prediction p̂(y|x), which already incorporates graph propagation. This regularized confidence then feeds into the reliability update (Eq. 13), which in turn modifies the graph structure for the next sample. This creates a self-reinforcing loop: the graph boosts confidence → reliability increases → graph edges strengthen → confidence boosts further. While R_j is clipped to [0,1], the paper provides no theoretical or empirical analysis of this loop's stability. The ablation in Table 4 shows net positive effects, but no experiment explicitly tests whether the loop can cause overconfidence collapse in adversarial or high-noise regimes.

- **Missing comparison with PROGRAM, the closest related work.** The paper explicitly positions itself against PROGRAM (Sun et al., 2024), which also uses a prototype graph for TTA, claiming SURE differs via "reliability-driven topology" and "VLM-specific design" (Sec. 2). Yet no empirical comparison is provided. Without this, the claim that reliability-driven topology offers unique benefits over feature-distance-based graphs is unverifiable. This is a notable gap given that PROGRAM is the most directly comparable prior method.

- **Reliability score formulation lacks ablation against alternatives.** The formula R_j = μ_j · (1 − σ_j/σ_max) with fixed σ_max = 0.5 is heuristic. The paper states it is "a practical proxy for inverse uncertainty" but provides no comparison against alternative formulations (e.g., normalized entropy of the confidence distribution, exponential moving variance, or simpler proxies like mean confidence alone). The ablation in Table 4 removes reliability entirely but does not test whether a simpler or differently parametrized reliability measure would achieve comparable gains. This leaves open whether the specific multiplicative form and the σ_max constant are necessary or arbitrary.

- **"Consistently outperforms" claim is overstated for specific datasets.** On ImageNet-R (ViT-B), SURE achieves 79.96%, below ZERO (80.75%) and DPE (80.40%). On ImageNet-A (RN50), SURE's 29.57% is below DPE's 30.15% and BCA's 30.35%. The overall average advantages over ZERO and DPE on ViT-B natural shifts are narrow (66.23% vs. 66.10% and 65.93%), within or near the reported standard deviations (±0.11–0.16%). The paper should acknowledge these specific deficits and discuss when graph-based semantic regularization is less beneficial (e.g., style-shifted domains where texture cues dominate over semantic structure).

- **C² scaling concern for large vocabularies is unaddressed.** The similarity matrix S and weight matrix W are C×C, computed and updated as prototypes evolve. For ImageNet (C=1000), this is manageable, but VLMs are increasingly deployed with vocabularies of 10k+ classes. The paper does not discuss memory or computational scaling, nor whether the top-k sparsification suffices to mitigate this at scale. This is relevant to the "generalizable approach" claim in the abstract.

- **No explicit analysis of sample ordering sensitivity beyond seed variance.** SURE's reliability statistics depend on the order of test samples. Appendix A.3 reports variance across seeds that "varying initialization and test-time orderings," which partially addresses this. However, structured ordering scenarios—e.g., temporally clustered by class, class-imbalanced streams, or abrupt domain transitions mid-stream—are not analyzed. For a method that explicitly relies on temporal stability of predictions, this is a meaningful gap.

## Nice-to-Haves

- Evaluation on ImageNet-C (synthetic corruptions), a standard TTA benchmark that would clarify whether PRG's benefits extend beyond semantic/natural shifts to the broader corruption-robustness regime.

- Experiments with other VLMs (ALIGN, BLIP, or larger CLIP variants like ViT-L/14) to substantiate the "generalizable approach" claim.

- Tracking pseudo-label error rate across the test stream to directly validate the "prevents error amplification" mechanism, rather than relying only on final accuracy.

- Per-class performance breakdown showing which specific classes benefit most from reliability gating versus those where it provides no improvement.

- Analysis of multi-domain continuous adaptation (domain transitions mid-stream) to test robustness in more realistic deployment scenarios.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Baseline fairness concern about TPT's augmentation**: The harsh critic questioned whether SURE (single-view) was unfairly compared to TPT (multi-view). TPT uses test-time augmentation as a core part of its method; comparing methods as designed is the correct approach. SURE's advantage of not needing augmentation is a legitimate design feature, not an unfair comparison.

- **"30,000 confident samples" initialization as a weakness**: The positive reviewer questioned the source of 30,000 samples for prototype initialization. This is a misunderstanding: N_i^proto = 30000 is a virtual count (Bayesian prior) in the running-average formula (Eq. 12), weighting the initial prototype 30,000× more than each new update. It is a standard dampening technique, not actual sample collection.

- **Demand for comparison with all cited related works**: The harsh critic noted Fuchs et al. (2025) and Lee et al. (2025) are cited but not compared. Not all cited works need to be baselines; the comparison set is already reasonable. However, the omission of PROGRAM (the most directly comparable method) is kept as a substantive weakness above.

- **Graph visualization limited to 5 classes**: While the 5-class visualization in Fig. 4 is limited, it serves an illustrative purpose. Demanding full-scale graph visualization is a nice-to-have, not a core flaw.

- **Clarification about k specification in method section**: The reviewer requested k = 3·log(C) be stated in the method rather than experiments. This is a minor presentation preference, not a substantive weakness.

## Novel Insights

The most insightful observation across the reviews concerns the positive feedback loop between regularized confidence and reliability estimation. This is not merely a theoretical concern—it reveals a fundamental architectural tension in SURE's closed-loop design: the very mechanism that enables co-evolution (predictions updating the graph, which updates predictions) also creates a self-reinforcing signal path that could amplify early errors rather than suppress them. The initialization assuming perfect reliability (R_j ≈ 1.0 at t=0) exacerbates this, because early noisy pseudo-labels face no reliability gating. The paper's own ablation (Table 4) shows that adding graph structure without reliability ("+Graph w/o Rel") can *hurt* performance (ImageNet-A drops from 57.92% to 57.68%), confirming that the graph can amplify noise when ungated. This suggests the reliability mechanism is not merely an enhancement but a necessity for the graph to be beneficial at all—a dependency that deserves more explicit acknowledgment and stress-testing.

## Suggestions

- Add an experiment comparing with PROGRAM to substantiate the claimed advantages of reliability-driven topology over feature-distance-based graphs.

- Ablate the reliability formulation: compare R_j = μ_j · (1 − σ_j/σ_max) against simpler alternatives (e.g., R_j = μ_j alone, or R_j based on entropy of the confidence window) to show the specific design is necessary.

- Address the cold-start vulnerability: consider initializing reliability conservatively (e.g., R_j < 1.0) or adding a burn-in period before the graph activates, and test whether this improves early-phase robustness.

- Report per-dataset results honestly: acknowledge that SURE underperforms on ImageNet-R and ImageNet-A (RN50), and provide analysis of when semantic graph regularization is less beneficial.

- Test with class-imbalanced or clustered test stream orderings to validate robustness of the temporal reliability estimates beyond random shuffles.

- Discuss C² scaling implications and potential approximations (e.g., approximate nearest-neighbor for top-k, or block-diagonal S for hierarchical label spaces) for deployment at larger vocabulary sizes.

# Actual Human Scores
Individual reviewer scores: [4.0, 4.0, 6.0, 4.0, 4.0]
Average score: 4.4
Binary outcome: Reject
