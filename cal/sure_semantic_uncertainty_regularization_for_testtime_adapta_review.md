=== CALIBRATION EXAMPLE 39 ===

# Harsh Critic Review
Now I have enough information to write a thorough review. Let me compose it.

---

## Section-by-Section Critical Review

### Title & Abstract

The title "SURE: Semantic Uncertainty Regularization for Test-Time Adaptation in Vision-Language Models" is accurate and descriptive. The abstract's core claims—(i) a dynamically evolving prototype-reliability graph, (ii) semantic consistency enforcement, (iii) error-amplification prevention, and (iv) consistent state-of-the-art—are all addressable through the paper's content. However, the abstract's framing that the method "consistently outperforms prior methods" is overstated: on individual datasets (e.g., ImageNet-R and ImageNet-A in Table 1), SURE loses to several baselines, and it only wins on *average*. This distinction matters scientifically.

---

### Introduction & Motivation

The motivation is reasonable: pseudo-label noise in TTA is a known and real problem, confidence thresholding discards useful signal, and exploiting semantic structure across classes is a natural solution. These points are clearly made.

However, the introduction claims that VLMs are *particularly* vulnerable to distribution shift compared to unimodal models (paragraph 1). This is an empirical claim that is neither cited nor demonstrated. It is plausible, but stated as fact without evidence.

The distinction between SURE and PROGRAM (Sun et al., 2024) is described in the related work section (§2) but never validated experimentally—PROGRAM is absent from all result tables. Given how closely PROGRAM resembles SURE conceptually (graph-based pseudo-label propagation at test time), this omission is the single most glaring weakness of the paper. A claim of superiority over a closely related graph-based method without direct comparison is not credible.

---

### Method (§4)

**Reliability score formulation (Eq. 4):** The reliability score $R_j = \mu_j \cdot (1 - \sigma_j / \sigma_{\max})$ with $\sigma_{\max} = 0.5$ is a heuristic. The paper invokes Shannon entropy to lend theoretical credibility, but $R_j$ is not an entropy measure—it is an ad hoc product of empirical mean and normalized standard deviation. The choice of $\sigma_{\max} = 0.5$ is not derived or justified; it is stated as a fixed hyperparameter "to ensure stable normalization." This works in practice, but the paper's information-theoretic framing is misleading.

**Cold-start behavior:** The initialization $\mu_j = 1.0$, $\sigma_j = 0.0$ makes $R_j = 1.0$ for all classes at the start, so the initial graph reduces to the pure semantic similarity matrix **S**. This means the reliability guard is entirely inactive during early adaptation—precisely when noisy pseudo-labels are most likely. The paper does not discuss this cold-start vulnerability or its effect on performance.

**Prototype initialization with $N_i^{\text{proto}} = 30000$:** Following Zhou et al. (2025), each prototype starts with a count of 30,000. A single test sample therefore contributes weight $1/30001 \approx 3 \times 10^{-5}$ to the prototype update. Even after processing all 50,000 ImageNet test samples, each prototype receives at most a handful of updates per class (1000 classes, 50 samples/class on average). The practical magnitude of prototype drift is therefore vanishingly small. This undermines the paper's framing of "prototype evolution" as a meaningful mechanism. The ablation study should quantify actual prototype displacement across the test run.

**Log base ambiguity in $k = 3 \cdot \log(C)$:** For ImageNet ($C = 1000$), this gives $k \approx 21$ (natural log) or $k = 30$ (base-10 log). The paper never specifies the base, making this hyperparameter non-reproducible.

**Single-step belief propagation (Eq. 9–10):** The paper justifies graph smoothing as "one-step belief propagation in a class-level MRF." No ablation studies multi-step propagation, nor is there any theoretical justification for why one step is sufficient. The equal-weight combination of raw and graph-based predictions in Eq. 10 (50/50 blending) is not ablated.

**Graph formalization (§4.1):** The formal system $G^{(\ell)} = (V, E^{(\ell)}, M, U)$ is presented as a "closed-loop dynamic system," but nodes are class prototypes—not test instances. The graph does not explicitly route information between test samples, only between class representations. This is quite unlike graph neural networks applied to sample graphs. The framing slightly overstates the structural complexity.

---

### Experiments & Results (§5)

**Marginal wins over strongest baselines.** On ViT-B (Table 1), SURE achieves 66.23% average vs. ZERO's 66.10% (+0.13%) and DPE's 65.93% (+0.30%). These margins are within or near the reported standard deviations (±0.11–0.16%). On RN50, SURE loses to no single competitor on *all* datasets simultaneously—it loses to BCA on ImageNet-A (29.57% vs. 30.35%). The paper does not report confidence intervals or statistical tests, so it is unclear whether the average gain is statistically significant.

**Inconsistent dataset-level performance.** The claim of "consistent outperformance" (abstract, §5.2) is contradicted by the tables themselves:
- RN50, ImageNet-A: SURE 29.57% < BCA 30.35%
- ViT-B, ImageNet-R: SURE 79.96% < ZERO 80.75%, TDA 80.24%, BCA 80.72%
- ViT-B, EuroSAT (Table 2): SURE 53.60% < TDA 58.00%, BCA 56.63%

SURE wins on *average* but loses on multiple individual benchmarks to multiple baselines. This pattern suggests the method trades off fine-grained performance in some domains for modest aggregate gains. The authors' narrative of "consistent" superiority requires more careful qualification.

**Missing comparison with PROGRAM (Sun et al., 2024).** PROGRAM is identified in §2 as the closest related graph-based TTA method, and two specific distinguishing claims are made (reliability-driven topology vs. static graph; VLM-specific vs. uni-modal). Yet PROGRAM does not appear in any result table. This is a critical omission for a paper whose key novelty is graph-based structure. Without this comparison, the advantage of SURE's specific design choices over PROGRAM cannot be assessed.

**Δ Gain column in Table 3 appears incorrect.** The table claims SURE achieves a +7.12% Δ Gain over the CLIP-ViT-B baseline (61.20%), but 66.23 − 61.20 = 5.03%, not 7.12%. Other entries are consistent: TPT = 62.44 − 61.20 = +1.24% ✓, ZERO = 66.10 − 61.20 = +4.90% ✓, SURE = 66.23 − 61.20 = +5.03 ≠ +7.12%. The origin of this figure is unclear and needs correction.

**Discrepancy in Table 3 vs. Table 1.** MTA is reported at 63.16% in Table 3 but 64.06% in Table 1 (both using ViT-B, same benchmark). If Table 3 uses a different subset or ordering, this should be stated explicitly.

**Visualization (§5.4) is limited.** The graph evolution is visualized for only 5 classes in a hand-crafted "micro-universe" under "simulated" distribution shift. This does not verify that the mechanism works correctly in the full 1000-class ImageNet setting. An actual measurement of reliability scores and edge weights across the real test run (e.g., tracking R_j for known confusable class pairs) would be more convincing.

**Online ordering effects.** For streaming TTA methods, test sample ordering can significantly affect results. The paper does not discuss the assumed test stream ordering (random? class-sorted? corrupted sequentially?) or whether sensitivity to ordering was measured. This affects reproducibility and fairness of comparisons.

**Ablation study (Table 4)** is valuable and clearly presented. The trajectory ProtoOnly → +Graph w/o Rel → +Graph + Rel → Full is informative. However:
- The gain from "+Graph w/o Rel" is marginal (+0.24% avg) and even slightly negative on ImageNet-A (−0.24%), supporting the authors' own admission that "graph smoothing alone may even hurt performance." This makes the case for the full SURE particularly dependent on the reliability component—yet the reliability component is the most heuristic part of the design.
- The ablation does not vary the logit combination weight in Eq. 10, leaving the 50/50 split unjustified.

---

### Writing & Clarity

The method section is generally clear. However, Algorithm 1 is split across two non-contiguous blocks (lines 1–3 in one location, lines 13–17 in another), presumably a PDF parsing artifact, but the structure makes the algorithm harder to follow. The paper would benefit from more explicit discussion of how edge self-loops (W_{jj} = 0) interact with the logit propagation—specifically, when all incoming neighbors have low reliability, what does Eq. 9 return?

---

### Limitations & Broader Impact

The paper provides no limitations section. Absent discussion includes:
- **Non-stationary distributions**: If class priors shift mid-stream (e.g., in the cross-dataset setting), reliability statistics accumulated from earlier classes may mislead. No mechanism resets or decays stale statistics.
- **Class imbalance**: If some classes appear very rarely in the test stream, their reliability statistics remain at initialization ($\mu_j = 1, \sigma_j = 0$), keeping their reliability artificially high. Rare-class edges will be governed by the uninformative initial state.
- **Scalability at very large C**: The $C \times C$ adjacency matrix scales quadratically. For 1000-class ImageNet this is manageable, but for datasets with tens of thousands of classes, storage and computation would be prohibitive.
- **Broader societal impact**: Not discussed.

---

### Overall Assessment

SURE presents a reasonable and well-motivated framework for test-time adaptation of VLMs, combining semantic graph structure with temporal reliability estimation. The closed-loop design and gradient-free nature are genuine practical advantages. However, the submission has several significant weaknesses that must be addressed before it meets ICLR's standards. First and most critically, the comparison with PROGRAM—the most closely related prior work—is entirely absent from all experiments, making the empirical novelty of SURE's graph design unverifiable. Second, the claimed "consistent outperformance" of prior methods is inaccurate: SURE loses to multiple baselines on multiple individual datasets (ImageNet-A, ImageNet-R, EuroSAT) and wins only on aggregate averages, with margins that approach or fall within the method's own variance. Third, the Δ Gain entry for SURE in Table 3 appears to contain a numerical error (+7.12% vs. the calculable +5.03%). Fourth, core design choices—the 50/50 logit blending, the $\sigma_{\max} = 0.5$ constant, the unspecified logarithm base in $k$, and the cold-start behavior—are insufficiently justified or ablated. The contribution is incremental rather than transformational relative to the TDA/BCA/DPE/ZERO family of methods, and the absent PROGRAM comparison leaves a significant gap in the positioning. In its current form, the paper is below the ICLR acceptance bar.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces SURE, a test-time adaptation (TTA) framework for Vision-Language Models (VLMs) that addresses error propagation in pseudo-labeling through a dynamically evolving Prototype-Reliability Graph (PRG). The method jointly leverages semantic affinity from text prototypes and temporal stability of prediction confidence to regularize logits via graph-based propagation. Extensive experiments demonstrate that SURE achieves state-of-the-art performance across multiple distribution shift benchmarks and backbones while maintaining competitive inference efficiency.

### Strengths
1.  **Innovative Integration of Semantic Structure and Uncertainty:** The core contribution of modeling class-level reliability within a graph structure ($W = S \odot R_{joint}$) is a well-motivated advancement over existing TTA methods that treat classes independently or rely solely on instance-level entropy. The ablation studies (Table 4) confirm that removing the reliability component significantly degrades OOD robustness ($+$Rel adds +1.07% average), validating the specific design choice.
2.  **Strong Empirical Validation and Efficiency:** The method achieves State-of-the-Art (SOTA) results across diverse benchmarks (Table 1 and 2), notably outperforming gradient-based methods like TPT and prototype-based methods like DPE. Crucially, the efficiency analysis (Table 3) shows SURE runs at 0.067s/sample, offering a >10x speedup over TPT (0.706s/sample) with superior accuracy. This addresses a critical need for deployment-ready TTA.
3.  **Comprehensive Calibration Analysis:** Beyond accuracy, the paper provides a thorough analysis of Expected Calibration Error (ECE) in Appendix A.6. It demonstrates that SURE avoids the common TTA pitfall of overconfidence (reducing ECE from 11.23% for ProtoOnly to 7.48%), suggesting the semantic regularization acts as a necessary constraint on confidence drift.

### Weaknesses
1.  **Computational Scalability for Large Label Spaces:** While the paper reports efficient inference (0.067s), the construction of the semantic similarity matrix $\mathbf{S}$ scales with $O(C^2)$ in label space (Eq. 3). For VLMs with large vocabularies (e.g., Open-Vocabulary models with $C > 10,000$), recomputing $\mathbf{S}$ at every step or updating it based on evolving prototypes $\mathbf{t}_i$ (Eq. 2) may introduce non-trivial overhead not fully quantified. The paper implies $\mathbf{S}$ might be static from text, but the "closed-loop" evolution of prototypes suggests $\mathbf{S}$ should theoretically track prototype shifts, which is computationally expensive.
2.  **Dependency on Text Semantics Validity:** The method relies heavily on the assumption that textual prototype similarity ($\mathbf{S}$) accurately reflects semantic affinity under distribution shift. If the distribution shift involves a semantic misalignment (e.g., a new class definition that visually resembles an existing text prototype but belongs to a different concept), the graph may propagate false confidence. The paper uses standard ImageNet labels; results on datasets with distinct semantic shifts where text priors are weak are not explicitly discussed.
3.  **Sensitivity to Graph Sparsity ($k$):** The paper reports optimal $k$ varies between 4 (natural shifts) and 3 (cross-dataset) (Fig 3). While variations are "smooth," the tight optimal range suggests the method requires careful tuning of neighbor density relative to the class count $C$ ($k = 3 \cdot \log(C)$). It would strengthen the contribution to show robustness of the log-scaling rule versus manual tuning in more extreme domain shifts.

### Novelty & Significance
**Novelty:** The proposal of a "Reliability-weighted" adjacency matrix in the context of VLM TTA is novel. While graph-based TTA exists (e.g., PROGRAM), SURE differentiates itself by conditioning edge weights on temporal stability metrics (reliability $R_j$) rather than just feature distance or static text similarity. This specific formulation of "semantic uncertainty regularization" is distinct from standard entropy minimization or prototype shifting.

**Significance:** TTA for foundation models like CLIP is a high-priority area for making AI systems robust in real-world deployment. Providing a solution that improves accuracy *and* calibration while remaining computationally efficient (faster than TPT) is highly significant. The closed-loop framework offers a generalizable principle for TTA that could extend to other foundation model architectures beyond CLIP.

### Suggestions for Improvement
1.  **Clarify Computational Complexity:** Explicitly state whether the semantic similarity matrix $\mathbf{S}$ is computed offline (using initial text prompts) or dynamically (using updated prototypes). If dynamic, provide a theoretical or empirical scaling analysis regarding the class count $C$ to assure readers of scalability to large-scale open-vocabulary tasks.
2.  **Analyze Failure Cases via Text-Visual Drift:** Include a case study or ablation where the semantic prior provided by text is arguably incorrect or weak under the specific domain shift (e.g., cross-lingual adaptation or novel concepts). This would demonstrate the method's resilience (or lack thereof) when the foundational assumption of text-based semantic structure is challenged.
3.  **Detail Hyperparameter Stability:** Provide a table or plot showing performance variance across different class sets (e.g., 100 classes vs. 1000 classes) to validate if the $k = 3 \cdot \log(C)$ heuristic holds consistently, or if it requires re-tuning for different benchmark label spaces.

---
**Reviewer Note:** I have treated formatting artifacts from the PDF parser (e.g., broken equation syntax in the text) as extraction errors and have evaluated the paper based on the apparent mathematical and conceptual content. The logical flow of the method remains clear despite these artifacts.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Warm-up/Initialization Sensitivity:** The claim of initializing prototypes with $N_i^{proto} = 30000$ contradicts the online TTA setting if interpreted as sample count, or freezes adaptation if interpreted as a virtual counter. Provide a learning curve showing accuracy vs. number of processed samples to prove adaptation actually occurs within a realistic test stream length.
2. **Stream Order Dependency:** TTA methods must be robust to data permutation. Report performance variance across multiple random shuffles of the test stream to verify the reliability statistics are not biased by specific sample ordering.
3. **Class Imbalance Robustness:** The reliability score $R_j$ relies on frequency-dependent statistics. Evaluate performance on long-tailed test distributions to ensure rare classes are not suppressed by the graph regularization.
4. **Label Space Scalability:** The graph construction is $O(C^2)$. Demonstrate runtime and memory scaling on datasets with $C > 1000$ (e.g., ImageNet-21k subset) to validate the "generalizable" claim for open-vocab settings.
5. **Initialization Ablation:** Vary the $N_i^{proto}$ initialization constant from 0 to 30000. If performance collapses without this large prior, the method relies on strong regularization rather than the proposed graph mechanism.

### Deeper Analysis Needed (top 3-5 only)
1. **Uncertainty Metric Validation:** Correlate the proposed reliability score ($R_j$) with actual class-wise error rates. Without this, the claim that the graph suppresses "uncertain" classes is an unverified assumption.
2. **Noise Accumulation Tracking:** Quantify the pseudo-label noise rate over time compared to baselines. This is necessary to substantiate the core claim of "preventing error amplification."
3. **Prototype Drift Quantification:** Measure the distance between adapted and source prototypes over time. Verify if "semantic drift" is actually mitigated or merely slowed down by the heavy initialization counter.
4. **Graph Necessity Isolation:** Compare against a simplified baseline that uses reliability weighting without graph propagation. This isolates whether the graph topology adds value over simple confidence filtering.
5. **Compute-Benefit Trade-off:** Analyze the marginal accuracy gain per unit of additional latency introduced by the graph updates. The efficiency claim is weak without comparing the graph overhead to the baseline adaptation cost.

### Visualizations & Case Studies
1. **Adaptation Learning Curve:** Plot accuracy over the stream index (0 to T) to reveal if the method requires a large burn-in period before outperforming CLIP.
2. **Reliability Calibration Plot:** Show predicted reliability vs. actual accuracy per class to expose whether the uncertainty estimation is well-calibrated.
3. **Prototype Trajectory:** Visualize 2D projections of prototype movement during adaptation to confirm they move towards target clusters rather than drifting randomly.
4. **Confusion Matrix Shift:** Compare confusion matrices before and after adaptation to verify if semantic errors (e.g., Cat vs. Dog) are specifically reduced.
5. **Edge Suppression Heatmap:** Visualize the adjacency matrix weights for specific confused class pairs to demonstrate the graph actively suppressing erroneous semantic links.

### Obvious Next Steps
1. **Clarify Initialization Mechanism:** Explicitly state whether the $N_i^{proto} = 30000$ implies pre-collected data or a virtual prior, as the current phrasing undermines the "source-free" claim.
2. **True Online Evaluation:** Remove the large initialization constant and test pure online performance to prove the method works without a warm-up phase.
3. **Memory Footprint Profiling:** Report peak memory usage for the graph cache and reliability buffers during streaming to assess deployment feasibility on edge devices.
4. **Simpler Baseline Comparison:** Include a non-graph reliability weighting baseline to ensure the graph complexity is justified by performance gains.
5. **Extended Backbone Evaluation:** Evaluate on larger VLMs (e.g., ViT-L/14) to ensure gains are not artifact of the smaller ViT-B/16 capacity.

# Final Consolidated Review
## Summary

SURE introduces a test-time adaptation framework for vision-language models that constructs a dynamically evolving Prototype-Reliability Graph (PRG) to regularize predictions. The PRG jointly captures semantic affinity from textual prototypes and class-wise reliability from temporal confidence statistics, enabling graph-based logit propagation that emphasizes semantically related and statistically stable neighbors. Experiments across natural distribution shifts and cross-dataset benchmarks demonstrate improved average accuracy over prior methods with competitive efficiency.

## Strengths

- **Novel integration of semantic structure with reliability-aware weighting:** The core contribution—combining semantic similarity S with class-wise reliability scores R_j to form a modulated adjacency matrix—is a well-motivated advancement over prior TTA methods that treat classes independently or rely solely on instance-level confidence. The ablation study (Table 4) confirms that removing the reliability component degrades OOD accuracy by +1.24%, validating that this specific design choice is not redundant.

- **Efficient inference with competitive accuracy:** SURE achieves state-of-the-art average accuracy (66.23% on ViT-B) while running at 0.067s/sample—more than 10× faster than TPT (0.706s/sample). This addresses a practical deployment constraint for test-time adaptation methods.

- **Calibration improvement demonstrated:** The appendix includes Expected Calibration Error analysis showing SURE reduces ECE from 11.23% (ProtoOnly baseline) to 7.48%, indicating the graph regularization mitigates the overconfidence typically induced by adaptation dynamics.

## Weaknesses

- **Missing comparison with the most closely related prior work:** The paper explicitly discusses PROGRAM (Sun et al., 2024) in Related Work (§2), claiming two key differences: reliability-driven topology versus static graph, and VLM-specific design versus uni-modal focus. Yet PROGRAM is absent from all experimental comparisons. For a paper whose central novelty is graph-based structure for TTA, the absence of this direct comparison leaves the claimed advantages unverifiable.

- **"Consistently outperforms" claim is inaccurate:** The abstract states the method "consistently outperforms prior methods," but the data shows losses on individual benchmarks: SURE loses to BCA on ImageNet-A (29.57% vs. 30.35%), loses to ZERO and TDA on ImageNet-R (79.96% vs. 80.75%), and loses to TDA on EuroSAT by 4.4 percentage points (53.60% vs. 58.00%). SURE wins on average but not consistently across individual datasets.

- **Numerical errors in Table 3:** The Δ Gain column reports SURE achieves +7.12% improvement over CLIP-ViT-B (61.20%), but the actual gain is 66.23 − 61.20 = 5.03%. This discrepancy needs correction. Additionally, MTA is reported at 63.16% in Table 3 but 64.06% in Table 1—both using ViT-B on natural distribution shifts—creating ambiguity about which figure is correct.

- **Cold-start vulnerability:** The paper initializes reliability scores with μ_j = 1.0 and σ_j = 0.0, yielding R_j = 1.0 for all classes. This means the reliability guard is entirely inactive during early adaptation—precisely when noisy pseudo-labels are most prevalent. The paper does not discuss this design choice or its implications.

- **Unspecified logarithm base for k:** The neighbor count is defined as k = 3·log(C) without specifying the logarithm base, affecting reproducibility. For C = 1000, this yields k ≈ 21 (natural log) or k = 30 (base-10 log).

- **No limitations section:** The paper does not discuss failure modes such as non-stationary distributions, class imbalance (rare classes retain artificially high R_j), or scalability to very large label spaces where the C × C adjacency matrix becomes prohibitive.

## Nice-to-Haves

- **Correlation between reliability score R_j and actual class-wise error rates:** Verifying that the proposed uncertainty metric corresponds to ground-truth noise would strengthen the claim that the graph "suppresses unreliable associations."

- **Analysis of stream ordering sensitivity:** Test-time adaptation methods can be sensitive to sample ordering; reporting variance across shuffled test streams would improve confidence in robustness.

- **Simplified baseline comparison:** An ablation testing reliability weighting without graph propagation would isolate whether the graph topology itself adds value beyond confidence-based filtering.

## Removed Points

These points are flagged to be removed, treat them with caution:
- *Harsh critic's claim that "VLMs particularly vulnerable to distribution shift" is uncited*: This is an intuition stated in the introduction, not a core empirical claim. It contextualizes the problem without affecting the method's validity.
- *Harsh critic's assertion that reliability score is "misleadingly framed" as entropy*: The paper states R_j "reflects an information-theoretic intuition"—this is softer language than claiming equivalence. The heuristic works empirically; the framing issue is minor.
- *Spark finder's concern that N_i^proto = 30000 "contradicts online TTA"*: The paper clarifies this is a virtual counter initialization (following Zhou et al., 2025), not actual pre-collected samples. The mechanism is standard practice for stabilizing prototype updates.
- *Balanced reviewer's concern about O(C²) scalability*: Valid in principle, but for the benchmarks tested (C ≤ 1000), this is manageable. It becomes relevant only for open-vocabulary settings outside the paper's scope.
- *Harsh critic's concern that prototype drift is "vanishingly small"*: The paper shows empirical improvements, so the graph propagation mechanism (not just prototype updates) contributes to gains. The relative contribution could be analyzed further but does not invalidate the method.

## Novel Insights

The visualization in Figure 4 reveals a hierarchy of trust that emerges during adaptation: reliable classes (e.g., "Carton") maintain high diagonal weights, hard fine-grained classes (e.g., "Tiger Cat") are preserved but modulated, while noisy classes (e.g., "Television") are actively suppressed. This demonstrates that SURE learns a soft, statistically grounded topology rather than applying uniform thresholding—suggesting the reliability mechanism successfully distinguishes inherent difficulty from semantic noise.

## Suggestions

- Add PROGRAM to the experimental comparisons to directly validate the claimed advantages of reliability-driven topology and VLM-specific design.
- Correct the Δ Gain value in Table 3 (should be +5.03%, not +7.12%) and resolve the MTA accuracy discrepancy between tables.
- Discuss cold-start behavior explicitly and consider alternative initialization strategies (e.g., lower initial reliability scores) that provide earlier noise suppression.
- Add a limitations paragraph addressing: (i) sensitivity to test stream ordering, (ii) behavior under class imbalance, and (iii) scalability considerations for very large label spaces.
- Clarify the logarithm base for k = 3·log(C) in the method description and hyperparameter specification.

# Actual Human Scores
Individual reviewer scores: [4.0, 4.0, 6.0, 4.0, 4.0]
Average score: 4.4
Binary outcome: Reject
