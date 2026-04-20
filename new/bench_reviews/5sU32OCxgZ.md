## Summary

The paper proposes TTVD, a geometric framework for test-time adaptation that formalizes neighbor-based TTA as Voronoi Diagram partitioning of feature space. It progressively extends this foundation to Cluster-induced Voronoi Diagrams (CIVD, leveraging rotation augmentation for multi-site influence) and Cluster-induced Power Diagrams (CIPD, for boundary-based noisy sample filtering). The method achieves state-of-the-art error rates and improved calibration (ECE) consistently across four standard benchmarks (CIFAR-10-C, CIFAR-100-C, ImageNet-C, ImageNet-R) under the standardized TTAB protocol. The geometric reframing of neighbor-based TTA is an interesting lens, and the ablation study cleanly isolates each component's contribution.

## Strengths

- **Novel geometric formalization of neighbor-based TTA.** Section 3.1 cleanly derives the distance-based loss (Eq. 3) as inducing a Voronoi partition over feature space, connecting prototype-based classification to computational geometry. This provides a principled vocabulary for understanding and extending neighbor-based TTA, rather than treating it as an ad-hoc heuristic.

- **Consistent SOTA results under standardized protocol.** Table 1 shows TTVD achieving the lowest error across all four datasets (CIFAR-10-C: 20.5%, CIFAR-100-C: 49.1%, ImageNet-C: 59.8%, ImageNet-R: 67.5%) and substantially lower ECE (reductions of 3.4–4.3%). All comparisons use the TTAB toolkit with grid-searched hyperparameters, which demonstrates strong commitment to fair evaluation.

- **Clean ablation isolating component contributions.** Table 2 demonstrates progressive improvement: VD alone (28.4% avg), CIVD (22.7%, −5.7%), and CIPD (20.5%, −2.2%). Table 4 further shows remarkable robustness to class mean computation precision (59.8–59.9% across 10%→5%→1% of ImageNet), suggesting practical scalability.

- **Elegant connection between geometry and confidence calibration.** The Power Diagram-based filtering mechanism (Section 3.3, Figure 2) provides a geometric alternative to entropy-based sample filtering, motivated by the observation that entropy landscapes fail to capture all decision-boundary-adjacent noisy samples.

## Weaknesses

### Fatal
None.

### Major

- **Circular dependency in the PD-based sample filtering mechanism.** Section 3.3 proposes subtracting Power Diagram cells from Voronoi cells to identify noisy samples near decision boundaries, excluding them from adaptation. However, in unsupervised TTA, cell assignment is determined entirely by the model's own pseudo-labels via Eq. 6 — the same predictions the filter is meant to vet. This creates a circular dependency: the geometric filter uses the model's current (potentially corrupted) predictions to decide which predictions are unreliable. Samples that are confidently but incorrectly classified near geometric boundaries will be retained, while informative hard samples will be discarded. Without an external or label-agnostic criterion for boundary proximity, the geometric subtraction reduces to a form of confidence-based weighting. This undermines the claim (Section 3.3) that PD filtering captures "unstable gradients" beyond what entropy-based methods achieve. The mechanism may still be useful empirically (as Table 2's +2.2% suggests), but the geometric robustness claim is overstated.

- **No computational overhead or latency analysis.** The paper claims suitability for "real-time adaptation during testing" (line 81) but reports no compute time, memory, or per-batch latency for TTVD. CIVD requires computing distances to 4 augmented sites per class — for ImageNet's 1000 classes, this means 4000 distance calculations per sample per batch. The paper only mentions that precomputing class means from 10% of ImageNet took "less than 10 minutes" (line 253), which is an offline cost. The online inference-time and adaptation-time overhead is a critical missing piece for evaluating a method claiming real-time TTA suitability, especially relative to methods like TENT that only update a few BN affine parameters.

### Minor

- **Unverified claim that CIVD's joint influence eliminates negative transfer.** Section 3.2 states that "the joint label \(\tilde{y}_k^{(\alpha)}\) avoids the negative transfer since the objective is now unified" (line 140). However, no empirical analysis is provided to substantiate this — no gradient cosine similarity, gradient norm comparison, or ablation showing that the joint CIVD objective does not produce conflicting update directions between the rotation-augmented and entropy-like components. The claim borrows from the cited Gandelsman et al. (2022) but does not demonstrate that the specific CIVD formulation actually resolves the gradient conflict issue.

- **Adaptation curves (Figure 4) suggest potential optimization asymmetry.** TTVD's error continuously descends over 750 online batches on static corrupted test sets, while TENT and SAR plateau. The paper attributes this to "robustness and resilience against overfitting," but continuous descent under fixed test distributions could equally indicate a more aggressive or differently scheduled learning-rate configuration for TTVD relative to baselines. While the paper states baselines were grid-searched, the stark divergence in trajectories would benefit from a controlled comparison with matched learning-rate schedules or an analysis confirming the descent is not overfitting to the test stream.

- **Baseline performance discrepancy warrants scrutiny.** Table 1 reports TENT at 24.0% error on CIFAR-10-C and 62.7% on ImageNet-C. Widely cited benchmarks using ResNet backbones typically report TENT errors of ~15–18% on CIFAR-10-C and ~55–60% on ImageNet-C. While the paper attributes its comparisons to the TTAB protocol, a 6–10% gap across baselines makes the 1–3% relative improvements claimed for TTVD difficult to contextualize. Clarifying whether these baselines match published peak performance under the specific TTAB configuration would strengthen the results.

### Trivial

- **Adaptation curves in Figure 4 lack variance bands.** TTA performance is highly sensitive to batch ordering and random seed; single-run curves for TTVD, TENT, and SAR provide no confidence about the statistical significance of the observed performance trajectories.

- **Table 4 reports identical results without variance.** The entries 59.8%, 59.8%, 59.9% across three data proportions (10% → 5% → 1%) are numerically identical to one decimal place. Without variance reporting, these cannot distinguish genuine insensitivity from measurement noise or a single seed.

## Nice-to-Haves

- A t-SNE/UMAP visualization of TTVD's adaptation trajectory in ResNet feature space would strengthen the 2D MNIST intuition (Figure 1) for high-dimensional settings (\(\mathbb{R}^{2048}\) ResNet-50 features).
- Analysis of sample retention rates under PD filtering across corruption severities (how many samples are discarded per batch, and whether discarded samples correlate with actual misclassification) would clarify the practical behavior of the geometric filter.
- Including AdaNPC directly in Table 1 (rather than the separate Table 3) would make the main comparison table more complete, since AdaNPC is the most directly comparable neighbor-based method.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Contradiction between static Voronoi sites and dynamic feature updates" (Harsh Critic, Point 1):** While the paper does pre-compute \(\mu_k\) and then updates the feature extractor via \(\sigma_{t+1} = \sigma_t - \lambda \nabla \mathcal{L}_{VD}\), this is not a contradiction. The method's goal is precisely to move features toward or away from the fixed anchor sites — that's what adaptation does. This is standard practice in all prototype-based TTA methods (T3A, SHOT, AdaNPC) where class means are fixed and features are updated. The harsh critic's claim that "the geometric alignment claim collapses" misreads the paper's intent: the alignment is the optimization objective, not a static property that must be preserved.

- **"CIVD unifies self-supervision and entropy minimization without empirical verification" (Harsh Critic, Section Note):** The paper *does* provide empirical verification through the ablation in Table 2: CIVD achieves 22.7% vs. VD alone at 28.4%, a 5.7% average improvement. The claim is supported by results even if the mechanism (gradient cosine analysis) is not probed.

- **"2D MNIST boundary figure does not scale to \(\mathbb{R}^{2048}\)" (Harsh Critic, Visualization suggestion):** The 2D visualization in Figure 1 is explicitly intended as an intuition builder, not a literal representation of high-dimensional behavior. The paper does not claim the VD boundaries look the same in \(\mathbb{R}^{2048}\); it uses the 2D case for illustrative purposes. This is standard practice.

- **"Missing appendix, missing proofs, absent references" and similar parser-stripped content concerns:** The parser strips these sections; they exist in the original submission.

- **Pure nitpicks about notation inconsistencies, figure formatting, and minor presentation details are removed per guidelines.**

## Novel Insights

The most valuable conceptual contribution of this paper is not the algorithm itself (which builds on well-established neighbor-based TTA) but the systematic geometric vocabulary it introduces. By recasting neighbor-based TTA as Voronoi partitioning, the paper opens a path toward principled extensions (CIVD, CIPD) rather than heuristic tuning. The robustness of TTVD to class mean precision (Table 4) is a particularly interesting finding: if 1% of ImageNet suffices to compute Voronoi sites without performance degradation, this suggests that the geometric prior is relatively invariant to prototype estimation quality, which has practical implications for large-scale deployment. However, the gap between the elegant geometric theory and its practical behavior (circular filtering, no compute analysis) limits the paper's current impact.

## Suggestions

1. **Report per-batch inference and adaptation time (wall-clock) for TTVD, CIVD, and CIPD compared to TENT/SAR** to demonstrate the method's practical feasibility for real-time TTA. This is the most actionable and impactful missing analysis.
2. **Provide gradient analysis for the CIVD joint objective** — at minimum, report gradient cosine similarity or gradient norm distributions between the rotation-augmented and entropy-like components to verify the "negative transfer" claim.
3. **Clarify whether baselines match their published peak performance** under TTAB's protocol. If the TTAB configuration is genuinely different from standard benchmarks, explain the protocol differences (e.g., batch size, ordering, corruption severity) to contextualize the 24.0% vs. ~15–18% TENT discrepancy.
4. **Add standard deviations or multiple-seed results** to the main table (Table 1) and adaptation curves (Figure 4) to establish whether the 1–3% improvements are statistically significant.

## Score and Decision

**Calibration against anchors:**
- Compared to **BmG88rONaU (TCR, scores 8,8,8,6, Spotlight Accept)**: TTVD has similarly strong empirical results and a novel framing, but TCR addresses a genuinely new problem domain (cross-modal retrieval TTA) with no prior TTA methods, while TTVD reframes an existing approach (neighbor-based TTA). This places TTVD below.
- Compared to **bdHjLCcMSP (NGTTA, scores 6,5,6,5, Reject)**: TTVD has cleaner experiments, more standard benchmarks, and a better-defined ablation story, making it stronger.
- Compared to **eXrUdcxfCw (EMA prototype CT-TTA, scores 3,5,5,6,5, Reject)**: TTVD has substantially larger improvements and a more systematic framework, making it clearly stronger.
- Compared to **75PhjtbBdr (TTA method, scores 8,6,6,5, Accept)** and **sEMJ1PLSZR (AEA, scores 8,6,8,3, Accept)**: These papers have similarly strong results but fewer methodological concerns.

The paper sits between the 5–6 and 7 ranges. It has genuine contributions (geometric reframing, consistent SOTA, clean ablation) but also real weaknesses (circular filtering, no compute analysis, unverified negative-transfer claim). The center of its anchor cluster places it at **5.5** — borderline Reject. The empirical strength is notable but不足以 overcome the methodological gaps without additional analysis in rebuttal.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>