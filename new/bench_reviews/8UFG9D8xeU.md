## Summary

This paper proposes DPA-OMF, a post-training preference alignment method for multi-agent motion generation that ranks a model's own rollouts against expert demonstrations using an optimal transport-based occupancy measure matching distance. The approach eliminates the need for human preference annotations while improving the realism of generated traffic simulations, with thorough empirical analysis of scaling laws and over-optimization phenomena.

## Strengths

- **Practical zero-annotation pipeline with clear motivation**: The method directly addresses the scalability bottleneck of human preference annotation in complex multi-agent settings (100+ agents), as argued in Sections 1 and 3.2. Table 3 shows ranking among model's own generations (0.739 composite realism, 0.84 classification accuracy) substantially outperforms adversarial AFD (0.720 realism, 0.52 accuracy), validating the core framing.
  
- **OT-based preference distance outperforms ADE for alignment quality**: Figure 3 empirically demonstrates that ADE saturates once simulations reach reasonable realism, while the OT-based preference distance maintains a stronger correlation with WOSAC realism across a wider range.

- **Transparent empirical analysis of scaling and over-optimization**: Figure 7 (left-right) systematically maps the relationship between preference data volume, KL divergence drift, and realism. The finding that small preference datasets initially degrade performance due to over-optimization is practically valuable for practitioners applying direct alignment to continuous control, extending Goodhart's law observations to this domain.

- **Honest feature ablation reveals multi-objective insight**: Table 2 shows collision-only features improve realism (0.724 vs. 0.721 baseline), while progress-only or comfort-only features degrade it (0.710, 0.705). This validates the need for comprehensive feature sets and provides concrete guidance for practitioners.

- **Contrastive alignment vs. SFT distinction validated**: Figure 4 shows DPA-OMF increases preferred rollout likelihood while decreasing unpreferred, whereas SFT-bestOA increases both. This explains why DPA-OMF achieves 0.739 realism vs. SFT-bestOA's 0.723 despite using the same top-ranked rollouts (Figure 6).

## Weaknesses

### Fatal
— None identified. No errors that completely invalidate the paper's core claims.

### Major

- **Evaluation metric contamination via shared features creates circularity concern**: Section 4 explicitly states: "These features are also used to encode the agent's state in the realism metric"—the five features [collision status, distance to road boundary, minimum clearance, control effort, speed] with weights [10, 5, 2, 1, 1] serve dual roles as both the OT preference distance ground and the WOSAC realism metric components. While the paper correctly distinguishes that preference distance measures alignment to expert demonstration while realism measures likelihood of expert given rollouts, optimizing the model on these exact features systematically biases it toward the evaluation metric. The primary headline improvement (Table 1: 0.739 vs. 0.721 baseline) could partially be an artifact of optimizing the same feature space that evaluation measures, rather than demonstrating genuine emergent behavioral alignment. Decoing training features from evaluation features would strengthen confidence in the generalizability of results.

- **OT-based distance ignores temporal ordering, undermining applicability to sequential behaviors**: Equation 2 allows arbitrary coupling between rollout timesteps $t$ and expert timesteps $t'$, computing optimal transport over marginal state distributions rather than enforcing temporal correspondence. For driving behaviors where causal ordering matters critically (e.g., yielding before versus during an intersection), this means the distance can reward sequences visiting the same states in different, potentially unsafe temporal order, provided feature distributions align. Figure 2 shows diagonal structure in coupling matrices for well-aligned cases, but this is an empirical observation rather than enforced by the method. The distance metric may not reliably capture behavioral alignment for dynamic multi-agent systems.

### Minor

- **No variance or error bars reported across multiple training seeds**: All tables (Tables 1-3) and figures report point estimates only without standard deviations, confidence intervals, or results across multiple random seeds. Given contrastive preference alignment's known sensitivity to initialization and data sampling, single-run results provide insufficient evidence for statistical significance of small improvements (e.g., the 0.018 gap between DPA-OMF and baseline in Table 1).

- **Missing learned reward model baseline**: The paper does not report a comparison against training a simple reward model (e.g., MLP scoring the same features) on auto-generated preference pairs, then applying standard DPO. Without this baseline, it is difficult to isolate whether gains come from OT machinery versus the contribution of using DPO on synthetic pairs with a direct learned reward scorer.

## Trivial

- **Framing of "implicit preference" may be misleading**: The paper abstract and introduction market the approach as learning from "implicit preferences encoded in demonstrations," but the preference signal is actually an explicit hand-crafted distance function with manually tuned scalar weights. The claim that the method derives preference from demonstrations rather than designer priors is supported by using demonstrations as reference, but the feature engineering and weighting represent substantial human prior knowledge about driving quality.

## Nice-to-Haves

- Provide side-by-side visualizations where OT distance is low but behavior is temporally misaligned (e.g., late yielding, phantom braking out of sequence) to empirically validate whether temporal correspondence matters in practice.
- Include discussion of why SFT-bestOA fails to suppress unpreferred log-likelihood as effectively as DPA-OMF (Figure 4), perhaps with brief theoretical or geometric analysis of how the contrastive loss regularizes the policy distribution compared to supervised fine-tuning.
- Report additional nuPlan benchmarks (Val14, Test14-Hard) for more comprehensive evaluation.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **Related work dismissal of AI feedback as incompatible**: The harsh critic flagged the paper's dismissal of synthetic AI feedback due to incompatible input modalities. While broadly framed, the paper's context is specific to multi-agent traffic simulation with 100+ agents, where unified foundation models may indeed be absent. This is scope-appropriate.

- **α and γ hyperparameters unexplained in Equation 4**: The paper references a multi-agent extension of contrastive preference learning (Hejna et al., 2023), and these parameters come from that framework. Their treatment as inherited from prior work makes this a reasonable assumption rather than a novel omission.

- **ADE as weak baseline**: ADE is a standard baseline in trajectory generation evaluation. The paper's use of it is appropriate given its prevalence in the autonomous driving literature.

- **Scaling/over-optimization analysis adds little novelty**: While DPO over-optimization has been studied in LLMs (Tang et al., 2024; Rafailov et al., 2024a), this paper's domain-specific finding that small preference datasets actively degrade performance provides practical insights for this specific domain. The observation that scaling mitigates over-optimization extends existing findings to multi-agent motion generation.

## Novel Insights

None beyond the paper's own contributions.

## Suggestions

1. **Decouple preference features from evaluation features**: Run the primary experiment using a disjoint set of features for ranking (e.g., purely kinematic or occupancy features) while evaluating on the full WOSAC metric suite. This would directly address the most significant methodological concern and strengthen confidence that improvements generalize beyond optimizing the evaluation proxy.

2. **Add error bars or multiple seed results**: Report results averaged across 3-5 random seeds with standard deviations for all tables. This would validate whether the observed improvements are statistically significant or within experimental noise.

3. **Include learned reward model baseline**: Train a simple reward model on the same auto-generated preference pairs using identical features, then apply standard DPO. This would clarify whether the OT component provides genuine value or if the gains come primarily from the DPO loop with any reasonable reward signal.

## Score and Decision

I compared this paper against several calibration anchors:
- **High-scoring (8+)**: VCbqXtS5YY (10,6,8,5, Spotlight) had both strong theoretical framework and comprehensive empirical validation without circular evaluation concerns. wM2sfVgMDH (8,8,8,6, Oral) had extensive baselines across multiple benchmarks.
- **Medium (6-7)**: xJbsmB8UMx (6,6,6,8, Poster) had a clear idea but was missing some key comparisons. This paper falls roughly in this tier empirically.
- **Borderline/Reject**: mjtCqmujYP (5,3,6,6,6, Reject) had solid empirical results but lacked theoretical backing and missing baselines. The paper under review is similar in being empirical but has the additional circular feature concern.

The paper demonstrates genuine empirical value with a well-motivated zero-annotation pipeline but is weighed down by the circular evaluation feature overlap and temporal ordering concerns with OT. These are substantive methodological weaknesses that cannot be fully resolved in rebuttal.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>