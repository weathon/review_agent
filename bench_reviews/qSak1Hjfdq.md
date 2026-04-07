## Summary
This paper formalizes the All-Day Multi-Scenes Lifelong Vision-and-Language Navigation (AML-VLN) problem, where agents must continually adapt to diverse scenes and environmental conditions (e.g., low-light, scattering) without catastrophic forgetting. It proposes Tucker Adaptation (TuKA), a parameter-efficient method that represents multi-hierarchical navigation knowledge as a high-order tensor decomposed via Tucker decomposition, and introduces a decoupled knowledge incremental learning strategy. The resulting AlldayWalker agent is evaluated on a new benchmark extending Habitat with degraded imaging models, showing consistent improvements over strong baselines.

## Strengths
- **Novel and well-motivated problem formulation:** The AML-VLN task addresses a critical gap in deploying VLN agents in dynamic real-world conditions, and the extension of Habitat with multiple degradation models (low-light, overexposure, scattering) provides a valuable benchmark for lifelong navigation research.
- **Methodological innovation beyond matrix-based adapters:** TuKA leverages Tucker decomposition to explicitly decouple shared, scene-specific, and environment-specific knowledge in a high-order tensor, offering a principled way to capture multi-hierarchical structure that existing LoRA variants cannot represent.
- **Extensive and thorough experimental validation:** The paper compares against 12 state-of-the-art baselines across 24 sequential tasks, demonstrates clear superiority in success rates and forgetting metrics, and includes insightful ablations (e.g., tensor order, shared components, scalability) and real-world deployment.

## Weaknesses
- **Incomplete ablation of the DKIL loss components:** The contribution of each regularization term (EWC for shared parameters, consistency for experts, orthogonality for new experts) is not isolated, making it unclear which mechanisms are essential for mitigating forgetting and enabling knowledge sharing.
- **Lack of analysis on forward transfer and expert interpretability:** While forgetting rates are reported, there is no measurement of whether learning new tasks improves performance on previous ones (forward transfer), a key aspect of lifelong learning. Additionally, the claim that experts decouple scene and environment knowledge lacks empirical validation through probing or clustering of the learned factor matrices.
- **Over-reliance on CLIP matching for inference without robustness analysis:** Expert selection during inference depends on matching CLIP features stored during training, but the accuracy of this matching under severe domain shifts or for novel scenes is not analyzed, leaving the reliability of the selection mechanism in doubt.
- **Limited comparison with broader continual learning strategies:** The baselines are predominantly LoRA-based variants; inclusion of strong non-LoRA continual learning methods (e.g., replay-based approaches) would better situate TuKA’s advantages within the broader lifelong learning landscape.

## Nice-to-Haves
- Sensitivity analysis of key hyperparameters (e.g., Tucker ranks, regularization weights) to assess robustness and reproducibility.
- Qualitative visualizations of agent trajectories across different environments to illustrate successes and failure modes.
- A dedicated limitations section in the main paper discussing fixed expert counts, inference matching assumptions, and real-world generalization.
- Improved readability of dense figures (e.g., Figure 3) and tables through simplification or supplemental visualizations.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Statistical significance from single runs:** In large-scale VLN benchmarks, single-run evaluation is common practice due to computational cost, and the paper follows established norms from cited prior work.
- **Placement of related work in the appendix:** While a succinct related work section in the main text is conventional, its absence does not undermine the technical contribution; this is a formatting preference.
- **Fixed expert count limiting unbounded lifelong learning:** The paper explicitly scopes the problem to known sets of scenes and environments; criticizing the absence of dynamic expansion is scope creep.
- **Hyperparameter sensitivity analysis:** Demanding exhaustive sensitivity analysis is not a standard requirement for methodological papers in this area.
- **Dense result tables:** This is a presentation issue that does not affect the substantive findings.
- **Negative values in forgetting metrics:** These are interesting observations that could be discussed but do not constitute a flaw in the method or evaluation.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Conduct an ablation study isolating each component of the DKIL loss (EWC, consistency, orthogonality) to clarify their individual contributions.
- Analyze forward transfer by measuring performance on earlier tasks after learning new ones, and provide interpretability analysis of the learned expert factors (e.g., via probing tasks).
- Evaluate the accuracy and failure modes of the CLIP-based expert matching mechanism, and consider fallback strategies for handling unseen scene-environment combinations.
- Include comparisons with replay-based continual learning methods to strengthen the baseline evaluation.