## Summary
This paper identifies that alternating training methods for multimodal learning fail to prevent classifier bias toward faster-converging modalities, perpetuating modality imbalance. To address this, the authors propose Classifier-Constrained Alternating Training (CCAT), a two-stage framework that pre-trains an unbiased shared classifier with contribution-aware regularization, freezes it as a stable decision anchor, and employs modality-specific LoRA adapters during alternating training. A sample-level re-optimization mechanism further targets severely imbalanced instances.

## Strengths
- **Well-motivated and targeted problem analysis**: The paper clearly identifies a critical, underexplored flaw in existing alternating training methods—persistent classifier bias—supported by empirical tracking of modality contributions (Figure 1). The connection drawn to class imbalance provides a coherent conceptual lens for the approach.
- **Comprehensive and convincing empirical validation**: The method achieves consistent and substantial accuracy gains across three diverse benchmarks (e.g., +6.76% on Kinetic-Sound). Ablation studies (Table 2) clearly demonstrate the value of each component (classifier freezing, alternating training, secondary updates, LoRA), and feature visualization (Figure 5) provides supporting evidence for improved discriminability.
- **Practical and reproducible design**: The integration of a frozen classifier with lightweight LoRA adapters is a clever and implementable solution to the distribution mismatch problem. The training pipeline is clearly detailed (Algorithm 1), and hyperparameter searches (Table 3, Figure 4) are documented, aiding reproducibility.

## Weaknesses
- **Incomplete comparison with relevant state-of-the-art baselines**: The main results table (Table 1) omits direct comparisons with key recent methods explicitly mentioned in the text (MLA, MMPareto, LFM) and other sample-level imbalance methods (e.g., SMSL). This omission makes it difficult to conclusively assess the claimed superiority and situate the contribution within the current landscape.
- **Theoretical section is informal and overstated**: Section 3.1 presents a valuable intuitive analogy between class and modality imbalance but frames it as a "unified theoretical framework" and "proof." The gradient analysis is heuristic, introducing fusion coefficients (γ) that are not part of the standard gradient derivation, and it lacks the rigor (e.g., formal assumptions, bounds) expected for a theoretical contribution. This section should be reframed to avoid overclaiming.
- **Limited analysis of secondary update mechanism and unimodal trade-offs**: The paper does not analyze how many samples are selected for secondary updates, how their contribution scores evolve, or whether this step genuinely rectifies imbalance on those samples. Furthermore, ablation results (Table 2, Kinetic-Sound) show that the full CCAT can sometimes yield lower unimodal accuracy for a modality than an ablated variant, a trade-off that is not discussed.

## Nice-to-Haves
- **Computational cost analysis**: A discussion of the training time and parameter overhead introduced by the two-stage process and LoRA modules compared to standard end-to-end or alternating training baselines would be informative.
- **Experiments on larger-scale or trimodal datasets**: While the three benchmarks are appropriate, testing on a larger dataset (e.g., AudioSet) or a trimodal task would strengthen claims about generalizability and scalability, as noted in the future work.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Weakness: "The mutual information estimator (Eq. 5) is unconventional... justification is lacking."** The paper cites prior work (Zhou et al., 2025b) for this metric. While its properties could be discussed further, its use is not a fundamental flaw.
- **Weakness: "Inference procedure... not specified."** The paper states: "These are fused at the decision level for final output." Common practice is averaging, and the exact method does not impact the core contribution.
- **Weakness: "Missing statistical significance tests."** While reporting variance is good practice, the paper follows the common norm in the field by reporting average accuracy over three seeds for these benchmarks. This is not a required standard for rejection.
- **Weakness: "Evaluated only on classification tasks."** The paper's scope is clearly modality imbalance for multimodal classification. Demanding evaluation on other tasks is scope creep.
- **Weakness/Strength: Generic statements about writing quality or topic importance.** These have been filtered out.

## Novel Insights
The core novel insight is the identification and mitigation of *classifier entrenchment bias* as a fundamental failure mode of modality-alternating training. While alternating updates decouple encoders, the classifier can become structurally biased toward early-dominant modalities, suppressing later learning from weaker ones. The paper's key innovation is treating this as analogous to decision boundary bias in class-imbalanced learning and applying a remedy—freezing a pre-regularized classifier as a stable anchor—within the multimodal setting. The integration of LoRA adapters to handle the unimodal/fused feature distribution mismatch while preserving this anchor is a clever and practical implementation of this insight.

## Suggestions
- **Include missing baselines in the main results table**: Add rows for MLA, MMPareto, LFM, and a recent sample-level method (e.g., SMSL or its variant) to Table 1 to provide a complete and fair comparison.
- **Reframe Section 3.1 as an analogy/motivation**: Revise the section title and text to present the gradient dynamics discussion as an insightful motivating analogy rather than a formal theoretical proof, to avoid overstatement.
- **Add analysis for the sample-level re-optimization**: Include a brief analysis tracking, for a subset of epochs, the number of samples selected for secondary updates and the change in their weak-modality contribution scores or accuracy, to validate the mechanism's operation.