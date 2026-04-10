=== CALIBRATION EXAMPLE 10 ===

# Final Consolidated Review
## Summary
This paper identifies a theoretical limitation of HiResCAM: its attention maps are not uniquely determined for a given prediction due to an arbitrary spurious shift. The authors propose ContrastiveCAM, which is invariant to this shift and provides granular class-versus-class explanations. Leveraging this, they introduce Core-Focused Cross-Entropy (CFCE), a novel loss function designed to suppress model reliance on non-core (spurious) image regions and explicitly encourage feature alignment.

## Strengths
- **Theoretical Contribution:** Provides a clear theoretical analysis and proof of a non-uniqueness problem in HiResCAM (Theorem 3.2) and a principled solution via ContrastiveCAM with a formal invariance guarantee (Theorem 3.5). This is a concrete advance in understanding CAM-based interpretability methods.
- **Novel Methodology:** Proposes a novel way to integrate corrected interpretability maps directly into the training objective (CFCE) to discourage reliance on non-core regions. The theoretical link to a constrained risk minimization objective (Theorem 4.6) is sound and well-motivated.
- **Comprehensive Empirical Scope:** Evaluates the method extensively across multiple challenging datasets (Hard-ImageNet, Oxford-IIIT Pets, PASCAL VOC) and tasks (multiclass, binary, multilabel classification, and downstream segmentation). The evaluation properly focuses on alignment metrics (RFS, IoU with core masks) beyond standard accuracy.
- **Practical Demonstrations:** Shows the method remains effective with weaker forms of supervision, such as auto-generated masks from Segment Anything (SAM) or simple bounding boxes, increasing its potential applicability.

## Weaknesses
### Major:
- **Potential Mathematical Issue in Core Loss Formulation:** There is an apparent inconsistency between the stated Core-Focused Cross-Entropy loss (Eq. 15, \(L_{\text{CFCE}} = \log\left( \sum_c \exp\left( -[H \odot \text{CAM} + (1-H) \odot |\text{CAM}|] \right) \right)\)) and its decomposition in the proof (Appendix A, Eq. 56, which separates terms into \(\exp(-H \odot \text{CAM})\) and \(\exp((1-H) \odot |\text{CAM}|)\)). This raises concerns about whether the loss as written correctly implements the intended suppression of non-core contributions. If the formulation is incorrect, it undermines the central methodological contribution.
- **Ablation of Architectural Modifications is Missing:** The proposed method requires specific architectural changes to the final layers of ResNet (removed final downsampling, bias, BatchNorm, ReLU) to maintain theoretical guarantees. The paper does not isolate the effect of these changes from the effect of the novel CFCE loss. Consequently, it is unclear how much of the observed performance change is attributable to the loss versus the altered architecture.
- **Performance Trade-off Lacks Deep Analysis:** Models trained with CFCE show a significant drop in standard (unablated) accuracy on Hard-ImageNet (~90.5% vs. ~94.3% for cross-entropy) in exchange for improved alignment. The paper acknowledges this trade-off but does not analyze its source. Is the drop due to discarding predictive but spurious signals (desirable), or is it harming the learning of legitimate core features (problematic)? A per-class or error-case analysis is needed.
- **Limited Comparison to State-of-the-Art:** The empirical comparisons are mostly against older baselines (CORM, DFR). To properly establish significance, the method should be compared against more recent state-of-the-art techniques for combating spurious correlations and improving feature alignment (e.g., GroupDRO, JTT, or other contemporary saliency-guided methods).

### Minor
- **Evaluation Metric Consistency Could Be Clearer:** In Table 2, alignment (IoU) for baselines is reported using GradCAM, while for the proposed method it is reported using both GradCAM and ContrastiveCAM. While done for "consistency with baselines," a fairer comparison would report ContrastiveCAM IoU for all methods or better justify why GradCAM is a sufficient common metric. This slightly weakens the evidence for improved alignment.
- **Dependence on Some Form of Region Annotation:** While the paper effectively shows the method works with approximate masks, it still requires some form of region specification (mask or bounding box) during training. This limits fully unsupervised application. However, the demonstrations with SAM and bounding boxes substantially mitigate this concern, making it a minor limitation.

### Trivial
- **Visualization of Real HiResCAM Failure Cases:** The paper uses a constructed example (Figure 1) to illustrate the theoretical spurious shift. Including a real example from the test set where HiResCAM yields a misleading explanation due to this shift would strengthen the motivational narrative but is not essential to the core claims.

## Nice-to-Haves
- **Extension to Vision Transformers (ViTs):** Demonstrating the applicability of ContrastiveCAM and CFCE to dominant architectures like ViTs would significantly strengthen the paper's generality and impact.
- **Deeper Failure Case Analysis for ContrastiveCAM:** A quantitative analysis of when ContrastiveCAM explanations might fail or become unreliable (e.g., on low-contrast or textured images) would provide a more complete understanding of its limitations.
- **Sensitivity Analysis to Mask Quality:** A controlled study showing how performance degrades with increasingly noisy or coarse masks would be valuable for practitioners assessing the method's robustness.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Strengths Removed:** "The paper is well-written" and "The topic is important" – These are generic and apply to many papers.
- **Weaknesses Removed:**
    1.  **"The loss function encourages large |CAM| in non-core regions"** – This specific claim about the sign of the effect is contingent on the potential mathematical inconsistency noted in the Major weaknesses. It is kept there as a verification issue, not as a separate removed point.
    2.  **"Reproducibility concerns about hyperparameters or implementation details"** – The paper provides training details and states code/data are published (though redacted). Hyperparameter details are sufficient by community standards.
    3.  **"Request for confidence intervals on all results"** – The paper provides standard deviations for key results (e.g., Table 2, Oxford Pets table). Demanding intervals for all large-scale benchmarks is not standard practice.
    4.  **"Criticism that cited models/datasets (e.g., SAM, Hard-ImageNet) do not exist or are unavailable"** – Per the hard rules, all cited entities are assumed to exist and be available.

## Suggestions
1.  **Clarify and Correct the Loss Formulation:** The authors must rigorously check and, if necessary, correct Eq. 15 to ensure it matches the intended behavior and the derivation in the proof (Theorem 4.6, Appendix A). The current ambiguity is a serious concern.
2.  **Perform an Ablation Study:** Conduct controlled experiments to disentangle the effects of the architectural modifications from the CFCE loss. A simple baseline of "CE w/ Arch" is not enough; an ablation where components are added incrementally is needed.
3.  **Analyze the Accuracy Trade-off:** Provide a deeper investigation into the source of the standard accuracy drop. Analyze misclassified cases to determine if the model is failing on core features or correctly ignoring spurious ones.
4.  **Benchmark Against Newer Methods:** Include comparisons with 2-3 recent state-of-the-art methods for feature alignment or shortcut removal to clearly position the improvement offered by CFCE.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 0.0]
Average score: 1.3
Binary outcome: Reject
