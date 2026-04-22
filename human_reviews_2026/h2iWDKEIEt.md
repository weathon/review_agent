# Probabilistic Robustness Analysis in High Dimensional Space: Application to Semantic Segmentation Networks

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
Semantic segmentation networks (SSNs) are central to safety-critical applications such as medical imaging and autonomous driving, where robustness under uncertainty is essential. However, existing probabilistic verification methods often fail to scale with the complexity and dimensionality of modern segmentation tasks, producing guarantees that are overly conservative and of limited practical value. We propose a probabilistic verification framework that is architecture-agnostic and scalable to high-dimensional input-output space. Our approach employs conformal inference (CI), enhanced by a novel technique that we call the **clipping block**, to provide provable guarantees while mitigating the excessive conservatism of prior methods. Experiments on large-scale segmentation models across CamVid, OCTA-500, Lung Segmentation, and Cityscapes demonstrate that our framework delivers reliable safety guarantees while substantially reducing conservatism compared to state-of-the-art approaches on segmentation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Semantic segmentation networks (SSNs) are important in safety-critical domains such as medical imaging and autonomated driving. Existing probabilistic verification techniques struggle to scale to the high complexity and dimensionality of modern SSNs, often resulting in guarantees that are overly conservative and of limited practical use. The authors introduce a scalable and architecture-independent probabilistic verification framework based on conformal inference (CI). They integrate a novel clipping block mechanism that refines uncertainty calibration while preserving formal guarantees. Experiments on large-scale segmentation benchmarks - including CamVid, OCTA-500, Lung Segmentation, and Cityscapes - show that the approach provides reliable safety assurances and tighter guarantees than state-of-the-art probabilistic verification methods.

### Strengths
- The authors deal with an interesting and important topic.
- Current research in this area is described by the authors.
- It's good that all the preliminaries are explained, which is quite a lot.
- The different used datasets are good.

### Weaknesses
- Right at the beginning, terms such as randomized smoothing and conformal inference are simply assumed to be familiar. A brief explanation and, in particular, a distinction between the two would be helpful for understanding. Not so easy to follow if you have no prior knowledge of CI - everything is derived, but little explanation is given as why it works.
- The preliminaries (section 2) and a large part of section 3 (including algorithms 1 and 2) are taken almost one-to-one from the reference Hashemi et al.
- I would like to see more experiments in comparison to the method used by Hashemi et al., as the approach presented by the authors is very similar to this.
- The computing power required for these experiments is already very high. It would be interesting to compare this with randomized smoothing.
- The Cityscapes results in Appendix D are not available.
- The introduction refers directly to Appendix A, where no notation is provided and is therefore difficult to understand.
- I find that the results are hardly described/explained.

### Questions
Conformal inference with reachset in the form of a convex hull, the surrogate model, and the PCA approach have all already been done in Hashemi et al.'s paper. What is the novelty of the approach presented here?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper is interested in verification of semantic segmentation networks and obtaining coverage guarantees of their outputs. The proposed approach defines a surrogate model by computing the convex hull of outputs obtained from a training set of adversarial data, and then projects new points on the convex hull by solving a linear programming problem. To deal with high dimensional data, the method relies on PCA learned from the training samples to reduce the dimension before computing the convex hull and the projection. The simple surrogate model can then be efficiently verified.

### Strengths
- **Originality:** The method is novel in its direct use of a convex hull of logits to define the surrogate reachset. This is a more direct geometric approach that avoids the need for an additional training step (unlike the ReLU network in Hashemi et al. 2025).

### Weaknesses
- **Clarity:** The paper suffers from a severe lack of clarity in defining its contribution. The method and explanation heavily rely on and closely mirror Hashemi et al. (2025). The only substantive difference appears to be the replacement of an additional learned ReLU surrogate network with the direct convex hull projection. This strong reliance makes it nearly impossible to evaluate the paper's unique contribution and necessitates a much clearer and more explicit discussion of the differences.
- **Significance:** The paper claims improved scalability and efficiency as key motivations, stating the prior work "could not handle our perturbation dimensions." This is directly contradicted by the results presented in Hashemi et al. (2025) on the same dataset and settings, often with faster runtime. This undermines the paper's core claim about the need for a new, more efficient approach.
- **Quality:** The experimental section is limited and not convincing. The absence of a comparison with the results obtained by the method in Hashemi et al. (2025) (or any other method) makes the presented results impossible to contextualize or evaluate.

### Questions
- Could you explicitly clarify the novel technical and conceptual contributions of this paper? Specifically, beyond replacing the ReLU network with the direct convex hull computation, what fundamental differences exist, and why are they necessary for this problem?
- Could you provide a direct, quantitative comparison of the coverage guarantees, runtime, or any other relevant metric, against their results?
- Could the proposed approach be extended to other pixel-wise prediction tasks, e.g. depth estimation?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this work, the authors propose a probabilistic verification framework for semantic segmentation networks (SSNs) using conformal inference enhanced with a novel clipping block. The method addresses limitations of existing work by replacing a trained ReLU surrogate model with a convex hull projection approach. They conduct experiments on 4 large-scale segmentation datasets, namely, CamVid, OCTA-500, Lung segmentation, and Cityscapes and demonstrate scalability to perturbations.

### Strengths
- The clipping block is training-free which is a big advantage compared. Hence it can be used in a plug-and-play manner with any existing model.
- The authors provide extensive formal proofs and guarantees regarding probabilistic coverage.
- Extensive experiments (4 large and popular segmentation datasets, several perturbation dimensions and magnitudes)

### Weaknesses
- The use of PCA and linear programming to project onto a convex hull in high dimensions is computationally expensive
- Authors do not provide any study of sensitivity to N (parameter in PCA)
- Authors should compare to other baselines such as randomized smoothing, hashemi etc, atleast on the toy example
- Authors should provide analysis of how changing confidence levels ($\delta_1, \delta_2$) affects tightness of robustness
- Currently the authors focus on norm perturbations but it would be interesting if authors could provide a discussion on other perturbations like blur, affine transformations, brightness etc
- The authors only use $l \infty$ but it would be interesting to compare $l_1,l_2$ too

### Questions
Please see weakness section

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a new probabilistic verification framework for semantic segmentation networks (SSNs). Existing methods for certified robustness (especially randomized smoothing) struggle to scale to high-dimensional segmentation tasks. The authors propose a scalable, architecture-agnostic approach combining conformal inference (CI) with a novel clipping block surrogate model. The clipping block projects network outputs onto a convex hull formed from training logits, avoiding the need for training a separate surrogate network (as in Hashemi et al., 2025). The approach provides provable $(\epsilon, l, m)$ probabilistic guarantees, supports general $L_p$ perturbations, and reduces conservatism. Experiments on CamVid, OCTA-500, Lung Segmentation, and Cityscapes demonstrate improved scalability and less conservative robustness bounds compared to prior CI and randomized smoothing methods.

### Strengths
- The “clipping block” is an elegant, training-free replacement for surrogate ReLU networks, avoiding fidelity and scalability issues.
- The paper extends CI to large-scale probabilistic reachability with formal $(\epsilon, l, m)$ guarantees.
- Demonstrated on realistic, high-dimensional datasets and large segmentation models (UNet, BiSeNet, HRNetV2).
- The paper systematically contrasts its approach with prior CI and randomized smoothing methods.
- Includes an anonymous toolbox and detailed algorithmic pseudocode.

### Weaknesses
- While multiple datasets are used, the evaluation focuses on a narrow type of perturbation (darkening) and may not reflect broader robustness (e.g., geometric or semantic transformations).
- The empirical section omits comparisons to recent certified robustness methods beyond Hashemi et al. (2025) and smoothing approaches.
- The convex hull projection step is computationally heavy; PCA-based dimensionality reduction mitigates this but may introduce approximation bias.
- The paper’s exposition is dense and overly mathematical in sections 3,4, which may obscure intuition for readers less familiar with CI.
- No ablation on PCA vs. clipping: It’s unclear how much each contributes to scalability and accuracy improvements.

### Questions
- How sensitive is the robustness value (RV) to the choice of calibration size $m$ and the PCA dimensionality $N$?
- Can the convex hull projection scale beyond the datasets tested?
- How would this approach handle distributional shift in test data; does the CI guarantee still hold under covariate drift?
- Could the clipping block approach be combined with randomized smoothing to strengthen guarantees?
- What is the empirical runtime or memory bottleneck for convex hull construction as t and n grow?
- Is it possible to simplify the presentation for the reader, and add additional retails (relevant background etc.) in the Appendix?

### Soundness
2

### Presentation
2

### Contribution
2
