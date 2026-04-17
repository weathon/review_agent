# An Efficient Structural Pruning for Spiking Neural Networks by Balancing Accuracy and Sparsification

- Decision: Reject
- Scores: 0, 6, 6, 6

## Abstract
The increasing scale of spiking neural networks (SNNs) poses significant challenges for deployment on resource-constrained neuromorphic hardware, necessitating lightweight and learnable structural solutions. Interestingly, biological neural systems employ an efficient organizational strategy—hierarchical structural reorganization around functional clusters, where new connections grow
orthogonally to existing ones to expand representational capacity. Inspired by this mechanism, we propose a dynamic pruning and regrowth framework with channel-level orthogonality for SNNs (DPRC-SNNs) to enable scalable and efficient structural learning for SNNs. DPRC-SNNs introduce the spiking column subset selection mechanism for SNNs, which integrates channel-level pruning with orthogonality-driven regrowth, selectively restoring diverse and complementary channels to minimize information loss from aggressive pruning. Through iteratively pruning redundant channels and regrowing orthogonal ones, DPRC-SNNs preserve functional diversity while enhancing sparsity at the channel level. Extensive evaluations on CIFAR10, DVS-Gesture, and DVS-CIFAR10 demonstrate that DPRC-SNNs achieve high compression rates and computational efficiency without compromising accuracy, showing strong potential for neuromorphic deployment.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper presents DPRC-SNNs, a Dynamic Pruning and Regrowth with Channel-level Orthogonality framework for spiking neural networks. Motivated by the hierarchical structural reorganization observed in biological neural circuits, the method enables adaptive structural learning by iteratively pruning redundant channels and regrowing new, orthogonal ones.

### Strengths
Structured and hardware-friendly sparsity

### Weaknesses
1. The overall pruning–regrowth paradigm has been explored in several works. The main innovation here—orthogonality-driven regrowth—while conceptually appealing, may require stronger empirical justification to establish substantial novelty.
2. This paper does not specify how orthogonality between channels is measured or enforced.
3. This paper should benchmark against modern event-based or structured sparsity approaches.
4. Although the method is motivated by neuromorphic deployment, the results are limited to simulation-level metrics (accuracy, FLOPs). Energy measurements or mapping to actual chips (e.g., Loihi 2, Tianjic) would substantiate the hardware efficiency claim.

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents DPRC-SNNs, a novel structural learning framework for Spiking Neural Networks (SNNs) that unifies dynamic channel pruning and orthogonality-driven regrowth to balance accuracy and sparsification. Unlike prior weight-level or static pruning strategies, the proposed spiking column subset selection mechanism enables channel-level optimization guided by temporal spiking dynamics, while the orthogonality-based regrowth restores diverse and complementary channels to preserve functional representation. The framework dynamically reorganizes network topology during training, achieving compact yet expressive SNN architectures. Extensive experiments on three benchmark datasets consistently demonstrate that DPRC-SNNs achieve substantial reductions in both parameters and computational cost while maintaining competitive accuracy. This highlights the model's strong potential for efficient neuromorphic deployment and scalable event-driven learning.

### Strengths
1. The paper proposes DPRC-SNNs, a novel structural learning framework that integrates channel-level pruning with orthogonality-driven regrowth, inspired by biological neural reorganization. By introducing the spiking column subset selection (SCSS) mechanism, it effectively captures temporal dependencies in SNN pruning, bridging the gap between fine-grained sparse optimization and hardware-efficient structured pruning.
2. The method is technically sound, with well-founded formulations for channel importance and orthogonality-based regrowth. Its dynamic pruning–regrowth strategy ensures stable optimization, and extensive experiments on both static and neuromorphic datasets validate its robustness and effectiveness.
3. The paper presents a coherent progression from biological motivation to algorithmic design and validation. The neuroscience-computation link is well-supported, with figures effectively illustrating channel evolution and pruning. The polished 
presentation is accessible to both neuromorphic and machine learning audiences.

### Weaknesses
1. The ablation study could more clearly isolate the effects of SCSS and orthogonality-based regrowth.
2. The paper provides limited analysis of computational overhead.
3. The experimental validation lacks support from larger datasets, which would strengthen the empirical evidence for the proposed approach.
4. The article's layout exceeds the margins.

### Questions
1. Can this pruning approach be extended to other mainstream neural network frameworks? If not, what do you consider to be the primary limitations or bottlenecks that may hinder its generalization?
2. The visualizations of pruning and regrowth dynamics are insightful but somewhat difficult to interpret. Would the authors consider improving the figure annotations or providing simplified visual aids to enhance clarity?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Inspired by the brain's ability to reorganize around functional clusters, this paper proposes a Dynamic Pruning and Regrowth framework with Channel-level orthogonality for SNNs (DPRC-SNNs). The goal is to enable scalable and efficient structural learning. The core innovation is a "spiking column subset selection mechanism" that iteratively performs pruning and regrowth.

### Strengths
​​The inspiration from hierarchical structural reorganization in the brain is a compelling and well-articulated foundation for the work.

The integration of dynamic pruning with an orthogonality-driven regrowth mechanism is a novel and interesting approach to structural learning in SNNs.

The paper is generally well-written and easy to follow.

### Weaknesses
The discussion on structural/unstructured learning in SNNs is lacking. A clear comparison with existing unstructured sparsity methods for SNNs is needed to position this work's unique contribution (structural/channel-level pruning) and its advantages/disadvantages.

The absence of results on a large-scale dataset like ImageNet is a significant gap. It is difficult to assess the scalability and true effectiveness of the method without this standard benchmark.

The purpose of comparing against TET (a method focused on improving accuracy, not sparsity) is confusing. Comparisons should primarily be against other state-of-the-art sparsity-inducing methods.

It is unclear if the proposed dynamic pruning/regrowth framework is specific to SNNs or a general technique. Showing its performance on standard CNNs would help demonstrate the generality and robustness of the concept.

The paragraph preceding the list of contributions largely repeats the items that follow. This section should be shortened and made more concise.

### Questions
How does DPRC-SNNs compare quantitatively against state-of-the-art methods that reduce unstructured sparsity in SNNs, particularly in terms of the trade-off between accuracy, sparsity level, and training/inference cost?

Can the DPRC framework be directly applied to standard CNNs (Artificial Neural Networks not SNNs)? If so, what are the results on a benchmark like ImageNet? If not, what aspects are specific to the dynamics of spiking neurons?

What is the specific rationale for including TET, a non-sparsity method, in the comparisons? Is the goal to show that DPRC-SNNs can alsoachieve high accuracy? This should be clarified in the text.

Have you conducted any experiments on ImageNet-scale datasets? If not, do you anticipate any challenges in scaling the dynamic pruning and regrowth process to such large networks?

What is the computational overhead of the iterative pruning-and-regrowth process during training compared to a standard one-shot pruning pipeline?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes DPRC-SNNs, a dynamic channel-level pruning and regrowth framework for Spiking Neural Networks inspired by biological neural reorganization mechanisms. The method introduces a spiking column subset selection mechanism that integrates channel-level pruning with orthogonality-driven regrowth to preserve functional diversity while enhancing sparsity. The experimental results on CIFAR and DVS datasets demonstrate competitive performance with significant parameter reduction.

### Strengths
The paper shows strong innovation in bridging biological principles with efficient SNN compression, and the orthogonality-based regrowth strategy represents a novel contribution to hardware-friendly neural network optimization.

### Weaknesses
1.The SCSS formulation would benefit from a clearer comparison to related methods in matrix approximation or subspace selection.

2.The evaluation scope is narrow; testing on more architectures or datasets could better establish generality.

### Questions
question 1: How does the adaptive sparsity mechanism, driven by batch normalization and spike activity, influence training stability and convergence?

question 2: The proposed method is only evaluated on convolutional architectures (ResNet, VGG). Can the proposed framework scale effectively to larger or more complex datasets?

### Soundness
4

### Presentation
3

### Contribution
4
