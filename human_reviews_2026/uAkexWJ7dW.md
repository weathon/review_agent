# Robust Selective Activation with Randomized Temporal K-Winner-Take-All in Spiking Neural Networks for Continual Learning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
The human brain exhibits remarkable efficiency in processing sequential information, a capability deeply rooted in the temporal selectivity and stochastic competition of neuronal activation. Current continual learning in spiking neural networks (SNNs) faces a critical challenge: balancing task-specific selectivity with adaptive resource allocation and enhancing the robustness with perturbations to mitigate catastrophic forgetting. Considering the intrinsic temporal dynamics of spiking neurons instead of traditional K-winner-take-all (K-WTA) based on firing rate, we explore how to leave networks robust to temporal perturbations in SNNs on lifelong learning tasks.
In this paper, we propose Randomized Temporal K-winner-take-all (RTK-WTA) SNNs for lifelong learning, a biologically grounded approach that integrates trace-dependent neuronal activation with probabilistic top-k selection. By dynamically prioritizing neurons based on their spatiotemporal relevance, RTK-WTA SNNs emulate the brain’s ability to modulate neural resources in spatial and temporal dimensions while introducing controlled randomness to prevent overlapping task representations. The proposed RTK-WTA SNNs enhance inter-class margins and robustness through expanded feature space utilization theoretically. The experimental results show that RTK-WTA surpasses deterministic K-WTA by 3.07–5.0\% accuracy on splitMNIST and splitCIFAR100 with elastic weight consolidation. Controlled stochasticity balances temporal coherence and adaptability, offering a scalable framework for lifelong learning in neuromorphic systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Randomized Temporal K-winner-take-all (RTK-WTA), a continual learning (CL) framework for Spiking Neural Networks (SNNs). The core idea is to integrate temporally accumulated neuronal traces with probabilistic top-k selection. Experimental results show that RTK-WTA outperforms state-of-the-art SNN continual learning methods across various tasks.

### Strengths
1. This paper provides detailed theoretical analyses of the effectiveness of the proposed RTK-WTA. It shows that the proposed RTK-WTA increases the effective feature space and improves the robustness of learning dynamics.
2. RTK-WTA outperforms state-of-the-art SNN continual learning methods, especially on the splitCIFAR100 benchmark.

### Weaknesses
1. As detailed in Fig. 1 and Sec. A.7, feature embeddings are extracted by a pre-trained ANN backbone (ResNet-18, ViT), with the SNN component reduced to only the final classification head. This design obscures the true role and benefit of the SNN and the RTK-WTA mechanism. The most challenging part of the task (hierarchical feature extraction from raw data) is offloaded to a static, pre-trained ANN. It is unclear why the authors did not employ an end-to-end SNN feature extractor or at least a pre-trained SNN backbone.
2. The background information in the main text is insufficient. The main text has only a limited background introduction, while the related work section is in the appendix. I recommend placing the related work section in the main text.

### Questions
1. Please explain the key role that the SNN classification head plays in the continual learning tasks and the reason why not employ an end-to-end SNN feature extractor or at least a pre-trained SNN backbone.
2. Please add more background introduction in the main text.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces a new biologically inspired mechanism called Randomized Temporal K-Winner-Take-All (RTK-WTA) to improve continual learning in spiking neural networks (SNNs). TK-WTA combines: the traces of recent spiking activity, and randomized neuron selection to dynamically choose neurons in a way that emulates biological neural selectivity and variability. Results show RTK-WTA achieves better performance on continue learning tasks than baseline methods.

### Strengths
1. This paper attempts to connect the proposed RTK-WTA mechanism to biological processes of neural selectivity and stochastic activation, which strengthens the connection between neuroscience and neuromorphic computing and enhances its conceptual depth and interdisciplinary relevance.

2. The proposed RTK-WTA method is novel and biologically grounded. Instead of relying on static, deterministic top-K firing rules, the model integrates temporal spike traces with controlled randomness, reflecting how biological neurons combine time-dependent excitation with stochastic competitive firing. This design allows the network to leverage temporal coherence in spiking activity while avoiding rigid specialization, which is a thoughtful and meaningful step toward more brain-like continual-learning systems.

3. The authors evaluate the proposed method on multiple standard continual learning benchmarks (SplitMNIST, SplitCIFAR-10, SplitCIFAR-100), demonstrating consistent improvements over baselines. The results convincingly show that RTK-WTA improves both performance and robustness.

### Weaknesses
1. The paper may lack a clear and structured background section to help readers unfamiliar with continual learning understand the fundamental concepts, challenges, and motivation behind this work. Additionally, the figures lack detailed, self-contained captions, making it difficult to interpret their meaning without referring back to the main text.

2. The discussion of prior literature, particularly in the context of continual learning within SNNs, is relatively insufficient. Important prior works are mentioned but not critically analyzed, making it difficult to discern how the proposed method advances beyond existing approaches or fills specific research gaps. As a result, the novelty and significance of the contribution are not made very explicit.

### Questions
1. The paper claims that RTK-WTA expands the spatiotemporal feature space, yet the underlying theoretical reasoning is only briefly mentioned. Could the authors provide a more rigorous theoretical explanation and empirical validation (e.g., through visualization or quantitative analysis) to support this claim?

2. How does RTK-WTA concretely enhance representational diversity within the network? It would be helpful if the authors could include additional experiments or visualizations to demonstrate this effect.

3. To better isolate and understand the contribution of each component, could the authors perform ablation studies comparing (a) deterministic vs. probabilistic selection, and (b) models with and without trace-based modulation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes a biologically inspired Randomized Temporal K-Winner-Take-All (RTK-WTA) mechanism for spiking neural networks (SNNs) aimed at continual learning. The method introduces controlled stochasticity in neuron selection by combining temporal traces with probabilistic top-k activation, improving adaptability and robustness against noise. Experiments on SplitMNIST, SplitCIFAR-10/100, and Tiny-ImageNet demonstrate enhanced performance and resilience compared to deterministic K-WTA baselines.

### Strengths
1.The proposed RTK-WTA mechanism is both innovative and simple to implement, integrating biological plausibility with computational efficiency.

2.The method exhibits strong noise robustness, as shown in Table 2 and additional Tiny-ImageNet results, maintaining high accuracy under perturbations.

3.The approach effectively balances sparse activation, adaptability, and robustness, offering a promising direction for scalable continual learning in neuromorphic systems.

### Weaknesses
1.In Table 2, key results demonstrating robustness could be bolded or highlighted to emphasize the superiority of the proposed method.

2.Further validation on larger or more complex datasets (e.g., Tiny-ImageNet or ImageNet-scale continual tasks) would strengthen the claim of generality and scalability.

### Questions
1.Can the proposed RTK-WTA mechanism be extended to deeper or hierarchical SNN architectures? How would temporal randomness interact with multi-layer trace dynamics?

2.Have the authors considered testing on larger datasets to further evaluate scalability and generalization performance?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses two core challenges in CL for SNNs: (1)catastrophic forgetting caused by overlapping task activations, and (2) insufficient robustness to input perturbations. 
Drawing inspiration from the brain’s temporal selectivity and stochastic neuronal competition, it proposes the Randomized Temporal K-Winner-Take-All（RTK-WTA） mechanism. RTK-WTA mechanism integrates temporally accumulated neuronal traces with probabilistic top-k selection to dynamically allocate neural resources while introducing controlled randomness.

### Strengths
(1) Originality: It integrates temporal neuronal traces with probabilistic top-k selection for SNN continual learning, differing from traditional rate-based or static trace-based K-WTA, and aligns well with biological neural mechanisms;
(2) Quality: The motivation is reasonable; Theoretical derivations strengthen the work’s rigor. But the experimental design is not comprehensive enough;
(3) Clarity: The writing of this paper is relative clear. For example, this paper has clear definitions of core concepts and step-by-step explanations of the RTK-WTA mechanism;
(4) Significance: RTK-WTA provides a new idea and practical solution for robust continual learning in SNNs.

### Weaknesses
(1) The experimental scenario of continuous learning is limited: The experiment only adopted the "task splitting" continuous learning paradigm, and did not verify more challenging scenarios.
(2) NEURONAL SELECTIVELY ACTIVATION ANALYSIS(3.4) is too simple and lack the depth.
(3) Although experiments identify α=0.1 as the optimal value, there is no quantitative analysis of how α correlates with task characteristics (e.g., task overlap, data dimensionality), making it hard to guide parameter tuning for new tasks;

### Questions
(1) Have you considered more complex CL scenarios (such as class-incremental or domain-incremental), and how RTK-WTA performs in these scenarios?
(2) Could you provide a more systematic analysis of the relationship between the randomness coefficient α and task characteristics, to guide parameter tuning?
(3)Can the neuronal selectivity analysis be extended to better support the claims of improved robustness and diversity?

### Soundness
3

### Presentation
3

### Contribution
3
