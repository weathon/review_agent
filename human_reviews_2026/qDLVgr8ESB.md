# Cannistraci-Hebb Training on Ultra-Sparse Spiking Neural Networks

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Inspired by the brain's spike-based computation, spiking neural networks (SNNs) inherently possess temporal activation sparsity. However, when it comes to the sparse training of SNNs in the structural connection domain, existing methods fail to achieve ultra-sparse network structures without significant performance loss, thereby hindering progress in energy-efficient neuromorphic computing. This limitation presents a critical challenge: how to achieve high levels of structural connection sparsity while maintaining performance comparable to fully connected networks. To address this challenge, we propose the Cannistraci-Hebb Spiking Neural Network (CH-SNN), a novel and generalizable dynamic sparse training framework for SNNs consisting of four stages. First, we propose a sparse spike correlated topological initialization (SSCTI) method to initialize a sparse network based on node correlations. Second, temporal activation sparsity and structural connection sparsity are integrated via a proposed sparse spike weight initialization (SSWI) method. Third, a hybrid link removal score (LRS) is applied to prune redundant weights and inactive neurons, improving information flow. Finally, the CH3-L3 network automaton framework inspired by Cannistraci-Hebb learning theory is incorporated to perform link prediction for potential synaptic regrowth. These mechanisms enable CH-SNN to achieve sparsification across all linear layers. We have conducted extensive experiments on six datasets including CIFAR-10 and CIFAR-100, evaluating various network architectures such as spiking convolutional neural networks and Spikformer. The proposed method achieves a maximum sparsity of 97.75% and outperforms the fully connected (FC) network by 0.16% in accuracy. Furthermore, we apply CH-SNN within an SNN training algorithm deployed on an edge neuromorphic processor. The experimental results demonstrate that, compared to the FC baseline without CH-SNN, the sparse CH-SNN architecture achieves up to 98.84% sparsity, an accuracy improvement of 2.27%, and a 97.5$\times$ reduction in synaptic operations, and the energy consumption is reduced by an average of 55$\times$ across four datasets. Our code is available at https://github.com/HuaGuaiGuai/CH-SNN.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces CH-SNN (Cannistraci-Hebb Spiking Neural Network), which is a four-stage dynamic sparse training framework for ultra-sparse spiking neural networks (SNNs). Extensive experiments on six datasets show CH-SNN achieves performance comparable to FC networks even at ultra-high levels of sparsity

### Strengths
Originality
(1) Integrates Cannistraci-Hebb theory, originally from complex network science, into SNN sparse training.
(2) Introduces two novel initialization schemes (SSCTI, SSWI) specifically designed for spike-based learning.

Quality
(1) Extensive experiments across six datasets show robustness and generalizability.
(2) Includes ablation, sensitivity, and hardware efficiency analyses, showing methodological thoroughness.

Clarity
The paper is clearly structured, with each stage of CH-SNN well explained. The biological and theoretical motivations are well linked to the computational framework.

### Weaknesses
(1) Insufficient analysis of temporal dynamics: 
The paper emphasizes structural sparsity but offers limited insight into the temporal spike dynamics.
(2) Clarity of comparison fairness:
It is not entirely clear whether all baseline methods were reimplemented under identical experimental conditions. The paper does not specify whether these results were reproduced using a unified experimental setup or directly taken from prior publications, which affects the transparency and comparability of the reported performance gains.

### Questions
(1) Temporal sparsity–accuracy trade-off: Have the authors analyzed the effect of increasing temporal sparsity on latency or robustness?
(2) Reproducibility: Please clarify whether all baseline methods were reimplemented under identical training conditions or if their results were directly adopted from previous studies.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents CH-SNN, a novel four-stage dynamic sparse training framework for spiking neural networks (SNNs) that achieves ultra-high structural sparsity while maintaining/improving accuracy compared to baselines. Extensive experiments on six datasets and three architectures demonstrate consistent performance and energy advantages.

### Strengths
1. Strong novelty and cross-disciplinary contribution: bridges network science (Cannistraci-Hebb theory) with neuromorphic learning, introducing a biologically and topologically inspired sparse training approach.
2. Comprehensive experimental validation: covers multiple datasets, architectures.
3. Clear modular structure: the four-stage design (SSCTI, SSWI, LRS, CH3-L3) is intuitive and extensible to other SNNs, which provides a reasonable baseline for SNN training.

### Weaknesses
1. Theoretical insufficiency: The paper lacks formal analysis of the convergence and stability of the CH3-L3 regrowth dynamics.
2. Scalability questions: Experiments are limited to medium-scale datasets. The framework’s behaviour on larger datasets (e.g., ImageNet or DVS-CIFAR100) remains untested.
3. Biological claim ambiguity: The connection to Hebbian principles is mostly conceptual; empirical neuroscientific grounding is minimal.

### Questions
1. Could the authors provide a theoretical argument or empirical evidence that the CH3-L3 regrowth mechanism guarantees stability or avoids redundant regrowth loops?
2. How sensitive is CH-SNN to the hyperparameters controlling sparsity ratio, pruning frequency, and regrowth sampling distribution?
3. The timestep of SNN is 8 according to this paper, could the authors explained the performance on different timesteps?
4. Is the CH3-L3 topological regrowth biologically interpretable in terms of synaptic rewiring or STDP-like plasticity?

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
2

### Summary
This paper introduces Cannistraci-Hebb Spiking Neural Network (CH-SNN), a dynamic sparse training framework for ultra-sparse SNNs. Extensive experiments are conducted on six datasets, demonstrating that CH-SNN achieves high sparsity while retaining or improving accuracy over fully connected baselines.

### Strengths
The paper presents an approach to ultra-sparse SNN training, integrating initialization, pruning methods, and Cannistraci-Hebb-inspired topological regrowth. 
Experiments were conducted on multiple datasets, and thorough ablation experiments and sensitivity analyses were carried out. 
The framework is implemented on a hardware-friendly algorithm S-TP, achieving significant gains in energy efficiency.

### Weaknesses
Some key areas of the mathematical description, particularly around pruning and regrowth, lack sufficient clarity.

### Questions
Can the authors clarify the multinomial sampling procedure in the pruning step, how exactly the link removal score (LRS) is converted into actual pruning decisions?
How accurate is the SSWI initialization method under varying degrees of input temporal sparsity, structural connection sparsity,and spike threshod？

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes CH-SNN, a dynamic sparse training framework for SNNs. The method sparsifies all linear layers in SNNs through four stages: (1) sparse topology initialization (SSCTI), (2) sparse weight initialization (SSWI), (3) hybrid pruning based on link removal scores (LRS), and (4) link regrowth using CH3-L3. Experiments shows that CH-SNN achieves ultra-high structural sparsity while maintaining accuracy. The authors also report significant energy savings when deploying CH-SNN on hardware-friendly algorithm S-TP.

### Strengths
1. CH-SNN achieves ultra-high sparsity (>90% on some datasets) without performance degradation.

2. The four-stage framework is well-structured and includes ablation studies showing the necessity of SSCTI and SSWI for stable training under extreme sparsity.

### Weaknesses
1. All experiments are conducted on relatively simple tasks using very shallow networks. The lack of evaluation on more complex datasets and deeper SNNs raises serious doubts about scalability. For example, the extremely high sparsity achieved on MNIST is likely attributable to the simplicity of the task, whereas the sparsity drops significantly on CIFAR. It can be inferred that on more challenging benchmarks like ImageNet, the claimed “ultra-sparse” may not be achievable. In contrast, works like SRigL (cited in Section 2.1) have been validated on ResNet-scale models. If the authors can provide relevant evidence, I am willing to increase my rating accordingly.

2. The Spikformer results lack meaningful comparison. First, no other sparse training method is evaluated on Spikformer, making any performance comparison meaningless. Second, Transformers are typically designed for large-scale datasets, applying them to MNIST-level tasks offers little insight. Moreover, the paper never specifies the depth or width of the Spikformer used, making it impossible to assess the result’s significance.

3. The paper repeatedly highlights marginal gains (e.g., outperforms FC network by 0.16% on MNIST) as evidence of superiority. However, MNIST is nearly saturated (~99% accuracy), and such a gain is statistically negligible. This risks overstating the method’s effectiveness.

### Questions
1. Section 3.2.1 states that for intermediate layers (e.g., after conv or attention), SSCTI is inapplicable and replaced by uniform random initialization. Does this mean the core topological initialization method is effectively limited to the first layer? How can CH-SNN ensure stable convergence or meaningful structure learning in deeper networks where feature correlations are nontrivial?

2. On CIFAR-100, sparse models report large improvements over the baseline (Table 1). Can the authors explain why there are such large improvements? Intuitively, such a large gap strongly suggests the baseline FC models may not have been sufficiently tuned.

3. Table 1 shows Grad R achieves 91.95% accuracy on DVS-Gesture with an accuracy improvement of +7.83%. However, CH-SNN reports 95.45% accuracy with only +0.38% improvement. Is there a reporting error in the accuracy or the improvement values?

### Soundness
2

### Presentation
3

### Contribution
2
