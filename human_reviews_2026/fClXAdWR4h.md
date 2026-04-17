# Stochastic Sample Approximations of (Local) Moduli of Continuity

- Decision: Reject
- Scores: 2, 6, 8

## Abstract
Modulus of local continuity is used to evaluate the robustness of neural networks and fairness of their repeated uses in closed-loop models. Here, we revisit a connection between generalized derivatives and moduli of local continuity, and present a non-uniform stochastic sample approximation for moduli of local continuity. This is of importance in studying robustness of neural networks and fairness of their repeated uses.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes non-uniform stochastic sampling methods, inspired by UCB bandit policies, to estimate local moduli of continuity (Lipschitz constants) of neural networks. The authors revisit the connections between generalized derivatives and Lipschitz constants in the o-minimal framework, claiming improved sample efficiency and scalability over standard uniform sampling, LipMIP, and LipSDP.

### Strengths
The topic is relevant to robustness and verification. The idea of casting Lipschitz estimation as a pure-exploration bandit problem is natural and can reduce unnecessary sampling in flat regions. The comparison with LipSDP/LipMIP is appreciated, and the experimental section shows improvements in some settings. The theoretical discussion linking Clarke Jacobians and definability is mathematically sound.

### Weaknesses
The novelty is limited. The core contribution is an adaptation of standard UCB sampling to the well-known stochastic Lipschitz estimation baseline. The theoretical section is lengthy but does not yield new guarantees of interest to the ML community beyond established facts (consistency, asymptotic unbiasedness, and asymptotic optimality from standard bandit theory). The o-minimal component feels disconnected from the algorithmic contribution and is unlikely to be valued by ICLR readers.

Experiments are insufficient to justify significance. Evaluations are conducted on small MLPs and a binary MNIST setting. There is no evidence that the method scales to modern architectures (CNNs, Transformers, diffusion models) or yields practical benefit in certified training or downstream robustness tasks. Reported gains, while present, remain modest. The computational overhead of the adaptive strategy is not fully analyzed.

The paper lacks a compelling narrative demonstrating why this improvement matters for robustness, certified training, or verification workflows in practice. As written, it reads as an incremental technical refinement rather than a contribution that changes understanding or capabilities.

### Questions
- Can you clearly highlight the theoretical contribution that is unique to this work?
- Does the method scale to larger architectures (e.g., Transformers, ViTs, LLMs)?  
- Are there new bottlenecks when moving to large-scale models?
- Can we show measurable improvements in robustness-aware training or certified robustness?  
- Does the approach improve performance on standard certification benchmarks? 
- What is the computational complexity relative to baselines?

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
3

### Summary
In this paper, the authors investigate the connection between generalized derivatives and local continuity, and further propose an algorithm for estimating the modulus of local continuity. Building on existing methods for evaluating local Lipschitz continuity, they extend the formulation to more general notions of local continuity. The authors also partition the input space of dimension 
$ d $ into $k^{d} subregions to support structured sampling. Finally, they incorporate a non-uniform UCB-based sampling strategy that concentrates samples in regions with higher estimated importance. The paper provides both theoretical analysis and empirical results demonstrating that the proposed approach yields more consistent and accurate estimates.

### Strengths
1: The authors propose a non-uniform sampling strategy based on UCB scores. This approach allows the sampling process to better adapt to different regions of the input space compared to uniform sampling.

2: The authors provide solid theoretical analysis demonstrating that their algorithm is more dynamic and consistent.

### Weaknesses
1: The proposed non-uniform sampling method relies heavily on hyperparameters such as the exploration coefficient "c" and the subdivision threshold. A detailed discussion or sensitivity analysis of these parameters would be important to assess the robustness and practical usability of the method.

2: For high-dimensional input spaces, partitioning the domain into more subregions may introduce significant computational overhead. The paper does not address the scalability implications of this partitioning strategy in high-dimensional settings.

3:Although the paper focuses primarily on theoretical contributions, evaluating the method on larger datasets or across different modalities, such as natural language processing or time-series data, would strengthen the empirical evidence and demonstrate broader applicability.

### Questions
Based on the identified weaknesses, I would appreciate clarification from the authors on the following points:
1: Could the authors discuss how the key hyperparameters (e.g., exploration constant c and subdivision threshold) affect the performance of the proposed method? Providing sensitivity analysis or practical guidance on selecting these values would help assess the robustness of the approach.

2: Could the authors analyze or comment on the computational efficiency of the method, particularly in high-dimensional settings?

3: While the focus is theoretical, demonstrating the method on more diverse datasets or tasks (e.g., larger vision benchmarks, NLP models, or time-series data) would strengthen the empirical validation. Can the authors provide results on additional datasets or comment on expected performance in such settings?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper presents a novel method for estimating the Lipschitz constant in neural networks, addressing the known scaling limitations of methods like LipMIP and LipSDP. The core contribution is Algorithm 3 in the paper, which frames the problem in an infinity-armed bandit setting without commonly considered assumptions, using an Upper-Confidence-Bound (UCB) policy to partition the input domain and concentrate samples in useful regions. The paper compares with LipSDP and LipMDP on RELU-activation based neural networks trained on synthetic data and on binary MNIST.

### Strengths
The paper is written very well. The results are also clearly presented.

The results on binary MNIST clearly depict the improvements over LipSDP and LipMDP.

I am glad that the paper does not limit itself to simple two layer neural networks and instead, in Figure 5, compares with the aforementioned methods across a number of hidden layers.

### Weaknesses
A potential related work missing: In [1], a extreme value analysis method is used to compute Lipschitz constants and could be relevant to this paper.

[1] Improving Neural Network Robustness via Persistency of Excitation, Sridhar et al, ACC 22

Scale: I am uncertain how easily this method can scale up to neural networks in use today from imagent scale models to LLMs, VLMs, and VLAs. Perhaps the authors can address the question of further scale?

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
