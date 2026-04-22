# Training Spiking Neural Networks with Real-Time Propagation Through Time

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
Online learning algorithms for Spiking Neural Networks (SNNs) offer a memory-efficient alternative to Backpropagation Through Time (BPTT), but suffer from two critical issues: training instability and membrane potential distribution drift. To address these challenges, we introduce Real-Time Propagation Through Time (RPTT), a novel online learning framework. RPTT computes gradients using only the spatial component and integrates two synergistic regularization mechanisms: Membrane Potential Distribution Regularization (MPDR), which statistically constrains membrane potentials to counteract distributional drift, and Spatio-Temporal Gradient Regularization (STGR), which smooths weight updates to ensure stable convergence. We theoretically prove that RPTT converges to a stationary point. Extensive experiments on CIFAR-10/100, ImageNet-1k, and DVS-CIFAR10 demonstrate that RPTT achieves state-of-the-art performance while significantly reducing memory consumption. Experimental analysis reveals that RPTT achieves strong performance by effectively alleviating the membrane potential drift. Our work thus provides an effective framework for the online training of SNNs, significantly advancing their application in dynamic and realistic environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the RPTT framework to address two critical challenges in online SNN training: instability and membrane potential drift. The method leverages an efficient spatial-gradient-only update scheme, augmented by two novel regularizers: Membrane Potential Distribution Regularization (MPDR) to counteract distributional drift and Spatio-Temporal Gradient Regularization (STGR) to stabilize the training process.

### Strengths
The paper offers a well-motivated and empirically effective solution to the important problem of online SNN training. The proposed regularizers, MPDR and STGR, are cleverly designed. The authors provide convincing experimental analysis to demonstrate their efficacy in mitigating membrane potential drift and stabilizing learning dynamics.

### Weaknesses
1. The paper fails to specify how the backpropagation for the STGR term, particularly $\|\nabla\ell_{t-1}(W_{t}^{l})\|^{2}$, is implemented. Differentiating this term with respect to $\mathbf{W}_{t}$ would seemingly require second-order information (i.e., a Hessian-vector product), which contradicts the paper's claim of O(N) memory complexity and the core goal of efficient online learning. 

2. The experimental evaluation lacks comparisons against several recent and highly relevant online learning methods, such as NDOT (Jiang et al., 2024) and OSR/OTS (Zhu et al., 2024). 

3. The authors claim the method is suitable for "dynamic environments," but the only dynamic dataset used is CIFAR10-DVS, which exhibits weak temporal correlations. The experiments do not provide sufficient evidence to support its efficacy in truly non-stationary or dynamic settings.

4. The ImageNet experiment is conducted by fine-tuning a pre-trained SLTT model rather than training from scratch. This severely weakens the claim of state-of-the-art performance on large-scale tasks. Disturbingly, this crucial detail is relegated to Appendix A.2.4 and omitted from the main manuscript, which could be misleading.

5. As reported in Appendix A.2.4, the hyperparameters for RPTT vary significantly across different datasets. This high sensitivity to parameter tuning suggests that the method may lack generalizability and could be difficult to apply to new tasks without extensive tuning.

6. The convergence proof for Theorem 1 relies on a critical assumption: a regularization term attenuation coefficient $c_t$ (Eq. 20) that decays over time. However, the experimental setup uses fixed, constant regularization parameters.

### Questions
1. Please provide a detailed complexity analysis for the STGR term. How is the gradient of $\|\nabla\ell_{t-1}(\mathbf{W}_{t}^{l})\|^{2}$ calculated in practice, and how does this align with the claimed O(N) memory complexity?

2. Please justify the choice of a fixed target distribution q=N(-60, 10) for all layers and datasets. I strongly suggest authors include a sensitivity analysis on these hyperparameters to demonstrate the robustness and generalizability of your method.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes RPTT, an online learning method for SNN learning. RPTT introduces MPDR to counteract distributional drift and STGR to ensure stable convergence during online learning.

### Strengths
1. Online learning methods save memory during training, which have constant memory cost with the number of time steps $T$.
2. This work provides theoretical proof of the stableness of RPTT.
3. The writing of this paper is easy to understand.

### Weaknesses
The final performance of this work is not high enough. Specifically, Table 1 does not include results of NDOT (Jiang et al. 2024) and OSR+OTS (Zhu et al. 2024), which generally perform better than this work.

### Questions
1. What is the necessity of updating weights at each step in SNN online learning? I don't see its advantage.
2. In Eq.4, authors say that $Var(V_t^l)$ is used to penalize overly small variance, but its coefficient is positive. Should the sign be '-'?
3. The STGR loss (Eq. 5) is similar to the loss in FPTT[1]. Could you give some comments on the difference between them?
4. Why the membrane potential is bimodal in Figure 4? It differs from the Gaussian target distribution in MPDR.

[1] Kag, A., & Saligrama, V. (2021, July). Training recurrent neural networks via forward propagation through time. In *International Conference on Machine Learning* (pp. 5189-5200). PMLR.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes RPTT, an online training method for SNNs with two regularizers: Membrane Potential Distribution Regularization (MPDR) and Spatio-Temporal Gradient Regularization (STGR). The authors provide a convergence argument, and present results on both static and neuromorphic datasets showing the memory efficiency and competitive performance.

### Strengths
1. Thoughtful empirical study. Experiments are conducted on both static and neuromorphic datasets, with ablations and distributional analyses.
2. Theoretical analysis. The paper attempts at a convergence analysis, showing the importance of the proposed techniques for deriving a kind of convergence.

### Weaknesses
1. Notation confusion. The paper overloads $t$ to mean both the SNN time step and the optimization iteration, making the description confusing.
2. Strong theoretical assumption. The theoretical analysis assumes a loss sequence with $|l_{t+1}(W)-l_t(W)|<\Delta_t$ and $\sum_t \Delta_t < \infty$, which is essentially an asymptotically stationary setting and almost implies convergence. As a result, the theorem contributes limited new insights.
3. Incremental novelty and practical gains. Using spatial-only gradients with low memory costs is already present in previous works. MPDR and STGR are mainly incremental regularizers rather than new training principles. In the largest-scale ImageNet experiment, the proposed method actually has no gain compared with SLTT.

### Questions
See Weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Real-Time Propagation Through Time (RPTT), a novel online learning algorithm for Spiking Neural Networks (SNNs). RPTT aims to address two key challenges in online SNN training: training instability and membrane potential distribution drift. The core of RPTT is the use of only spatial gradients for parameter updates, augmented by two synergistic regularization mechanisms:

Membrane Potential Distribution Regularization (MPDR): A statistical constraint that uses KL-divergence and a variance penalty to keep membrane potentials close to a target Gaussian distribution, counteracting drift.

Spatio-Temporal Gradient Regularization (STGR): A smoothing mechanism that uses a moving average of weights and a penalty on previous gradients to stabilize updates and suppress noise.

The authors provide a theoretical convergence guarantee for RPTT and demonstrate its effectiveness on static (CIFAR-10/100, ImageNet-1k) and neuromorphic (DVS-CIFAR10) datasets, showing it achieves competitive or state-of-the-art performance while significantly reducing memory consumption compared to BPTT.

### Strengths
Originality: The explicit formulation of the membrane potential drift problem in the context of online learning and the proposal of MPDR as a direct, layer-wise solution is a novel and valuable contribution. While drift has been studied in offline settings, its exacerbation by frequent online updates is a fresh and meaningful insight. The combination of MPDR with STGR to jointly address drift and instability is a creative and well-motivated design.

Quality: The paper is technically sound. The experimental section is comprehensive, covering multiple datasets and network architectures. The inclusion of a theoretical convergence analysis, despite relying on some strong assumptions, adds rigor and depth to the work. The ablation studies and membrane potential visualization (e.g., the emergence of a bimodal distribution) provide empirical support for the method's mechanisms.

Significance: The work addresses a critical bottleneck for the practical deployment of SNNs: achieving memory-efficient and stable online training. By significantly reducing memory overhead (O(N) vs. BPTT's O(TN)) and mitigating a fundamental performance-limiting phenomenon (drift), RPTT represents a tangible step towards making SNNs viable for dynamic, real-world applications on resource-constrained neuromorphic hardware.

### Weaknesses
Insufficient Comparison with Related Work: The paper's academic impact is limited by its superficial comparison with existing methods. The performance comparison in Table 1 is useful but does not provide insights into why RPTT performs better.

Missing Analysis on Membrane Potentials: A core claim is that RPTT better mitigates drift. However, there is no direct, quantitative comparison of membrane potential distributions (e.g., Z-score trajectories, KL-divergence from target) between RPTT and other online methods like OTTT or SLTT. The analysis in Fig 3(b) only compares RPTT with BPTT and OSBP (a weak baseline). Does SLTT, which uses delayed updates, also suffer less from drift than OSBP? How does RPTT compare to OTTT in this regard? This is a major missed opportunity to validate the central motivation.

Baseline Currency: While some recent methods are included (e.g., SLOT, NDOT), the choice of the primary online baseline, OSBP, is weak. OSBP is not a established, strong benchmark from the literature. A more convincing comparison would involve directly integrating MPDR and STGR into a stronger and more recent online method like OTTT or the framework of SLOT to perform an ablation, demonstrating the generalizability and additive value of the proposed regularizers.

Weak and Partially Inaccurate Motivation:

The statement "to the best of our knowledge, the problem of membrane potential distribution drift has not yet been studied under online learning algorithms" (Page 2) is too strong. While perhaps not the primary focus, works like Zhu et al. (2024) ("Online Stabilization of Spiking Neural Networks") directly address firing rate stability across time, which is intrinsically linked to membrane potential distribution. The authors should tone down this claim and more precisely articulate their unique focus on distributional drift via online, layer-wise regularization.

The motivation would be stronger if it included a preliminary analysis showing that existing online methods (OTTT, SLTT) indeed exhibit more severe drift than offline BPTT, thereby creating a clear gap that RPTT fills.

Limited Ablation Study:

The ablation in Section 4.3 only reports final accuracy. It does not quantify the individual contribution of MPDR and STGR to reducing distribution drift. For instance, how much does the Z-score improve with MPDR alone? How does STGR alone affect the variance of the membrane potential distribution? Linking each component's effect directly to the underlying problem it is designed to solve would greatly strengthen the paper.

Theoretical Limitations:

The convergence proof relies on strong assumptions (e.g., $\beta$-smooth loss, bounded gradients, Gaussian membrane potentials) that may not hold perfectly in practice. A discussion of these limitations and the proof's practical relevance would be beneficial.

The assumption that the task sequence change is bounded ($\sum \Delta_t < \infty$) is particularly strong for a non-stationary online learning setting and deserves clarification.

### Questions
None

### Soundness
2

### Presentation
2

### Contribution
2
