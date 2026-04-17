# Sample-Efficient Online Distributionally Robust Reinforcement Learning via General Function Approximation

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
The deployment of reinforcement learning (RL) agents in real-world tasks is frequently hampered by performance degradation caused by mismatches between the training and target environments. Distributionally Robust RL (DR-RL) offers a principled framework to mitigate this issue by learning a policy that maximizes worst-case performance over a specified uncertainty set of transition dynamics. Despite its potential, existing DR-RL research faces two key limitations: reliance on prior knowledge of the environment -- typically access to a generative model or a large offline dataset -- and a primary focus on tabular methods that do not scale to complex problems. In this paper, we bridge these gaps by introducing an online DR-RL algorithm compatible with general function approximation. Our method learns an optimal robust policy directly from environmental interactions, eliminating the need for prior models and enabling application to complex, high-dimensional tasks. Furthermore, our theoretical analysis establishes a near-optimal sublinear regret for the algorithm under the total variation uncertainty set, demonstrating that our approach is both sample-efficient and effective.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a near-optimal algorithm in online distributional robust reinforcement learning under the general function approximation setting, demonstrating sample efficiency. The algorithm is built on the classical UCB algorithm, making it easy to implement and robust. The authors provide both the regret and the sample complexity, clearly comparing their results with other papers, which makes their contribution easy to follow.

### Strengths
Clear Presentation: The paper is well-structured, effectively defining robust reinforcement learning and the robust value function. The definitions and assumptions are well-cited, enhancing the paper's credibility. The proofs are straightforward and easy to follow.

UCB-Based Algorithm: The authors provide a comprehensive analysis of both sample complexity and regret for their proposed algorithm in an online learning context, demonstrating its efficiency. The identification of the appropriate robust coverage is crucial for learnability, making this the first online DR-RL algorithm suitable for large-scale problems with minimal structural assumptions.

### Weaknesses
Lack of Regret Lower Bound: While the paper presents upper bounds for sample complexity and regret, it fails to include lower bounds for both metrics, which are essential for establishing the optimality of the algorithm. The absence of lower bounds may give the impression of insufficient rigor, as deriving the regret should be straightforward once the sample complexity is established.

Lack of Novelty: Despite the low workload of the paper, the algorithm appears to build on existing works, lacking originality. Additionally, the technical lemmas seem to follow established research, raising concerns about the novelty and technical depth of the contributions. It appears that the authors merely adapted existing methods to a new setting.

Unclear Contribution: It would be beneficial for the authors to clearly outline their contributions in the introduction. Given that the technical lemmas are derived from existing works, the overall problem appears to be relatively straightforward, which may not meet the expected quality standards for such a conference.

### Questions
1. Why is there no regret lower bound provided in the paper? What do you think are the main technical difficulties in deriving such a lower bound? I am considering raising my score if you can add the right lower bound in the final version of the paper.
2. What is the main technical difficulty in proving the upper bound of the proposed algorithm? In my opinion, it all follows standard methods. Choosing the right coverage and the assumptions seems to also follow existing works.
3. This algorithm is designed for the online RL setting. Can we extend it to the offline setting? Will simply changing the UCB to LCB be sufficient for the extension?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper propose to use function family to facilitate the robust RL. 
Theoretical analysis are provided to justify the advantages of the proposed methods.

### Strengths
The paper is clearly written and the topic is interesting. The idea of using function family to improve robut RL sounds reasonable. 
The theoretical part is useful to provide some insights.

### Weaknesses
There is no specific examples even toy examples are provided to illustrate the proposed methods.
This makes it is difficate to judge whether it is easy to implement and train the models
and how to choose the family of functions in pratice.

### Questions
What kind of large-scale problems could be used to demonstrate the advantages of the proposed framework?
If experiments are designed, what kinds of alternative methods could be used to compare?
How to choose the function family for practical problems?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper, *Sample-Efficient Online Distributionally Robust Reinforcement Learning via General Function Approximation (RFL-TV)*, proposes one of the first **online DR-RL algorithms** that operate under **general function approximation**.

Unlike tabular RMDP works (e.g., *NeurIPS 2024 – Near-Optimal DR-RL with General Lp Norms*), this study focuses on the online setting, where agents interact with an unknown environment and must learn robust value functions via dual functional optimization.

The authors reformulate the robust Bellman operator under TV-divergence uncertainty as a **functional convex optimization** problem, introducing a dual function $g(s,a)$ to replace state-action-wise scalar dual variables.

This yields a tractable empirical loss function, enabling robust fitted learning through convex minimization.

Building on this, the proposed **RFL-TV algorithm** constructs global confidence sets based on the fitted dual loss and uses the **optimism in the face of uncertainty (OFU)** principle to ensure exploration.

The resulting **robust regret bound** is

$$ \tilde{O}\!\left(H \sqrt{C_{\text{rcov}} H \min\{H, 1/\sigma\} K}\right) - O(C_{\text{rcov}} \xi_{\text{dual}}),$$

where the newly defined **robust coverability** $C_{\text{rcov}}$ quantifies exploration difficulty under model uncertainty.

To make function approximation theoretically valid, the paper introduces **Assumption 3 (Dual Realizability)**, requiring the dual function class $\mathcal{G}$ to approximate the true dual optimizer.

This assumption is stronger than the Bellman completeness condition used in *Jin et al., 2021* but is necessary to bound bias from empirical dual optimization.

In summary, RFL-TV provides one of the first theoretically grounded frameworks for **distributionally robust online RL with general function approximation**, albeit under additional realizability assumptions and bounded coverage conditions.

### Strengths
1. **general-function-approximation framework for distributionally robust online RL**
    
    The paper establishes the first principled framework for online reinforcement learning under distributional robustness, extending beyond tabular or offline DR-RL settings.
    
2. **Dual functional optimization for robust Bellman approximation**
    
    The authors reformulate the TV-divergence-based robust Bellman operator into a tractable **convex functional optimization** problem by introducing a global dual function $g(s,a)$, replacing pointwise scalar dual variables and enabling efficient empirical learning.
    
3. **Introduction of Robust Coverability and Sublinear Regret Guarantees**
    
    The work introduces **robust coverability** $C_{\text{rcov}}$, a weaker and more general complexity measure than standard coverability or Bellman rank, capturing exploration difficulty under uncertain transitions.
    
    Based on this metric, RFL-TV achieves provable efficiency with the regret bound
    
    $$    \tilde{O}\!\left( H \sqrt{C_{\text{rcov}}\, H\, \min\{H, 1/\sigma\}\, K}    \right)     - O(C_{\text{rcov}}\, \xi_{\text{dual}}),$$
    
    ensuring **sublinear regret** and **PAC-style sample efficiency** even under distributional uncertainty.

### Weaknesses
1. **Strong dual realizability assumption (Assump. 3)**
    
    The theoretical results hinge on the additional **dual realizability** assumption, which requires the dual function class $\mathcal{G} $ to approximate the true dual optimizer within $ \xi_{\text{dual}} $.
    
    This condition is non-verifiable in practice and notably stronger than the Bellman completeness assumption in *Jin et al., 2021*, limiting the theoretical generality and real-world robustness of the framework.
    
2. **Lack of empirical validation despite a concretely defined algorithm**
    
    Although the paper defines a clear algorithmic structure based on well-specified optimization objectives and assumptions, it does not provide any experimental results or numerical demonstrations.
    
    Given that the proposed RFL-TV method is theoretically implementable, empirical evaluation—even in simplified environments—would have significantly strengthened the paper’s completeness and practical credibility.
    
    In its current form, the work remains purely theoretical under restricted assumptions, leaving open whether the proposed dual functional optimization and confidence-set construction yield tangible performance benefits in practice.
    
3. **Restricted uncertainty model (TV-divergence only)**
    
    The analysis is limited to **TV-divergence** uncertainty sets, which, while foundational, restricts the generality of the conclusions.
    
    Extending the framework to broader f-divergence or Wasserstein uncertainty metrics—as explored in *NeurIPS 2024 Lp-norm DR-RL*—would enhance the universality and comparative significance of the results.

### Questions
Compared with [A], this paper’s theoretical development remains less consolidated and would benefit from clearer connection between assumptions, algorithmic implementation, and empirical outcomes.

---

**Questions to the Authors**

1. Could you treat the uncertainty level \( \sigma \) and **Assumption 3’s approximation constant** \( \xi_{\text{dual}} \) as tunable hyperparameters and show how performance or regret changes as these values vary?
2. The paper defines a seemingly implementable algorithm. Could you provide **empirical results** or demonstrations to verify its practical effectiveness under realistic settings?

**Reference**

[A] Pierre Clavier, Laixi Shi, Erwan Le Pennec, Eric Mazumdar, Adam Wierman, and Matthieu Geist,

*Near-Optimal Distributionally Robust Reinforcement Learning with General Lp Norms*,

NeurIPS 2024.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces RFL-TV, an theoretical online distributionally robust reinforcement learning algorithm that supports general function approximation under a total-variation (TV) uncertainty set. The authors adopt a functional optimization formulation of the robust Bellman operator that enables efficient estimation and learning without requiring state-action-wise optimization. They further use the notion of robust coverability to characterize the information deficit between the nominal and worst-case dynamics. Under  certain assumptions, the proposed RFL-TV algorithm achieves a sample complexity of $\widetilde{O}\left(\frac{H^2 \min(H, \sigma^{-1})\, C_{\text{rcov}}}{\varepsilon^2} + \frac{C_{\mathrm{rcov}}\cdot\xi_{\mathrm{dual}}}{\varepsilon}\right)$, independent of the state–action space size, confirming its scalability.

### Strengths
1. The paper is the first one to provide a provably sample-efficient algorithm for the theoretical problem of distributionally robust reinforcement learning with both (i) online data collection and (ii) general function approximations.
2. The paper potentially introduces new techniques to handle the error accumulation of functional-optimization-based robust Bellman equation estimator in the non-stationary online learning situation.

### Weaknesses
1. The comparison of the results with previous arts could be further improved and made more detailed, in terms of the comparison of sample complexity, model structure assumiptions, coverage assumptions, etc.

2. The paper lacks empirical or simple numerical validation of the proposed algorithm.

3. Some other issues and questions that I detailed in the Questions part.

### Questions
1. The theory relies on the completeness assumption that the robust Bellman operator with robust radius $\sigma$ is closed in the function class $\mathcal{F}$. I am curious would the assumption implicitly incur any necessary dependence of the function class size $|\mathcal{F}|$ (which appears in the sample complexity bounds) on the robust radius $\sigma$? 
2. The theory is developed based-on the assumption of finite robust coverability assumption proposed in He et al., 2025. In another series of works on robust RL with online data collection the sample complexity does not depend on the robust coverage coefficient due to the usage of other types of structural assumptions, e.g., failure states assumption. The authors said that the  the failure state assumptionguarantees the finiteness of the robust coverability. Then how to compare the sample complexity results obtained by these two types of theoretical assumptions?
3. Can you highlight more what kind of new techniques are developed to handle the error accumulation of functional-optimization-based robust Bellman equation estimator? 
4. Inconsistent equations (8) and (9) for the terms $(1-\sigma)\cdot g(s,a)$ and $(\sigma/2-1)\cdot \eta$.

### Soundness
3

### Presentation
3

### Contribution
2
