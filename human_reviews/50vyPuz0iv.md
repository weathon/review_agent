# Iteratively Refined Behavior Regularization for Offline Reinforcement Learning

- Decision: Reject
- Scores: 6, 3, 3

## Abstract
One of the fundamental challenges for offline reinforcement learning (RL) is ensuring robustness to data distribution. 
Whether the data originates from a near-optimal policy or not, we anticipate that an algorithm should demonstrate its ability to learn an effective control policy that seamlessly aligns with the inherent distribution of offline data. Unfortunately, behavior regularization, a simple yet effective offline RL algorithm, tends to struggle in this regard. In this paper, we propose a new algorithm that substantially enhances behavior-regularization based on conservative policy iteration. Our key observation is that by iteratively refining the reference policy used for behavior regularization, conservative policy update guarantees gradual improvement, while also implicitly avoiding querying out-of-sample actions to prevent catastrophic learning failures. We prove that in the tabular setting this algorithm is capable of learning the optimal policy covered by the offline dataset, commonly referred to as the in-sample optimal policy. We then explore several implementation details of the algorithm when function approximations are applied. The resulting algorithm is easy to implement, requiring only a few lines of code modification to existing methods. Experimental results on the D4RL benchmark indicate that our method outperforms previous state-of-the-art baselines in most tasks, clearly demonstrate its superiority over behavior regularization.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper aims to address the suboptimality that arises from the behavior regularization w.r.t the suboptimal datasets in policy-constraint-based offline RL methods. Specifically, this paper casts policy constraints on an iteratively evolved reference policy $\bar{\pi}$ rather than the suboptimal behavior policy $\pi_D$. Theoretical analysis proves that this iteratively evolved policy constraint can not only avoid OOD queries but also achieves in-sample optimality when the initial reference policy $\bar{\pi}$ is initialized to $\pi_D$ under the full-data-coverage assumption that the optimzied policy $\pi$ should always stay in the support of the reference policy $\bar{\pi}$. However, for practical implementation, this paper also applies constraints on the suboptimal behavior policy to stablize training, where the constraint strength between $\pi_D$ and $\bar{\pi}$ are balanced by an newly introduced hyper-parameter. This paper also provides extensive experimental results to demonstrate the effectiveness of the proposed method.

### Strengths
1. This paper is well organized and well written, presenting good motivations in Figure 1 and theoretical analysis in Theorem 1 to show the significance of the proposed method.
2. The proposed method is simple and easy to implement.
3. The experimental results are sufficient.

### Weaknesses
## Major weakness
The performance improvement seems to primarily stem from comprehensive parameter tuning, rather than the proposed  iterative refinement of policy constraints. CPI can be viewed as adding an additional iteratively refined policy constraint to TD3+BC, while introducing a new hyperparameter $\lambda$ to adjust the constraint strengths. In my view, this is doing some kind of conservatism relaxation and introduces an additional hyperparameter, providing a more precise adjustment of conservatism strengths than solely tuning the $\tau$. Consequently, this leads to performance gains compared to the base method, TD3+BC.

See from Table 3 and Table 4 that hyper-parameters are thoroughly tuned for each individual task. Moreover, Table 5 shows that TD3+BC (dynamic $\alpha$) is on par with CPI and CPI-RE. This indicates that TD3+BC might also achieve in-sample optimality and meanwhile prevents OOD issues with a carefully sweeped conservatism strength $\alpha$.

I would consider raise my score if the major weakness is well resolved.

## Minor weakness
The idea about utilizing an iteratively refined policy constraint is already introduced by a recent offline2online RL paper[1], which is stated as a future research in Conclusion.

[1] PROTO: Iterative Policy Regularized Offline-to-Online Reinforcement Learning. 2023.

## Typo
1. In proposition 1, the (34) would be better (5).

### Questions
1. I'm wondering how do CPI and CPI-RE perform using one group of hyperparameters for each types of tasks including Gym-Mujoco, Antmaze and Adroit, respectively, just like previous works such as IQL and TD3+BC does.
2. In Equation (10), I'm questioning the term $\tau(1-\lambda)\lambda$. Could there possibly be an error? Should it be $\tau(1-\lambda)$ instead?

Please refer to the weakness for details.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents an innovative approach to refining behavior-regularized offline RL algorithms, introducing the concept of Iterative Self Policy Improvement (ISPI) that progressively optimizes within the behavior policy's support. The authors also propose a practical implementation trick for ISPI, demonstrating its effectiveness against strong baselines across various tasks.

### Strengths
1. Introduces a novel and promising concept of refining behavior policy to address suboptimal behavior.
2. Section 3 adeptly explains the iterative solution and establishes a clear connection with online learning.
3. Comprehensive and robust experimental section, with comparisons against the latest SOTA methods, clearly highlighting the improvements over the TD3+BC backbone algorithm.

### Weaknesses
1. There's a noticeable discrepancy between the theoretical framework and the practical implementation. The theory is grounded on in-sample learning assumption, yet the Competitive Policy Improvement (CPI) paradigm isn't in line with this. The implementation appears to violate the in-sample property suggested by the theory.
2. This paper lacks a clear link to in-sample learning despite a strong motivation, potentially leading to reader confusion, especially due to the mismatch between theory and practice. A restructuring of the manuscript to improve clarity and coherence is recommended.

minors: 
There are two hyperparameters in ISPI, which may be hard to tune in practice.

### Questions
n/a

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper introduces an approach to improve behavior regularization through conservative policy iteration. This is achieved by incorporating an additional KL regularization term between the current policy and a static target policy network into the pre-existing TD3+BC objective function. The authors offer a theoretical analysis for the tabular case. They evaluate their proposed algorithm, CPI, as well as an ensemble policy variant, CPI-RE, using toy discrete datasets and the offline RL benchmark D4RL datasets.

### Strengths
The experiments conducted are comprehensive and of high quality.

### Weaknesses
1. The paper's clarity and organization could benefit from a thorough revision. Currently, the flow is disjointed, with readers needing to frequently navigate back and forth, e.g., Proposition 1 references an equation found in the appendix. Additionally, the algorithm box lacks clarity, which is elaborated upon in the questions section.

2. The paper's originality and novelty appear limited. The newly introduced component merely utilizes a target policy network for KL regularization—a technique already employed in algorithms like TRPO and PPO. My interpretation suggests that the paper's primary contribution is a trick that demonstrates how integrating a conservative policy update can enhance offline RL.

3. The scope of testing is narrow. Evaluating only on TD3+BC doesn't provide enough evidence to support the universal benefits of adding conservative policy learning. The authors claim that implementing their method requires minimal modifications to existing algorithms. If so, it would be beneficial to test it on other frameworks like IQL, CQL, and Diffusion-QL.

### Questions
1. In Figure 1, does the dashed line represent a BC trained solely on the curated dataset?

2. You mentioned, "We provide the pseudocode for both CPI and CPI-RE here." I assume this "here" refers to the algorithm box?

3. The algorithm box seems ambiguous in its description:
  * Right after Equation (9), the reference policy is identified as the target policy. However, in the algorithm for CPI-RE, the reference policy is described as the optimal policy between $\omega_1$ and $\omega_2$.
  * Can you clarify what is meant by the "cross-update scheme"?

4. The paper notes that $\pi_\omega$ is initialized as $\pi_\omega = \pi_D$. Could you specify where this initialization is reflected in the algorithm box?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
