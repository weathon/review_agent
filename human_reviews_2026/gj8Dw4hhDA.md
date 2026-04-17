# A KL-regularization framework for learning to plan with adaptive priors

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Effective exploration remains a central challenge in model-based reinforcement learning (MBRL), particularly in high-dimensional continuous control tasks where sample efficiency is crucial. A prominent line of recent work leverages learned policies as proposal distributions for Model-Predictive Path Integral (MPPI) planning. Initial approaches update the sampling policy independently of the planner distribution, typically maximizing a learned value function with deterministic policy gradient and entropy regularization. However, because the states encountered during training depend on the MPPI planner, aligning the sampling policy with the planner improves the accuracy of value estimation and long-term performance. To this end, recent methods update the sampling policy by minimizing KL divergence to the planner distribution or by introducing planner-guided regularization into the policy update. In this work, we unify these MPPI-based reinforcement learning methods under a single framework by introducing Policy Optimization-Model Predictive Control (PO-MPC), a family of KL-regularized MBRL methods that integrate the planner’s action distribution as a prior in policy optimization. By aligning the learned policy with the planner’s behavior, PO-MPC allows more flexibility in the policy updates to trade off Return maximization and KL divergence minimization. We clarify how prior approaches emerge as special cases of this family, and we explore previously unstudied variations. Our experiments show that these extended configurations yield significant performance improvements, advancing the state of the art in MPPI-based RL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents PO-MPC, a unified KL-regularized framework for model-based reinforcement learning that integrates learned policies with MPPI planning. By treating the planner’s action distribution as a prior, PO-MPC balances return maximization and policy–planner alignment, generalizing prior methods like TD-MPC2 and BMPC. An adaptive prior further improves stability and sample efficiency, leading to state-of-the-art performance on DMControl and HumanoidBench tasks.

### Strengths
1. **Practical algorithmic insights.**
The introduction of an adaptive intermediate prior to mitigate variance from outdated planner samples is a simple yet effective innovation. It provides a practical solution to a known instability issue in planning-based RL systems.

2. **Comprehensive empirical validation.**
Experiments on comprehensive high-dimensional continuous control tasks (DMControl and HumanoidBench) are extensive and consistent. The ablation studies on λ and KL direction clearly illustrate the effects of the proposed components and the interpretability of the trade-offs.

### Weaknesses
1. **Lack of theoretical analysis on the adaptive prior.**
While the adaptive prior empirically reduces variance, the paper does not provide a formal justification or theoretical understanding of why or when it helps. A short analytical discussion or ablation on its effect across different environments would make the contribution more solid.

2. **Unclear practical advantages over baselines.**
The performance of PO-MPC improvements appear moderate in some tasks (Table 1). This omission further weakens the contribution of the paper, especially given that the work does not include substantial theoretical analysis to compensate for the limited theoretical insights or evidence of improvement.

3. **Limited discussion of related work.**
The paper omits explicit discussion of several highly relevant studies, especially there are several prior works focused on exact the same point: TD-M(PC)²[1] and TDMPBC[2]both analyzes how varying the policy regularization parameter (β) bridges TD-MPC and BMPC variants (**even in TD-M(PC)^2, the issue of MPPI-based RL is also summarized as "policy mismatch"**), and empirically outperforms TD-MPC2. For the planning part, Residual-MPPI[3] also formulates MPPI planning through a KL-regularized objective and provides theoretical insights into this formulation.


[1] Lin, Haotian, et al. "TD-M (PC) $^ 2$: Improving Temporal Difference MPC Through Policy Constraint." arXiv preprint arXiv:2502.03550 (2025).

[2] Zhuang, Zifeng, et al. "Tdmpbc: Self-imitative reinforcement learning for humanoid robot control." arXiv preprint arXiv:2502.17322 (2025).

[3] Wang, Pengcheng, et al. "Residual-mppi: Online policy customization for continuous control." arXiv preprint arXiv:2407.00898 (2024).

### Questions
1. Could the authors discuss in which regimes PO-MPC is most beneficial (e.g., high-dimensional control, noisy dynamics, sparse rewards) and whether its stability advantages justify the added complexity?
2. Could the authors provide a clear discussion comparing the mentioned prior works in weaknesses?

If the authors can adequately address these concerns, I would be inclined to raise my score.

### Soundness
2

### Presentation
4

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
This paper presents PO-MPC, a framework that integrates Model Predictive Path Integral (MPPI)-based methods with KL-regularized RL to improve the performance of model-based RL (MBRL) in high-dimensional continuous control tasks. The central idea is to adaptively regularize the learned policy with a KL-divergence term, using the planner's action distribution as a prior. The framework is evaluated on several high-dimensional control tasks, where it is claimed to outperform state-of-the-art methods, such as TD-MPC and BMPC.

### Strengths
- The paper introduces a unifying framework that brings together previous works in MPPI-based RL, presenting them under the umbrella of KL-regularized RL, which is useful for organizing existing methods.
- In certain tasks, the proposed method demonstrates modest improvements over existing methods. The inclusion of an intermediate policy prior to reduce variance in the KL divergence term is a reasonable approach and shows some potential.

### Weaknesses
- My main concern is that  the core contribution of this work does not appear to be a novel idea. The concept of regularizing RL policies using a prior has been widely explored (e.g., in prior works like TD-MPC[1] and BMPC[2]). 
- Experimental Validation is not strong.
   - How were the hyperparameters, such as λ, tuned? What effects did variations in λ have on performance?
   - The experiments are insufficiently detailed, with missing information about key parameters, such as the architecture used in the tasks and the precise reward structure in the environments.
   - The results in Figure 7 (showing the trade-off between forward and reverse KL loss) suggest that while forward KL helps some tasks, it degrades others, but there is no analysis of this behavior across all environments tested.
- The paper presents a solution to a problem (policy-prior mismatch) that is neither clearly demonstrated nor adequately explained. The experiments do not convincingly argue why this issue is critical and how it differs from traditional generalization challenges in RL.
- minor issue: Why is there no numbering for the formula on P3?

---

[1] Temporal difference learning for model predictive control

[2] Bootstrapped Model Predictive Control

### Questions
- Can you provide a theoretical analysis of why aligning the learned policy with the planner’s behavior using KL regularization leads to the claimed improvements? Specifically, how does this regularization prevent the over-generalization problem in non-linear function approximation settings?
- How sensitive are the results to the choice of hyperparameters, particularly λ? The experiments do not provide enough insight into this.
- Can you clarify the specific experimental setup, particularly the architecture and parameter settings used for the control tasks?
- For the KL-regularized action-value in Eqs. (2)–(3), can you formalize the Bellman operator
   $$
   (\mathcal{T}_{\pi,\lambda}Q)(z,a)=\mathbb{E}\big[r(z,a)+\gamma\big(Q(z',a')-\lambda\log \tfrac{\pi(a'|z')}{\pi_p(a'|z')}\big)\big]
   $$
   and prove contraction (w.r.t. which norm) and existence/uniqueness of the fixed point for $\lambda>0$? How does this fixed point relate to (i) standard soft-Q (entropy-regularized) and (ii) the unregularized TD objective used in TD-MPC2? 
- Eq. (2) is written under $d^{\pi_{\theta_s}}$, but training is off-policy from a replay buffer generated by a changing planner and actor. What assumptions (or corrections) justify dropping importance weights? Any known bias in $Q_{\pi,\lambda}$ estimation under your pipeline? 
- Can you connect Eqs. (1)–(3) to known RL-as-inference derivations and Maximum a Posteriori Policy Optimization, clarifying what is new in your derivation beyond using the planner as the prior?
- You motivate the intermediate prior as “shielding” variance. Can you report the variance (or SNR) of $\nabla_{\theta_s}J$ with vs without the intermediate prior, not just the KL-term’s variance proxy?

### Soundness
2

### Presentation
2

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
This paper is in the area of planner-based Model-based Reinforcement Learning. The authors propose Policy Optimization-Model Predictive Control (PO-MPC) which builds on a recent approach, TD-MPC2, by using KL-regularization in learning the sampling policy, where the sampling policy is regularized to be close to another policy, an adaptive prior, that is itself trained to follow the planning-produced samples in the replay buffer. As a result, PO-MPC subsumes a version of another approach, BMPC, as well as TD-MPC2, depending on how the KL-regularization term is traded off against the return maximization term. 

The authors perform experimental evaluations on different tradeoffs for KL regularization, using a learned adaptive prior vs. the planning samples as the prior, as well as how the learning of the prior affects performance, showing the benefits of different parameters for different tasks.

### Strengths
I like that this paper attempts to unify closely-related approaches and provides a more general way to think about them. I think the exploration of forward vs reverse KL for learning of the adaptive prior is interesting, particularly showing that minimizing the forward KL biases the policy towards more exploration but delayed convergence which is beneficial in some tasks but detrimental in others. 

I think the exposition and writing of the paper are also good overall. The experiments are quite thorough and answer most of the questions raised in the paper.

### Weaknesses
I found some of the core ideas of the paper difficult to follow and had to re-read a few times. Since there are a lot of different parts, and there is also another policy being added here, I think a diagram or illustration would have helped to put the pieces together. 

I think the main message of the paper, that it unifies existing approaches, could be conveyed more clearly. For example, it is mentioned in several places that $\lambda \rightarrow \infty$ recovers Variant 3 of (Wang et al., 2025), but what is this variant? Why is this variant recoverable but not the others? I think the variant itself should be described in the present paper as well, and the details of which parts are included and which parts are not.

### Questions
- In L390, it is mentioned that the $\lambda$ values tried were $0.1, 0.5, 0.9$, but in Table 1 and Figure 1 the $\lambda$ values are stated to be $0.1, 1, 9$. Which is the correct one?
- Could you comment on the effectiveness of learning an adaptive prior as opposed to using lazy reanalyze like in BMPC and the planner samples as the prior? From the results in Figure 5, it seems there is almost no difference between the two priors for the same $\lambda$. 
- Related to the above question: was lazy reanalyze or a similar approach used to update the samples in the replay buffer in combination with the adaptive prior as well? Did you observe an improvement?

### Soundness
3

### Presentation
3

### Contribution
3
