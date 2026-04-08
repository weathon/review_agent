## Human Reviewer 1

### Summary
The paper proposes **Constraint-aware Reward (Re)Labeling (CARL)** for offline safe RL. It learns a cost-to-go critic and relabels rewards in each mini-batch: if $Q_c^\pi(s,a)>\kappa$, assign a large negative penalty; otherwise keep the original reward. An off-the-shelf offline RL backbone (e.g., TD3-BC, IQL) is then trained on the relabeled data.

### Strengths
* **Practical motivation:** avoids brittle dual updates seen in some Lagrangian methods.
* **Intuitive method:** convert safety constraints into state-action pair penalties rather than tuning Lagrange multipliers.
* **Simple backbone-agnostic wrapper:** CARL can be wrapped around standard offline RL backbones.

### Weaknesses
## Positioning & Objectives
* **Tight-budget regime but CMDP formulation:** The paper targets small cost limit $\kappa$ and *pointwise safety*, which is closer to *hard-constraint / shielded* or *risk-sensitive* formulations. However, problem setup is based on CMDP and baselines are CMDP-style OSRL only.
* **Theory-metric mismatch:** Theory enforces *statewise* safety; but evaluation reports *episodic normalized cost*. If pointwise safety is the goal, episodic evaluation metric may be a mismatch.
* **Feasibility under offline coverage:** Pointwise constraints may be *infeasible* for a given $\kappa$ with partial data coverage while the expectation constraints are *feasible*. Current submission does not discuss this scenario in detail.

## Method & Claims
* **Theorem 1 conditions (notation/assumption):** The proof sketch relies on a sufficiently punitive $-V_{\max}$, feasibility of the pointwise-constrained problem, and bounded nonnegative (or shifted-to-nonnegative) rewards. Note that nonnegative reward is not specified in problem setup, but it is required for $V_r^{\tilde{\pi}*}(s) > 0$ used in the proof of Theorem 1. In addition, the main experiments use $-R_{\max}$, not $-V_{\max}$, weakening the applicability of Theorem 1.
* **CARL stability vs Lagrangian:** CARL replaces dual updates with reward relabeling using Q-estimates. However, this introduces non-stationarity between the policy, fitted Q evaluation (FQE), and relabeling; and inherits FQE’s estimation noise. With the absence of convergence analysis, it’s unclear that CARL is less brittle than Lagrangian methods. Rather, it may swap dual-instability for OPE-instability. 
* **OPE dependence:** Unsafe detection hinges on FQE accuracy under distribution shift; miscalibration can over- or under-penalize. 
* **“Neighborhood suppression” claim:** With function approximation, penalizing one transition can generalize unpredictably beyond a "local neighborhood".
* **"No extra hyperparameters" claim:** The value of *penalty magnitude* has to be determined. The paper uses ($R_{\max}$ vs $V_{\max}$) for experiments. However, this value is offline dataset dependent and there may not be a sufficiently good estimate at deployment time. In addition, this value varies significantly across tasks.

## Evaluation
* **Baselines:** For tight budget setting, **hard-constraint/shielded** and **risk-sensitive** baselines may be more appropriate for small $\kappa$.
* **Budgets & metrics:** If the aim is strict/tight safety, violation rate and/or per-state safety metrics should be reported alongside episodic cost.
* **Statistical confidence:** Evaluation results average over 20 episodes and 3 random seeds. This seems too few to establish statistical confidence. 

## Reproducibility
* The anonymous code link given in the paper is broken. No code is accessible, the link returns "The requested file is not found." 

## Minor Typo
* **Line 088:** “ation-value”.

### Questions
## Theory & Guarantees
1. The paper uses $-R_{\max}$ as the penalty magnitude; what guarantee remains in practice, since Theorem 1 depends on $-V_{\max}$?

## Method & Stability
2. How is the **penalty scale** chosen at deployment (task-agnostic guidance)? 
3. How much noise do you witness in the OPE relative to threshold $\kappa$? How often are actions misclassified as unsafe/safe?
4. Does CARL update exhibit contraction/monotone improvement? My assessment is that the paper should temper the claim that CARL alleviates Lagrangian brittleness. It is not a strictly better stability trade-off.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper presents Constraint-Aware Reward Relabeling (CARL), a straightforward and hyperparameter-free approach to offline safe reinforcement learning (OSRL). CARL enforces safety constraints without using Lagrange multipliers or dual optimization by alternating between cost estimation and policy improvement. During training, rewards for actions predicted to be unsafe are replaced with strong negative penalties, effectively turning the constrained objective into an unconstrained one. The method serves as a lightweight wrapper that can be integrated with existing offline RL algorithms such as TD3-BC and IQL, enabling the learning of safe yet high-performing policies even from datasets containing many unsafe samples. Experiments on the DSRL benchmark show that CARL achieves consistently strong results, meeting cost constraints while maintaining high returns across various tasks and cost settings, highlighting its stability, generality, and practical utility for safety-critical offline RL.

### Strengths
- CARL provides a clean and practical approach to offline safe RL, removing the need for complex constrained optimization yet achieving strong safety and reward trade-offs. The method’s simplicity and consistent empirical gains make it a notable contribution. 
- Because CARL operates solely through reward relabeling at the data-processing stage, it can be seamlessly paired with diverse offline RL methods, making it a flexible and practical choice for safety-critical deployment.
- The paper offers a clear theoretical contribution by showing that optimizing a relabeled reward function with large penalties for unsafe actions leads to policies that satisfy pointwise safety constraints, and is formally equivalent to the original constrained CMDP formulation under mild assumptions.
- The paper is well written, with clear organization and logical flow throughout. In particular, the results are effectively visualized—figures and tables are well designed, easy to interpret, and enhance the overall clarity and impact of the presentation.

### Weaknesses
- In the CARL framework, performance is highly dependent on the accuracy of the cost evaluation function. Inaccurate cost-to-go estimates can result in incorrect reward relabeling, which introduces significant uncertainty and may compromise both safety and reward performance. However, the paper does not include any sensitivity analysis to assess how estimation errors in the cost value impact the overall behavior of the learned policy, leaving an important aspect of robustness unaddressed.
- Although the paper reformulates the OSRL objective as an unconstrained optimization problem, it lacks a theoretical convergence analysis of the proposed iterative procedure, especially in the context of value function approximation. Providing such an analysis, or clearly outlining conditions under which convergence is guaranteed, would significantly improve the theoretical soundness and credibility of the method.
- This paper claims that CARL can combine with any other offline RL algorithm framework, but only two frameworks were tested — both are value-based, not generative or transformer-based (e.g., CQL, Diffuser, Decision Transformer) [1-3].
- This paper lacks comparisons with several recent and strong baselines, which makes it difficult to fully assess the claimed performance improvements and the novelty of the proposed method relative to the current state of the art [4-6].
- While CARL is designed as a minimalist wrapper, it introduces an additional value function training phase for cost estimation in each iteration. This added computational step—particularly when used with deep offline RL backbones—may impact training time and resource usage. However, the paper does not provide any analysis or empirical comparison of runtime efficiency, training wall-clock time, or computational complexity relative to baseline methods [7].
- As shown in Table 1, CARL occasionally fails to satisfy the cost constraint in certain tasks, and in some settings, it does not achieve the highest reward compared to other baselines. The paper would benefit from a deeper analysis of these cases to understand the conditions under which CARL underperforms, and to clarify the trade-offs between safety and performance [8].

### Questions
- How sensitive is CARL’s performance to errors in the cost-to-go estimator Qc?
- Are there theoretical guarantees that CARL will respect constraints under bounded Qc​ estimation error?
- Could you evaluate CARL’s compatibility with non–value-based offline RL frameworks such as CQL, AWAC, or Decision Transformer?
- Could you provide a performance comparison between CARL and additional recent baselines, including those in references [4–6]?

## Reference

**[1]** Janner, Michael, et al. "Planning with diffusion for flexible behavior synthesis." arXiv preprint arXiv:2205.09991 (2022).

**[2]** Chen, Lili, et al. "Decision transformer: Reinforcement learning via sequence modeling." Advances in neural information processing systems 34 (2021): 15084-15097.

**[3]** Kumar, Aviral, et al. "Conservative q-learning for offline reinforcement learning." Advances in neural information processing systems 33 (2020): 1179-1191.

**[4]** Wang, Ruhan, and Dongruo Zhou. "Safe Decision Transformer with Learning-based Constraints." 7th Annual Learning for Dynamics\& Control Conference. PMLR, 2025.

**[5]** Guan, Jiayi, et al. "Voce: Variational optimization with conservative estimation for offline safe reinforcement learning." Advances in Neural Information Processing Systems 36 (2023): 33758-33780.

**[6]** Wei, Honghao, et al. "Adversarially trained weighted actor-critic for safe offline reinforcement learning." Advances in Neural Information Processing Systems 37 (2024): 52806-52835.

**[7]** Chemingui, Yassine, et al. "Constraint-adaptive policy switching for offline safe reinforcement learning." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 15. 2025.

**[8]** Gong, Ze, Akshat Kumar, and Pradeep Varakantham. "Offline safe reinforcement learning using trajectory classification." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 16. 2025.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper proposes a minimal approach, named Constraint aware Reward (Re)Labeling (CARL), to translate offline safe RL to offline RL via reward shielding (relabeling). 

CARL contains the following content:

1. **Iterative Policy Improvement**
   1. offline policy evaluation $\to$ reward relabeling $\to$ offline policy optimization.
2. **Batch Reward Relabeling**
   1. Relabel the reward, which violates the constraint to a large negative value.

### Strengths
This paper has done comprehensive experiments on many benchmarks compared with latest baselines. 

This paper writes a detailed related work on offline safe RL.

This paper translates the constrained optimization problem into a shielding optimization problem, where the reward relabeling can be regarded as an improved shielding method in Safe RL.

### Weaknesses
### The key idea of this paper shares a very similar essence to CPQ [1]. However, the main difference between CPQ and CARL, and the improvement/superiority of CARL over CPQ are not explained clearly. The contribution may be limited before clarifying the following issue: 

1. Both CPQ and CARL share the same **Sketch of an Iterative Policy Improvement Algorithm** in Eq. (4) of this paper.
   1. In CPQ, it follows: offline policy evaluation $\to$ reward value relabeling $\to$ offline policy optimization.
   2. That is to say, CPQ already utilizes this approach in 2022.

2. **The only difference is that CPQ relabels the value to 0, while CARL relabels the reward to a large negative value.**
   1. CPQ utilizes value function relabeling:
      1. $Q_r=1_{\{Q_c\leq k\}}\cdot Q_r'$
      2. This is to say that CPQ relabels the unsafe condition as 0.
   2. While CARL utilizes reward relabeling in Eq. (3) and (5).
      1. $r_\pi(s,a)=1_{\{Q_c\leq k\}}\cdot r(s,a)-1_{\{Q_c> k\}}\cdot V_{max}$
      2. This is to say that CARL relabels the unsafe condition as $-V_{max}$.
   3. Consider the value function is a cumulation of rewards; there seems to be no large difference between relabeling rewards and relabeling the value function directly.
      1. Originally, the reward value follows: $Q_r = r+\gamma Q_r'$
      2. In CPQ, the reward value follows: $Q_r = r+\gamma \cdot 1_{\{Q_c\leq k\}}\cdot Q_r'$, which is
         1. $Q_r = r+\gamma \cdot 1\cdot Q_r'$ for $Q_c\leq k$
         2. $Q_r = r+\gamma \cdot 0\cdot  Q_r'$ for $Q_c> k$
      3. In CARL, the reward value follows: 
         1. $Q_r = r\cdot 1+\gamma \cdot Q_r'$ for $Q_c\leq k$
         2. $Q_r = -V_{max}+\gamma \cdot  Q_r'$ for $Q_c> k$
      4. Considering that the rewards are usually positive values (Even if they are not, the rewards can be regularized to be positive without loss of generality
         1. **Both CPQ and CARL aim to set the reward value that violates the constraint to a small value.**
         2. Is there an essential difference between relabeling to 0 or $-V_{max}$?
         3. Is there an essential difference between relabeling the reward signal or the reward value function?
   4. **This difference is hard to be regarded as the central contribution of a paper if it is not explained or compared clearly.**
3. CPQ naturally follows $K=M=1$, where the OPE and OPO are jointly optimized.
   1. What is the necessity of discussing the content of this part if this finding is already utilized in previous work?
   2. CPQ also suffers from the oscillation in Figure 1 even if $K=M=1$. Is there any explanation about the problem? Or are $K$ and $M$ really the main reason for this problem?
4. CPQ shares all **Summary of CARL’s advantages** in line 309, page 6, as they share a similar essence:
   1. CPQ can also be wrapped around existing offline RL algorithms.
   2. CPQ also doesn’t introduce any additional hyperparameters. (The additional hyperparameters in CPQ are related to another OOD action problem, which is not considered in CARL.)
   3. The main reason is that the only (or main) difference between CPQ and CARL is the above relabeling methods.  

5. Besides, CPQ additionally addresses the OOD action problem. However, the performance of CARL is much better to CPQ.
   1. Why does CARL not consider the OOD action problem, considering that the Bellman backup procedure is also utilized in CARL?

   2. Why CARL works much better than CPQ? 

6. It is very important to clarify that from Section 4 to 5,
   1. which content is already proposed in previous work (like CPQ)? 
   2. which content is completely innovative part of CARL? 
   3. why and how do the innovative parts improve CARL against CPQ?
   4. where does the performance gain come from?



### Although there is no doubt that TD3BC can be applied to CARL, there are still problems about how to apply IQL to CARL, which is glossed over in this paper. 

1. TD3BC has explicit policy during the training procedure. Thus, CARL can estimate $Q^\pi_r$ and $Q^\pi_c$ under the same policy. 
   1. The OOD problem can be mitigate by BC but can not be avoided as long as the Bellman backup procedure is updated with explicit policy.
2. To avoid the OOD problem completely, IQL propose to update the $Q^\pi_r$ implicitly via expectile regression, where the policy is implicitly hidden in the expectile regression to avoid explicitly sampling action during Bellman backup procedure.
3. Considering this implicit design, it is impossible to estimate $Q^\pi_r$ and $Q^\pi_c$ under the same policy without extracting the policy out.
   1. Think about why IQL regards "value function update" and "policy extraction" as two separate procedure.
      1. To avoid the OOD problem, the value function update cannot utilize the extracted policy.
4. This is why C2IQL [2] is proposed to to apply IQL's idea in Offline Safe RL to address this problem.
   1. C2IQL tries to combine CPQ and IQL without breaking the implicit property to avoid OOD problem.
   2. Since CPQ is similar to CARL, C2IQL is also similar to IQL+CARL, while C2IQL additionally points out and addresses this problem
   3. To understand the above content, please understand the idea of IQL, IDQL, C2IQL instead of merely treating them as an algorithm.
5. If this paper utilize the extracted policy to estimate the $Q^\pi_c$, it has no difference between TD3BC and IQL when applied to CARL. 
   1. Extracted policy breaks IQL's key idea. When this policy is utilized to influence value function update, IQL has no difference from other actor-critic method.
   2. Thus the backbone analysis in line 369, page 7 seems to be not very meaningful.
6. If this paper utilize framework that implicitly keeping both functions under the same policy like C2IQL, please explain the design in detail with proper citation.
7. Considering the similarity between C2IQL (CPQ+IQL) and CARL+IQL, please verify from section 4 and 5 again:
   1. which content is already proposed in previous work; 
   2. which content is completely innovative part of CARL; 
   3. why and how does the innovative parts improve CARL previous work;
   4. where does the performance gain come from;

[1] Xu, H., Zhan, X., & Zhu, X. (2022, June). Constraints penalized q-learning for safe offline reinforcement learning. In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 36, No. 8, pp. 8753-8760).

[2] Zifan, L. I. U., Li, X., & Zhang, J. C2IQL: Constraint-Conditioned Implicit Q-learning for Safe Offline Reinforcement Learning. In *Forty-second International Conference on Machine Learning*.

**We highly hope that this paper can clarify these issues, as they may potentially lead to academic plagiarism in severe cases.**

### Questions
Please see the Weakness part.

### Soundness
1

### Presentation
2

### Contribution
1

### Rating
2

### Confidence
5

---

## Human Reviewer 4

### Summary
This paper proposes CARL (Constraint-aware Reward Labeling), a simple offline safe RL method that enforces safety by relabeling rewards with large penalties for unsafe state-action pairs identified through cost estimation. Experiments on DSRL tasks show that it achieves better reward performance while satisfying safety constraints under small cost budgets.

### Strengths
The method is simple and can be easily integrated into existing offline RL algorithms without introducing new hyperparameters or major architectural changes.

### Weaknesses
The proposed method lacks theoretical convergence analysis and does not provide any formal guarantee of safety during test-time deployment.

### Questions
1. In Eq. (2), although the constraint is defined state-wise, it remains an expectation-based formulation, effectively a soft constraint since the Q-function represents the expected cumulative cost. In contrast, FISOR formulates offline safe RL with Hamilton–Jacobi reachability constraints that treat safety violations as hard constraints. Why is the formulation in Eq. (2) better at ensuring safety than the hard-constraint formulation?

2. When relabeling the reward, $V_{max}$ represents the maximum possible infinite-horizon value. Since only an offline dataset is available, $V_{max}$ is approximated as the maximum within the dataset, which can introduce bias. How does this bias affect policy learning and safety performance, and how do the authors mitigate or correct for this bias during training?

3. Regarding the results on varying cost limits, can the proposed method generalize to new cost thresholds without retraining, as methods like CAPS and CCAC can? If retraining is required for each threshold, how do the authors ensure a fair comparison? According to Figure 2, while it is reasonable that the normalized reward increases as the cost budget increases, why does the normalized cost decrease?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
4