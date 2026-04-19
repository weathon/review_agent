# Rethinking Behavior Regularization in Offline Safe RL: A Region-Based Approach

- Decision: Reject
- Scores: 3, 3, 5, 5, 3, 5, 6

## Abstract
Behavior regularization is a widely adopted technique in offline reinforcement learning (RL) to control distributional shift and mitigate extrapolation errors from out-of-distribution (OOD) actions by keeping the learned policy close to the behavior policy used to collect the dataset. However, directly applying behavior regularization to offline safe RL presents several issues. The optimal policy in safe RL should not only favor actions that prevent the agent from entering unsafe regions but also identify the shortest escape path when the agent finds itself in unsafe states. Enforcing safety and behavior regularization constraints simultaneously is inherently difficult and can often lead to infeasible solutions, especially when multiple constraints are involved. Furthermore, adding behavior regularization may cause the learned policy to imitate the behavior policy, even in states where the behavior policy performs poorly (not safe). This issue becomes particularly severe in offline safe RL, where the quality of the dataset collected by the behavior policy heavily impacts the learned policy’s effectiveness. To address these challenges, we propose $\textit{BARS}$ ($\underline{B}$ehavior-$\underline{A}$ware $\underline{R}$egion-Based $\underline{S}$afe offline RL), a novel algorithm that distinguishes between safe and unsafe states and applies region-specific, selective behavior regularization to optimize the policy. Extensive experiments show that BARS significantly outperforms several state-of-the-art baselines in terms of both rewards and safety, particularly in scenarios where the behavior policy is far from optimal. Notably, when dataset quality is low, BARS continues to perform well and ensure safety, while all other baselines fail to guarantee a safe policy in most of the environments. Our work has great potential to address a previously overlooked issue in offline safe RL.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes a safe offline RL method called BARS. The proposed method basically adds incremental modification to FISOR [1] and contains a number of theoretical flaws. The general framework and practical implementation are almost the same as FISOR. The only change is actually adding an additional term in the reward-maximizing objective, which can be ill-behaved. For detailed comments, please refer to the strengths/weaknesses.


[1] Zheng, Y. et al. Safe offline reinforcement learning with feasibility-guided diffusion model. ICLR 2024.

### Strengths
- The paper is easy to read.
- The reported results look good.

### Weaknesses
The paper lacks novelty and contains many technical issues:
- The paper shares too many similarities with FISOR, including paper structure, overall framework, and diffusion-based policy extraction. More specifically, 
  - The general framework (3) of decoupling safe offline RL to maximize reward in feasible regions and minimize cost in infeasible regions is the same as FISOR.
  - One change claimed by the author is to remove the behavior regularize $D(\pi || \pi_{\beta})$ in the cost value minimization in the infeasible region. However, the authors use the same policy extraction objective for infeasible regions as in FISOR, i.e., Eq. (12), this actually corresponds to $\pi^*\propto \exp(-\mu_2 A_c(s,a))\pi_{\beta}(a|s)$, which is exactly the closed-form optimal solution of $\max_\pi -V_c^\pi$, s.t., $D_{KL}(\pi || \pi_{\beta})$. Hence the authors actually **used** implicit behavior regularization in their infeasible region optimization objective, which contradicts with their claim and motivation.
  - The authors argued that the FISOR used HJ reachability to identify the feasible region, which has several issues (L233-243). However, in their algorithm, they used the **same** reverse expectile regression objective for feasible region identification method as in FISOR (Eq.(6-7)). The only difference is that they changed the symbol of discount factor $\gamma$, cost value $h$ in the FISOR paper to $\alpha$ and $c$ in this paper. How can you claim that you have addressed the limitation in FISOR, but still used exactly the same scheme? 
  - The introduction of a combined loss with two additional weight terms $b_1, b_2$ is entirely unnecessary, since it is equivalent to scaling the temperature parameter $\mu_1, \mu_2$. If you remove these two redundant hyperparameters, the policy extraction objective reduce to exactly the same as the one used in FISOR.
  - The only different design in the paper that I find different from FISOR is the additional term $Q_r^{\pi}/Q_c^{\pi}\cdot \mathbb{1}(Q_c^{\pi}>\delta)$ added to the reward maximization objective (Eq. (8)) in the feasible region. However, this is also highly problematic and lacks theoretical support. As $Q_c^{\pi}$ appears in the denominator, if $Q_c^{\pi}$ is small or takes 0, this term will explode. Since you are operating in the feasible region, ideally, the $Q_c^{\pi}$ should be 0 or a very small value, hence the optimization problem in the feasible region can be highly unstable and ill-conditioned. If one wants to avoid this, dense, non-zero cost signals need to be used, which severely limits the applicability of this method to many practical safe offline RL problems. As in many cases, we might only have sparse 0 or 1 cost signals to indicate safe/unsafe conditions, especially in complex problems where cost functions are hard to specify.
   - In summary, most parts of the algorithm actually replicate the original FISOR algorithm, while the only change is problematic.
- For the toy example in Figure 1, note that FISOR uses weighted behavior cloning ($\pi^*=\pi_{\beta}(a|s) \cdot w(s, a)$, with the weight $w$ take different forms for feasible and infeasible regions) to extract the policy. This means that if we have a higher proportion of unsafe data, we practically need to adjust the temperature parameters ($\alpha_1, \alpha_2$ in the FISOR paper, $\mu_1, \mu_2$ in this paper) to reduce the weight on the unsafe region, otherwise the imitative nature of policy extraction will cause too much learning on the unsafe data. Actually, this is exactly what the author did in their paper, they essentially introduced different loss weights $b_1, b_2$ (equivalent to adjusting $\mu_1, \mu_2$) to different tasks. In other words, the case motivated by the authors in Figure 1 can be entirely addressed by tuning the hyperparameter in FISOR. 
- The use of the "normalized divergence metric" $\bar{D}$ in Eq. (5) is problematic. If you look at most offline RL literature, the behavior regularization is imposed for each state, i.e., $D(\pi(a|s) || \pi_{\beta}(a|s))$ for each $s$, and the reward value maximization is also conducted for the given state. Here, the state is the input to the policy, and the $\pi(a|s)$ is a probability distribution over the action. You can only compute KL by integrating the action but not on states. No one used the $\sum_s D(\pi(a|s) || \pi_{\beta}(a|s))$ as the behavior regularization. If you do this, then the behavioral regularization will be the same for all states when you optimize $\max_{\pi} V_r^{\pi}(s)$ for a state $s$. The proof of Theorem 3.3 (Appendix C) is thus wrong due to this error. The $D(\pi || \pi_{\beta})\leq \epsilon$ in Eq. (16) is defined for each state, but the author mistakenly treats it as the sum of KL-divergence over all states. This makes Theorem 3.3 flawed and hardly provides any theoretical support for the proposed method.

### Questions
- What do you mean by "learning the **least safe** region"? Any non-safe region can be considered as the "least safe" region. To guarantee safety while maximizing the reward, we actually need to identify the largest feasible region.
- The authors use different loss weights $b_1, b_2$ for different tasks, basically equivalent to adjusting the temperature hyperparameter $\mu_1, \mu_2$ for different tasks. What are the results if you use only one set of hyperparameters? Note that FISOR only uses one set of hyperparameters in all experiments. If you tune the hyperparameter for each task, for sure you will get better results.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
BARS is developed to tackle challenges in offline safe reinforcement learning by applying selective behavior regularization according to the safety of states. This method shows improved performance in terms of safety and rewards compared to existing approaches, especially in scenarios involving suboptimal behavior policies and low-quality datasets.

### Strengths
- Learning a safe policy from an offline dataset is important and useful in real-world applications.
- Ensuring that every state meets safety requirements has practical needs.

### Weaknesses
## Key components lack theoretical analysis and proofs

- The author removes distribution constraints in unsafe areas to avoid limits that hinder obtaining safe actions. However, this might lead to OOD issues, which could still result in unsafe actions. How to resolve this remains an issue.
- It's unclear whether Equation 7 is the Hamilton-Jacobi (HJ) value function or the cost value function. If it is the cost value function, not having a discount factor could undermine the Bellman iteration, which needs discounting for convergence. If it's the HJ value function, it does not align with its usual definition[1]. Also, if it is the HJ function, how it ensures safety with different cost limits is not clear, given that HJ usually focuses on zero constraint violations. These issues suggest there might be errors in the training algorithm's formulation.
- It's unclear how Equation 5 transforms into Equation 9. Additionally, the training loss in Equation 12 lacks theoretical backing. Although it resembles the idea used in TRPO, it's uncertain if this approach is correct in a safe offline setting.

[1] Fisac, Jaime F., et al. "Bridging hamilton-jacobi safety analysis and reinforcement learning." *2019 International Conference on Robotics and Automation (ICRA)*. IEEE, 2019.

## Practical application issues

The training of the cost value function and the value function requires diffusion policy sampling, which is time-consuming. Additionally, the train-collect-train framework is problematic as it offers no safety guarantees during the collection procedure. This is in conflict with the goal of offline safe learning.

### Questions
- The largest feasible region shown in Figure 1 should not be related to data quality, yet it appears different in panels b, c, and d. Why?
- The cost threshold in the code is set to 8. How was this value determined?
- It's strange that in Table 3, fisor achieves different performances under different cost limits. fisor should consider hard constraints, meaning its performance should be independent of the cost limit, right?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper tackles challenges in offline safe RL caused by behavior regularization, which keeps the learned policy close to the behavior policy but can lead to unsafe actions when the behavior policy itself is suboptimal. The authors introduce BARS, an algorithm that distinguishes between safe and unsafe states and applies selective behavior regularization accordingly. BARS ensures safety and high performance even with low-quality datasets, outperforming current baselines in both reward and safety metrics.

### Strengths
(1) Interesting topics. Safety issues are important to Reinforcement Learning applications, and this paper has shed some insights on how to achieve safety in offline RL.
 
(2) Good insight and intuitive idea: Behavior regularization (data quality) indeed has a strong impact on the final performance of machine learning systems, as revealed in many data-centric learning works [1]

Reference:
[1] Chunting, Zhou, et al. "Lima: Less is more for alignment." NeurIPS 2023.

### Weaknesses
(1) Typo: Line 305, Liu et al. 2023b did not use a diffusion model to parameterize the policy network.

(2) Concerns about the experiment performance. In Table 1, some results of the proposed method raise concerns about its effectiveness. For example, in CarButton1 and SwimmerVel tasks, the rewards are negative. However, as reported in the original DSRL paper, some baselines such as CDT and Safe-BC should perform well with a high reward and low cost. The authors also did not report the comparison between these baselines.

(3) Confusion about the new setting: Safe offline RL with new Data Collection. Why did the authors discuss this part I believe this part should fall into the categories of continued learning or active learning. If the authors want to claim the contribution using this part, I believe baseline comparison for active learning/offline-to-online adaptive learning should be necessary.

(4) The proposed method shares many insights over the previous work: FISOR [1]. For example, using a diffusion model to parameterize the policy, using a weighted loss for training, and using control theory (feasibility) to help describe the algorithm and derive theorems. Thus, I suggest the authors to further clarify the difference between the proposed method and FISOR.

Reference:

[1] Yinan, Zheng,  et al. "Safe offline reinforcement learning with feasibility-guided diffusion model." ICLR 2024.

### Questions
(1) Could you clarify if you also used a weighted loss (Line 307) in baseline training (CPQ and COptiDICE) for a fair comparison?

(2) Can you provide more baseline results such as BCQ-Lag, CDT, and Safe-BC? As they are also implemented in the DSRL benchmark [1] you use, it would be easy to make a comparison.

(3) How do you choose the hyperparameters b1 and b2 in equation 13?

(4) Can you clarify what results you believe are not consistent with the FISOR paper as you stated in Lines 429-431?

(5) Is equation (13) the key component to mitigate the claimed regularization issues? If so, I believe more discussion about the extension of the proposed method to the general method including those that do not require diffusion policy such as BCQ-Lag and CDT is necessary. If not, I suggest the authors to further clarify and locate the core components to solve this issue.

(6) In Table 3, the authors show the experiments with different thresholds. The authors only show one experiment task and one baseline (FISOR). All the results are safe for BARS and FISOR, and the strength of the BARS is shown by higher rewards. However, based on the previous description “BARS is a novel algorithm that distinguishes between safe and unsafe states and applies region-specific, selective behavior regularization to optimize the policy.” (lines 24-26), the strength of BARS should be the ability to achieve safety under safety-critical situations. I expect the authors to further clarify the strength of BARS.


Reference:

[1] Zuxin Liu, et al. "Datasets and benchmarks for offline safe reinforcement learning." arXiv preprint arXiv:2306.09303 (2023).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces Behavior-Aware Region-Based Safe offline RL (BARS), an algorithm designed to enhance safe offline RL by distinguishing between safe and unsafe states and applying region-specific behavior regularization. BARS adjusts its approach based on the safety level of each state, maximizing rewards in safe regions and minimizing costs in unsafe ones. BARS is theoretically demonstrated to achieve higher rewards while satisfying safety constraints, and experimental results on standard benchmarks show its superiority over baselines in both reward and safety performance.

### Strengths
1. The motivation behind BARS is clear and relevant to safe offline RL, particularly in optimizing policies while ensuring safety.
2. The design of BARS appears reasonable, as it introduces a novel approach to behavior regularization that adapts to the safety level of each state.
3. Evaluation results indicate that BARS significantly improves rewards and reduces costs compared to state-of-the-art methods, especially as dataset quality varies.

### Weaknesses
The primary concern is that BARS seems to achieve safety by over-conservatism. By learning value functions under the "most conservative policy" to identify safe and unsafe regions, BARS could limit its performance. In general, such a conservative policy can be the policy with minimal action, e.g., an agent choosing not to move at all to stay in a safe state to avoid constraint violation. Additionally, the evaluation results in Section 5 seem to show that BARS achieves both low costs and low rewards, e.g., the average normalized reward is 0.282 in Table 1 and 0.195 in Table 2.

### Questions
1. Following the weakness, could the authors compare their method with 1) the top 20 safe trajectories in the dataset that satisfy the cost limit, i.e., 10, and 2) Behavior cloning that only uses the safe trajectories in the dataset (BC-Safe in DSRL) for a more clear comparison?
2. The authors use IQL to learn the value function and Q-value function of the most conservative policy (Eq.(6), and (7)). However, according to Eq.(6) and (7), the states and actions are simply sampled from the whole dataset $\mathbb{E}_{(s, a) \sim \mathcal{D}}$. Could the authors clarify how this update can result in the value functions of the most conservative policy? The learned functions appear to represent those of the behavior policy used to collect the dataset.
3. Could the authors clarify the purpose of introducing an additional threshold, $\delta$, in Eq.(8)? $\delta$ seems to act as another cost threshold, similar to $l$ in Eq.(4) and (5). How is this $\delta$ selected, and is it related to the cost threshold $l$?
4. The authors use standard bellman loss to learn the $Q$ functions of the current policy (Eq.(14) and (15)). However, directly applying this technique in the offline RL setting is known to suffer from OOD actions. Why are OOD actions not an issue for BARS?
5. Regarding the experiment involving new data collection (lines 470-483), could the authors provide more details to assess if the proposed method is deemed safe enough for exploration (line 475)? For example, how many trajectories are collected in each environment, and what is the cumulative cost, i.e., total costs, incurred during this process?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper studies behavior regularization in offline safe reinforcement learning. This paper argues that in unsafe regions where no safe policy exists, the optimized policy should not consider behavior regularization anymore. This is because (1) Enforcing safety constraints alongside behavior regularization often leads to conflicting objectives, making it hard to minimize the cost in unsafe regions; (2) Behavior policy can be very sub-optimal, strictly enforcing behavior regularization can be sub-optimal in this sense. To address these challenges, this paper proposes BARS (behavior-aware region-based safe offline RL), which firstly identifies safe and unsafe regions through expectile regression, and then discards the behavior regularizations for unsafe regions. The experiments on the DSRL benchmark demonstrate the effectiveness of the proposed method.

### Strengths
1. This paper is well-written, presenting a clear motivation for the problem setups.

2. Behavior-aware regularization is necessary for safe offline RL.

### Weaknesses
## The motivation is good but some method details are quite strange
1. Eq. (7) tries to ensure $(1-\alpha) c + \alpha(c + V_c)=Q_c$. This is not a common Bellman equation for the cost value Q and V function. Instead, this equation is similar to the one for the feasible value function but is still different: $(1-\alpha) h + \alpha \max (h, V_h) =Q_h$. Specifically, for the feasible value function, a maximization term exists, but in Eq. (7), the authors directly replace it as a summation term.

2. The definition of $A_r^\pi$ in Eq. (9) is somehow ad-hoc, solely providing some intuitive explanations.

3. It is strange that directly minimizing the $Q_c$ value in Eq.(11) will not lead to a large bootstrapping error accumulation. 

4. If not, this means that the policy under unsafe regions still stay near the behavior policy. So the introduction of TRPO-style optimization in Eq. (12) still tries to ensure a relatively relaxed behavior regularization, which contradicts the motivation of this paper that the behavior regularization for unsafe regions should be dropped.

5. In my view, the potential benefits of BARS are primarily attributed to the definition of $A_r^\pi$. Under its definition, the policy will exhibit a more conservative behavior to avoid unsafe regions and so is safer. For unsafe regions, it is quite hard for me to judge if the policy can obtain a reasonable behavior since no behavior regularization exists anymore and the policy can easily exploit the approximation errors of the Q_c value function. It would be better if the authors could show more rollout trajectories for BARS in Figure 1 when starting at one unsafe region.

6. The authors have identified the safe and unsafe regions through expectile regression, but still need to learn additional Q and Qc value functions through standard bellman update in Eq.(14-15). This can be unstable due to error accumulations and inefficient due to the costly diffusion sampling process. 

## Evaluations
7. Table 3 shows that FISOR produces different results for varied cost limits. However, FISOR is one cost limits-agnositc method that studies hard constraint and should not behave differently with different cost limits.

### Questions
Please see Weakness for details.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 6

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces BARS (Behavior-Aware Region-Based Safe offline RL), which differentiates between safe and unsafe states and applies selective behavior regularization tailored to each state’s safety level. BARS prioritizes reward maximization and adhers to safety constraints in safe states while minimizing risks in unsafe ones. Experimental results show that BARS outperforms existing methods, achieving higher rewards and better constraint satisfaction across varying data quality levels.

### Strengths
- The paper is supported by both theoretical and experimental results.
- Solving different optimization problems for safe states and unsafe states sounds effective to me in the offline safe RL setting.
- The issues of infeasible solutions and poor data quality identified in the paper are important.
- The experimental setting in Figure 1 (safe offline RL with new data collection) appears to have high practical value.
- The empircal performance of BARS appears good.

### Weaknesses
- I am not quite certain about the contributions of this work compared to FISOR, especially before noticing Appendix B. Since BARS and FISOR are closely related, it would enhance the clarity of the paper to include this discussion in the main text, if space permits. Additionally, pointing out which component or improvement of BARS, in comparison to FISOR, contributes the most to its performance would further clarify the contributions of the paper.
- While the gap between theoretical and practical is inevitable, the authors are encouraged to elaborate on the differences between the practical algorithm and the theoretically analyzed algorithm (Eq. (5)).

[1] Zheng et al. Safe Offline Reinforcement Learning with Feasibility-Guided Diffusion Model.

### Questions
- Table 4 shows that policy restriction increases reward at the risk of higher cost. It may be difficult to determine which one is better. How do the authors recommend balancing the increased reward against the higher cost when using policy restriction?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 7

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a novel algorithm, BARS, aimed at addressing the issue of behavior regularization in offline safe reinforcement learning. The authors effectively tackle the challenge of learning safe strategies in unsafe states by differentiating between safe and unsafe states and applying region-based selective behavior regularization to optimize the policy. 

The authors start from **two main drawbacks** of prior offline safe RL methods: one is the importance of finding less harmful action, and the other is the imitation of behavior policy. The method is based on FISOR [1], a SOTA method in offline safe RL. It first finds the largest safe region and an absolute unsafe region by HJ-reachability. Then, it further separates the safe region into absolute safe and margin safe by using a threshold. A **main improvement** is removing the behavior constraint in the unsafe region to better learn a safe policy. The experiments involving different coverage of datasets demonstrate its effectiveness.

However, the experiment lacks analysis of hyperparameters and some derivations are confusing.

Reference:
[1] Zheng, Y., Li, J., Yu, D., Yang, Y., Li, S. E., Zhan, X., and Liu, J. (2024). Safe offline reinforcement learning with feasibility-guided diffusion model. arXiv preprint arXiv:2401.10700.

### Strengths
- The paper removes behavior cloning in unsafe regions, which is straightforward, effective and relatively novel.
- It solves the main weaknesses of some prior work and clearly explains how to improve previous methods. 
- The authors improve the previous optimization problem and replace it with a new one. They also provide a theorem as to why it is better than the original one.
- The main experiments show the superiority of BARS and ablation studies provide interesting findings.
- The additional experiments (adding restrictions when updating the policy and different data collection) are interesting. It explores the performance under different behavior policies and extends it to more practical experiments.

### Weaknesses
- The overall method is based on FISOR, which lacks some novelty.
- Extensive experimental results indicate that BARS seems to struggle to balance the reward and the cost. While the main results in some tasks have lower costs with lower rewards (e.g. CarGoal1), the results in other tasks show higher rewards and costs (e.g. AntVelocity).
- The introduction section lacks a summary of contributions, which would help readers quickly understand the paper. (minor suggestion)
- I suspect the good performances are due to adjustment of hyperparameters (such as safe loss control).
- The paper lacks a Limitation section which is important.
- Some derivations of the formulas seem to be confusing. See the Question for details.

### Questions
1. In line 264, Why can IQL be used to obtain results $V_c(s) = \min_a Q_c(s,a)$?

2. In line 270, expectile regression in IQL is $L^{\tau}(u) = |\tau - 1(u<0)| u^2$. Why can it use $u>0$?

3. In line 300, why does $A^{\pi}_r(s,a)$ have this form? Why do not use the advantage function?

4. In line 318, if there are no distributional shifts, how can equation 11 be deviated? (I think there is no exp.)

5. Since the method introduces new hyperparameters (such as threshold $\delta$, trust episode $K_{trust}$), how do they affect the performances? The threshold $\delta$ is not contained in Table 7.

6. In the Algorithm 1, what's the meaning of $k$?

Recommendation:
Due to the absence of significant inherent errors in the method and some experimental settings are quite interesting, I would like to give 'borderline accept'.

### Soundness
3

### Presentation
3

### Contribution
2
