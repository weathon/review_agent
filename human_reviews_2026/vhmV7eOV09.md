# Exploiting Reflectional Symmetry in Heterogeneous MORL

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
This work studies heterogeneous Multi-Objective Reinforcement Learning (MORL), where objectives exhibit considerable discrepancies in, amongst others, sparsity and magnitude. The heterogeneity can cause dense objectives to overshadow sparse but long-term rewards, leading to sample inefficiency. To address this issue, we propose Parallel Reward Integration with reflectional Symmetry for heterogeneous MORL (PRISM), a novel algorithm that aligns reward channels and enforces reflectional symmetry as an inductive bias. We design ReSymNet, a theory-inspired model that aligns time frequency and magnitude across objectives, leveraging residual blocks to gradually learn a `scaled opportunity value' for accelerating exploration while maintaining the optimal policy. Based on the aligned reward objectives, we then propose SymReg, a reflectional equivariance regulariser to enforce reflectional symmetry in terms of agent mirroring. SymReg constrains the policy search to a reflection-equivariant subspace that is provably of reduced hypothesis complexity, thereby improving generalisability. Across MuJoCo benchmarks, PRISM consistently outperforms the baseline and oracle (with full dense rewards) in both Pareto coverage and distributional balance, achieving hypervolume gains of over 100\% against the baseline and even up to 32\% against the oracle. The code is at \url{https://anonymous.4open.science/r/reward_shaping-1CCB}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper addresses heterogeneous Multi-Objective Reinforcement Learning (MORL), where different reward channels vary significantly in sparsity and magnitude, leading dense objectives to overshadow sparse but crucial long-term ones. The authors propose PRISM, a novel framework that enhances sample efficiency and generalisation in such settings. Theoretical analysis shows that PRISM constrains the policy search to a lower-complexity, reflection-equivariant subspace, leading to tighter generalisation bounds.

### Strengths
1. The paper introduces a fresh perspective on heterogeneous MORL, an underexplored yet realistic challenge. The combination of reward alignment through residual networks and symmetry-based inductive bias is novel and creative. Using reflectional symmetry as a regulariser for MORL agents is particularly original, bridging theoretical symmetry principles with multi-objective RL.
2. The work is thorough, combining rigorous theoretical analysis (covering-number and Rademacher-based generalisation bounds) with extensive experiments and ablation studies.
3. The proposed approach addresses a practical and fundamental problem in MORL—reward heterogeneity—which often arises in robotics and control. The demonstrated ability to outperform even dense-reward oracles suggests real impact for efficient multi-objective learning in sparse or imbalanced settings.

### Weaknesses
1. The reflectional symmetry assumption may hold for certain robotic domains (e.g., locomotion) but is less applicable to asymmetric or high-dimensional tasks. The method’s generality could be questioned—evaluating on more asymmetric tasks would clarify its broader utility.
2. ReSymNet relies on at least one dense reward signal to align sparse ones, which limits applicability in fully sparse or partially observable scenarios. A discussion or ablation on settings without dense channels would strengthen the claim of general applicability.
3. The notation is somewhat inconsistent and insufficiently defined throughout the paper, such as $D$, $r_t^{sp}$, and parameters like $\psi$ used in Eq. (1) are either not introduced clearly.
4. The problem is formulated as a multi-objective RL task, yet much of the theoretical and analytical development is effectively single-objective. The generalisation analysis, reflectional-equivariance projection, and covering-number proofs are all framed in terms of a scalar return rather than vector-valued multi-objective returns.

### Questions
See the Weaknesses.

### Soundness
3

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
This paper proposes an algorithm called PRISM to address the overshadowing problem in heterogeneous multi-objective reinforcement learning (MORL), where some objectives are dense while others are sparse. The proposed architecture, ReSymNet, performs reward decomposition for sparse objectives using residual blocks. By learning pseudo-dense signals for these sparse objectives, the method enables policy training within a standard MORL framework. To enhance policy generalization, an auxiliary loss is incorporated into the policy loss to enforce reflectional equivariance. Experimental results demonstrate that integrating the proposed method into a standard MORL algorithm (specifically, CAPQL) improves performance according to several widely used MORL metrics.

### Strengths
- Provides theoretical support.

- Explores an interesting direction in MORL.

- Modular design: can be equipped with any (multi-policy) MORL algorithm.

### Weaknesses
1. Lack of comparison with reward decomposition methods in delayed-reward RL

- To my surprise, the paper does not adequately compare the proposed approach with existing reward decomposition methods [A,B,C,D], both in the literature review and in experiments. A simple extension (e.g., adding the dense reward component as an additional input) in [A,B,C,D] could make such experimental comparisons feasible in MORL settings. Comparison might even be possible without modification, since “state-action features already contain substantial signal (Line 465)”. The authors should review and discuss the relevant works including [A,B,C,D], clarifying whether the only difference lies in architectural choices.

 -  A. https://arxiv.org/pdf/2410.20176 
    B. https://arxiv.org/pdf/2402.03771 
    C. https://arxiv.org/pdf/2111.13485
    D. https://arxiv.org/pdf/2010.12718

2. Unclear motivation for SymReg and its connection to ReSymNet

- 2-1. The motivation for incorporating SymReg alongside ReSymNet is unclear, as they appear to represent orthogonal approaches. An ablation study combining ReSymNet with other well-known policy feature learning or regularization methods would help clarify the contribution. 

- 2-2. The authors should also discuss why SymReg synergizes with the ReSymNet architecture, whereas other representation methods may not. 

3. Manual decomposition of $s_{asym}$ and $s_{sym}$ 
​
- 3-1. It seems that $s_{asym}$ and $s_{sym}$ are manually decomposed for each environment, which raises concerns about the generality of the proposed method.
- 3-2. Can we also consider more general augmentation other than mirroring in the PRISM framework?

4. Lack of key experiments for ablation study

- 4-1. For sanity check, the authors should include (i) a uniform reward distribution (i.e., 1/T * last-reward for each timestep), and (ii) a random reward generation.

- 4-2. The paper should visualize the final 2D/3D Pareto fronts of each algorithm in addition to the quantitative results in Table 1 (similar to Fig. 4). Plus, using these visualizations, please discuss why the proposed method outperforms the oracle.

- 4-3. It would be helpful to qualitatively analyze the behavior of the learned reward signals, similar to Fig. 5 in reference [A].

- 4-4. Since the proposed method is modular, applying it to other MORL algorithms beyond CAPQL could further demonstrate its generalizability.

- 4-5. Testing in environments with different $p_{\text{rel}}$ values would also strengthen claims of generalization.

- 4-6. The first objective in each environment corresponds to the velocity component. It would be meaningful to investigate whether similar improvements occur when delaying other quantities, such as energy (i.e., the second objective).

5. Need for truly sparse reward settings

- It is interesting that the “w/o refinement” variant performs moderately well. This may be because non-zero non-trivial rewards are still provided at the end of episodes in delayed reward settings. To better reflect the truly sparse nature of certain objectives, the authors could consider other sparse environments in nature. For example, MO-Lunar-Lander [E], where the first objective represents the success or failure of each trajectory, could serve as a suitable testbed.

- E. https://openreview.net/forum?id=jfwRLudQyj

6. Additional issues

- How to split D train and D val in the algorithm? 
- If the authors used MO-Gymnasium, please cite reference [E] in addition to the original MuJoCo citation.
- In the related work section, papers categorized as “(iii) meta-policy” could also be considered “multi-policy” in many MORL studies. Note that in the MORL community, the term meta-policy is sometimes reserved for works that explicitly adopt meta-learning approaches, such as [F].
- Minor: A left parenthesis is missing in Line 722.
- F. https://arxiv.org/pdf/1811.03376

### Questions
Please see the above weaknesses.

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
In multi-objective reinforcement learning (MORL), the decision maker's utility function is typically unknown. As such, MORL algorithms aim to find policies for each of the optimal trade-offs over the objectives. Often, the utility function is assumed to be a weighted sum over the objectives, with unknown weights. However, these scalarization functions are known to be very sensitive, with a small change in the weights potentially leading to drastically different policies [1]. They are also heavily impacted by the difference in magnitudes of different objectives. Moreover, not all objectives appear as frequently. Some might be sparse, while others are dense, making the learning of optimal policies more difficult. This work tackles this problem, by proposing a resnet-like network for reward shaping, that can be plugged in any MORL algorithm to train on the shaped rewards. Additionally, they propose a symmetry-enforcing regularizer that allows for jointly learning symmetric motions.

### Strengths
Reward sparsity and scaling differences result in optimization challenges in MORL. A generic reward shaping method that can be applied to many MORL algorithms could have a high impact.

### Weaknesses
However, I have some concerns, which I list here:
- My main concern is a presentation issue. The paper proposes 2 completely different ideas (reward shaping and symmetry preservation) packaged into one algorithm. There does not seem to be any special synergy between the 2 components, and the paper does not describe a specific interaction where one component's mechanism enables or enhances the other. The main challenge presented in this work is, to my understanding, the reward sparsity and magnitude. I would be curious to understand how the symmetry component helps to mitigate this issue, as it is not clear to me.
- The symmetry component here seems to be specific to Mujoco, and the symmetric and asymmetric sets are hardcoded. Knowing these sets in advance seem like a strong assumption, making its use limited in practice.
- The abstract mentions that ReSymNet aligns time frequency and magnitude across objectives. While it is clear ReSymNet tackles time frequency (Eq1), it is not so clear for the magnitude. Since the individual reward predictions need to sum to the environment return, how are the magnitudes impacted? Def7 (line 688) mentions a scale parameter $k$ applied on the TD error, but it is my understanding that the reward predictions are used as-is, in any MORL algorithm. I do not see how $k$ can be used in CAPQL.
- Although this work tackles sparsity, the sparsity is enforced artifically, by providing zero-rewards at intermediate timesteps, and then providing the sum-or-rewards at the terminal timestep (in the extreme case, line 126). But this breaks the Markov property, since the reward $r_t$ now not only depends on the current state $s_t$, but also on the past visited states of the trajectory $s_{0:t}$. Many of the results show how much worse CAPQL is on the sparse reward variant. But, as it stands, it is impossible to know how much this is due to sparsity compared to non-markovian rewards. A solution would be to augment the state-space with the accumulated rewards (like [2]) or, better, to use benchmarks that naturally mix sparse and non-sparse rewards. One example is the Minecart environment [3] (implemented in MO-Gymnasium), where fuel is consumed at each timestep, but the ore values are provided once the agent returns to base.
- I wanted to check the code to make sure I understood the method, but the anonymous repo mentions "requested file is not found" for all the files in `src/reward_shaping`.

### Questions
Also, I noticed in Table 10 that the "w/o loss" variant seems systematically better than the oracle. This surprises me, since the non-sparse rewards are learned over time, while the oracle uses the ground-truth dense rewards. Could the authors explain why this happens?

Overall, the tackled issue is important for the MORL community, but only half of the proposed method actually tackles sparse rewards. Moreover, the difference in magnitude does not seem to be addressed, and the reward sparsity in the experiments breaks the Markov property.

[1] Vamplew, P., Dazeley, R., Berry, A., Issabekov, R., & Dekker, E. (2011). Empirical evaluation methods for multiobjective reinforcement learning algorithms. Machine learning, 84(1), 51-80.

[2] Reymond, M., Hayes, C. F., Steckelmacher, D., Roijers, D. M., & Nowé, A. (2023). Actor-critic multi-objective reinforcement learning for non-linear utility functions. Autonomous Agents and Multi-Agent Systems, 37(2), 23.

[3] Abels, A., Roijers, D., Lenaerts, T., Nowé, A., & Steckelmacher, D. (2019). Dynamic weights in multi-objective deep reinforcement learning. In International conference on machine learning (pp. 11-20). PMLR.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes PRISM, an algorithm for dealing with reward imbalance in MORL objectives. It utilizes supervised learning of a dense reward model, as well as a symmetry aware loss term, to improve performance in cases where different objectives of the MORL task differ in sparsity. The paper presents experiments comparing the proposed algorithm to ablations on the method, finding improvement in a few MuJoCo multi-objective locomotion tasks.

### Strengths
The various sections of the paper are well written and easy to read. 

The motivation behind the proposed method is valid and the authors do a good job of explaining how their contributions aim to solve the presented problem. 

The theoretical analysis is clear and easy to follow, and while it is not surprising that introducing symmetry simplifies the solution space, it is good to see a non-theory-focused paper where the “obvious” algorithmic improvement isn’t taken for granted, but instead rigorously proven.

### Weaknesses
My central concern with this paper is that it feels like two separate papers attempting to be condensed into one. This lack of consistent storyline hurts both aspects of the paper’s contribution.  

Both the introduction and the related work sections are unclear on the connection between the heterogeneous rewards problem in MORL and the conversation regarding reflectional symmetry. At first glance, these seem orthogonal (i.e. each relevant in a specific subset of problems which may or may not overlap), and these first two sections do nothing to draw the connection between the two. In particular, for the symmetry side of the paper, the related work section is lacking in references to approaches dealing with symmetry - a plethora of methods exist in geometric DL and adjacent fields, which seem to be neglected here. 

This issue persists throughout other sections of the paper - it seems like the idea of heterogeneous reward mitigation and reflectional symmetry are two orthogonal contributions of this paper, mashed together. Section 4.2 discussing symmetry is a sharp pivot from section 4.1 discussing reward shaping, with no apparent connection.

In the same manner, the theoretical analysis section, while sound, seems like it deals with the symmetry regularization objective only, and not with the reward learning network. 

In the experiments section, the focus shifts back to MORL, with symmetry in the state and action spaces treated as an afterthought. 

Some more issues with the experiments section: 

- The comparison is lacking in baselines - while the oracle and baseline comparisons defined in the paper are decent basic comparisons, a glaring omission is other baselines dealing with reward heterogeneity in reward signals in MORL.
- It is unclear whether the baseline and oracle models utilize the symmetry aware loss or not: are the gains over the compared methods obtained by the dense reward prediction learning or by the symmetry loss?
- At the bottom line, it is hard to tell whether results are statistically significant  over the oracle - especially in the halfcheetah and swimmer environments.

Some other concerns are listed as questions below.

### Questions
1. How does a purely random policy (line 161) ensure broad state-space coverage? Wouldn’t this depend on the structure of the state space and the parameters of the random policy? Clarification on what “purely random” means in this context would be helpful. 
2. Equation 1: how does the sparse objective operating on the sum of trajectory rewards ensure shaping of rewards per step? One optimal solution could be to provide the entire reward at the last step. What enforces the distribution of rewards along the steps of the episode?
3. Section 4.2: do the symmetric and asymmetric parts of the state and action spaces have to be defined manually for each task? If this is the case (as is implied by the tables in appendix D), how scalable is this method to more realistic robotic scenarios and more complex tasks?
4. What are the various objectives for the multi-objective MuJoCo tasks? What correlations exist between them?

### Soundness
3

### Presentation
2

### Contribution
2
