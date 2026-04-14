# An Optimal Discriminator Weighted Imitation Perspective for Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 5, 6, 6, 6

## Abstract
We introduce Iterative Dual Reinforcement Learning (IDRL), a new method that takes an optimal discriminator-weighted imitation view of solving RL. Our method is motivated by a simple experiment in which we find training a discriminator using the offline dataset plus an additional expert dataset and then performing discriminator-weighted behavior cloning gives strong results on various types of datasets. That optimal discriminator weight is quite similar to the learned visitation distribution ratio in Dual-RL, however, we find that current Dual-RL methods do not correctly estimate that ratio. In IDRL, we propose a correction method to iteratively approach the optimal visitation distribution ratio in the offline dataset given no addtional expert dataset. During each iteration, IDRL removes zero-weight suboptimal transitions using the learned ratio from the previous iteration and runs Dual-RL on the remaining subdataset. This can be seen as replacing the behavior visitation distribution with the optimized visitation distribution from the previous iteration, which theoretically gives a curriculum of improved visitation distribution ratios that are closer to the optimal discriminator weight. We verify the effectiveness of IDRL on various kinds of offline datasets, including D4RL datasets and more realistic corrupted demonstrations. IDRL beats strong Primal-RL and Dual-RL baselines in terms of both performance and stability, on all datasets.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper proposes Iterative Dual-RL (IDRL), a new algorithm for solving offline RL. The paper claims that an iterative "filtering weight" for imitation learning outperforms other offline RL methods. This point can be understood with the well-known dual formulation of RL, and by iterative self-distillation, the authors argue that RL can gradually correct state-action distribution between train and expert datasets. To validate this claim, the authors have provided theoretical justification for IDRL techniques and experimental results to support their claims.

### Strengths
1. The paper is well-written; the core contribution is straightforward to understand and supported by the theoretical arguments (I did not fully read the proof line-by-line). 
2. This paper offers a novel perspective on offline RL. The authors successfully demonstrated that combining the dual formation of RL and imitation learning algorithms brings synergy to solve various tasks.
3. The paper contains realistic imitation learning experiments with corrupted datasets.

### Weaknesses
1. I believe the performance of IDRL is not stellar. Considering that the algorithm requires multiple dataset filtering steps, the performance gains from extra computation might not necessarily suggest the significance of the results.
2. Since it works with filtering, the algorithm might fail in scarcity of data. In MuJoCo, single demonstration imitation learning is a standard setting. In this case, I suspect IDRL's performance will converge with (single demonstration) behavior cloning performance.
3. An analysis scalability (such as computation costs) of various offline RL tasks should be reported and experimentally validated.

### Questions
1. Is the denoising process of diffusion-based offline RL (such as Diffusion-QL) similar to filtering datasets? How is IDRL conceptually different?

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
The authors point out that current Dual-RL methods incorrectly estimate the visitation distribution ratio. As a remedy, they propose a method to recover the true visitation distribution ratio by solving an OPE problem using Fenchel-Rockafellar duality. Additionally, they introduce a method to iteratively refine the offline dataset using the learned distribution ratio. They theoretically analyze the performance bound and the monotonic improvement property of the filtering procedure. The authors perform experiments on a gridworld toy case and the D4RL benchmarks to validate their claims.

### Strengths
1. This work theoretically demonstrates that semi-gradient Dual-RL only learns an action-distribution ratio, and derives a method for recovering the full state-action visitation ratio with tractable objectives.
2. The proposed iterative filtering procedure is supported by theoretical analysis and empirical evaluations.

### Weaknesses
1. There should be a comparison of compute costs (e.g., run time, memory usage), given the substantial amount of modifications introduced (e.g., additional updates and iterative dataset refinement).
2. The proof of Theorem 1 lacks clarity for readers not familiar with Fenchel-Rockafellar duality, as the authors have omitted some details (e.g., solving for $w^{*}(s)$). A more detailed explanation would be helpful.
3. Line 240 states that Deep RL algorithms are prone to overestimation errors caused by fragmented trajectories. And the authors claim that the proposed method avoids this issue (Line 448-449). However, this fragmentation effect does not seem to be supported by any theoretical/empirical analysis in the paper or in a previous work. Please cite relevant texts if any.

### Questions
1. Equation 12 shows that $w^{*}(s, a) = w^{*}(s) * w^{*}(a | s)$, which implies that state-action pairs filtered by $w^{*}(a | s)$ would also be filtered by $w^{*}(s, a)$. If $w^{*}(a | s)$ produces fragmented trajectories during dataset refinement, the trajectories produced using $w^{*}(s, a)$ will only be more fragmented. Also, from looking at Figure 2(e), it appears that using $w^{*}(s, a)$ produces incomplete trajectories as well. How does correcting the visitation distribution address the fragmented trajectory problem?
2. Line 240 states that Deep RL algorithms are prone to overestimation errors caused by fragmented trajectories. Is this conclusion based on a previous study? To the best of my knowledge, the "stitching" challenge (which is a task design factor of D4RL) requires offline RL algorithms to assemble sub-trajectories in order to solve a task [1].
3. (Line 263, 283) Which equation are you referring to? I assume it is Equation 9?
4. Does the "IDRL w/ $w^{*}(a | s)$" result in Table 2 apply the iterative refinement procedure? If so, does iterative refinement contribute negatively with $w^{*}(a |s )$? Without distribution correction, one might expect the algorithm to produce results similar to conventional Dual-RL methods (e.g., IQL). However, the average score in Table 2 seems to be significantly worse (56.8 vs. 77.8 of IQL on Mujoco). A more detailed ablation study may help.

[1] Fu, Justin, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. 2020. “D4RL: Datasets for Deep Data-Driven Reinforcement Learning.” _arXiv Preprint arXiv:2004.07219_. [http://arxiv.org/abs/2004.07219](http://arxiv.org/abs/2004.07219).

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
3

### Summary
This paper presents IDRL (iterative Dual RL), an algorithm for dual reinforcement learning that aims to solve two issues in current dual RL methods -- the semi gradient update and data regularized policy extraction. IDRL is a method which iteratively refines the dataset based on a trained discriminator. The paper proves both a theoretical iterative update guarantee and empirically shows that this method has superior performance compared to primal RL and dual RL offline methods.

### Strengths
- This paper does a good job with outlining the main issues with current dual RL algorithms and provides a theoretically grounded solution.
- While the idea is simple, it is well explained and well founded.
- The proposed method also has strong empirical results.

### Weaknesses
- It is unclear whether this method will suffer from poor generalization to other states which may have been ignored during dataset filtering.
- Further, this method seems to be computationally more expensive compared to other methods. It would be nice if this was discussed.

### Questions
- How does this policy generalize to states that were filtered out?
- How does filtering the dataset in round $k$, change the approximation of previously removed s,a pairs in later rounds?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a new framework, Iterative Dual-RL (IDRL), which utilizes an optimal discriminator-weighted imitation approach to enhance offline reinforcement learning (RL). This method iteratively refines the dataset to approximate the optimal visitation distribution by filtering out suboptimal transitions, thus aiming to overcome limitations of previous Dual-RL methods. IDRL is evaluated on D4RL benchmarks and several corrupted datasets, showing promising improvements in stability and performance over existing offline RL methods.

### Strengths
1.	The motivation of this paper is interesting and meaningful, which tries to combine offline RL with expert datasets.
2.	The proposed IDRL offers a novel discriminator-weighted imitation view that extends Dual-RL to better handle offline datasets by iteratively optimizing the dataset. 
3.	Detailed theoretical derivations and empirical validations make the methodology clear and support the proposed approach’s effectiveness. 
4.	The empirical results show that IDRL outperforms Primal-RL and existing Dual-RL methods on various benchmarks, indicating IDRL’s superior policy performance and dataset filtering effectiveness.

### Weaknesses
1.	This paper misses some literature in RL trained with weighted loss, such as EDP, and QVPO [1, 2].

[1] Kang B, Ma X, Du C, et al. Efficient diffusion policies for offline reinforcement learning[J]. Advances in Neural Information Processing Systems, 2024, 36.

[2] Ding S, Hu K, Zhang Z, et al. Diffusion-based Reinforcement Learning via Q-weighted Variational Policy Optimization[J]. arXiv preprint arXiv:2405.16173, 2024.

### Questions
1.	The reviewer believes the author should provide more explanation on how the additional variance introduced by the training of the U and W networks affects the overall stability of the algorithm. 
2.	In Algorithm 1, the reviewer wonders whether the W network is updated based on (12) rather than (10) on line 10? 
3.	The reviewer is confused about the value of M used in the experiments, and considers further clarification is needed here.

### Soundness
4

### Presentation
3

### Contribution
3
