# Self-Improving Robust Preference Optimization

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Online and offline $\mathtt{RLHF}$ methods, such as $\mathtt{PPO}$ and $\mathtt{DPO}$, have been highly successful in aligning AI with human preferences. Despite their success, however, these methods suffer from fundamental limitations: $\mathbf{(a)}$ Models trained with $\mathtt{RLHF}$ can learn from mistakes or negative examples through RL mechanism or contrastive loss during training. However, at inference time, they lack an innate self-improvement mechanism for error corrections. $\mathbf{(b)}$ The optimal solution of existing methods is highly task-dependent, making it difficult for them to generalize to new tasks. To address these challenges, we propose Self-Improving Robust Preference Optimization ($\mathtt{SRPO}$), a practical and mathematically principled offline $\mathtt{RLHF}$ framework. The key idea behind $\mathtt{SRPO}$ is to cast the problem of learning from human preferences as a self-improvement process, mathematically formulated as a min-max objective that jointly optimizes a self-improvement policy and a generative policy in an adversarial fashion. Crucially, the solution for this optimization problem is independent of the training task, which makes it robust to its changes. We then show that this objective can be reformulated as a non-adversarial offline loss, which can be efficiently optimized using standard supervised learning techniques at scale. To demonstrate $\mathtt{SRPO}$’s effectiveness, we evaluate it using AI Win-Rate (WR) against human (GOLD) completions. When tested on the XSum dataset, $\mathtt{SRPO}$ outperforms $\mathtt{DPO}$ by a margin of $\mathbf{15}$% after $5$ self-revisions, achieving an impressive $\mathbf{90}$% WR. Moreover, on the challenging Arena-Hard prompts, $\mathtt{SRPO}$ outperforms both $\mathtt{DPO}$ and $\mathtt{IPO}$ (by $\mathbf{4}$% without revision and $\mathbf{6}$% after a single revision), reaching a $\mathbf{56}$% WR against against $\mathtt{Llama-3.1-8B-Instruct}$.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors propose Self-Improving Robust Preference Optimization (SRPO) to address two of the main drawbacks of existing RLHF methods. That is, (a) models trained with RLHF can learn from mistakes or negative examples through RL mechanism or contrastive loss at the time of training. However at the time of inference they are not equipped with an innate mechanism to correct mistakes by self-improvement. (b) The optimal solution of existing methods is highly task-dependent and thus it is difficult for them to generalize to new tasks. SRPO overcomes these shortcomings, and its effectiveness is shown also empirically.

### Strengths
Although this is not my field of research, the paper was sufficiently clear. Its motivations are well-stated, and the results easy to grasp. They seem relevant to the RLHF literature.

### Weaknesses
Probably due to my unfamiliarity with this field, I am not able to find major weaknesses to the paper. Some minor considerations follow.

Line 62: "at the time of inference It would be very" should be changed to "at the time of inference, it would be very".

What is $\mathbb{R}^*_+$ in line 162?

Line 195: "Eq. equation 2" should be "equation 2". This typo is repeated throughout the manuscript.

### Questions
Is the problem studied by the authors related to continual learning under specific trade-offs [1]?

---

[1] https://arxiv.org/abs/2305.14782

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Self Improving Robust Preference Optimization (SRPO), a new method of LLM fine tuning. The authors propose using in-context learning to iteratively generate higher quality completions while being robust to the underlying offline dataset. The paper proposes a min-max objective which aims to maximize the quality of the improvement (of the improvement model) and minimize the amount by which you can be improved (for another language model). The paper then analyzes optimizations for each of the models independently and derive a convex combination of these losses. Finally, the authors experimentally verify their theory with experiments on TL;DR and XSum.

### Strengths
The observations made by this paper are well supported empirically. In particular, they use a nice synthetic bandit example to show robustness. Further, they experimentally verify on the TL;DR dataset and XSum on out of distribution examples. This paper further does a good job elucidating the derivation of the objective.

### Weaknesses
- The proposed method empirically performs similar to other methods under the same amount of inference compute.
- The paper does not provide much theoretical justification of the combination loss.

### Questions
- DPO and IPO both have "revisions"? Does this mean they received the same in context prompt as SRPO?
- The primary thesis of this work is that current preference optimization methods should not expect the ideal completion to be written and PPO/DPO/IPO are solving a fundamentally different task. In that vein, how does PPO compare under these same conditions? Is the lack of robustness also observed?
- Overoptimization has been observed in DPO and other direct alignment algorithms. Is this observed with SRPO as well? Why/Why not? [0]


[0] https://arxiv.org/abs/2406.02900

### Soundness
4

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
4

### Summary
This paper introduces a novel method, Self-Improving Robust Preference Optimization (SRPO), which aims to address two primary limitations in current preference optimization techniques. First, existing methods generally lack an inherent self-improvement mechanism at inference time, which limits adaptability. Second, they often rely heavily on the training task and distribution used to generate preference data, which affects the robustness of solutions. To address these issues, SRPO is designed as an offline preference optimization method that leverages a min-max formulation to learn a robust generative policy. This policy generates completions that require minimal adjustment when optimized through a self-improvement policy. The authors apply a DPO/IPO-inspired derivation to show that this min-max objective can be effectively optimized through standard supervised learning techniques. Experimental results on the TL;DR Summarization dataset indicate that SRPO achieves better in-distribution (ID) and out-of-distribution (OOD) performance compared to DPO and IPO baselines.

### Strengths
The paper is clearly written and easy to follow, with each step of the SRPO algorithm systematically derived.

The derivations provided in the paper appear rigorous, demonstrating a well-founded approach to preference optimization. The use of a min-max objective with a focus on self-improvement mechanisms is an interesting contribution.

### Weaknesses
The evaluation is currently limited to a single dataset (TL;DR Summarization) and only compares SRPO against two baselines, DPO and IPO. Conducting experiments on additional datasets would strengthen the empirical claims related to robustness.

The paper lacks (empirical) comparisons to more recent preference optimization methods, such as SimPO (Meng et al., 2024) and RPO (Liu et al., 2024), which integrate recent advancements over DPO. 

In Figure 3, SRPO achieves improved performance with revisions, likely due to its self-improvement mechanism. However, unlike DPO and IPO, which lack this self-correction feature, SRPO appears to incur additional inference costs to achieve its performance benefits.

(Meng et al., 2024) SimPO: Simple Preference Optimization with a Reference-Free Reward

(Liu et al., 2024) Provably Mitigating Overoptimization in RLHF: Your SFT Loss is Implicitly an Adversarial Regularizer

### Questions
Please see above.

### Soundness
3

### Presentation
3

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
This paper studies the problem of reinforcement learning from human feedback (RLHF). It designs a novel RLHF algorithm called SRPO based on training a robust self-improving policy by solving a min-max problem. It explains the math in detail on how to find the solution to this min-max problem. The experiment results show that SRPO achieves a higher winning rate against the gold dataset than existing state-of-the-art RLHF algorithms like DPO and IPO.

### Strengths
1. This paper studies the critical problem of finding a more efficient RLHF algorithm. It could have great potential in many important real-world applications.

2. The idea of using a self-improving policy in solving the RLHF problem is novel.

3. The mathematical explanation of how to solve the min-max learning problem they propose is thorough.

3. The SRPO algorithm aims at training a robust policy against the quality of the dataset, making it more trustworthy than many other RLHF algorithms.

4. The experiment design is reasonable in general, and the results look positive.

### Weaknesses
1. There is no theoretical guarantee of the learning outcome. This makes the whole theoretical part weak. Is there a chance to provide any theoretical guarantee on the performance of the policy learned by SRPO under some assumptions?

2. The design of SRPO algorithm is novel, but the description of its motivation can be improved. Currently, the motivation is that 'Instead, it is more natural to learn that given a query x and a completion y what would be the improved completion upon y'. However, in general, you can also say that to some other learning algorithms as long as they improve the quality of their policy to generate better completion after each training iteration. It is better to find some more unique motivation for the design of the SRPO algorithm.

3. The SRPO algorithm's winning rate against other algorithms is not provided. This paper only measures the winning rate against a gold-standard dataset. A policy that has a higher winning rate against a fixed dataset than that of another policy does not necessarily mean this policy is better than that policy. To verify that the policy learned by SRPO is better than the policies learned by other algorithms, it is necessary to compare these policies against each other directly.

### Questions
1. It can be beneficial to provide more explanation of the self-improving policy. This concept is not common in many RLHF literatures. Readers may be curious about how these policies work and how they are implemented. Providing more details can make this paper easier to follow.

2. Can you provide more details about the evaluation? For example, what are the parameters, such as the accuracy compared to humans, of the AI evaluation platform? Exactly how do we measure the winning rate between any two models? Revealing such details can make this work much easier to reproduce.

### Soundness
3

### Presentation
3

### Contribution
3
