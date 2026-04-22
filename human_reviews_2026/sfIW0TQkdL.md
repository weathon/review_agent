# Boosting Multiagent Reinforcement Learning at High Replay Ratios with Ensemble Reset

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 2

## Abstract
Reinforcement learning with a high replay ratio, where the agent's network parameters are updated multiple times per environment interaction, is an emerging way to improve sample efficiency. However, this paradigm remains underexplored in multiagent reinforcement learning (MARL). In this paper, we investigate how to efficiently train MARL at high replay ratios to accelerate learning. Surprisingly, we found that simply increasing the replay ratio leads to severe dormant neurons in the centralized global Q-value network, where neurons become inactive thereby undermining network expressivity and hindering the learning of MARL. To tackle this challenge, we propose Ensemble Reset (EnSet) to boost MARL at high replay ratios from two aspects. First and for the first time, EnSet utilizes an ensemble of global Q-value networks with parameter reset to reduce dormant neurons when updated at a high frequency. Second, EnSet diversifies the replay experience using a multiagent translation invariance prior of the global Q-function to prevent overfitting. Extensive experiments in SMAC, MPE, and SMACv2 show that EnSet substantially speeds up various MARL algorithms at high replay ratios.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The proposed paper introduces an algorithm named EnSet to improve sample efficiency in multi-agent reinforcement learning environments. To achieve this, EnSet employs an ensemble of global Q-value networks and periodically performs a reset operation to reduce dormant neurons. Furthermore, it utilizes translation invariance to prevent overfitting of the global Q-value function.

### Strengths
* The paper is easy to understand.
* EnSet can be applied to various algorithms and demonstrates improved sample efficiency.
* The effectiveness of EnSet is verified across diverse environments.

### Weaknesses
* The paper attributes improved sample efficiency to the combination of ensemble, reset, and multi-agent translation invariance. However, given that the reset mechanism is adopted from prior work [1] and the empirical impact of the multi-agent translation invariance appears marginal in the results, the primary contribution of the paper seems reduced to the mere application of a standard ensemble technique.

* While achieving strong performance with fewer environment steps is intriguing, this approach, coupled with the ensemble method and a high replay ratio, likely incurs a substantial computational overhead (e.g., training time). A thorough analysis quantifying the computational cost vs. performance gain trade-off is currently missing and required for a complete evaluation.

* The overall clarity of the paper's narrative is inconsistent in specific sections: (1) The purpose and relevance of the content related to Equation 1 in Section 2.2 are ambiguous, particularly as the concept is not subsequently utilized, suggesting it may be superfluous. (2) The third paragraph of Section 4.1 requires significant revision for structural clarity and flow.

* The periodic 'Shrink & Perturb' operation is not clearly defined. It is uncertain whether this mechanism is applied to each individual network ($\phi^{h}$ and $\theta^{i}$) within the ensemble or to the ensemble as a whole($\phi^{1},\cdots,\phi^{H}$ and $\theta^{1},\cdots,\theta^{N}$), which hinders a complete understanding of the algorithm's functional mechanism.

* The experimental results lack justification for using different evaluation metrics (median with 25-75% percentiles and average with 95% confidence interval) across various experiments. Furthermore, the excessive smoothing applied to several plots significantly diminishes the graphical fidelity and reliability of the presented data.

* A comprehensive table summarizing all crucial experimental setup parameters and algorithmic hyperparameters would greatly enhance the reproducibility and clarity of the work.

&nbsp;

[1] Yang, Y., Chen, G., & Heng, P. A. (2024, July). Sample-efficient multiagent reinforcement learning with reset replay. In Forty-first international conference on machine learning.

### Questions
* Figures 2 and 4 suggest that merely reducing dormant neurons is insufficient to guarantee performance improvement. Could the authors elaborate on why this occurs, and, in light of these specific control results, what is the mechanism by which EnSet successfully translates reduced dormant neurons into superior performance?

* Considering the goal of mitigating dormant neurons, the 'redo' [2] mechanism has been explored in related literature as an alternative to 'reset'. Could the authors provide experimental results showing the performance of the proposed algorithm when the 'Shrink & Perturb' component is replaced solely with the 'Redo' technique while maintaining other components?

* Could the authors quantify the difference in computational cost (e.g., wall-clock time, GPU/CPU utilization) between the baseline algorithm and the proposed algorithm?

* Previous research [3] in single-agent RL utilizes ensemble methods and reset mechanisms, demonstrating improved sample efficiency as the replay ratio increases, which feels similar to EnSet. What is the authors' opinion on comparing EnSet with this method?

&nbsp;

[2] Sokar, G., Agarwal, R., Castro, P. S., & Evci, U. (2023, July). The dormant neuron phenomenon in deep reinforcement learning. In International Conference on Machine Learning (pp. 32145-32168). PMLR.

[3] Kim, W., Shin, Y., Park, J., & Sung, Y. (2023). Sample-efficient and safe deep reinforcement learning via reset deep ensemble agents. Advances in neural information processing systems, 36, 53239-53260.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the challenge of improving sample efficiency in multiagent reinforcement learning (MARL) by utilizing high replay ratios. It introduces a novel method called Ensemble Reset (EnSet), which combines an ensemble of global Q-value networks with parameter resets to mitigate the issue of dormant neurons that arise when training at high replay ratios. The authors assert that this approach enhances the training efficiency of various MARL algorithms in environments such as SMAC (StarCraft Multi-Agent Challenge) and MPE (Multiagent Particle Environment).

### Strengths
1. This paper provides a new perspective on enhancing sample efficiency by addressing the dormant neuron phenomenon
2. The authors conduct extensive experiments across multiple environments (SMAC, MPE, and SMACv2) and a variety of tasks, demonstrating the effectiveness of EnSet. The results show that EnSet significantly improves the performance of existing MARL algorithms at high replay ratios.
3. The identification and analysis of the dormant neuron phenomenon in the global Q-network is a crucial finding. The paper effectively links this issue to the challenges of high replay ratios, providing a clear motivation for the proposed solution.

### Weaknesses
1. The distinction between MARR and EnSet is not clearly explained. It seems like an ensemble version of MARR with added translation, but the ablation study shows that translation does not contribute significantly.
2. It’s not clear that whether the dormant neuron phenomenon only occur in the global Q-network.

### Questions
1. Please provide a more detailed explanation of the key distinctions between MARR and EnSet, particularly regarding their motivations, design choices, and the specific problems they aim to solve.
2. We recommend adding ATM(RR=10) and similar baselines to Figure 3 for direct comparison.
3. Why do all the comparison methods in Figure 4 choose to build on QMIX rather than ATM? We recommend adding this experiments.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates training MARL algorithms under high replay ratios. The authors identify that increasing the replay ratio exacerbates the dormant neuron phenomenon in the centralized global Q-network, which leads to representational collapse and unstable learning. To address this, they propose EnSet, combining an ensemble of global Q-networks with periodic parameter reset (Shrink & Perturb) and a data augmentation method based on multi-agent translation invariance.
Comprehensive experiments across SMAC, MPE, and SMACv2 demonstrate that EnSet stabilizes training and improves sample efficiency for various MARL algorithms at high replay ratios.

### Strengths
1. The paper addresses an important problem: how to effectively scale MARL to higher replay ratios for improved sample efficiency, which is increasingly relevant for large-scale and expensive environments.

2. Experiments span multiple benchmarks and show consistent improvements across algorithms and replay ratios. The dormant neuron visualizations and gradient analyses are insightful and help justify the design choices.

3. The paper connects dormant neuron growth with high replay ratios and empirically shows that ensembles and resets mitigate this effect.

### Weaknesses
1. Overstated novelty

The dormant neuron issue in QMIX has already been discussed in [1]. Although this paper focuses on high replay ratios, it is not accurate to claim being the first to identify dormant neurons in global Q-networks.
The Shrink & Perturb reset mechanism originates from [2]. This paper reuses the idea without substantial conceptual innovation.
In addition [3] also demonstrates that ensembles can improve QMIX’s performance in SMAC and SMACv2.

2. Limited training horizon and convergence analysis

Most experiments stop at 1M environment steps to emphasize sample efficiency, but many methods (including the baselines) have not yet converged. Showing converged performance or stability at longer horizons would help assess whether EnSet maintains strong final performance and robustness rather than only accelerating early learning.

3. Framework inconsistency

The paper adopts pymarl2 for SMACv2 experiments but uses the older pymarl implementation for SMAC. Since pymarl2 provides stronger and more optimized baselines, the comparison on SMAC may not fairly reflect the advantage of the proposed method.

4. Unclear analysis of ensemble benefits

The paper states that ensembles are applied only to global Q-networks because local Q-networks do not suffer from dormant neurons. However, in experiments, MADDPG’s critic (which corresponds to individual Q-value) also benefits from ensembles.
This raises an important open question: is the dormant neuron phenomenon primarily due to high-dimensional joint action/state input or replay ratio scaling? A deeper analysis of when and where the problem arises would strengthen the contribution.

[1] The Dormant Neuron Phenomenon in Multi-Agent Reinforcement Learning Value Factorization

[2] Sample-Efficient Multiagent Reinforcement Learning with Reset Replay

[3] Revisiting Cooperative Off-Policy Multi-Agent Reinforcement Learning

### Questions
1. How significant is the computational overhead of EnSet (e.g., ensemble size H=5) compared to standard QMIX or MARR?

2. In the main experiment results, does MARR use the same parallel environments with EnSet?

### Soundness
3

### Presentation
3

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
This paper proposes a method that uses an ensemble of global Q-networks to increase the replay ratio while avoiding the problem of dormant neurons. In addition, the authors adopt an augmentation technique called multi-agent translation invariance to diversify the replay experience.

### Strengths
This paper tackles an important aspect of deep MARL regarding sample efficiency, and I believe this line of research is important.

### Weaknesses
**(Lack of novelty)**

The contributions include having multiple hypernetworks (called the global Q-network) and introducing multi-agent translation invariance. I understand from the ICLR review guideline that creative combinations of existing ideas can be a good contribution. However, naively using an ensemble network is not novel at all. In addition, there is a work in the single-agent RL domain [1], which further reduces the novelty of this work. Even though [1] is in the single-agent setting, [1] and this work are highly related in the sense that both use an ensemble of networks to allow a high replay ratio. Moreover, this work uses an ensemble of global Q-networks—which can actually be seen as a single agent (since centralized methods treat multiple agents as a single agent). The authors should mention [1] in the related work section.

**(Lack of clarification of multi-agent translation invariance)**

 It is hard to understand the motivation and effectiveness of translation invariance. It looks to me like injecting noise (z is sampled uniformly). Why does it diversify the replay experience? This is a common way to improve robustness. How is it related to improving sample efficiency and preventing dormant neurons? The authors should provide a clear motivation and results on this.

**(Inefficient and unclear experiments)**

– The experiments are generally not well organized. For example, Figure 3 shows the performance with and without EnSet on top of ATM, QPLEX, and QMIX, and it seems ATM outperforms the others in all tasks. It is good to have the results of many baselines. However, Figure 4 only shows the results of all reset methods and the proposed method on top of QMIX. I believe this is the primary result showing the contribution, but why do the authors not use ATM as the base algorithm? Basically, the algorithm performing best is EnSet-ATM, and the primary baseline is MARR, so the result of MARR-ATM must be included. In addition, Figure 3 shows the result of ATM when RR = 1 and EnSet-ATM when RR = 10, which is unfair (I acknowledge that EnSet-ATM when RR = 1 is provided in the appendix, but the authors should be careful about what results are included in the main paper).

– Again, I appreciate that the authors use different base algorithms. However, in my opinion, the authors use different baselines in terms of reset methods, not RL algorithms—for example, EnSet-Reborn vs. Reborn or ReDo-Reborn vs. ReDo. This also applies to MPE and SMACv2. The results that EnSet-FACMAC is better than FACMAC in MPE and EnSet-QMIX (RR = 15) is better than QMIX (RR = 1) are not enough. The results of other reset methods must be included. Again, since the contribution is using an ensemble method, the primary baselines should be other reset methods, not just naïve MARL algorithms.

[1] Kim et al., “Sample-Efficient and Safe Deep Reinforcement Learning via Reset Deep Ensemble Agents,” NeurIPS 2023

### Questions
See Weaknesses

### Soundness
1

### Presentation
2

### Contribution
1
