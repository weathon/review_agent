# Energy-Based Transfer for Reinforcement Learning

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Reinforcement learning is a powerful framework for sequential decision-making, but its sample inefficiency limits its scalability, especially in multi-task or continual learning settings. A common solution is to transfer knowledge from a teacher policy to guide exploration in new tasks. However, blindly applying such guidance can degrade performance by biasing exploration toward low-reward behaviors. We propose an introspective transfer learning method that selectively guides the student only when the teacher is likely to be helpful. Using energy-based models for out-of-distribution detection, the teacher issues advice only in familiar states -- those within its training distribution. We theoretically show that energy scores reflect the state visitation density under the teacher policy, and empirically demonstrate improved sample efficiency and returns in single-task and multi-task settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the low sample efficiency of reinforcement learning and the "suboptimal guidance" flaw of traditional transfer learning. It proposes Energy-Based Transfer Learning (EBTL): using energy scores to detect in-distribution or out-of-distribution states. The teacher only guides the student in ID states, while OOD states let the student explore freely. Energy regularization is added to enhance ID/OOD separability without harming teacher performance. Experiments on single-task and multi-task scenarios show EBTL outperforms 5 baselines in sample efficiency and final rewards. Theoretically, energy scores are proven proportional to the teacher’s state visitation density, ensuring reliable guidance decisions.

### Strengths
1. The core insight is good. The paper ties the timing of intervention to the teacher’s actual competence, ensuring guidance is only deployed when it adds value.
2. The experiments target the exact pain points EBTL aims to solve. The team tests on both Minigrid navigation and Overcooked cooking scenarios, two settings where knowledge transfer can facilitate exploration. And the results show that EBTL outperforms all baselines.

### Weaknesses
1. A notable limitation of EBTL lies in its narrow reliance on the current single-state representation to calculate energy scores. It overlooks the critical role of the agent’s impending actions and historical state sequences when evaluating whether knowledge transfer is appropriate. This oversimplification may lead to misjudgments about "guidance value" in scenarios where state meaning depends on context.
2. The paper also fails to explore obvious, promising extensions that would use the energy score’s graduality more effectively. It ignores the energy score’s inherent value as a continuous measure of familiarity and skips more adaptive approaches like mixed action weighting or importance-guided selection. Instead of a binary “guide or not,” use the energy score to directly weight the mix of teacher and student actions. A higher score could mean a higher probability of sampling the teacher’s action. Besides, since the energy score correlates with the teacher’s state visitation density, it could refine the importance weights EBTL already uses to correct offline bias.
3. I want the authors to consider more practical transfer settings like sim2real.

### Questions
See weaknesses.

### Soundness
2

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
2

### Summary
This paper introduces energy-based transfer learning (EBTL), which improves sample efficiency in reinforcement learning through selective teacher guidance. EBTL employs the teacher’s energy as a familiarity proxy, issuing advice only in likely in-distribution states and thereby avoiding additional networks, mappings, or handcrafted OOD detectors. Empirically, EBTL outperforms baselines in both single- and multi-task transfer.

### Strengths
1. Originality:
The paper introduces a novel energy-based mechanism to control knowledge transfer in RL. While energy models have been used in OOD detection, applying them to decide when to transfer across tasks is an original and creative idea.

2. Quality:
The technical development is sound and theoretically grounded. The algorithm design is coherent with the underlying theory. Experiments are comprehensive, covering both single- and multi-task settings. The results consistently demonstrate the method’s efficiency and robustness.

3. Clarity:
The paper is clearly written and well-organized. The motivation, methodology, and theoretical insights are presented logically, with helpful figures and algorithm descriptions. 

4. Significance:
EBTL addresses a core limitation in transfer RL—negative transfer under distributional shift—and provides a general framework applicable to various domains. Its ability to adaptively decide when to transfer makes it both practically relevant and conceptually impactful, offering a promising direction for improving sample efficiency in multi-task and continual RL.

### Weaknesses
1. Limited theoretical depth:
The link between energy scores and teacher visitation density is only intuitively discussed, without formal guarantees on convergence or optimality. Providing stronger theoretical analysis—e.g., on transfer efficiency or sample complexity—would enhance rigor.

2.Incomplete ablation analysis:
The effects of key components (energy threshold τ, decay schedule, regularization) are not fully disentangled. More systematic ablations would clarify each component’s contribution and the model’s behavior over training.

### Questions
1. Scalability to complex domains:
 Could the authors discuss whether EBTL can handle continuous-action tasks ? Are there computational issues when estimating energy scores in such settings?

2. Robustness to poor teachers:
While the paper claims robustness to imperfect teachers, there is little quantitative evidence. How does EBTL perform when the teacher policy is partially suboptimal or even misleading? Could the authors include or elaborate on experiments that vary teacher quality?

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
3

### Summary
This paper addresses the challenge of low sample efficiency in reinforcement learning (RL), particularly in multi-task and continual learning settings. The authors propose an energy-based transfer learning framework that leverages a previously trained teacher policy to guide exploration in new tasks. To avoid negative transfer when the target task diverges from the teacher’s domain, the method employs out-of-distribution detection based on energy scores, ensuring that teacher interventions occur only in familiar states. The authors provide a theoretical justification linking energy scores to the teacher’s state-visitation density and present empirical results demonstrating that the approach improves both sample efficiency and final performance across single-task and multi-task benchmarks.

### Strengths
* This work tackles a good problem, in real world settings, it has been shown that with respect to a reward function the teacher may be considered sub-optimal in parts of the task. 
* The authors discuss between same task transfer, and mult-task transfer, which makes allows insight into both perspectives, unlike other works which may focus on one or the other.
* The performance seems to be good on two applicable grid based environments and the authors clearly show that their method has advantages over the other advising algorithms.
* The paper is well written, and the diagrams chosen make things clear to the reader.

### Weaknesses
Recently, there have been ways introduced to see if a policy is out of distribution specific to the RL domain, see [1,2,3,4], the energy function may not work in OOD scenarios, potentially in partially observable environments, and it might be necessary to take insight from [1,2,3,4] or discuss the expected changes from a unsupervised ood method to a suitable RL ood method. This may provide insight to section ```Higher covariate shift makes OOD detection more challenging ```. 

Although it is an older work and is focused on sub-optimal ensembles of teachers, I was surprised that the authors did not have much insight on the issues [5] encountered, specifically one of the main takeaways: 
* Behavioral policies can be sensitive to contradictory teachers
  - When two policies contradict, i.e potentially the teacher and the student, there are works that find it very difficult for the task to be completed [6], intuitively it may find itself stuck in a state as one policy recommends to leave, where the other (potentially student) keeps recommending to return. There does not seem to be any explanation to why the authors would not encounter this issue. 

The decay parameter seems to be very important to the task, yet rarely talked about. Unlike other works [7], t is not budget based. One scenario that comes to mind is to suppose the teacher is considered OOD for the first 3 quarters of the task. It appears that as the horizon increases, the teacher will never be called to do any action advising depending on the decay and would reduce to a single agent. There are pros and cons of using a horizon based decay vs a budget based decay, however there doesn't seem to be any talking points on this either, perhaps ablation experiments should be added? Why was decay chosen instead of budget?  


[1] Mohamad H Danesh and Alan Fern. Out-of-distribution dynamics detection: Rl-relevant benchmarks and results. arXiv preprint arXiv:2107.04982, 2021.

[2] Hongming Zhang, Ke Sun, Linglong Kong, Martin Muller, et al. A distance-based anomaly detection framework for deep reinforcement learning. Transactions on Machine Learning Research,
2024.

[3] Geigh Zollicoffer, Kenneth Eaton, Jonathan C Balloch, Julia Kim, Wei Zhou, Robert Wright, & Mark Riedl (2025). Novelty Detection in Reinforcement Learning with World Models. In Forty-second International Conference on Machine Learning.

[4] Tom Haider, Karsten Roscher, Felippe Schmoeller da Roza, and Stephan Gunnemann. Out-of- ¨
distribution detection for reinforcement learning agents with probabilistic dynamics models. In
Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems, pp. 851–859, 2023.

[5] Andrey Kurenkov, Ajay Mandlekar, Roberto Martin-Martin, Silvio Savarese, & Animesh Garg. (2019). AC-Teach: A Bayesian Actor-Critic Method for Policy Learning with an Ensemble of Suboptimal Teachers.

[6] Shenfeld, I., Hong, Z.W., Tamar, A., & Agrawal, P. (2023). TGRL: An Algorithm for Teacher Guided Reinforcement Learning. In Proceedings of the 40th International Conference on Machine Learning (pp. 31077–31093). PMLR.

[7] Da Silva, F. L., Hernandez-Leal, P., Kartal, B., & Taylor, M. E. (2020). Uncertainty-Aware Action Advising for Deep Reinforcement Learning Agents. Proceedings of the AAAI Conference on Artificial Intelligence, 34(04), 5792-5799.

### Questions
Although I don't believe it is necessary due to the increased difficulty of the analysis, but have the authors considered continuous action spaces or continuous state spaces? I would be willing to increase my score if these concerns were adequately addressed.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the problem of improving teacher–student transfer in reinforcement learning. The proposed method uses an energy score to decide whether the current state lies within the teacher’s training distribution. If the score exceeds a threshold, the teacher provides action guidance; otherwise, the student acts independently.

### Strengths
1. The transfer learning in RL is an important and timely topic.
2. The proposed approach is simple yet effective, and conceptually easy to follow.
3. The paper is clearly written and well structured.

### Weaknesses
1. The central theoretical claim (Proposition 4.1) states that the logarithm of the stationary visitation density is proportional to the negative free energy $\phi(s)$. This relies on a very strong assumption, that is, the policy network optimized for reward maximization (e.g., PPO) implicitly forms a *realizable energy model* $p_{\theta^*}$ that perfectly fits the visitation distribution. In practice, the policy’s logits are trained for control, not density estimation, so equating them with a likelihood model is conceptually weak.
2. The student’s training data mixes *on-policy* samples from itself (sampled in training by current policy) and *off-policy* samples from the teacher, whose policy arises from a different distribution. The paper doesn't address how this distribution mismatch is handled or whether importance weighting or correction is applied. This omission raises concerns about biased gradient estimates and unstable policy updates.
3. The experiments are limited to MiniGrid and Overcooked, where the ID–OOD distinction is relatively trivial (goal position or recipe type). These toy setups do not convincingly demonstrate that an energy-based metric is necessary, e.g., simple rule-based distinctions might achieve the same effect. Evaluation in more complex or continuous-control environments would strengthen the paper.

### Questions
1. The paper introduces a decay schedule $\delta(t)$ that gradually reduces teacher intervention, similar to $\epsilon$-greedy annealing. However, this design blurs attribution: performance gains may come from the decay itself rather than from energy-based selection. Could the authors run ablation studies removing the decay, or removing the energy gating while keeping only the decay, to isolate the contribution of each component?
2. Regarding the introduction of $L_{energy}$ to augment teacher training, it uses both in-distribution $D_{in}$ and out-of-distribution $D_{out}$ samples. This raises some questions: does this introduce a “future leakage” paradox? That is, since we can sample OOD data from new environments the teacher model has never encountered, does this imply the teacher model has received specialized training in these “unseen” environments beforehand? Does this contradict the principle of transfer learning, which emphasizes learning effectiveness in completely unseen environments?
3. Could the author provide the proportion of teacher intervention during training? I wonder what percentage of actions are sampled from the teacher model, to better understand how crucial the teacher's role truly is. If the proportion at different training stages could be provided, it would allow us to explore how teacher intervention evolves throughout the training process.

### Soundness
3

### Presentation
3

### Contribution
2
