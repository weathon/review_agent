# APC-RL: Exceeding data-driven behavior priors with adaptive policy composition

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
Incorporating demonstration data into reinforcement learning (RL) can greatly accelerate learning, but existing approaches often assume demonstrations are optimal and fully aligned with the target task.
In practice, demonstrations are frequently sparse, suboptimal, or misaligned, which can degrade performance when these demonstrations are integrated into RL.
We propose Adaptive Policy Composition (APC), a hierarchical model that adaptively composes multiple data-driven Normalizing Flow (NF) priors.
Instead of enforcing strict adherence to the priors, APC estimates each prior's applicability to the target task while leveraging them for exploration. 
Moreover, APC either refines useful priors, or sidesteps misaligned ones when necessary to optimize downstream reward.
Across diverse benchmarks, APC accelerates learning when demonstrations are aligned, remains robust under severe misalignment, and leverages suboptimal demonstrations to bootstrap exploration while avoiding performance ceilings caused by overly strict adherence to suboptimal demonstrations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work builds upon substantial advancements in behavioral priors for online reinforcement learning, and proposes an end-to-end technique to balance and control the guidance received from multiple, potentially misaligned priors. Given N data sources, the method trains N+1 policies: one acting over real actions, and N acting over the latent space of frozen normalizing flows estimating the state-conditional action distribution in each data source (as pioneered by PARROT). These policies, as well as their values, are simply trained through SAC; the softmax of the value can be used as a policy-selection criterion for online data collection. Finally, the connected data by each policy can be relabeled according to the transformations induced by each prior, and used for training every other policy. The authors evaluate the proposed methods on a simple 2D maze, as well as on simulated kitchen and driving environments. Across the board, results confirm that the proposed method is less sample efficient than PARROT when the prior is well aligned (likely due to the overhead of value learning for high-level action estimation), but prevent catastrophic failure when the prior is misaligned, while consistently outperforming a solution learning from scratch (as long as the prior is not adversarial). The submission finally provides ablations and a visualization of the policy switching pattern in the racing environment.

### Strengths
- The method is clear, principled and well motivated.
- The paper is well-written and easy to follow; claims are well supported and not overstated.
- The experimental section is well-designed and sufficient to analyze the most important facets of the algorithm. Notably, it includes both simulated and human-collected data.

### Weaknesses
- The computational overhead induced by the method is perhaps the biggest limitation: a linear number of agents need to be trained as the number of data sources increases. Have the authors considered conditional actor and critic architecture (i.e. a single critic network, conditioned on the policy's index)?
- A further constraint is the reliance of this method on invertible generative models in order to effectively relabel actions. Normalizing flows are in general capable models, but other methods (such as flow matching) seem to be prevalent empirically. Can the method be generalized to a broader class of architectures for modeling priors?
- The work fails to acknowledge that the proposed method may only alleviate partially misaligned priors. Consider a setting in which the agent is initialized in the top left corner of the maze, and the prior is trained on data reaching the top right corner, while the downstream task involves reaching the bottom left corner. Exploration, in this case, will be biased towards an useless direction, and it is hard to imagine that the method would outperform learning from scratch. Moreover, the normalizing flow trained on the prior would often be queried OOD. This is of course understandable, but I believe it should be acknowledged explicitly in a limitation section.
- There are several options for dynamically incorporating behavioral priors into downstream RL, including [1] and [2] (although I believe [2] does not directly apply as uncertainty estimates are not available nor informative). Although they have not been applied in the exact same setting, a comparison would be informative.

References:
[1] Bagatella et al., SFP: State-free Priors for Exploration in Off-Policy Reinforcement Learning, TMLR 2022
[2] Cramer et al., CHEQ-ing the Box: Safe Variable Impedance Learning for Robotic Polishing, arXiv 2025

### Questions
## Minor concerns and questions
- line 202: a definition and discussion of "primacy bias" might be informative.
- equation 4: I would recommend simply expressing the policy through $\propto$, and avoiding an explicit normalization constant.
- equation 4: did the authors also evaluate a greedy high-level policy? i.e. simply an argmax of the values
- line 218: did the authors actually measure high variance? Would more actions samples result in better empirical performance?
- line 244: why is a non-standard shape chosen for the D4RL maze.
- line 310: Fig.3 right instead of left
- Figure 5: how is the network-based high-level actor trained?
- line 716: this would be appreciated.

## Conclusion
This work is clearly motivated, the algorithm is principled and sufficiently supported by empirical evidence, particularly using both simulated and human-collected data. Despite computational and modeling limitations (as highlighted above), and assuming that questions are sufficiently addressed, I would currently recommend acceptance.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes APC-RL, which combines multiple prior-based actors (using normalizing flows) with a prior-free actor under a value-based selector. It introduces a “reward-sharing” trick that reuses experience across actors via NF invertibility. Experiments on PointMaze, FrankaKitchen, and CarRacing claim faster learning and robustness to misaligned demonstrations.

### Strengths
1. Tackles a relevant problem: overcoming poor demonstrations in RL.

2. Simple and modular architecture that integrates easily with SAC.

3. Reward-sharing via NF inversion is conceptually neat and potentially sample-efficient.

### Weaknesses
1. Efficiency and justification gaps: The approach may be inefficient in multiple ways: maintaining several actors, evaluating each at every step, and managing multiple replay buffers. The paper neither analyzes this computational overhead nor justifies why keeping multiple actors is preferable to simply retraining a single policy once demonstrations start degrading performance.

2. Limited generality: The method is evaluated only with SAC, with no discussion of applicability to other off-policy algorithms.

3. Invertibility not clearly needed: Since SAC is off-policy, rewards could be shared directly without NF inversion. The necessity of invertibility is not well justified.

4. No clear intuition: The paper details how training occurs but not why the composition improves learning or stability.

5. Weak experimental design: Few baselines, undefined notion of “misalignment,” and only 2–3 seeds. The paper should also compare against established demonstration-filtering methods such as HER with demonstrations (Q-filter) to contextualize the claimed benefits.

### Questions
1. How is demonstrator misalignment quantified?

2. Why not directly share rewards without invertibility if the training is off-policy?

3. How efficient is the selector? Does it scale with many actors?

4. Would this method still outperform if retraining from scratch without demos?

5. How would APC-RL compare to HER with Q-filter on similar tasks?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Adaptive Policy Composition (APC), a framework that learns multiple data-driven Normalizing Flow (NF) priors from different demonstration datasets. Building on these NF priors, the method trains multiple latent policies and corresponding low-level actors in online environments. A reward-sharing trick leverages the invertibility of NF priors, allowing different actors to learn from the same transition. A high-level selector is then learned to determine which low-level actor to invoke at each state. The proposed approach effectively addresses the challenge of suboptimal demonstration data in real-world scenarios. Experimental results show that APC outperforms the PARROT baseline.

### Strengths
1. The reward-sharing trick is particularly interesting, as it allows different actors to learn from a single data source, thereby improving sample efficiency.

2. The paper presents comprehensive experimental results across multiple environments, demonstrating the effectiveness of the proposed method.

### Weaknesses
1. Since the proposed APC method learns multiple actors from the environment, it may incur additional computational costs compared to traditional methods (e.g., PARROT). Moreover, the approach requires multiple demonstration datasets, which may not be applicable in real-world scenarios.

2. The ablation study in the paper is also not sufficiently strong. For instance, it lacks an analysis of performance under limited demonstration sources (e.g., using only a single dataset) and does not include comparisons with state-of-the-art Learning-from-Demonstration (LfD) methods using optimal or suboptimal data.

### Questions
1. If I understand correctly, the authors use the selector to choose a single actor at each step rather than forming a mixture of multiple actors. How would the performance be affected if the method instead used a mixture of different actors? If such a mixture could hinder actor training, could it still be beneficial after the actors have been trained?

2. What is the main difference between APC and existing methods that rely on a single data source? If other Learning-from-Demonstration (LfD) methods were extended to integrate the multiple demonstration datasets used in this work, would they achieve similar performance?

3. From Figures 3(b) and 4(b), the proposed APC method does not show substantial improvement over PARROT or SAC on aligned data (with slower convergence). Can I assume that the method mainly provides advantages when the demonstration data are misaligned?

4. I am interested in the performance of APC on the D4RL benchmarks. How does it compare with state-of-the-art LfD methods trained on datasets with different levels of expertise? Would APC still outperform those methods if only a single expert demonstration dataset were used?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a hierarchical RL approach to leveraging suboptimal/misaligned demonstration data; specifically, they propose Adaptive Policy Composition (APC), which trains multiple Normalizing Flow (NF) prior policies and chooses one based on alignment with the target task. Crucially, APC uses two key mechanisms to maintain performance in the face of non-optimal / misaligned demonstrations: (1) reward-sharing, which exploits NF invertibility to map each action token to all policies' latent spaces, enabling continuous updates across all policies and not just the one which took the action, (2) a parameter-free "arbitrator" high-level selector, which simply selects the policy based on the value estimate of the lower-level actors.

This method surpasses PARROT [1] as well as standard from-scratch RL and imitation learning baselines in the misaligned and suboptimal demonstration settings and is only slightly slower to converge compared to PARROT given a fully-aligned prior. Ablations demonstrate that both mechanisms are critical for performance: the arbitrator avoids overcommitting to the prior-based actor and maintains balanced usage of actors, and reward-sharing ensures all actors are updated continuously.

[1] Singh, Avi, et al. "Parrot: Data-driven behavioral priors for reinforcement learning." arXiv preprint arXiv:2011.10024 (2020).

### Strengths
Leveraging suboptimal / misaligned demonstration data for reinforcement learning is a significant research topic, given that real-world scenarios are unlikely to have perfectly aligned demonstration data.

The proposed method is a novel contribution; in particular, the reward-sharing trick which exploits invertibility of normalizing flows which enables continuous updates for all actors.

The experimental section is executed thoroughly and with high quality; the ablations are convincing that both mechanisms are critical for final performance. Overall the paper is well-motivated and design choices at each step is clear.

### Weaknesses
APC’s framework trains multiple parallel SAC agents and evaluates each, so (as typical for methods that target the low-sample regime) while the sample-efficiency is increased, the total compute and (likely) wall-clock time may actually increase. Clarifying the scale of this extra compute (and e.g. memory for the additional replay buffers) per task suite could be helpful.

Besides Franka Kitchen, evaluation settings are relatively simple (Maze Navigation and Car Racing); demonstrating this works on additional task suites would be much more convincing.

### Questions
Could you elaborate on how to get the Normalizing Flow reward-sharing method to work for discrete action spaces?

Performing some sensitivity analysis on the temperature parameter for the "parameter-free" arbitrator (which is therefore a bit of a misnomer) could be useful as well.

Are there any pathological situations in which, due to the distribution of demonstration data, APC would learn significantly slower than the learning-from-scratch baseline, or is there an argument for why this cannot occur?

It would be clearer to explicitly state the number of actors per evaluation environment (e.g. one of them only had a single prior-based actor), and it would be very interesting to find the effect on performance as the number of actors (and therefore demonstration datasets) scaled. Could you have number of actors != number of datasets (e.g. suppose the number of tasks is extremely high)?

### Soundness
3

### Presentation
4

### Contribution
3
