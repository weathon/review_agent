# Scaling Curriculum Learning for Autonomous Driving

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Batched simulators for autonomous driving have recently enabled the training of reinforcement learning agents on a massive scale, encompassing thousands of traffic scenarios and billions of interactions within a matter of days. 
Although such high-throughput feeds reinforcement learning algorithms faster than ever, their sample efficiency has not kept pace: 
As the standard training scheme, domain randomization uniformly samples scenarios and thus consumes a vast number of interactions on cases that contribute little to learning.
Curriculum learning offers a remedy by adaptively prioritizing scenarios that matter most for policy improvement. 
We present CL4AD, the first integration of curriculum learning into batched autonomous driving simulators by framing scenario selection as an unsupervised environment design problem.
We introduce utility functions that shape curricula based on success rates and the realism of the agent's behavior, in addition to existing regret-estimation functions.
Large-scale experiments on GPUDrive demonstrate that curriculum learning can achieve 99% success rate a billion steps earlier than domain randomization, reducing wall clock time by 77%, and by 40% compared to traffic density-based heuristic curricula.
An ablation study with a computational budget further shows that curriculum learning improves sample efficiency by 67% to reach the same success rate. 
To support future research, we release an implementation of CL4AD in GPUDrive.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents CL4AD, a scalable curriculum learning framework for batched autonomous driving simulators such as GPUDRIVE. It integrates unsupervised environment design methods to adaptively prioritize driving scenarios based on regret, success, and realism. The authors design new utility functions tailored for autonomous driving and show that CL4AD improves sample efficiency and convergence speed compared to uniform domain randomization.

### Strengths
1. Addresses an important and timely problem: improving sample efficiency in large-scale driving RL through adaptive scenario selection.
2. Provides a well-engineered integration of UED methods into high-throughput simulators, with clear algorithmic formulation and open-source implementation.
3. Large-scale experiments convincingly show faster convergence and reduced compute cost compared to domain randomization, backed by detailed ablations on compute limits and dataset size.

### Weaknesses
1. The novelty is moderate: the method mainly adapts existing PLR-based UED techniques to the AD domain, with limited conceptual innovation beyond utility function design.
2. The proposed realism-based utility functions are interesting but insufficiently justified. It is unclear whether they consistently outperform regret-based metrics or simply correlate with progress.

### Questions
1. Could the authors quantify the computational overhead introduced by CL4AD compared to standard domain randomization, especially in terms of GPU time and memory?
2. How sensitive is the performance to the choice and scaling of different utility functions? Are some functions redundant or correlated?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Based on GPUDrive, this paper investigates how curriculum learning (CL) can accelerate training driving agents in the self-play setting. Besides 3 existing CL unitility functions, this work proposed 3 new utility functions and benchmarked these methods on GPUDrive and compared the results with uniform sampling. The results show that introducing CL into the training can indeed largely speed up the policy training in terms of success rate and other metrics.

### Strengths
1. This is the first work that integrates CL into batched autonomous driving simulators by framing scenario selection as an unsupervised environment design problem.
2. The effectiveness of CL method is confirmed in this setting
3. The paper is easy to read

### Weaknesses
1. This work benchmarks existing Unsupervised Evironment Design (UED) methods, especially PLR-based ones, under the bacthed driving simulator setting. Though three new utility functions used in PLR are proposed, they didn't outperform those proposed in previous research (PLR-learn and PLR-MaxMC). In addition, a large part of the method section is to introduce previous works, making this paper looks like a benchmark paper. Considering that CL helps training is a common sense, this paper lack of novelty on both the method and conclusion.

2. As this paper is engineering-oriented, one good direction to revise is to tune/benchmark the agents on widely-used benchmarks like WOSAC and nuPlan like what the Giga flow did. This can bring real-world impact. It would be better to see how CL can help self-replay agents achieve SOTA on those benchmarks in terms of performance and training speed. The outcome will add a lot of values to this paper, making it an new paradigm for addressing the agent simulation & planning. In addition, I believe new problems will appear in this process, which may even inspire authors to design new UED methods to improve the novelty of this paper.

3. Several related works are missing:
- Discovering General Reinforcement Learning Algorithms with Adversarial Environment Design by Matthew T. Jackson et al., which combines the idea of meta-learning and UED.
- ScenarioNet: Open-Source Platform for Large-Scale Traffic Scenario Simulation and Modeling By Quanyi Li et al., which also applied CL into the training of driving agents.

### Questions
N/A

### Soundness
3

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
This paper addresses the sample inefficiency of training autonomous driving (AD) agents in modern, large-scale batched simulators. While simulators like GPUDRIVE achieve massive throughput (billions of interactions per day), the standard training method, domain randomization (DR) or uniform sampling, wastes compute on scenarios that are either too easy or too hard.

The authors propose CL4AD, the first framework to integrate and scale curriculum learning (CL) within such a batched, self-play simulator. The core idea is to frame scenario selection as an Unsupervised Environment Design (UED) problem. The system uses Prioritized Level Replay (PLR) as its backbone, which adaptively samples scenarios from a large dataset (Waymo Open Motion Dataset) based on their "utility" for learning.

### Strengths
- The paper tackles a clear and significant bottleneck. As simulation throughput increases, sample efficiency (i.e., what to simulate) becomes the next logical and practical problem to solve. This work is at the forefront of this shift.
- The performance gains are not minor; they are massive. Achieving a 99% success rate 1 billion steps earlier (or 77% faster in wall time) is a very strong and convincing result. The paper does an excellent job of validating this at different scales (1k, 10k, 80k scenarios) and under compute constraints, showing the method is robust.
- While the core CL algorithm (PLR) is an existing method, its application and integration into a batched, multi-agent, self-play simulator at this scale is a non-trivial and valuable engineering contribution. This is the first work to successfully do so.

### Weaknesses
- The primary weakness is that the paper does not propose a new curriculum learning algorithm. The core method is PLR, which is from 2021. The contribution is one of integration, scaling, and new utility functions, making this more of a systems and application paper than a fundamental RL methods paper.
- The paper is titled "Scaling Curriculum Learning," but it only scales one specific UED algorithm (PLR). It does not explore other prominent UED methods cited in its own related work, such as ACCEL or RE-PAIRED, which might have different scaling properties or performance characteristics.
- The only baseline used for comparison is Domain Randomization (DR). While DR is the current standard and thus the most important baseline, the paper would be stronger if it compared against other, simpler curriculum heuristics (e.g., replaying failed episodes, or an annealed curriculum based on scenario complexity like the number of agents) to better situate the gains from a sophisticated method like PLR.

### Questions
- The algorithmic contribution is primarily the integration of PLR. Why was PLR chosen over other UED methods like ACCEL, which also scales and mutates scenarios to create new ones (as mentioned in your limitations)?
- Your realism-based utility functions (UGC-ADE, UAct-MAE) are a key contribution. Did you observe any qualitative differences in the policies trained with these metrics versus regret-based metrics? For example, did they result in more "human-like" (but perhaps less optimal) driving behavior?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
- This paper proposes Curriculum Learning for Autonomous Driving (CL4AD) for batch AD simulators.
- The authors frame the CL problem as an unsupervised environment design (UED) problem. They try to improve upon random sampling of the levels/scenarios/environment for the batched sims for RL training. 
- The authors introduce utility functions to score the scenarios/level of curricula to shape the replay distribution. 
  - The scores are dependent on regret, success and realism of the AD  trajectories.
  - They combined the prioritized level replay (PLR) to adaptively sample high utility scenarios with different scoring functions and analyse their effects on the overall AD policy performance.
- With the introduction of curriculum, the large scale experiments claimed to achieve 99% success in GPUDrive a billion steps earlier than random sampling and also reduced the wall clock time by 77%.
- The authors perform experiments to find if CL can accelerate learning in AD policies, is it effective under limited compute resources and can it be scaled with more scenarios. 

It is appreciated that while the authors have proposed a good integration of CL for batched simulators and presented multiple experiments and ablations to support the approach, it would be more advantageous if there were some explicit analysis done on why a particular utility function is better or worse than the other and which one can be preferred under which conditions.

Given the comments related to weaknesses and limitations, I can have the rating as 4. Flexible to move this during or after rebuttal.

### Strengths
- The paper did a good job laying out the theoretical background required for integrating UED based CL for AD. 
- The paper presents three novel utility functions from which UAct-MAE (a realism metric calculating distance between RL agents’ actions and logged trajectories) had relatively better results in terms of returns and success rate.
- The paper visualized the evolution of the utility score values (replay distributions) with the number of policy updates for different scoring functions. This helps in analysing how those values evolve as the policy improves with increasing steps.
- The paper is able to bring the advantages of CL in terms of sample efficiency for training AD policies using batched simulators. 
- The approach seems to be scalable and extensible and can be applied to other simulators as well.
- A well written paper with clear figures, plots and diagrams and ablations.

### Weaknesses
1. The plots were not clear enough in showing how the utility score evolved with training. They are of almost the same color throughout the number of policy updates.

2. No analysis or comments on why a certain utility score function is performing relatively better than the other.


**Limitations**
- There can be other study/experiments related to heuristic-based curriculum design as in increasing the number of agents, different traffic densities, etc as a proxy for difficulty levels. This can be compared to the automated curriculum.
- The paper might be strengthened by framing this as a designs-space formulation on how CL for AD and directly used a plug-and-play option for batched simulators.

### Questions
1. Line 151: Definition 3.2 - The POSG is underspecified. Can we have some examples of what could be some unknown components or uncertainties that are making the observations underspecified?

2. Fig 5: We might assume more scenarios with high utility scores in the beginning of the training and a gradual decrease with more updates as the policy improves. However, such a trend was not observed from the plot (first row) and the scores seem to be pretty consistent as the number of policy update steps increased. 

3. Fig 5 (middle row): Would that mean scenarios that have around > 10-15 agents are not useful for learning. 
  - Either they are too easy to effect the utility score, or
  - Too difficult to match the score to get any useful signal from it.

4. How are the scenarios/levels evicted from the buffer?

5. What is the general update-to-data ratio (UTD ratio) for policy training?

**Suggestion:**
- It would be helpful to have some of the acronyms used in the figures to have their full-form in the figure description. It could be confusing as some figures appear way before the text that introduces those terms (for example Fig 2).
- It would be nice to have configuration of the replay buffer (sample to insert ratio, eviction policy, etc.)

### Soundness
3

### Presentation
3

### Contribution
2
