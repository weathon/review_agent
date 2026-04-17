# Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents

- Decision: Accept (Poster)
- Scores: 2, 4, 4, 6

## Abstract
Large Language Models (LLMs) as autonomous agents are increasingly tasked with solving complex, long-horizon problems. 
Aligning these agents via preference-based methods like Direct Preference Optimization (DPO) is a promising direction, yet it faces a critical granularity mismatch. 
Trajectory-level DPO provides stable signals but blur where credit should be assigned within long trajectories, whereas step-level DPO offers fine-grained supervision but can be statistically noisy and data-inefficient when Monte Carlo rollouts are limited, and can be hard to fully exploit multi-step structured behaviors that only reveal their effect over several actions.
To balance this trade-off, we introduce **H**ierarchical **P**reference **L**earning (HPL), a hierarchical framework that optimizes LLM agents by leveraging preference signals at multiple, synergistic granularities. 
While HPL incorporates trajectory- and step-level DPO for global and local policy stability, its core innovation lies in group-level preference optimization guided by a dual-layer curriculum. 
Our approach first decomposes expert trajectories into semantically coherent action groups and then generates contrasting suboptimal groups to enable preference learning at a fine-grained, sub-task level. 
Then, instead of treating all preference pairs equally, HPL introduces a curriculum scheduler that organizes the learning process from simple to complex. 
This curriculum is structured along two axes: the group length, representing sub-task complexity, and the sample difficulty, defined by the reward gap between preferred and dispreferred action groups.
Experiments on three challenging agent benchmarks show that HPL outperforms existing state-of-the-art methods. 
Our analyses demonstrate that the hierarchical DPO loss effectively integrates preference signals across multiple granularities, while the dual-layer curriculum is crucial for enabling the agent to solve a wide range of tasks, from simple behaviors to complex multi-step sequences.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Hierarchical Preference Learning (HPL), which combines preference labeling at three different granularities to improve credit assignment in DPO-based training for LLM agents. Specifically, the method constructs preference pairs at: (1) trajectory level, (2) step level, and (3) segmented sub-trajectory (group) level. Chosen examples are sampled from demonstrations while rejected examples come from a reference policy. For group-level preferences, the authors segment trajectories and use Monte Carlo value estimation with reference policy rollouts to assign preference labels. The approach incorporates curriculum learning for group-level preference optimization. The final objective combines DPO losses from all three levels plus a behavior cloning term. Experiments on three agent benchmarks demonstrate that HPL outperforms methods using only step-level or trajectory-level preferences, with ablations showing positive contributions from each granularity level.

### Strengths
1. **Technically sound approach:** The paper presents a well-designed framework that thoughtfully integrates curriculum learning and multi-granularity preference labeling to enhance agent performance.
2. **Clear presentation:** The paper is well-written with effective visualizations. Figure 2 clearly illustrates the HPL framework and training procedure, making the methodology easy to follow.

### Weaknesses
1. **Insufficient motivation:** The paper argues that step-level DPO is "myopic" and cannot capture "synergistic value" of sub-trajectories, but this strong claim lacks adequate support. Diverse planning tasks has been successfully achieved via PPO learning with step-level rewards. The paper needs to better position itself relative to existing literature and provide more rigorous justification for why step-level preferences are fundamentally insufficient.
2. **Contradictory problem setting:** There is conceptual confusion about whether this is truly an offline alignment method. While the authors frame this as offline preference alignment, the group-level labeling procedure involves reference model rollouts and online reward labeling. If online reward queries are required, this fundamentally cannot be offline alignment.
3. **Incomplete experimental comparison:** Given the online reward labeling involved in HPL (per weakness 2), the experimental setup is incomplete and potentially unfair: i) Missing online RL baselines (e.g.PPO, GRPO), as the current trajectory-level and step-level DPO are purely offline methods. ii) Missing IL and IRL baselines.
4. **Unclear role of curriculum learning:** The motivation for incorporating curriculum learning is not well established. How does curriculum learning specifically address the granularity mismatch problem?

### Questions
1. How does your approach compare to critic learning in PPO for credit assignment? If fine-grained step-level feedback is available, what is the specific advantage of segmenting trajectories into groups rather than using all step-level information directly?
2. Is training instability the primary motivation for the curriculum scheduler in group-level preference learning? Does the high variance in trajectory lengths contribute to this instability?
3. My hypothesis is that group-level preference learning may work better than step-level primarily because the prior knowledge injected during trajectory segmentation improves exploration efficiency, while step-level preferences could potentially learn the targeted credit assignment given sufficient data. I would like to ask for an in-depth analysis on this.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the "granularity mismatch" in preference-based alignment for long-horizon LLM agents—trajectory-level DPO is too coarse for credit assignment, while step-level DPO is too myopic—and proposes **Hierarchical Preference Learning (HPL)** that combines trajectory-, step-, and a crucial **action-group** DPO operating on semantically coherent sub-tasks. HPL segments expert trajectories into action groups, constructs contrastive pairs, estimates group rewards, and trains with a dual-layer curriculum scheduled by group length (complexity) and reward-gap difficulty, yielding structured, sub-task credit assignment without sacrificing global stability. Experiments on ALFWorld, WebShop, and InterCode-SQL show HPL—especially with semantic segmentation—consistently outperforms SFT, RFT, ETO, and IPR; ablations highlight the group-level loss and the curriculum as the primary performance drivers.

### Strengths
1. **Principled mid-granularity learning:** The action-group DPO provides a theoretically grounded bias–variance trade-off—achieving lower variance than trajectory/step DPO while adding at most ε bias with group length (k=\Theta(\log(1/\epsilon))); within-trajectory sample independence further avoids the large covariance that plagues step-level pairs.   

2. **Strong and well-validated gains:** Across ALFWorld, WebShop, and InterCode-SQL, HPL (especially the semantic segmentation variant) outperforms SFT/RFT/ETO/IPR—e.g., ALFWorld-unseen 86.57%—with ablations showing the group-level loss is most critical and the dual-layer curriculum (length + difficulty) delivers additional, consistent improvements.

### Weaknesses
1. **Limited novelty:** Framing DPO at an intermediate “action-group” level feels incremental—more a structured decomposition of known step/trajectory DPO than a fundamentally new alignment principle.

2. **Theory under-validated:** The bias–variance claims and independence assumptions are not directly stress-tested (e.g., no gradient-variance or signal-to-noise measurements, nor controlled sweeps that confirm the predicted (k)–(\epsilon) trade-off).

3. **Sensitivity to segmentation/curriculum:** Success appears to hinge on segmentation quality and curriculum schedules; robustness to noisy grouping, alternative segmenters, and hyperparameter perturbations is insufficiently examined.

4. **Scope and scalability limits:** Benchmarks and models are narrow (scripted simulators, limited agents); reliance on MC rollouts and extra segmenters raises training cost and may hinder transfer to real-web or human-preference settings.

### Questions
See weaknesses.

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
The paper aims to address the issue of granularity mismatch in preference-based alignment for LLM agents and proposes Hierarchical Preference Learning (HPL), which integrates preference signals across three levels of abstraction. The main focus of this work is providing action-group level of abstraction, which is done by semantically partitioning the trajectories into semantically coherent sub-tasks and optimizing a hierarchical DPO loss. The paper also suggests dual-layer curriculum learning strategy, which uses the information of sample discriminability and action group length. Experiments focused on long-horizon agent benchmarks show that HPL outperforms popular baselines.

### Strengths
1. The overall flow of the paper is well organized, allowing readers to effectively understand the paper.
2. Figures and experiment results are well polished.
3. Experiments such as ablation study in Figure 5 allow the readers to understand the effectiveness of components involved in the proposed algorithm.

### Weaknesses
1. The paper does not have very strong theoretical motivation or support of the proposed method.
2. The experiment results are not having standard deviation information, which is crucial for reinforcing the credibility of the results.
3.  The proposed method seems to be much more complicated than the compared baselines, but the potential limitation coming from this approach is not well analyzed. HPL requires expert trajectory, generations from the reference model, reward signal, and access to a powerful model such as gpt-4o when semantic segmentation is used.
4. Some of the crucial experimental details seem to be omitted from the paper (ex. the value of $\gamma$ for the experiments, the number of generations $M$ used to evaluate the reward in Equation 2.

### Questions
## Questions
1. How many additional datapoints were created by action group-level data generation in Section 3.2.2? How big is the dataset compared to step-level and trajectory-level datasets?
2. If the value of $\gamma$ actually used for experiment was practically 1 (no discount in the trajectories), does the analysis of Proposition 1 not become incoherent?
3. How much money in total was spent for the API call for semantic segmentation of the action groups, per each train-test run in Table 1?

## Suggestions
1. Could you provide a table which present a comparison between HPL and the presented baselines, in terms of the use of external powerful LLM, number of new generations for the training, and wall-clock time for generation and training phase?

### Soundness
2

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
2

### Summary
This paper proposes Hierarchical Preference Learning (HPL) to address the granularity mismatch in preference-based offline alignment for long-horizon LLM agents. Conventional Direct Preference Optimization (DPO) methods operate either at the trajectory level or the step level. HPL introduces an intermediate group-level DPO that decomposes trajectories into semantically coherent action groups, paired with a dual-layer curriculum along task complexity and sample difficulty. Experiments on ALFWorld, WebShop, and InterCode-SQL demonstrate consistent improvements over existing benchmarks.

### Strengths
1. This work is well motivated. The paper starts with an intuitive problem, the granularity mismatch in preference optimization, and then proposes a novel solution that bridges trajectory/step-level supervisions.

2. The sample code is provided. It facilitates the reviewer to verify the numerical results independently.

3. The numerical performance is sound. The proposed methods beat the existing baselines by a clear margin.

### Weaknesses
1. The presentation could be improved.  While technically rich, some methodological descriptions (e.g., the Monte Carlo approach in reward estimation and curriculum scheduler) could benefit from more intuitive explanations.

### Questions
Please see the weakness part.

At the current stage, I tend to recommend acceptance. However, I'm open to revising my final decision after the rebuttal and further discussions.

### Soundness
2

### Presentation
2

### Contribution
2
