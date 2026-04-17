# Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6

## Abstract
Exploration remains the key bottleneck for large language model agents trained with reinforcement learning. While prior methods exploit pretrained knowledge, they fail in environments requiring the discovery of novel states. We propose EMPO$^2$, a hybrid RL framework that leverages memory for exploration and combines on- and off-policy updates to make LLMs perform well with memory while also ensuring robustness without it. On ScienceWorld and WebShop, EMPO$^2$ achieves 128.6% and 11.3% improvements over GRPO, respectively. Moreover, in out-of-distribution tests, EMPO$^2$ demonstrates superior adaptability to new tasks, requiring only a few trials with memory and no parameter updates. These results highlight EMPO$^2$ as a promising framework for building more exploratory and generalizable LLM-based agents.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a reinforcement learning framework, EMPO², designed to enhance exploration and generalization in large language model (LLM) agents. Existing RL-based LLM agents often overfit to pretrained knowledge and fail to discover novel states. EMPO² addresses this by combining on-policy and off-policy updates with a self-generated memory mechanism—where the agent reflects on past failures to produce “tips” that guide future rollouts.

It internalizes useful behaviors into model parameters while maintaining adaptability via external memory through hybrid optimization. Evaluations on ScienceWorld and WebShop show substantial gains over GRPO with strong out-of-distribution adaptability requiring no parameter updates. The framework is proposed to be a step toward building self-improving, memory-aware LLM agents capable of efficient exploration and continual learning.

### Strengths
- Novelty in integration of memory and RL optimization: The paper introduces a hybrid framework (EMPO²) that unifies parametric (on-policy) and non-parametric (off-policy) learning, bridging memory-augmented reasoning and reinforcement learning. This combination is conceptually novel and addresses a long-standing gap between reflection-based and RL-based LLM agents.

- Effective exploration mechanism: By incorporating self-generated reflective memory (“tips”), the method enables autonomous correction of past errors and promotes deeper exploration without additional supervision. It is a meaningful improvement over prior static-memory or fixed-parameter approaches.

- The method demonstrates substantial gains on two challenging multi-step reasoning benchmarks, ScienceWorld and WebShop, with comparably significant improvement over GRPO. It also shows good out-of-distribution generalization with zero-shot adaptability.

- The study includes comprehensive ablation analyses, comparisons across offline, online, and non-parametric baselines, and even computational cost breakdowns, supporting the robustness of the findings.

- The paper is clearly written, with structured algorithmic explanations, detailed pseudocode, and implementation appendices.

- Broader significance: EMPO² provides a promising direction toward self-improving, memory-aware, and generalizable LLM agents, with potential applications in embodied AI, web interaction, and general decision-making systems.

### Weaknesses
- The novelty is limited: The core contribution of EMPO² lies in combining existing components, such as memory reflection, on/off-policy RL, and intrinsic rewards, rather than introducing a fundamentally new algorithmic principle or theoretical insight. The innovation is primarily architectural rather than conceptual, which makes its novelty kind of limited. 

- The paper does not provide a formal or empirical analysis explaining why the hybrid on/off-policy mechanism stabilizes exploration or improves generalization. Key hyperparameters such as the rollout and update probabilities (p,q) are heuristic, with no clear sensitivity or convergence analysis.

- Evaluation is confined to two benchmarks (ScienceWorld and WebShop), both text-based and reasoning-oriented. The framework’s effectiveness in broader or more complex environments—such as robotics, code synthesis, or multimodal RL—remains untested.

- The study does not analyze the semantic quality of generated “tips” or demonstrate how they concretely guide exploration. Without such analysis, it is unclear whether the model truly learns generalized reasoning strategies or simply memorizes patterns.

- Naming of the Method: the name of the method is kind of confusing at the first time. The "square" symbol is like the footnote, which is ambiguous. Since it is an acronym, it is better to show the full name at the first time of the appearance (e.g. in the abstract, you should show the full name at the first time it appears).

### Questions
- Q1: Justification of the hybrid update design: Could the authors elaborate on why combining on-policy and off-policy updates yields more stable or effective exploration in LLM agents? A theoretical or empirical rationale (e.g., ablation across different ratios of on/off-policy updates) would strengthen the methodological foundation.
- Q2: Sensitivity to hyperparameters p and q: The rollout and update probabilities seem chosen heuristically (p=0.75, q=1/3). How sensitive is EMPO² to these settings? Have the authors explored how different sampling ratios affect training stability, exploration depth, or convergence?
- Q3: Role and quality of generated “tips”: The memory mechanism is central to EMPO². Could the authors provide qualitative examples or a deeper analysis of what kinds of tips are most beneficial? Do these tips generalize semantically across tasks, or do they mainly encode task-specific heuristics?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Online reinforcement learning (RL) has recently emerged as a powerful method to improve reasoning and agentic capabilities of large language models (LLMs). However, these methods generally employ on-policy rollouts, and past failed attempts deliver no information other than single scaler reward. For hard tasks where models can be consistently wrong, on-policy samples do not recover any new information, and models may never learn how to solve these tasks via RL.

A possible alternative to this approach is to incorporate memory into LLM agents — LLMs can read their past rollouts, figure out where they had gone wrong/what they could have done differently, and use them to collect future experience. This paper proposes EMPO$^2$, an end-to-end framework where LLMs generate memory in the form of hints from prior rollouts, incorporate them in future rollouts to collect better experience, and uses a mixture of on and off-policy optimization to both improve the model with these hints, and distill the behavior with hints into the model via off-policy learning to retain good behavior on these tasks even when hints are not present. Experiments on multi-turn agentic environments like WebShop and ScienceWorld show superior performance of the proposed method compared to pure on-policy online RL methods like GRPO.

### Strengths
**Overall, the paper is quite strong and I recommend acceptance of the paper.**

## Novelty

The paper proposes a mechanism for self-generating memory, incorporating memory into the rollout mechanism in order to avoid past mistakes, promote exploration and achieve better rollouts. Moreover, to the best of my knowledge, this paper is the first to use off-policy learning to then distill back these **hint-augmented** prompts back into the model’s parametric knowledge. This is remarkable and also what I was looking for in the online RL system, Kudos to the authors for making it work so nicely.

To the best of my knowledge, this was only possible in an offline manner using SFT via context distillation [1] (this is an important work that should be cited and discussed however), but no one has done it using online RL before. 

The hints are self-generated and do not use a stronger model, which makes the work more appealing.

## Strength of results

The results provided on two benchmarks are very strong, showing remarkable improvement over regular on-policy online RL. This may also unlock improvement in cases where on-policy RL has failed, and have potential implications beyond what the paper presents, i.e., on reasoning tasks like math/coding.

## Memory

The fact that models can generate their own memory and use it in future effectively, **despite not being an entirely novel idea**, is very nice to see in practice.

### Weaknesses
As mentioned above, I really like this paper. However, I would note the following weaknesses:

## Comparison on single turn reasoning tasks

The idea of off-policy updates using previously generated hints can be useful beyond the tasks used in this paper. Particularly, this can help regarding single turn reasoning tasks like math/coding. 

This is the single most important point where the paper's results can be improved. **If the authors can demonstrate the usefulness of their framework on these tasks, and respond to the other weaknesses/questions I mention below, I am very likely to increase my score on this paper further.**

## Adding a comparison of performance vs GPU hours

The proposed method is inherently more computationally expensive compared to regular on-policy GRPO. **While Appendix E provides a rudimentary analysis of the breakdown of compute spent on various components of the proposed method, no comparison with GRPO is given.** The authors should include a plot with

1. X-axis: GPU hours/flops/some other measure of compute

2. Y-axis: performance

In the main paper, to make the comparison fairer with GRPO.

## No ablation for intrinsic exploration reward

The paper uses an intrinsic exploration reward for novel states, in order to encourage the model to explore unseen/sufficiently novel states. However, I could not find an ablation of the proposed method showing how the performance differs in case this reward is not added/for different choices of the intrinsic exploration reward. To simply put it, it is unclear what the effect of this component of the proposed method is on the performance of the method.

## Experiments using only one base model

All the experiments in the paper are done using only one base model. While the results are strong, it is unclear if the gains come from model specific pretraining/finetuning for Qwen2.5-7B-Instruct. Including results on models from different companies/different pre-training would make the paper significantly stronger.

## No learning component for reward generation

**This is more of an after-thought for future work instead of a serious weakness.** The proposed algorithm does not incentivize better memory generation. However, some proposed memory/hints can be better at steering future generations compared to others, and the model is never incentivized **directly** to generate better memory/hints (it may get incentivized indirectly via reward on memory-augmented rollouts). This needs to be addressed to make the learning better.

(**Minor**)

An important prior work for section 3 discussing LLM agents to be information seeking is Paprika [2]. Similarly, context distillation [1] for learning to behave the same way without hints as whenever hints are available is an important prior work that should be cited and discussed.

### Questions
(**Question 1: Advantage Estimation**)

Based on the definition of advantage in line 106-107, am I correct to understand that there is no per-turn advantage? It seems like the advantages are calculated using the entire sum of rewards in each trajectory.

(**Question 2: Calculating Importance Sampling Ratio**)

It is not clear to me how the importance sampling ratios are calculated for off-policy updates. Based on the text and Algorithm 1 from Appendix A, it seems the old log probs ($log \pi_{\theta_{old}}$) are calculated using an off-policy manner (i.e., without generated hints). But how is the current log probs ($log \pi_\theta$) calculated for the off-policy updates? With or without the hints? More generally, could you elaborate how $log \pi_\theta$ is calculated for all different cases (regular on-policy updates, on-policy updates with hints, off-policy updates without hints)? Adding a table clarifying these cases to the main paper would help a lot regarding the clarity of the paper.

(**Question 3: Figure 8**)

I am a bit confused about this figure.

Why are the EMPO and GRPO plots split across two different panels and not on the plot? It is much harder to compare, at least at the first glance, what the performance difference is.
Why does the starting point between EMPO and GRPO vary? Is it because they already went through one round of training with their respective algorithm on the previous task?
What happens if you take the checkpoint resulting from GRPO and run EMPO on top of it, and vice versa (run GRPO on the checkpoint from EMPO) on the new task?

(**Question 4: ScienceWorld**)

What does return/reward mean for ScienceWorld? Is there some other metric like task success rate/completion rate beyond just reward coming from different subgoals/components within a task? Could the authors report that?

(**Question 5: Performance Difference between ScienceWorld and WebShop**)

The proposed method seems to outperform online GRPO quite heavily on ScienceWorld, but not so much on WebShop. Is there a reason/explanation for this?

(**Question 6: Example of how the hints help**)

Could the authors put rollouts with/without hints side-by-side in the appendix, to showcase an example of how the hints help generate better rollouts/avoid common or repeated mistakes?

# References

[1] Learning by Distilling Context, https://arxiv.org/abs/2209.15189

[2] Training a Generally Curious Agent, https://arxiv.org/abs/2502.17543

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a memory-augmented RL algorithm for learning effective LLM based policies. The key idea is to use a memory buffer in sampling (rollouts) and policy learning by using the memory to enable more effective exploration in the rollout phase (by being able to reason about past experiences) and using the memory to enable learning a more generalizable policy by performing a combination of on-policy and off-policy (where main policy is not conditioned on memory) learning. Results demonstrate significant improvement in ScienceWorld and moderate improvement in WebShop benchmarks.

### Strengths
- Paper is well-motivated and well-written. Justification for improved exploration in RL for LLMs is sound.
- Use of memory in both rollout and update phase is simple yet novel in the context of RL for LLMs. 
- Strong results on ScienceWorld which demonstrate the OOD generalization of their method (due to generality of memory).

### Weaknesses
- Lack of ablations. The method introduces additional hyperparameters and components, the effects of which are largely undocumented.
    - Effect of intrinsic reward component. What is the effect of this component on the performance of the final policy (paper only documents the effect on policy entropy)? How generalizable is this reward term? It seems as if it may require further reward-shaping (i.e. tuning similarity threshold) to generalize to newer domains where naive state similarity may lead to poor performing rollouts (e.g. see static noise TV example from Pathak et al).
    - Effect of sampling proportion ($p$) between memory-free and memory-augmented rollouts. What is the effect of varying $p$?
    - Effect of update proportion ($q$) between on- and off-policy updates. What is the effect of varying $q$?
    - The authors set the KL coefficient to 0.0; does the final model lose its broad generality (e.g. on standard LLM benchmarks)?

[1] Pathak, Deepak, et al. "Curiosity-driven exploration by self-supervised prediction." International conference on machine learning. PMLR, 2017.

### Questions
Please see questions listed in weaknesses.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces the EMPO2 algorithm, which works as follows
- agents roll out trajectories
- at the end they write "tips" for solving the environment
- in future runs, some fraction of the time the agent conditions on hints form past rollouts
- we train on both types of rollouts, including a 3rd variant where the tips are stripped out of the rollout.
- they show improvements above prior works on ScienceWorld and WebShop.

### Strengths
Although each individual part of the proposed method is not original, combining them all together under one framework is an important contribution and has not been done before, particularly in the important but still early field of RL on LLMs.

The quality is good, mainly focusing on showing saturation of two in-distribution benchmarks. The paper also demonstrated signs of life on out of distribution benchmarks. The paper also sought to understand each component’s importance by doing ablations, as well as proposed future interesting extensions to the work, such as a similarity-based bonus for novel contributions to the memory bank.

I appreciated the clarity of the communication, the plots, figures, and charts all are well-designed and get the most important points across.

The significance is important, since the standard of RL for LLMs (GRPO) mainly focuses on parametric updates for each training batch rather than encouraging exploration or learning over time across training or episodes.

### Weaknesses
I am confused about the hyperparameter choices for choosing to sample between memory and non memory for rollouts and on-policy and off-policy for updates. There isn’t explanation for these choices (½ and ⅓ respectively), and there are no ablations or sweeps (although 6.3 does ablate the entire components).

Some of the plots could have better reporting. For example, figure 1 B not having error bars across the seeds, or figure 8.

For some of the baselines, I am concerned about the reported numbers being derived from other papers . For example, for WebShop, the Naive, Reflexion, GRPO, and GiGPO are taken from the paper. The paper states that all of the hyperparameters are the same for the training methods, which absolves most of my concerns, but RL methods are notorious for subtle implementation differences in the algorithm or environment which may not even be highlighted in the paper making big difference.

I also think scaling this to more “non-toy” training and evaluations would improve the paper’s significance (although not necessary since this is proposing a new method).

### Questions
- The off-policy updates seem like they could introduce stability (as you mentioned). Did you consider either using a real off-policy algorithm or introducing an importance-sampling correction where the numerator is the prob with no tips and the denom is prob with tips?

- The off-policy stabilization approach in Fig 6 is interesting. Would be nice to see error bars or multiple seeds there (to make sure it's not that one seed got unlucky in Fig 6).

- A bit ambiguous to me what numbers are reported in Table 1 (and other results) for EMPO2 -- which rollout mode were used to create these? If the "with tips" one, how many rollouts were used to create the memory bank? Same question about baselines that use multiple episodes.

- How were the hyperparameters chosen for sampling from on or off policy?

- Why are there no error bars in table 2 for naive or reflexion?

- Why are there no error bars in figure 8?

### Soundness
2

### Presentation
4

### Contribution
3
