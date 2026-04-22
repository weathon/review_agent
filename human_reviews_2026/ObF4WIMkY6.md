# EEPO: Exploration-Enhanced Policy Optimization via Sample-Then-Forget

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Balancing exploration and exploitation remains a central challenge in reinforcement learning with verifiable rewards (RLVR) for large language models (LLMs). Current RLVR methods often overemphasize exploitation, leading to entropy collapse, diminished exploratory capacity, and ultimately limited performance gains. Although techniques that increase policy stochasticity can promote exploration, they frequently fail to escape dominant behavioral modes. This creates a self-reinforcing loop—repeatedly sampling and rewarding dominant modes—that further erodes exploration. We introduce **E**xploration-**E**nhanced **P**olicy **O**ptimization (**EEPO**), a framework that promotes exploration via two-stage rollouts with adaptive unlearning. In the first stage, the model generates half of the trajectories; it then undergoes a lightweight unlearning step to temporarily suppress these sampled responses, forcing the second stage to explore different regions of the output space. This *sample-then-forget* mechanism disrupts the self-reinforcing loop and promotes wider exploration during rollouts. Across five reasoning benchmarks, EEPO outperforms GRPO, achieving average relative gains of 24.3\% on Qwen2.5-3B, 33.0\% on Llama3.2-3B-Instruct, and 10.4\% on Qwen3-8B-Base.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenge of insufficient exploration and resulting entropy collapse in Reinforcement Learning with Verifiable Rewards (RLVR) used for training large language model (LLM) reasoning capabilities. Current methods often over-exploit dominant trajectories, and while adding randomness increases stochasticity, it fails to effectively shift the policy away from these modes. To counter this, the authors propose Exploration-Enhanced Policy Optimization (EEPO), a framework that modifies the rollout phase with a novel "sample-then-forget" mechanism. EEPO performs rollouts in two stages: after sampling the first half of trajectories, it applies a temporary, lightweight unlearning step to the rollout model to suppress the just-sampled responses, using a complementary loss function gated by an entropy threshold. This forces the second stage of sampling to explore different, less probable regions of the output space, actively steering away from dominant modes. The standard GRPO objective is then used to update the main policy model using the combined trajectories from both stages, decoupling the exploration enhancement from the policy optimization itself. Experiments across five reasoning benchmarks and three different LLMs show that EEPO consistently outperforms GRPO and other exploration-focused baselines, achieving significant average accuracy gains.

### Strengths
1. The paper investigates the critical problem of insufficient exploration and entropy collapse in Reinforcement Learning with Verifiable Rewards (RLVR), a central challenge that limits the performance gains and generalization of LLMs trained for reasoning tasks.
2. The proposed Exploration-Enhanced Policy Optimization (EEPO) method is conceptually intuitive, employing a straightforward "sample-then-forget" mechanism within a two-stage rollout process to actively suppress dominant modes and encourage exploration.
3. The effectiveness and robustness of EEPO are demonstrated through comprehensive experiments conducted across five different mathematical reasoning benchmarks and three distinct LLM architectures and sizes.

### Weaknesses
1. The comparison primarily focuses on standard GRPO and variations based on randomness or rollout count, neglecting direct benchmarks against more advanced, contemporary RLVR algorithms like the full DAPO framework.
2. Experimental validation is confined to LLMs with fewer than 10 billion parameters (specifically 3B and 8B models), leaving the scalability and effectiveness of EEPO on larger foundation models untested.
3. While EEPO aims to improve performance via enhanced exploration, the provided training curves show its mean reward is comparable to or even slightly lower than standard GRPO (Fig. 6), and the paper lacks a detailed explanation for why the proposed method results in slightly faster wall-clock training times despite incorporating an additional unlearning step.

### Questions
Please see paper weaknesses.

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
2

### Summary
This work introduces EEPO, a framework that addresses the exploration-exploitation dilemma in RLVR. EEPO enhances the rollout process through a sample-then-forget mechanism, which temporarily suppresses recently sampled trajectories during generation. This deliberate exclusion discourages the policy from repeatedly collapsing into dominant, high-probability modes and instead promotes the discovery of alternative, underexplored reasoning pathways. By converting passive stochasticity into purposeful, diversity-driven exploration, EEPO steers the policy toward more comprehensive coverage of the output distribution.

Extensive experiments across three model families and five mathematical reasoning benchmarks show that EEPO outperforms existing methods in final performance while preserving comparable training efficiency.

### Strengths
(1) Well-written (2) Detailed experiment (3) The problems related to training efficiency that have been solved are distinctive and seem valuable to the industrial sector

### Weaknesses
N/A

### Questions
I do not know this specialized research field very well. I will adjust my score and optimize my review document based on the evaluations of other expert reviewers and my performance during the rebuttal period.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors introduce a novel method to maintain exploration while learning in LLMs with RLVR. Current approaches overemphasize exploitation, causing premature entropy policy collapse. The current method avoids this by adaptive unlearning of the dominant modes and focusing on secondary modes. Improving performance compared to baselines is shown.

### Strengths
The problem solution is simple and elegant. This is complemented by simple and useful intuitions that help us to understand the proposed algorithm.

### Weaknesses
Nice intuitions are presented, like in Fig. 2, but no theory is provided to back up those intuitions. This means that improvement can strongly depend on parameters and on a case-by-case basis. 

Ablations studies are missing, so it is unclear what aspect of the introduced algorithm is critical: is it enough to have an entropy-conditioned activation of unlearning just based on Eq. 8, assuming a constant $p_{clip}$?

Is it possible that in some problems no secondary mode exists? Then how would the method fare in discovering the most promising actions in the vast space of tokens and reasoning trajectories?

While performance increases compared to other methods, the performance is very close to GRPO with updated parameters, so it seems that the gains are relatively small. 

As far as I can see, no comparison with other methods of RL exploration is introduced (epsilon-greedy with optimal epsilon, entropy regularization, and such). Does the method outperform the obvious modifications of the algorithms with those additional exploration tricks?

### Questions
See weaknesses.

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
The authors proposes Exploration-Enhanced Policy Optimization (EEPO), a novel framework for reinforcement learning with verifiable rewards (RLVR) that addresses policy entropy collapse.

### Strengths
1. The "sample-then-forget" idea is a clever and novel approach to the exploration-exploitation problem. Instead of just adding indiscriminate noise (like increasing temperature), it actively and strategically steers the policy away from dominant modes it is starting to overfit on.

2. The method demonstrates consistent and significant performance gains over strong GRPO baselines across five challenging mathematical reasoning benchmarks and three different language models. The average relative improvements are substantial (e.g., +33.0% on Llama3.2-3B-Instruct).

### Weaknesses
1. The method introduces at least two key new hyperparameters: the entropy threshold $\alpha$ for activating unlearning and the unlearning rate $\eta$. The paper uses fixed values ($\alpha=0.3$, $\eta=3\times10^{-3}$) without providing an ablation study or discussion on how these values were chosen or how sensitive the model's performance is to them. This could be a point of fragility.

2. The unlearning step modifies the rollout policy ($\pi_{\theta^{\prime}}$) in the middle of generating a single batch of trajectories. It is unclear how the importance sampling (IS) ratio for the standard GRPO objective (Eq. 2) remains valid when trajectories in the same batch ($O$) are drawn from two different policies (pre-unlearning and post-unlearning). This potential violation of the IS assumption needs a more rigorous justification.

### Questions
Please address the concerns in weakness section.

### Soundness
2

### Presentation
3

### Contribution
2
