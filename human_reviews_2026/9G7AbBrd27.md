# Diversity-Incentivized Exploration for Versatile Reasoning

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a crucial paradigm for incentivizing reasoning capabilities in Large Language Models (LLMs). Due to vast state-action spaces and reward sparsity in reasoning tasks, existing methods often struggle with deficient exploration and poor sample efficiency. In the paper, we propose **DIVER** (**D**iversity-**I**ncentivized Exploration for **V**ersatil**E** **R**easoning), an innovative framework that highlights the pivotal role of global sequence-level diversity to incentivize deep exploration for versatile reasoning. We first conduct a primary empirical study to reveal a strong positive correlation between global diversity and reasoning capacity. Building on this insight, we introduce global diversity incentives as an intrinsic reward to promote deep exploration in a semantically structured space. Incorporating the intrinsic reward, we develop a potential-based reward shaping mechanism to preserve optimal policy invariance and design simple heuristics to mitigate possible reward hacking. Experimental results show that DIVER outperforms competitive RLVR baselines with various exploration strategies on both in-domain and out-of-domain tasks, excelling in both Pass@1 and Pass@k evaluations. Our code is available at https://github.com/NJU-RL/DIVER.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies RL from verifiable rewards for LLM reasoning. The authors argue that token-level exploration is not enough, and we should explore entire reasoning paths. They propose a small change to GRPO: for each prompt, sample G rollouts, measure sequence-level diversity across that group, and add this as an intrinsic reward. The bonus is only given when a rollout is verified correct and is clipped. On math benchmarks (AIME/AMC/Minerva/OlympiadBench/MATH-500) the method improves Pass@1/Pass@k. Overall, the contribution is mixed: conceptual (global/trajectory exploration) and algorithmic (a training-time diversity bonus) with solid empirical support in the math setting.

### Strengths
- Very simple, plug-and-play idea. Easy to add to GRPO and existing RL-LLM training frameworks.
- Clear empirical gains on hard math tasks and strong improvements.
- Training diagnostics are helpful (diversity/entropy/“solve-none”).
- The correctness mask and clipping are practical and effective to avoid length explosion.
- Works across several backbones and seems to be applicable to various LLM architectures.

### Weaknesses
- Theory vs practice mismatch. The shaping argument (policy invariance) assumes a state-only potential. Here the bonus is group-dependent and masked by correctness. The paper does not provide a formal justification for invariance in this setting.
- Equational diversity extraction is under-specified. The paper defines the set metric but does not describe the extraction pipeline (parser/regex, failures, error rate). Hard to reproduce precisely.
- The paper lacks ablation studies to understand the impact of the hyperparameters or the performance of the method. 
- Scope limited to single-turn math. The “versatile reasoning” claim is not tested on code, multi-turn, or tool-use tasks.
- Code is not available which hinders reproducibility.

### Questions
- How exactly do you extract equations for ED (parser, regex, normalization)? What is the failure rate?
- What is the training overhead (tokens/sec, wall-clock) vs GRPO?
- How sensitive is the method to λ, group size G, and temperature?
- Could you make the code available for reproducibility?

### Soundness
2

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
The paper identifies global sequence-level diversity as a driver for better exploration and improved reasoning capability in RL finetuning of LLMs, and proposes DIVER: an intrinsic-reward framework that (1) measures sequence diversity (Textual Diversity via inverted BLEU, and Equational Diversity via formula uniqueness), (2) injects diversity as a potential-based shaping reward to preserve optimal policy invariance, and (3) uses simple heuristics to mitigate reward-hacking. Experiments on several math benchmarks and general domains show consistent gains over GRPO and other exploration baselines; ablations and diagnostics support the claims.

### Strengths
1. The motivation to enhance exploration through global sequence-level diversity is clear and compelling.
2. The method is simple and pricipled.
3. The experimental evaluation is thorough.

### Weaknesses
1. Limited generality of diversity metrics: The choice of diversity metrics lacks clear guidance. The two proposed ones have different strengths and weaknesses and may not generalize to all scenarios. It would be natural to consider combining them or testing alternatives such as LLM-based or embedding-based measures.
2. No analysis of computational cost: The paper does not discuss whether computing diversity metrics adds significant overhead or affects training efficiency.
3. Theorem 1 offers limited novelty: It follows directly from prior reward-shaping theory[1], and the reward-hacking solution breaks its assumptions, reducing its relevance.
4. Insufficient ablations and sensitivity analysis: Key components and hyperparameters (e.g., λ, reward-hacking solution) are not thoroughly tested for robustness.
5. Missing discussion of potentially related prompt selection methods: Prior work promotes uncertainty by selecting prompts with success rates near 0.5 [2,3,4]. Since both pursue response uncertainty, their connection deserves discussion.

[1] Policy invariance under reward transformations: Theory and application to reward shaping.\
[2] Self-Evolving Curriculum for LLM Reasoning\
[3] Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning\
[4] Can prompt difficulty be online predicted for accelerating rl finetuning of reasoning models?

### Questions
see weaknesses

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
The paper proposes DIVER, a reinforcement learning framework that enhances reasoning in LLMs by introducing global sequence-level diversity as an intrinsic reward.Built upon GRPO, DIVER uses a potential-based reward shaping to preserve optimal policy invariance and mitigate reward hacking.Experiments on multiple math reasoning benchmarks show consistent but modest gains in Pass@1 and Pass@k (up to 32) compared to Entropy-RL and Pass@k Training.

### Strengths
1. The paper presents a clear motivation and a well-structured method for enhancing exploration in LLM reasoning.

2. Experiments are comprehensive and demonstrate consistent improvements across multiple benchmarks and model scales.

### Weaknesses
1. The improvement on Pass@k is relatively small and only evaluated up to k = 32, while [1] tests up to k = 1024, including the base model.

2. The comparison on Pass@1 in Table 1 seems not entirely fair, since methods like Pass@k Training [2] mainly aim to improve Pass@k. Their Pass@1 performance is usually enhanced through a two-stage training process (first optimizing Pass@k, then fine-tuning for Pass@1).

3. In Table 3, GRPO mostly remains the best baseline, and the base model results are not included, making DIVER’s advantage less consistent.

4. The diversity metrics (BLEU, formula overlap) may not effectively capture semantic-level reasoning diversity.

### Questions
1. One confusing point is that in Table 3, GRPO is actually the best baseline in most cases (compared to Entropy-RL and Pass@k Training), which seems inconsistent with the expected performance of these methods.

2. Would DIVER still outperform GRPO and the base model at larger values of k?

[1] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? arXiv preprint arXiv:2504.13837, 2025.

[2] Zhipeng Chen, Xiaobo Qin, Youbin Wu, Yue Ling, Qinghao Ye, Wayne Xin Zhao, and Guang Shi. Pass@K Training for Adaptively Balancing Exploration and Exploitation of Large Reasoning Models. arXiv preprint arXiv:2508.10751, 2025.

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
This paper introduces DIVER (Diversity-Incentivized Exploration for Versatile Reasoning), a framework that uses global sequence-level diversity as an intrinsic reward to enhance exploration in reinforcement learning for Large Language Models (LLMs). The framework incorporates diversity incentives as an intrinsic reward and utilizes a potential-based reward shaping mechanism to ensure policy invariance and mitigate reward hacking. Empirical results demonstrate that DIVER outperforms existing RLVR baselines, improving performance on both in-domain and out-of-domain tasks, with notable improvements in the Pass@1 and Pass@k metrics.

### Strengths
1. This paper studies an important problem on how to boost the exploration 

2. The paper provides comprehensive experimental results, evaluating DIVER on multiple benchmarks (in-domain and out-of-domain). It also assesses performance from different perspectives, including final benchmark performance and training dynamics.

3. DIVER achieves performance improvement across different benchmarks, but the improvement is not consistent compared to baselines methods.

### Weaknesses
1. While Textual Diversity (TD) and Equational Diversity (ED) are designed to incentivize exploration, it is unclear whether they truly measure meaningful diversity in reasoning. For example, ED could be easily hacked by generating irrelavent formulas. As shown in figure 4, DIVER leads to a decrease in the solve_all rate, which raises concerns about whether the model is exploring paths of lower quality.


2. The paper claims that "Longer Horizons Improve Performance", but as shown in Figure 7 and 8, all experiments use response length at 1-2k tokens. It remains unclear whether this finding holds for thinking models with longer horizon reasonings. Verification on such models is needed.

3. More comprehensive ablations are necessary to demonstrate the effectiveness of the proposed method. As show in Figure 9, a larger $\lambda$ could bring almost no performance improvement.

### Questions
For rollout data that already exhibits high entropy, is it still necessary to introduce additional diversity incentives? Could this approach unintentionally encourage false positives, where the model generates diverse but incorrect responses? How does DIVER ensure that diversity incentives do not lead to an over-exploration of irrelevant or incorrect reasoning paths?

### Soundness
3

### Presentation
3

### Contribution
2
