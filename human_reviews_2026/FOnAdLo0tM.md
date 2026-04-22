# Do Not Let Low-Probability Tokens Over-Dominate in RL for LLMs

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 2, 8, 6

## Abstract
Reinforcement learning (RL) has become a cornerstone for enhancing the reasoning capabilities of large language models (LLMs), with recent innovations such as Group Relative Policy Optimization (GRPO) demonstrating exceptional effectiveness. In this study, we identify a critical yet underexplored issue in RL training: low-probability tokens disproportionately influence model updates due to their large gradient magnitudes. This dominance hinders the effective learning of high-probability tokens, whose gradients are essential for LLMs' performance but are substantially suppressed. To mitigate this interference, we propose two novel methods: Advantage Reweighting and Low-Probability Token Isolation (Lopti), both of which effectively attenuate gradients from low-probability tokens while emphasizing parameter updates driven by high-probability tokens. Our approaches promote balanced updates across tokens with varying probabilities, thereby enhancing the efficiency of RL training. Experimental results demonstrate that they substantially improve the performance of GRPO-trained LLMs, achieving up to a 46.2% improvement in K&K Logic Puzzle reasoning tasks. Our implementation is available at https://github.com/zhyang2226/AR-Lopti.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work first identifies the pattern that tokens with low probability could induce significantly higher gradients than those with high probability tokens. Basing on this observation, this work then theoretically prove that the gradient norms are proportional to (1-$\pi$), where $\pi$ is the token probability.  Then, this work proposes two algorithms, advantage reweighing and low-probability token isolation to alleviate the issue.

The proposed method are evaluated on two kinds of tasks: K&K logic puzzles and mathematical reasoning. On both tasks, the proposed methods can outperform the original GRPO. Furthermore, on K&K, the results show that further improvement can be obtained by combining advantage reweighing and low-probability token isolation.

### Strengths
1, This work rigorously describes how the gradient norm is decided by token probability, during online RL training. This provides valuable insights to the community, and encourages exploration on token-level reward design.

2, The two proposed methods are sound, and straightforward. According to empirical results, they can effectively improve performance over the original GRPO.

3, I especially appreciate the in-depth ablation studies. They demonstrate: 1, how different tokens with different probs affect the training dynamics; 2, different hyper-parameters affect the training dynamics compared to GRPO.

### Weaknesses
1, The first concern is with the hyper parameters. According to Figure 6, the choices of $\alpha$ and $\eta$ significantly affect the training dynamics. But considering that the training and test curves are similar, introducing a validation mechanism would be very helpful. Would you like to discuss such mechanism? 

2, On a philosophical level, I am not sure if such fixing (improving the effect of high-probability grad norms) is necessary, especially on positive samples. Specifically, those tokens are already of high probability so that improving their grad norm can only marginally change their distribution. Figure (6) is helpful in demonstrating the effects of low and high probability tokens. But those experiments are different from gradient update. Specifically, inclusions of different probability tokens in figure (6) is hard: tokens are either updated or not updated; but in gradient update, all tokens are updated, but the update magnitudes (gradient norm) are different. 

3, I do appreciate the bounds over the gradient norm. But I am also concerned that $c_l$ and d_l$ could be sufficiently large, dominating the effect of token probability.

### Questions
1, Could you also report the curves on mathematical tasks (e.g., MATH 500)? 

2, How did you select the hyper parameters in Table 3?

3, Do you have any hypothesis for whether combining advantage reweighing and Lopti did not improve for math problems?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This research identifies a critical but previously overlooked inefficiency in the reinforcement learning (RL) process for Large Language Models (LLMs): the disproportionate dominance of low-probability tokens in policy gradient updates. The paper argues that this phenomenon, termed the "tyranny of the unlikely token," actively hinders the model's ability to refine its core knowledge. To address this, the paper proposes the advantage reweighting and Low-probability Token Isolation (Lopti). Experiments on complex reasoning and math tasks using GRPO demonstrate that both methods significantly outperform the baseline.

### Strengths
1. The research is built on a solid theoretical foundation, with Proposition 4.2 mathematically demonstrating that a token's gradient norm is inversely related to its probability. This elevates empirical observation to a predictable phenomenon.
2. The paper excels at communicating a complex technical subject. The narrative is logical, and the framing of the issue as the "tyranny of the unlikely token" is both memorable and effective.

### Weaknesses
1. The author claim that low-probability tokens dominate model updates during RL training and that this dominance may impede the precise adjustment of the probability distribution across all tokens.   In fact, the high-entropy minority tokens drive effective reinforcement learning for llm reasoning, we do not need to adjust the probability distribution across all tokens based on RL.

2. RL Algorithm Specificity: Experiments were conducted exclusively with GRPO. All experiments used models from the Qwen2.5 family.  There is no other strong comparable baselines on the benchmark.

3. Acknowledging that Lopti doubles update time is treated as a minor caveat. A 100% increase in backward pass cost is prohibitive for most large-scale training operations, rendering Lopti impractical for all but the most specialized applications.

4.The paper offers no guidance on how to co-tune the two new hyperparameters (α and η).




[1] Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning, arXiv 2025.

### Questions
The solutions are presented without comparison to existing techniques for managing noisy gradients.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper identifies and analyzes an overlooked source of update bias in RL training of LLMs: low-probability tokens produce larger gradient magnitudes and can therefore dominate parameter updates. The authors (1) derive bounds showing token-wise gradient norms scale with $1−\pi$, (2) propose two mitigation methods: Advantage Reweighting and Low-Probability Token Isolation (Lopti), and (3) demonstrate consistent empirical gains on several benchmarks.

### Strengths
1. The theoretical derivation and supporting experiments are well aligned; the motivation is lucid and convincing.
2. The proposed methods are low-cost, effective, and easy to implement in practice.
3.  The authors provide careful experiments and analysis, including evaluations on multiple algorithms, multiple domains, and several ablations.

### Weaknesses
See questions below.

### Questions
1. The paper evaluates only naive GRPO and REINFORCE++ baselines. It’s better to consider whether widely used GRPO variants (for example, DAPO) might interact with or mitigate the same gradient-magnitude imbalance.
2. To demonstrate broader generalization, It’s better to add experiments across different model families and sizes (for example, Qwen-3 and Llama-3.1).

### Soundness
4

### Presentation
4

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
This paper claims that low-probability tokens have higher gradient norms which over-dominates the gradient update when doing RLVR to an extent that high-probability tokens might update even in the opposite direction. After showing this empirically, the show via mitigating this via isolating or re-weighting the low-probability tokens, we have a better RLVR algorithm.

### Strengths
I think this paper's main claim is new to the field: the gradient of the low probability tokens are dictating the changes in the probability of high probability tokens (not the advantages of those high-probability tokens) because their gradient magnitude is large. The paper shows mitigating this improves results on K&K puzzle. I am not familiar with this puzzle, but it seems it is a challenging task. I think the K&K results are the strongest in supporting the evidence.

### Weaknesses
The paper points out a very interesting observation. However, the math results are not consistent with the theory: their methods and GRPO are almost achieving the same score. I think it is because they are testing this on a R1-Zero style scenario where they start from a base model. Papers that do RL on base models show this quick recovery of some performance the curves are flat afterwards. I think the paper should have been done on a native reasoning model on R1-Distill-1.5B as they show usually continued progress even in math problems. Therefore, I would say the experiments are not providing the  significant signal that this algorithm scales. In terms of the reasoning to support the conclusion that mitigating the dominance of low-probability tokens is key in RLVR, while the K&K results support the claim, we don't know if this holds in more serious scenarios like math problems. I don't know if K&K is considered a task that its performance forecasts the performance on downstream hard reasoning.

### Questions
1-Is there evidence that K&K is a hard reasoning benchmark? 
2-Have you tried this method on native reasoning models, not base models, and observe improvement?

### Soundness
3

### Presentation
3

### Contribution
3
