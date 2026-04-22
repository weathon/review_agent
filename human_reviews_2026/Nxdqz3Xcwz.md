# MAPO: Momentum-Aware Policy Optimization

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a promising framework to enhance the reasoning capabilities of Large Language Models (LLMs), yet the samples from the policy model are not fully exploited during training. We propose Momentum-Aware Policy Optimization (MAPO), a critic-free, drop-in framework that preserves the simplicity of GRPO while improving exploration and stability. MAPO introduces (i) a Momentum Group Baseline that yields non-vanishing learning signals under group-standardized rewards; (ii) confidence-based prioritized replay that reuses verified successes to increase sample efficiency; and (iii) entropy-weighted token updates that concentrate gradient mass on uncertain decision points. Evaluated on math reasoning benchmarks, MAPO outperforms strong baselines—including GRPO and DAPO—in best-of-$N$ accuracy (pass@$N$), demonstrating superior exploration and discovery of correct reasoning trajectories. Ablation studies attribute the primary gains to the momentum advantage, which reduces the steps required to reach the target, alleviates stalls on homogeneous reward groups, and reduces across-seed variance. The replay and entropy components provide complementary improvements in sample utilization and gradient allocation. Overall, MAPO achieves target performance in fewer optimization steps while maintaining training stability, offering a practical enhancement to group-based RLVR methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes MAPO (Momentum-Aware Policy Optimization), a reinforcement learning framework for large language model (LLM) reasoning under verifiable rewards (RLVR). The method builds upon GRPO and DAPO by adding three main components:

1. A Momentum Group Baseline, which introduces an exponential moving average (EMA) to preserve non-vanishing learning signals even when group reward variance is small;

2. Confidence-Based Prioritized Replay, reusing verified successful trajectories to increase sample efficiency;

3. Entropy-Weighted Token Updates, allocating higher gradient weight to uncertain tokens based on entropy deviation.

### Strengths
1. The paper is clearly written, with intuitive figures and a clean derivation of the three MAPO components. It’s easy to follow the logic, and the experiments are organized coherently.
2. The paper correctly identifies real challenges in GRPO-style training, like gradient collapse under uniform rewards and inefficient credit assignment. The proposed EMA baseline and entropy weighting are sensible heuristic fixes for these problems.

### Weaknesses
1. The core ideas—EMA baseline, replay buffer, and entropy weighting—are standard heuristics widely known in reinforcement learning. MAPO simply reuses them in a GRPO context. There is no fundamentally new algorithmic insight, and the theoretical analysis restates well-known bias-variance trade-offs rather than introducing new theoretical depth. As a result, the work feels more like an engineering refinement than a conceptual breakthrough.
2. The improvements over DAPO are marginal (often within ±2%), and sometimes inconsistent across models and tasks. For example, on the 14B model, MAPO slightly underperforms DAPO on Pass@1 despite higher Pass@N. The paper doesn’t analyze whether these differences are statistically significant. The overall gains are too modest to justify acceptance at a top-tier conference.
3. The so-called Momentum Group Baseline is just an exponential moving average of past group means — a very standard variance-reduction technique. The “momentum advantage” term doesn’t introduce fundamentally new learning dynamics; it’s a reweighted baseline update similar to existing adaptive control variates. The name “momentum” may oversell what is effectively a smoothed baseline.
4. The replay buffer stores only one “best” trajectory per prompt, which risks biasing toward early found solutions and limits diversity. Entropy weighting sometimes reduces accuracy on hard datasets like AIME24. 
5. There’s no analysis of sensitivity to hyperparameters like EMA decay α or temperature γ.

### Questions
1. Are the improvements statistically significant across multiple seeds?
2. How sensitive are results to α (EMA decay) and η (momentum coefficient)?

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
4

### Summary
In this paper, the authors propose a novel critic-free policy-gradient-based reinforcement learning algorithm called Momentum-Aware Policy Optimization (MAPO). The proposed method comprises three core components: (1) a momentum-based approach for advantage estimation, (2) a confidence-based prioritized replay mechanism for historical trajectory reuse, and (3) an advantage reweighting scheme that emphasizes tokens with increasing entropy.

The proposed method demonstrates several improvements over baseline approaches such as GRPO and DAPO. The ablation studies are well-executed, and the theoretical analysis of each component is well-structured.

However, from my perspective, the motivation underlying each component lacks sufficient conviction. Additionally, the experimental evaluation is limited to a single mathematics-related dataset, leaving the validity and scalability of the proposed method yet to be fully established.

### Strengths
1. Despite the complexity of the notation, the presentation remains clear and accessible. Each component is introduced in a well-organized manner, avoiding potential confusion.

2. The ablation studies are comprehensive. The experiments convincingly demonstrate that each proposed component contributes to enhancing the baseline algorithm's performance.

3. Appendix C provides theoretical analysis supporting the methods presented in the main text. The proof sketches are easy to follow and closely aligned with the proposed methodologies.

### Weaknesses
1. Regarding the proposed "Momentum Advantage" component, the authors argue that the motivation is to address vanishing advantages when batch rewards are homogeneous. Personally, I have serious doubts about this motivation. Compared with traditional REINFORCE, the vanishing of homogeneous rewards is precisely the key to GRPO's success [1]. Similarly, DAPO discards homogeneous batched samples, thereby improving training efficiency. From the perspective of policy gradient methods, the objective is to emphasize favorable trajectories (with positive advantages) and suppress unfavorable ones (with negative advantages). It remains unclear why samples with zero advantages should be utilized, as this appears to contradict the fundamental principle of policy gradient-based methods. Furthermore, the introduced momentum mechanism results in non-zero mean advantages within each update batch, which, in my experience, may lead to faster entropy collapse or entropy blow-up.

2. The motivation underlying the proposed "Experience Replay" component is also flawed from my perspective. It is designed to address prompts that initially yield good responses but fail to do so after one epoch of training. As the policy's capability continues to improve, how is such deterioration even possible? I suspect that the activation of this module would be extremely rare during the training process of any model. Moreover, the replayed samples are approximately one epoch old, rendering them highly off-policy. Incorporating such samples into the update batch may result in excessively large KL penalties.

3. The motivation behind "entropy-guided advantage reweighting" remains questionable. Why should tokens exhibiting "higher-than-usual uncertainty" receive greater emphasis?


[1]. Xiong et.al, A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce.

### Questions
1. Could the authors provide a more detailed explanation of the motivation underlying each proposed component?

2. The training curves of DAPO appear unusual compared with those of GRPO in Figure 3. Is the comparison fully fair? Did the authors strictly adhere to the experimental settings reported in the original DAPO paper?

3. Could the authors provide additional experimental results on other domains, such as coding or general reasoning tasks?

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
This paper proposes Momentum-Aware Policy Optimization (MAPO), an extension to GRPO. The method combines a momentum group baseline to prevent gradient collapse under low reward variance, a confidence-based prioritized replay mechanism to reuse verified successes, and an entropy-weighted token update to focus learning on uncertain reasoning steps. Experiments on several mathematical reasoning benchmarks show improvements over GRPO and DAPO in Pass@1 and Pass@N, with ablation studies indicating complementary effects among the three components.

### Strengths
1. The paper targets an important weakness of GRPO in realistic low-variance reward settings and provides an elegant baseline mechanism with quantifiable variance reduction.
2. The theoretical analysis is detailed, including finite group and variance decomposition results.
3. Experimental results are clearly demonstrate sample efficiency and learning stability gains.

### Weaknesses
1. The integration between the three proposed components is not fully clear. The paper could discuss a unifying motivation or design principle, since the “momentum‑aware” name primarily describes only the first component.
2. Some evaluation benchmarks, particularly AIME24/25, are small in size, which limits the statistical reliability of reported improvements. Confidence intervals or significance tests would make the results more convincing.
3. To better position MAPO among recent methods, the related work discussion should be expanded to include both replay-based optimization (e.g., RePO, RLEP) and recent work on token reweighting (e.g., 'Beyond the 80/20 Rule...'; 'Do Not Let Low-Probability Tokens...').
4. The off‑policy nature of replayed samples is a potential concern. Appendix C.2 provides a theoretical bound, but the main text should summarize this clearly to justify the design.
5. The hyperparameters α and η are fixed throughout. A short sensitivity analysis would help confirm that the method is not overly dependent on these values.
6. In Table 1, MAPO’s improvements over baselines are sometimes small or inconsistent across datasets and metrics, and clarification on these fluctuations would help interpret the results.

### Questions
1. How sensitive is MAPO to EMA decay α and momentum weight η? Could a high α lead to stale baselines?
2. Is there a deeper conceptual connection among the three mechanisms beyond their empirical complementarity?
Strengths

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
4

### Summary
The paper introduces a novel framework that enhances Large Language Model reasoning through Reinforcement Learning with Verifiable Rewards (RLVR). It tackles two key limitations of group-based methods like GRPO: vanishing learning signals due to low reward variance and suboptimal token-level credit assignment. MAPO's solution integrates three core components: a Momentum Group Baseline for stable learning signals, a Confidence-Based Prioritized Replay for sample efficiency, and Entropy-Weighted Token Updates for precise gradient focus. Extensive evaluations on mathematical reasoning benchmarks demonstrate MAPO's superiority over GRPO and DAPO in both Pass@1 and Best-of-N accuracy, with faster convergence and improved training stability.

### Strengths
1. The Momentum Group Baseline stands out as a significant contribution. It elegantly resolves a fundamental issue in group-based policy optimization by employing an Exponential Moving Average (EMA) of historical rewards. This ensures a persistent learning signal, directly countering a primary cause of training stagnation and fostering more robust and continuous improvement.

2. The framework is highly practical, functioning as a "drop-in" enhancement to existing methods like GRPO. Its three components are complementary yet conceptually independent, facilitating straightforward implementation and enabling clear ablation studies to understand the contribution of each part.

### Weaknesses
1. The design of the prioritized replay buffer appears somewhat conservative, as it stores only the single highest-confidence successful trajectory per prompt. This strategy may inadvertently limit the diversity of positive examples during training, potentially constraining the explored solution space. There is a risk of the model overfitting to a specific style of successful trajectory, particularly for problems that admit multiple valid solution paths.

2. While the results on mathematical reasoning benchmarks are compelling, the empirical evaluation is confined to this single domain. The paper does not demonstrate the framework's efficacy on other prominent RLVR tasks, such as code generation or symbolic reasoning. This leaves its generalizability and performance across a broader range of domains as an open question.

### Questions
The hyperparameters for the momentum mechanism (EMA decay α and momentum coefficient η) are crucial to the method's performance. How sensitive are the results to the specific values chosen (0.9 and 0.1, respectively), and is there a rationale or empirical evidence for these choices? Could an adaptive scheduling mechanism for these parameters further improve performance across different training phases or tasks?

The entropy-weighting mechanism aims to focus learning on "critical decision points," which is assumed to correlate with high token-level entropy. However, could this approach inadvertently amplify the gradient for tokens that are highly uncertain but semantically unimportant (e.g., transitional phrases or stylistic variations)? How does the method ensure that high entropy reliably indicates a critically reasoning-relevant token?

### Soundness
3

### Presentation
3

### Contribution
3
