# Low-probability Tokens Sustain Exploration in Reinforcement Learning with Verifiable Reward

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has propelled Large Language Models in complex reasoning, yet its scalability is often hindered by a training bottleneck where performance plateaus as policy entropy collapses, signaling a loss of exploration. Previous methods typically address this by maintaining high policy entropy, yet the precise mechanisms that govern meaningful exploration have remained underexplored. Our analysis suggests that an unselective focus on entropy risks amplifying irrelevant tokens and destabilizing training. This paper investigates the exploration dynamics within RLVR and identifies a key phenomenon: the gradual elimination of what we term \textbf{\textit{reasoning sparks}}: a crucial subset of low-probability tokens such as ``wait'', that initiate diverse reasoning paths. We find that while abundant in pre-trained models, these sparks are systematically extinguished during RLVR due to over-penalization, leading to a degeneracy in exploration. To address this, we introduce Low-probability Regularization (Lp-Reg). Its core mechanism regularizes the policy towards a heuristic proxy distribution. This proxy is constructed by first applying a probability threshold to filter out noise tokens and then re-normalizing the distribution over the remaining candidates. This process effectively shields the exploratory tokens from destructive updates. Experiments show that Lp-Reg enables stable on-policy training for around 3,000 steps over 81,204 GPU-Hours, a regime where baseline entropy-control methods collapse. This sustained exploration leads to state-of-the-art performance, achieving a $60.17\%$ average accuracy on five math benchmarks, an improvement of $2.66\%$ over prior methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the exploration dynamics within RLVR and identifies that the gradual elimination of what the authors term "reasoning sparks"—a crucial subset of low-probability tokens  that initiate diverse reasoning paths. While these tokens are abundant in pre-trained models, they are systematically extinguished during RLVR due to over-penalization, leading to a degeneracy in exploration. Previous methods typically address this issue by maintaining high policy entropy, but the precise mechanisms that govern meaningful exploration have remained underexplored. The authors' analysis suggests that an unselective focus on entropy risks amplifying irrelevant tokens and destabilizing training. To address this, the paper introduces Low-probability Regularization, which regularizes the policy towards a heuristic proxy distribution.

### Strengths
1.  The motivation that low-probability tokens might indicate valuable exploration is well-argued and supported by analysis.
2.  The authors compare their method with up-to-date baselines.

### Weaknesses
1.  The idea of training on low-probability or high-entropy tokens has been heavily investigated in prior work (as cited in the related work section). Compared to these works, the main difference introduced here is the filtering of noisy data, which may make the contribution somewhat limited.
2.  While the authors claim that probability might be a better metric than entropy, the provided support is mainly heuristic; a more theoretical explanation would strengthen the argument.
3.  The authors select different probability percentile thresholds for different base models. It would be helpful to clarify how these hyperparameters are tuned and whether this increases the tuning cost for future applications.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work identifies that exploration collapse in Reinforcement Learning with Verifiable Rewards (RLVR)  might be caused by the loss of valuable, low-probability reasoning actions. To counter this, the authors propose Low-probability Regularization (Lp-Reg), a method that distinguishes meaningful low-probability tokens from noise by leveraging their relatively higher likelihood in local predictive contexts. 

By filtering out noise and regularizing the policy toward these high-value sparks, Lp-Reg preserves the policy’s useful low-probability tail, enabling more stable training and achieving about 2.66% improvement in test accuracy over baselines.

### Strengths
(1) This paper is well-written
(2) The visualizations are clear and rich in information
(3) The design method is simple and effective, and has good generalization potential.
(4)The exploration issues that this article focuses on are indeed the key problems recognized in the current RLVR field. 
(5) The baselines in the experiment were all relatively new
(6) The correlation between low-probability tokens and entropy was explored for the first time

### Weaknesses
- Expert dependence in hyperparameter configuration: The method’s parameter settings rely heavily on expert prior knowledge, which may hinder its accessibility and ease of use for practitioners without domain-specific experience.

- Computational overhead of the regularization procedure: The proposed regularization appears to involve, for each training batch, a full-vocabulary statistical analysis and ranking of token logits. Given the size of modern language model vocabularies (often >50K tokens), this step could incur prohibitive computational costs, raising concerns about scalability.

- Need for trajectory-level analysis of low-probability tokens: The current analysis lacks a trajectory-level characterization of low-probability tokens. Under Monte Carlo sampling, advantages within a single trajectory are typically uniformly positive or negative. It would be highly informative to examine how low-probability tokens are distributed across positive versus negative trajectory segments—this could reveal whether such tokens are genuinely associated with high-quality reasoning or merely stochastic noise.

- Clarification of conceptual distinction from prior work: A related study [1] also investigates the role of low-probability tokens in policy learning but appears to reach a contrasting conclusion. The authors are encouraged to provide a detailed comparative analysis that clearly articulates the fundamental differences in assumptions, definitions (e.g., what constitutes a “valuable” low-probability token), and methodological approaches between this work and [1]. Highlighting this distinction will strengthen the novelty and conceptual contribution of the current paper.

[1]: Do Not Let Low-Probability Tokens Over-Dominate in RL for LLMs

### Questions
Please refute the above-mentioned weakness through experiments or discussions. I will adjust the score based on the author's feedback and the opinions of other reviewers

### Soundness
3

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
This paper studies why RLVR often stops improving when models lose exploration ability. The authors find that during RLVR training, certain rare but important reasoning sparks gradually disappear because they are over-penalized. To fix this, they propose Lp-Reg, which protects these exploratory tokens by regularizing the policy toward a filtered, re-normalized proxy distribution. This approach stabilizes training and maintains exploration for longer, leading to better reasoning performance, with a 60.17% average accuracy on five math benchmarks, outperforming previous methods by 2.66%.

### Strengths
1. Novel insight into exploration collapse: this work introduces the concept of ***reasoning sparks***, offering a fresh perspective on why exploration diminishes during RLVR training.


2. Simple yet effective method: the proposed Lp-Reg is lightweight and easy to implement, improving exploration stability without architectural changes.


3. Strong empirical performance: this work achieves state-of-the-art results on five math reasoning benchmarks, outperforming previous methods by 2.66% on average.


4. Improved training stability: Lp-Reg effectively prevents early entropy collapse, enabling longer and more stable on-policy training phases.

### Weaknesses
1. Heuristic dependency: the proposed method relies on manually selected probability thresholds and re-normalization heuristics, lacking strong theoretical justification. As shown in Eq. 6, additional hyperparameters such as $\tau$ and $\delta_\rho$ are introduced, which require extra tuning and may increase the method’s sensitivity to hyperparameter selection.


2. Incomplete mechanistic understanding: while the paper identifies the disappearance of ***reasoning sparks***, the precise connection between these tokens and the underlying reasoning structures remains ambiguous. The authors state that “while both are empirically low-probability tokens within a long trajectory, a reasoning spark often has a higher relative probability than a noise token in the immediate next-token prediction.” Please provide quantitative evidence to support this claim, and clearly define what constitutes a reasoning spark to distinguish it from random low-probability tokens.


3. Potential computational cost: the additional filtering and redistribution of token probabilities may introduce non-trivial computational overhead, especially for large-scale models. Please include an analysis or estimation of the extra computational complexity incurred by Lp-Reg, and discuss whether it affects training efficiency or scalability.

### Questions
Q1. In Figure 1 (c) and (d), what does the variable $n$ represent?


Q2. When comparing GRPO in Eq. 3 and Lp-Reg in Eq. 6, the normalization strategies differ: the former uses $\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}$ while the latter adopts $\frac{1}{\sum_{i=1}^{G} |o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|}$. As far as I know, the latter formulation provides a more accurate normalization over all tokens. Could the reported performance improvement be partially attributed to this modification rather than the proposed regularization itself?

### Soundness
2

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
They first reveal that the existing methods that have an unselective focus on entropy risks amplifying irrelevant tokens and destabilizing training. To address this, they propose Lp-Reg, which applies a probability threshold to filter out noise tokens and then re-normalizes the distribution over the remaining candidates. They compare their method with other training methods and show a high stability. Their method achieves higher performances than others based on five math benchmarks.

### Strengths
- They introduce a new training scheme, Lp-Reg, a method that creates a more stable exploratory environment by filtering out presumed meaningless noise to protect the remaining low-probability tokens.
- They show that Lp-Reg achieves state-of-the-art performance with five math benchmarks.
- Their methodology is principled and simple. 
- They also conduct ablation studies for robustness.

### Weaknesses
Even though they claim that their methodology achieves a higher stability and accuracy than other existing methodologies, the current experiments are limited to the math domain, and the results are not consistent across all benchmarks (i.e., their method is not always the best). Because the reported improvement numbers themselves look small, it is unclear whether the differences are statistically significant. Moreover, their domain is only math, and they evaluate only models trained with one math dataset. This raises concerns about the robustness of the results. For example, if the models were trained on different math datasets, would the results remain the same? Similarly, would the same result appear in other domains such as coding? The tested model family is also limited to Qwen, which further questions the robustness.

I'm also wondering whether the proposed technique would eventually show a performance collapse after 1,000 training steps. Reporting the point at which their method begins to collapse (if it does) could be useful and interesting.

### Questions
Please see the weaknesses

### Soundness
3

### Presentation
2

### Contribution
2
