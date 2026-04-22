# Policy Likelihood-based Query Sampling and Critic-Exploited Reset for Efficient Preference-based Reinforcement Learning

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 8, 6, 6

## Abstract
Preference-based reinforcement learning (PbRL) enables agent training without explicit reward design by leveraging human feedback. Although various query sampling strategies have been proposed to improve feedback efficiency, many fail to enhance performance because they select queries from outdated experiences with low likelihood under the current policy. Such queries may no longer represent the agent's evolving behavior patterns, reducing the informativeness of human feedback. To address this issue, we propose a policy likelihood-based query sampling and critic-exploited reset (PoLiCER). Our approach uses policy likelihood-based query sampling to ensure that queries remain aligned with the agent’s evolving behavior. However, relying solely on policy-aligned sampling can result in overly localized guidance, leading to overestimation bias, as the model tends to overfit to early feedback experiences. To mitigate this, PoLiCER incorporates a dynamic resetting mechanism that selectively resets the reward estimator and its associated Q-function based on critic outputs. Experimental evaluation across diverse locomotion and robotic manipulation tasks demonstrates that PoLiCER consistently outperforms existing PbRL methods. Our code is available at https://github.com/JongKook-Heo/PoLiCER.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a new method called PoLiCER to improve query efficiency in previous PbRL work. It proposes two improvements: policy likelihood sampling (PLS) aims to make the reward model focus on the trajectories that are likely under the current policy; critic exploited reset forces the reward estimator to reset occasionally to reduce overestimation bias that emerges from repeated feedback over the same region of behaviour. Experiments involved in locomotion and manipulation show that PoLiCER outperforms prior PbRL algorithms.

### Strengths
1. Preference-based RL is a growing area with practical relevance. This work provides a new perspective to improve the algorithms, promoting the development of the field.

2. The motivation is clear, and the authors use experiments to demonstrate that these issues are indeed important, which enhances the readability and coherence of the paper.

3. The two mechanisms the authors propose are beneficial to each other: reset the critic and reward model to make them more plastic to adapt to preference data that is more consistent with the current strategy.

4. The authors provide empirical evaluation across multiple tasks (locomotion + robot manipulation), which helps test generality.

5. The code is included in the Supplementary Material, demonstrating the reproducibility of this work.

### Weaknesses
1. PLS seems to lack some theoretical insight to support. It is not clear why directly maximizing the policy likelihood can find more informative trajectories.

2. Human-in-the-loop experiments are relatively simple.

3. In the main experiments, using TA as another trick to the algorithm may cause SURF to completely become an ablation of this method, and the additional setting of the ratio may lead to unfairness in the experiment.

### Questions
1. What are the details in human-in-the-loop experiments? How did the volunteers provide feedback? I hope the authors can provide some examples to prove that their experiments are fair, i.e., this feedback may show personalized preferences or mixed preferences, and PoLiCER can handle them. Meanwhile, explain what factor will affect the process.

2. In the main experiments, why is QPA better than PoLiCER in Hammer? PLS improves QPA's sampling method, so PoLiCER should definitely be better than QPA in all experiments, but the results are not.

3. What is the time consumption for pixel-based tasks? Since these tasks generally take more time, it is important to report your time consumption and compare it with previous methods.

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
4

### Summary
PoLiCER addresses two central challenges in preference-based RL: query-policy misalignment and reward overestimation caused by primacy bias. It proposes Policy Likelihood-based Sampling (PLS) to select queries aligned with the current policy and Critic-Exploited Reset (CER) to dynamically reset the reward model and critic when overestimation is detected.

### Strengths
The paper focuses on (1) query–policy misalignment in query sampling and (2) reward overestimation caused by primacy bias in reward learning, which are key issues of high importance in PBRL.

The design of the approach is reasonable, including the likelihood computation in PLS and the CRITIC-EXPLOITED RESET mechanism; it is simple and low-cost.

It shows clear performance advantages on both vector-observation and pixel-observation tasks, and provides human-in-the-loop validation along with fairly comprehensive ablations.

### Weaknesses
PLS is sensitive to policy entropy/scale and may prefer low-entropy trajectories, reducing diversity.

The discussion of related work could be more complete. PPE~[1] advocates proximal policy exploration to expand the coverage of the preference buffer and improve reward model quality; like this paper’s PLS, it aims to make labeled data/queries closer to the current policy, but PPE emphasizes active generation/exploration, whereas PLS passively selects from the replay buffer and ranks by likelihood. The difference and advantage of PLS over PPE should be discussed in the manuscript.

Minor issues such as typos: “primary bias” vs. “primacy bias”; a unified wording is recommended.


[1] Zhu, Y., ... . (2024). Optimizing reward models with proximal policy exploration in preference-based reinforcement learning. In NeurIPS 2024 Workshop on Behavioral Machine Learning.

### Questions
Does the PLS likelihood score bias toward low-entropy policies? It is recommended to add sensitivity experiments on policy entropy/temperature.

Is a diversity constraint necessary?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes PoLiCER, which introduces two mechanisms to improve preference-based reinforcement learning. 1) Policy Likelihood-based Sampling (PLS) selects feedback queries most aligned with the current policy. 2) Critic-Exploited Reset (CER) prevents reward overestimation from primacy bias by adaptively resetting the reward and critic networks. The goal of PLS is to ensure feedback queries remain representative of the current policy, avoiding outdated or irrelevant samples. Compared to recency-based methods, it directly measures alignment between data and current policy. It is more computationally efficient than disagreement sampling, requiring only $2 \times L \times K$ forward passes than $N$ for disagreement sampling. And it does not increase training cost. The goal of CER is to counteract primacy bias, where the reward estimator overfits to early feedback and inflates Q-values, leading to overoptimistic policies. It dynamically stabilizes reward learning by resetting networks only when critic overestimation is detected. Experiments are conducted on Meta-World and DMControl tasks. Authors compared to several existing baselines and show improvements.

### Strengths
PoLiCER offers several strengths over prior preference-based reinforcement learning methods such as disagreement sampling and recency-based query selection. Its Policy Likelihood-based Sampling (PLS) improves data–policy alignment by measuring how representative each trajectory is under the current policy, rather than assuming recency implies relevance. This allows the model to select feedback that directly reflects its current behavior, improving sample efficiency and policy convergence.

Another advantage is computational efficiency. Unlike ensemble-based disagreement sampling, which requires multiple forward passes per query, PLS operates with a fixed, small number of policy evaluations. Its inverse-rank likelihood weighting also provides robustness to outliers, enabling stable performance across diverse continuous control tasks. Together, these factors make PoLiCER both efficient and reliable in selecting informative feedback.

PoLiCER’s second component, Critic-Exploited Reset (CER), effectively mitigates reward overestimation caused by primacy bias. Instead of using fixed reset intervals, CER dynamically resets the reward estimator and critic only when critic outputs exceed an adaptive threshold. This approach reduces overestimation while allowing normal learning to continue when stable, leading to better long-term returns and less training disruption.

Overall the writing is clear and easy to follow. Experimental setups are well explained, together with baseline methods. In the DMControl suite, it achieved performance comparable to QPA and clearly outperformed earlier PbRL methods like PEBBLE, SURF, RUNE, and MRN. In the more challenging Meta-World benchmarks, PoLiCER further distinguished itself by achieving over 80% average success, significantly higher and more consistent than competing methods.

### Weaknesses
1. PLS depends on accurate policy likelihood estimation, which may be unreliable in highly stochastic or multimodal policies. 
2. The rank-based weighting, though robust, can blur distinctions between highly relevant and moderately relevant samples. 
3. CER’s adaptive resets introduce temporary instability as networks reinitialize, requiring careful tuning of replay ratios. 
4. Because PoLiCER omits ensemble-based uncertainty modeling, it may handle noisy or inconsistent human feedback less effectively than Bayesian or disagreement-based methods.

### Questions
It seems that PoLiCER has more empirical performance gain in pixel-based environments, is there any intuition on why this happens compared to state-based environments?

### Soundness
2

### Presentation
3

### Contribution
2
