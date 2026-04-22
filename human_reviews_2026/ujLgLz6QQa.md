# Random Policy Valuation is Enough for LLM Reasoning with Verifiable Rewards

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
RL with Verifiable Rewards (RLVR) has emerged as a promising paradigm for improving the reasoning abilities of large language models (LLMs). Current methods rely primarily on policy optimization frameworks like PPO and GRPO, which follow generalized policy iteration that alternates between evaluating the current policy's value and improving the policy based on evaluation. While effective, they often suffer from training instability and diversity collapse, requiring complex heuristic tricks and careful tuning. We observe that standard RLVR in math reasoning can be formalized as a specialized finite-horizon Markov Decision Process with deterministic state transitions, tree-structured dynamics, and binary terminal rewards. Though large in scale, the underlying structure is simpler than general-purpose control settings for which popular RL algorithms (e.g., PPO) were developed, suggesting that several sophisticated techniques in existing methods may be reduced or even omitted. Based on this insight, we prove a surprising result: the optimal action can be recovered from the Q-function of a fixed uniformly random policy, thereby bypassing the generalized policy iteration loop and its associated heuristics. We introduce \underline{\textbf{R}}andom P\underline{\textbf{o}}licy \underline{\textbf{V}}aluation for Div\underline{\textbf{e}}rse \underline{\textbf{R}}easoning (ROVER) to translate this principle into a practical and scalable algorithm for LLM math reasoning, a minimalist yet highly effective RL method that samples actions from a softmax over these uniform-policy Q-values. ROVER preserves diversity throughout training, allowing sustained exploration of multiple valid pathways. Across multiple base models and standard math reasoning benchmarks, ROVER demonstrates superior performance in both \textbf{quality} (\textbf{+8.2} on pass@1, \textbf{+16.8} on pass@256) and \textbf{diversity} (\textbf{+20.5\%}), despite its radical simplification compared to strong, complicated existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents ROVER, a simplified reinforcement learning algorithm for improving the reasoning abilities of large language models. The authors argue that standard methods like PPO are overly complex for the specific structure of math reasoning tasks, which they model as deterministic, tree-structured Markov Decision Processes. The core insight is that an optimal policy can be derived directly by evaluating a fixed, uniform random policy, thus avoiding the unstable iterative loops of generalized policy iteration. ROVER implements this by learning Q values based on a random policy and using a softmax sampler to promote diversity. Experiments show that ROVER significantly outperforms strong baselines in both solution quality and diversity.

### Strengths
1. The paper proposes an elegant simplification of the reinforcement learning process for this specific problem domain. By identifying the unique properties of the math reasoning MDP, the authors justify moving away from complex generalized policy iteration. 

2. The empirical results are consistently strong and well presented. The proposed method, ROVER, not only achieves higher scores on pass@1 metrics but also shows substantial gains in diversity and pass@k at larger k values. 

3. The work offers a valuable conceptual contribution by reframing the problem. The key insight is the mismatch between general-purpose RL algorithms and the specialized structure of LLM reasoning tasks. This perspective is insightful.

### Weaknesses
1. There is a gap between the theoretical foundation and the practical implementation of the Q function. The theory relies on the true Q values of a uniform random policy. However, the algorithm approximates this with a relative Q function based on the log probabilities of the current and previous policies. The paper does not provide a clear justification for why this specific practical choice is a good proxy for the theoretical quantity.

2. While the baselines are relevant, the comparison could be strengthened by including methods that explicitly aim to improve diversity through mechanisms other than softmax sampling. For instance, algorithms that incorporate an entropy bonus in the reward function or use other diversity-seeking objectives.

3. The novelty of the algorithmic components in the practical implementation is somewhat limited. The use of a relative Q function, mean-centered rewards, and reward broadcasting are techniques inspired by or used in prior works like GRPO and REINFORCE++. The main novelty lies in the theoretical motivation for combining them in this simplified framework, not in the components themselves.

### Questions
1. Could the authors discuss how you expect ROVER's performance to be affected in environments that violate the assumptions, such as tasks involving tool use with a chance of failure or those with partial credit rewards?

2. Algorithm 1 presents the loss function for updating the model. Could the authors please provide the explicit mathematical formula for the stop gradient target used in the L_ROVER calculation? The description in lines 295 and 296 is slightly ambiguous about how the target for the Q value regression is constructed from the next state.

3. The paper shows superior diversity compared to PPO-style methods. Did the authors consider comparing ROVER to other approaches that explicitly encourage diversity, for example, by adding a learned or fixed entropy bonus to the model's reward?

### Soundness
3

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
3

### Summary
This paper finds that in LLM math reasoning, the optimal action can be recovered from the Q-function of a fixed uniformly random policy, thereby bypassing the generalized policy iteration loop and its associated heuristics. Based on this observation, the authors introduce an algorithm called Random Policy Valuation for Diverse Reasoning (ROVER).

### Strengths
1. The paper is clearly written and well motivated.
2. The experiments validate the effectiveness of the method.
3. The method is simple and seems to be scalable.

### Weaknesses
1. The motivation and the method itself are very similar to [1]. Although the authors cited [1] in the paper, more detailed discussions are needed, given the similarities. Specifically, [1] provided theoretical results, while the authors stated that "these observations have
remained primarily empirical, with limited theoretical justification." Besides, the authors stated that the main difference lies in that their paper provides the first theoretical analysis that accounts for LLM math reasoning. What are the exact differences that prevent their method and analysis from being useful in math reasoning?
2. Regarding the experiment results, the performance of the baselines is too low in my opinion. For example, in [2], they report 33.33 and 25.42 for AIME 24 and 25, fine-tuned from Qwen3-8b-base using DAPO. The results are 10+ points higher than the same baseline reported in this paper, even outperforming the proposed method by a considerable margin. I understand that the training dataset and hyperparameters are different. But given that (1) their training dataset is barely MATH, instead of DeepScaler which is more in-distribution of AIME, and (2) the performance gap is huge, I believe more careful hyperparameter tuning is needed for both the baselines and the proposed method for a fair comparison.
3. The proposed method does not use random policies. It rather uses the old policy for the rollouts. Thus, the naming is not that appropriate in my opinion. Besides, the main experiments are conducted on Qwen models, which fail to fully demonstrate the effectiveness of the method. From the appendix, fine-tuning on the DeepSeek-R1-Distill-Qwen models seems to give small improvements.

[1] Laidlaw et al., Bridging Reinforcement Learning Theory and Practice with the Effective Horizon.\
[2] Wang et al., Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning.

### Questions
See weakness.

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
4

### Summary
This paper simplifies reinforcement learning with verifiable rewards (RLVR) for LLM reasoning. Exploiting the deterministic, tree-structured MDP with binary rewards, the authors prove that evaluating a uniform random policy's Q-function suffices for optimal reasoning—bypassing iterative policy improvement (PPO/GRPO). The proposed ROVER algorithm samples via softmax over these Q-values, balancing diversity and performance. Experiments show significant gains in reasoning quality and diversity versus complex RL baselines.

### Strengths
1. Algorithmic Minimalism: ROVER dispenses with conventional policy improvement loops and ad hoc stabilizers (e.g., KL penalties or clipping), yielding a strikingly simple and easy-to-tune algorithm, as clearly depicted in Figure 3 (p. 3).

2. Strong Empirical Performance and Strategic Diversity: Across standard math reasoning and countdown tasks, ROVER consistently outperforms strong baselines (e.g., GRPO, DAPO, REINFORCE++) in both solution accuracy (Table 1, p. 8; Figure 9) and the generation of diverse, correct reasoning paths (Figures 7, 10, 17).

3. Reproducibility: Public release of source code enhances transparency and facilitates replication.

### Weaknesses
1. Narrow Theoretical Scope: The optimality guarantee for ROVER strictly requires deterministic, tree-structured MDPs with binary terminal rewards. The paper offers no empirical or theoretical exploration of how—or whether—the method degrades gracefully under violations of these assumptions (e.g., stochasticity, intermediate rewards, or graph-structured environments), limiting its applicability to more realistic RLVR settings.

2. Oversimplified Reward Modeling: ROVER assumes only terminal binary feedback, ignoring process-based signals, partial credit, or intermediate rewards increasingly used in LLM alignment. The paper provides no discussion of how the framework might extend to richer reward structures, hindering practical deployment.

3. Limited Model Scale in Experiments: All evaluations use models up to 8B parameters, falling short of current state-of-the-art scales (e.g., 70B+). This constraint tempers claims about ROVER’s readiness for deployment in cutting-edge LLM systems.

### Questions
1. Scaling to Longer Horizons and Larger Action Spaces. How does ROVER’s computational and memory footprint compare to PPO/GRPO at the largest vocabularies and problem sizes considered? Can the authors clarify if action batching, logit pruning, or any approximations are required to keep training feasible?

2. Reward Shaping/Process-level Signals. Have the authors tested ROVER with partial or intermediate rewards (e.g., for partially correct math steps or process rewards)? If so, what is observed? If not, are there theoretical or practical obstacles?

3. Extension to Non-binary, Non-deterministic Environments. How does ROVER behave when rewards are non-binary or transitions are stochastic? Is there a degradation in quality/diversity, and are there principled ways to extend the theorem to these cases?

4. Failure and Negative Cases. Can the authors discuss cases where ROVER’s performance flatlines or drops compared to PPO/GRPO (e.g., pathological tasks, extremely long horizons, or ambiguous reward settings)?

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
4

### Summary
This paper proposes ROVER (Random Policy Valuation for Diverse Reasoning), the key idea is the optimal policy can be derived simply by greedily selecting actions based on the Q-values of a fixed, uniformly random policy. This bypasses the need for complex Generalized Policy Iteration (GPI) loops used in standard algorithms like PPO and GRPO. To maintain diversity (which pure greedy selection sacrifices), ROVER samples actions via a softmax over these uniform-policy Q-values. Empirically, ROVER outperforms strong baselines (GRPO, REINFORCE++, DAPO) on math benchmarks (AIME, MATH500, etc.) in both solution quality.

### Strengths
1. The core finding—that a uniform random policy's Q-values are sufficient for optimal control in this specific MDP structure—is a significant and surprising theoretical contribution that challenges prevailing RL paradigms for this domain.
2. ROVER drastically simplifies the RL pipeline by removing actor-critic loops, reference models (explicitly), and complex hyperparameter tuning associated with PPO/GRPO. It is computationally efficient and easy to implement.
3. The method consistently outperforms state-of-the-art baselines across multiple models (Qwen-4B, 8B, DeepSeek-1.5B) and challenging benchmarks (AIME, HMMT), showing gains in both pass@1 and pass@k (diversity).

### Weaknesses
1. The theoretical guarantee relies on a strict MDP structure (deterministic, tree-structured, binary terminal rewards). While standard for current math benchmarks, this may not hold for more complex reasoning tasks involving tool use, intermediate feedback, or stochastic environments, potentially limiting broader applicability.
2. While the paper claims robustness, the ablation shows performance is sensitive to the temperature parameter $\rho$. An extremely low $\rho$ causes collapse, while a high $\rho$ reduces to random sampling. The default $\rho=1$ works well here but might need tuning for other domains.

### Questions
1. Theorem 1 proves that acting greedily according to the random policy's Q-values is optimal. Why, then, does the practical ROVER algorithm use a softmax sampler instead of a purely greedy approach?
2. How would this be helpful if extend to other more complicated domain than math?

### Soundness
3

### Presentation
2

### Contribution
3
