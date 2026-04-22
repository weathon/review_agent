# Rectifying LLM Thought from Lens of Optimization

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Recent advancements in large language models (LLMs) have been driven by their emergent reasoning capabilities, particularly through long chain-of-thought (CoT) prompting, which enables thorough exploration and deliberation. Despite these advances, long-CoT LLMs often exhibit suboptimal reasoning behaviors, such as overthinking and excessively protracted reasoning chains, which can impair performance. In this paper, we analyze reasoning processes through an optimization lens, framing CoT as a gradient descent procedure where each reasoning step constitutes an update toward problem resolution. Building on this perspective, we introduce RePro (**Re**ctifying **Pro**cess-level Reward), a novel approach to refine LLM reasoning during post-training. RePro defines a surrogate objective function to assess the optimization process underlying CoT, utilizing a dual scoring mechanism to quantify its intensity and stability. These scores are aggregated into a composite process-level reward, seamlessly integrated into reinforcement learning with verifiable rewards (RLVR) pipelines to optimize LLMs. Extensive experiments across multiple reinforcement learning algorithms and diverse LLMs, evaluated on benchmarks spanning mathematics, science, and coding, demonstrate that RePro consistently enhances reasoning performance and mitigates suboptimal reasoning behaviors.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents REPRO, a framework for improving LLM reasoning by viewing chain-of-thought (CoT) as an optimization process. The authors define a surrogate objective function to model reasoning progress, introduce dual metrics—magnitude and stability—to assess optimization quality, and integrate them as process-level rewards within RL post-training. Experiments across several RL algorithms (PPO, GRPO, REINFORCE++) and reasoning benchmarks in math, science, and coding show consistent accuracy improvements and reduced reasoning inefficiencies.

### Strengths
The proposed method itself is coherent and easy to follow.  The authors conduct thorough experiments on several benchmarks to validate the effectiveness of the proposed method.

### Weaknesses
While the proposed framework is conceptually appealing, several limitations temper its overall contribution.

**(1) Limited model scale:** Most experiments are conducted on small- to medium-sized models (e.g., ≤8B parameters), which constrains the generality of the findings. Modern reasoning research increasingly focuses on 30B–70B or larger models, where optimization dynamics, token efficiency, and stability behavior differ substantially. It remains unclear whether REPRO’s process-level rewards would scale effectively or remain computationally practical in such large models.

**(2) Theoretical abstraction:** The analogy between chain-of-thought reasoning and gradient descent optimization, though intuitively appealing, is primarily heuristic and lacks rigorous formal grounding. The paper would benefit from a stronger mathematical justification or empirical validation of this analogy.

**(3) Simplified proxy objective:** Using ground-truth log-likelihood as a surrogate for reasoning quality oversimplifies complex internal cognitive states and may not generalize to open-ended or ambiguous reasoning tasks.

**(4) Modest performance gains:** Despite consistent improvements, the absolute accuracy increases are relatively small, raising questions about practical significance versus added complexity. Overall, the approach is promising but under-evaluated at scale.

### Questions
1.	Could the process-level reward interact negatively with other RLVR objectives, especially in long-horizon reasoning?
2.	Have the authors considered token-level or adaptive segment granularity for more fine-grained control?
3.	How sensitive is REPRO to inaccuracies in the surrogate objective when applied to open-ended reasoning without clear ground truth (e.g., creative tasks)?

### Soundness
3

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
3

### Summary
This paper’s main focus is on how to obtain a process reward *without* introducing anything akin to a PRM.  
The proposed method centers on two components: the Magnitude Score, which captures the change in the model’s confidence brought about by the reasoning trajectory, and the Stability Score, measured with Kendall’s τ correlation coefficient.  
In small-scale experiments these two scores produce observable improvements.

### Strengths
1. The paper is written in a fluent and highly readable style.  
2. The experimental content is rich and concrete, and the ablation studies are comprehensive.  
3. The idea is clear, and the algorithmic design is relatively concise.

### Weaknesses
1. Please refer to the "Questions" section for details.

### Questions
1. In the process of computing the advantage (accumulation), what does k represent? If it is consistent with the meaning of k mentioned earlier, should this formula still be expressed this way?
2. In the “Entropy-Based Selection Strategy”， why is “\n\n” chosen as the delimiter? Are there better ways to split the sequence? And why use the entropy of the *first* token of a segment—can it really characterize the whole segment?
3. The fact that larger k yields better results seems to imply that the selection scheme itself is flawed. Have you run an ablation exp that selects all the segments?
4. As can be seen, the proxy objective function J changes sharply during the first half of the reasoning chain and then flattens out in the second half. How should this be interpreted? Does it indicate insufficient representational capacity? Is there a need for additional corrective terms?

### Soundness
2

### Presentation
2

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
REPRO introduces a modification to the LLM RL objective by augmenting the advantage function with a surrogate objective that quantifies the posterior of (ground truth answer | current reasoning chain). The idea is that by quantifying the magnitude and stability of this posterior and encouraging confident and stable reasoning path, we can improve model performance while reducing thinking length. Authors performed experiment by adding the REPRO objective on top of PPO,RF++ and GRPO for DeepSeek-R1-Distill-Qwen-1.5B and Qwuen3-1.7B, and demonstrated consistent performance gain on math and code benchmarks.

### Strengths
The biggest strength of the paper is in its simplicity. The idea of reward shaping with posterior of ground truth given intermediate reasoning is simple and elegant and appears effective.

It requires no additional rollouts during RL process and therefore can be readily adapted to existing approaches without much overhead.

The paper is generally well written and easy to follow.

### Weaknesses
The biggest drawback is that it is not clear from the manuscript what is the fundamental mechanism that made the difference in the proposed algorithm. In particular, since the objective of P(answer | thinking) can be potentially by artificially inflated by increasing model confidence across the board (collapsing entropy). It would be interesting to see and control for entropy collapse during training to see how much the conditioning actually contributes to model performance improvement, other than collapsing model entropy. Additionally, would naive length penalty also achieve the same result? Is the length reduction leading to increasing confidence? These important counterfactual questions are mostly unaddressed in the current work.

One minor thing is that the reward shaping is very similar in spirit to PPO's critic model, except the critic here is hardcoded and not learned. In my opinion, this is a potentially more interesting question to address, which is what if the critic is trained using this surrogate objective.

### Questions
It is interesting that REPRO is uniformly better than GRPO across different values of alpha. I'm quite curious to see what happens if we just train on the model confidence and not on the ground truth reward.

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
3

### Summary
This work aims to address the "overthinking" and inefficiency found in the long CoT reasoning of LLM. The authors innovatively frame the CoT reasoning process as a form of gradient descent optimization. Based on this, the paper proposes REPRO, a framework that defines a surrogate objective function based on the perplexity of the ground-truth answer to evaluate the "magnitude" and "stability" of the reasoning process. These two scores are combined into a process-level reward and integrated into the RLVR training loop to rectify the model's reasoning behavior. Experiments show this method improves performance on several benchmarks while increasing token efficiency and reducing invalid reasoning steps.

### Strengths
1. This paper tackles a critical problem in RLVR, i.e., how to optimize the reasoning process itself, rather than just rewarding the final answer, which is key to solving overthinking.

2. The empirical results are the paper's greatest strength, particularly in efficiency gains. The data clearly shows that REPRO not only boosts task accuracy but also significantly reduces the number of tokens required for reasoning, directly addressing its stated goal.

3. The method demonstrates good generality, proving to be a plug-and-play component that can be combined with various major RL algorithms like PPO, GRPO, and etc.

### Weaknesses
1. The analogy of CoT as gradient descent is illustrative but lacks rigorous support. Using the log-prob of the correct answer as the surrogate objective essentially assumes that a better reasoning path is one that converges to the answer "sooner and smoother." This design may excessively penalize exploration, trial-and-error, and backtracking (which might be necessary for hard problems, but would look like oscillation or low magnitude under this metric). I am concerned it rewards simple, direct paths rather than truly rectifying complex reasoning.

2. This work involves several new and critical hyperparameters to tune. The ablation studies suggest that performance is highly sensitive to the values of these hyperparameters, which significantly increases the difficulty of reproducing the results and applying the method in practice.

### Questions
1. See weaknesses 1.

2. Given all the extra computations (entropy, surrogate objective function), what is the actual total increase in training overhead for REPRO compared to a baseline RL algorithm?

### Soundness
3

### Presentation
3

### Contribution
3
