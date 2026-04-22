# BAPO: Stabilizing Off-Policy Reinforcement Learning for LLMs via Balanced Policy Optimization with Adaptive Clipping

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Reinforcement learning (RL) has recently become the core paradigm for aligning and strengthening large language models (LLMs). Yet, applying RL in off-policy settings—where stale data from past policies are used for training—improves sample efficiency, but remains challenging: policy entropy declines sharply, optimization often becomes unstable and may even collapse. Through theoretical and empirical analysis, we identify two key insights: (i) an imbalance in optimization, where negative-advantage samples dominate the policy gradient, suppressing useful behaviors and risking gradient explosions; and (ii) the derived Entropy-Clip Rule, which reveals that the fixed clipping mechanism in PPO-like objectives systematically blocks entropy-increasing updates, thereby driving the policy toward over-exploitation at the expense of exploration. Building on these insights, we propose BAlanced Policy Optimization with Adaptive Clipping (BAPO), a simple yet effective method that dynamically adjusts clipping bounds to adaptively re-balance positive and negative contributions, preserve entropy, and stabilize RL optimization. Across diverse off-policy scenarios—including sample replay and partial rollout—BAPO achieves fast, stable, and data-efficient training. On AIME 2024 and AIME 2025 benchmarks, our 7B BAPO model surpasses open-source counterparts such as SkyWork-OR1-7B, while our 32B BAPO model not only achieves state-of-the-art results among models of the same scale but also outperforms leading proprietary systems like o3-mini and Gemini-2.5-Flash-Thinking.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a new adaptive and asymmetric clipping for PPO algorithm applied to LLM. Indeed the authors first observe that a common symmetrical clipping $clip(r_t, 1-\epsilon,1+\epsilon)$ for some $\epsilon > 0$ implies a decrease of policy's entropy with its eventual collapse, which prevents from further exploration. They argue that it happens because of misbalance between positive and negative tokens. The final algorithm BAPO adaptively tunes the clipping bounds to include more low probability positive tokens, and potentially reducing the number of negative tokens that are retained for further update steps. This approach is theoretically motivated and is further validated on AIME 2024 and AIME 2025 benchmarks, showing the consistent improvement over GRPO and SFT, demonstrating even some competitive performance compared to some proprietary models.

Overall, the paper is easy to read, providing new interesting results, but it can benefit from more comparisons with other prior assymetric clipping mechanisms. Therefore, I recommend weak accept.

### Strengths
The paper is nicely written and in a concise manner. BAPO represents an easy to implement and flexible clipping mechanism for finetuning LLM with RL. It is well motivated, with some theoretical insights, and its advantage over a manual symmetrical clipping is evident. Their models trained with BAPO achieve on-par results on AIME 2024 and 2025 compared to some proprietary algorithms.

### Weaknesses
Despite that the paper mentions other works that consider asymmetric clipping mechanisms, like TOPR, DAPO, and even dynamic clipping — DRPO, there is no systematic comparison between different methods for fixed original models like BP-Math-7B or BP-Math-32B. The only baselines that the authors are comparing to are GRPO, SFT and a single fixed asymmetric clipping with strictly predefined bounds. 

Minor:

- L141: provide the specific function for A_t that you use
- Figure11: why not checking [0.8, 1.5] as in figure 7?
- L841: provide a more precise definition for $z_{y,x}$

### Questions
- Is there a computational overhead of computing BAPO? How big?
- How does BAPO perform for longer training, does it only delay the entropy collapse or the entropy is expected to stabilise at some small non-zero value?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles instability in RL for LLMs with respect to a fundamental challenge in off-policy training: data staleness or distribution shift, i.e., the distribution of rollout policy differs from the distribution of the optimization target policy. The paper proposes that the instability mainly comes from two sources: i) imbalanced positive and negative samples, and ii) lack of exploration. To address both issues from empirical practice, the authors propose BAPO, which dynamically adjusts lower/upper importance sampling weight clip bounds to balance the negative and positive tokens. Experiments on reasoning tasks show competitive improvement over open baselines without clip-bound adjustment.

### Strengths
1. The empirical performance with an asymmetric clip constant is promising.
2. Some intuition-level justification from the policy gradient contribution perspective and the entropy dynamic perspective is given to pave the motivation.
3. The paper is quite well-written and easy to follow.

### Weaknesses
I'm somewhat skeptical of both perspectives on the training instability discussed in the paper:

1. From the policy gradient perspective, e.g., Eq. (3), in general RL, both positive and negative samples are important in estimating the value function gradient. Taking out either part will result in biased estimation and harm the performance improvement. On the other hand, if a random minibatch contains only negative samples, the policy can still be improved in expectation. So, the dominance of negative samples does not necessarily prevent learning and lead to training instability or failure.

2. The decrease of entropy also does not directly imply instability of training. The entropy decreases as the policy converges to a deterministic policy, which may or may not be the optimal solution. Therefore, entropy collapse should be expected for all RL training. If the policy converges to the optimal policy, the entropy also collapses, but we should not witness instability or sub-optimality. The instability could come from other places, such as the distribution shift or simply the high variance of using importance sampling. Whether to preserve or decrease entropy is the classic trade-off in exploration and exploitation, which this paper oversimplifies.

3. Essentially, the problem studied in this paper is somewhat artificial and stems from the clip-trick instead of the RL problem itself. In the ideal case, one should clip outlier events symmetrically over the probability measure such that the clipped estimation is still an unbiased estimator of the genuine policy gradient, but symmetric clipping on numerical values fails to coincide, since the irregular distribution of advantages in practice, and potentially depends on the state as well. These shortcomings have been quite well-studied in classic winsorization literature with application to RL, and, quite surprisingly to me, the paper does not discuss any of them. Therefore, I do not see much ideological contribution in the field. See below:


4. It seems all the downsides and limitations of PPO-clip mentioned in the paper, i.e., one single trust region for all states, lack of exploration due to fast entropy decrease, could be naturally solved with PPO-KL or TRPO, i.e., posing the trust region on policy instead of the action of each state, and using the trust region to prevent fast entropy decrease. It is very unclear why the authors do not discuss this approach. The paper should include a comprehensive discussion (from theory to empirical evaluation) between dynamic clipping and the policy trust region approaches, to convince the value of their proposed approach. 

Orenstein, Robust Importance Sampling with Adaptive Winsorization, 2018

Vehtari et al., Pareto Smoothed Importance Sampling, 2015

Su et al., Doubly robust off-policy evaluation with shrinkage, 2020

### Questions
See Weaknesses.

Some minor question:

1. From Figure 8, the fluctuation of both bounds across time is very minor compared to the asymmetry between higher and lower bounds. It seems we could also use a stationary asymmetric clip constant across time. Why would practitioners resort to dynamic clipping, which needs to compute the clip constant every step, versus a stationary asymmetric clip? Why are the authors in favor of the first approach?

2. In (8), why do you use $c_{low} = 0$ in the numerator?

### Soundness
2

### Presentation
4

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
The paper proposed a new RL loss for llm training which stablize off policy training

### Strengths
* Entropy clipping rule is a good contribution
* the result shows strong improvement

### Weaknesses
* hyper parameter complexity
* eval task diversity is very limited (only math)

### Questions
For a real LLM RL training, it might need to combine multiple tasks. but they might have different extent of imbalance. How will the algorithm tackle that?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper identifies two issues with standard PPO-like RL algorithms when applied to stale data (e.g., due to sample replay or partial rollouts):
(1) optimization imbalance, where negative-advantage samples dominate the policy gradient;
(2) systematic suppression of entropy-increasing updates, which limits exploration.
To address these issues, the authors propose BAPO, a method that adaptively adjusts the clipping range to ensure that the proportion of loss contributed by positive-advantage samples remains above a pre-specified threshold.
Empirical results indicate that BAPO-trained models outperform many existing models on the AIME 2024/2025 benchmarks.

### Strengths
* This work provides a sensible and relevant analysis of issues commonly encountered in practice.
* The proposed method is reasonable, practical, and straightforward to implement.
* Strong model performance on AIME 2024 / 2025 benchmarks.

### Weaknesses
- **Limited evaluation scope.**
Evaluation are restricted to AIME 2024/2025, each containing only 30 problems. Broader evaluation on more diverse and widely used benchmarks would strengthen the claims.

- **Attribution of improvements is unclear.** 
Comparisons are made with prior models trained under substantially different setups, notably in terms of training data. This paper lacks detail on the SFT stage (before RL), making it difficult to determine whether performance gains of the final models stem primarily from BAPO or from the SFT data. Surprisingly, the SFT models (BP-Math-7B/32B-SFT) already outperform baseline models in most cases, as shown in Table 1, suggesting that SFT data might play a major role.

- **Hyperparameter tuning.**
BAPO introduces 7 new hyperparameters (Line 272). The authors claim these are not finely tuned and that BAPO reduces manual tuning compared to methods like DAPO (for which the upper and lower clip ranges are tunable).
However, two different hyperparameter sets are used for the main experiments in Section 5 and the additional experiments in Appendix B, suggesting that some setting-specific tuning is still required. 
Moreover, Figure 8 shows that the adaptive clip range (determined by BAPO) remains rather stable throughout, raising the question of whether well-tuned GRPO/DAPO could achieve comparable results.

- **Potential exaggeration of instability issues.** 
Some claims and empirical results may overstate the instability caused by data staleness or off-policy RL. In fact, splitting a batch into 4 or 8 mini-batches (corresponding to staleness 4 or 8, if I understand correctly) has been common practice and worked well with standard GRPO/DAPO algorithms in broad settings.

- **Clarify issues about the analysis.**
See questions below.

### Questions
Questions about Eq. (5), which rewrites the standard PPO-style policy gradient:
* For both terms on the right-hand side, should the leading $\pi_{\theta}(y_t)$ terms be replaced with the probability ratio $r_t$? 
If so, how does this affect the theoretical analysis about entropy later in this work, e.g., Eq. (6)?
* On the left-hand side, please make explicit that $\nabla J^{PPO}$ is defined for a specific $x, y_t$, in contrast to the loss $J^{PPO}$ in Eq. (4).


Questions about Eq. (8): 
* Why does the left-hand side contain $\pi_{rollout}$ terms? 
* Is there a guarantee that the numerator should be smaller than the denominator?
Note that the denominator sums both positive terms (same as numerator) and negative terms before taking the absolute value.

### Soundness
2

### Presentation
2

### Contribution
2
