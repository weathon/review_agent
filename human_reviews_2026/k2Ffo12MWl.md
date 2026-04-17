# Value-as-Return: A Two-Stage Framework to Align on the Optimal Score Function

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Reinforcement learning with diffusion models has shown strong potential, but existing approaches such as variants of Direct Preference Optimization (DPO) often rely on an inaccurate simplification: they equate trajectory likelihoods with final-state probabilities. This mismatch leads to suboptimal alignment. We address this limitation with a principled framework that leverages the optimal value function as the return for short trajectory segments. Our approach follows a two-stage procedure: (i) learning a value-distribution function to estimate segment-level returns, and (ii) applying our VRPO to refine the score function. We prove that, under sufficient model capacity, the resulting model is equivalent to training a diffusion process on the tilted distribution proportional to $p(x)\exp(\eta r(x))$. Experiments on large-scale diffusion models validate our analysis and show stable and consistent improvements over prior methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses a critical and often-overlooked problem in the alignment of diffusion models via reinforcement learning. The authors identify a key limitation in existing methods based on Direct Preference Optimization, which incorrectly equate the likelihood of a full generation trajectory with the probability of the final output. The paper convincingly argues that this fundamental mismatch between trajectory-level processes and final-state rewards leads to suboptimal policy alignment and hinders the model's ability to learn complex preferences effectively.

### Strengths
The authors first formally articulate that existing DPO-based methods for diffusion models rely on a flawed oversimplification: they equate the intractable marginal probability of a final image with the joint probability of a single sampled trajectory. This leads to unstable training and suboptimal alignment, because the policy is updated using an incorrect and misleading reward signal. This is indeed a valid problem, and it would be beneficial to solve it efficiently without incurring high costs.

### Weaknesses
1. The paper does not discuss the practical computational cost of the proposed framework compared to baselines.
2. I hope the authors can share the hyperparameter sensitivity analysis regarding the segment length.
3. The transition from the continuous-time Stochastic Optimal Control (SOC) formulation in Section 2.2 to the discrete "bandit view" of DPO in Section 2.3 is abrupt. The writing can be improved.
4. The paper's theoretical conclusion points to a more direct alternative: guiding the diffusion process with the gradient of the reward function. This technique is well-established in the context of energy-based models and may represent a powerful baseline that can be included.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper identifies a fundamental limitation in existing Direct Preference Optimization (DPO) methods for diffusion models: the oversimplification of equating the probability of a full generation trajectory with the marginal probability of the final image. To address this problem, the authors propose a two-stage reinforcement learning framework: First, a value-distribution function is learned to estimate the distribution of returns (future rewards) from any intermediate state. Second, a novel Value-as-Return Preference Optimization (VRPO) method is applied to short trajectory segments, using the learned value function to label preferences. The paper proves that under sufficient model capacity, their method converges to an optimal policy that samples from a "tilted" target distribution, and experiments on large-scale diffusion models demonstrate that VRPO achieves some improvements over several strong baselines.

### Strengths
To the best of my knowledge, the proposed method is novel and addresses a key shortcoming of prior work, where the probability of full generation trajectories are treated as the marginal probability of the final image. The writing is also generally clear and easy to follow, and the authors evaluate their method on large scale models relevant to practitioners working with diffusion models. Additionally, the approach is grounded theoretically.

### Weaknesses
My main concern revolves around the significance of the author’s empirical results. To me it looks like VRPO marginally outperformed SPO on SDXL (Table 1) with respect to the metrics considered, except for ImageReward where gains are more significant. I am having difficulty gaging why the authors claim that these are substantial improvements over the SPO baseline; have similar gains by other methods (e.g., 1-3% improvements) been considered significant in the past? Or is the ImageReward a metric readers might care about more than the others?

### Questions
Can the authors elaborate on why their main empirical results in Table 1 are meaningful improvements over the strongest baseline, SPO?

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
3

### Summary
The paper is presenting an algorithm for modifying a diffusion model’s induced distribution to align it with a specified reward function, and in particular learning a generative model for the tilted distribution $p(x) e^{r(x)}$. The approach is split into two stages: first learn a value or cumulative distribution over short diffusion segments, then fine-tune the diffusion model by ranking segment endpoints using that learned value.

### Strengths
- The theory section builds from reasonable assumptions and gives good theoretical intuition of the algorithm.
- The method is easy to follow and to implement.
- Results show some practical improvements on metrics on Stable Diffusion 1.5. While not uniformly large, the gains are nontrivial in places and indicate the approach is at least competitive.

### Weaknesses
- The paper does not position itself clearly against [Adjoint Matching](https://arxiv.org/abs/2409.08861), which is a closely related framework that also targets a tilted terminal distribution. Similar to e.g. [Diffusion-QL](https://arxiv.org/abs/2208.06193) and e.g. [QSM](https://arxiv.org/abs/2312.11752) for diffusion RL, there are both pros and cons for using a value-function-based vs. vector field-based approach. Without any comparisons, it is hard to know when a practitioner should choose this method instead of something like adjoint matching.

- While results on SD 1.5 are nonnegligible, the improvements on SDXL seem very marginal. When there are many different methods for learning a tilted distribution for score-based generative models, it makes it further unclear as to when this method should be used vs. alternatives.
- The paper does not quantify compute cost or runtime relative to strong baselines. Likewise to the above points, lack of this information makes it difficult for future researchers to evaluate when they would want to use this method.

### Questions
- Under what conditions should a practitioner prefer this method over Adjoint Matching?
- How sensitive is performance to segment length, to misranking in the value model over time steps, and to the inverse temperature?
- What are the compute and runtime characteristics compared with other baselines?

### Soundness
4

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
2

### Summary
introduces Value-as-Return (VRPO), a two-stage reinforcement learning framework for aligning diffusion models with human preferences. It aims to address a flaw in prior DPO methods that misinterpret trajectory and final-state probabilities. VRPO learns a value-distribution function for short segments, then refines the model using these learned returns. the authors report more stable and a consistently superior alignment method for large-scale diffusion models.

### Strengths
- provides a theoretically grounded solution using stochastic optimal control, correcting a major flaw in existing DPO-based diffusion training
- the value-distribution + VRPO pipeline improves reward consistency and stability across long diffusion trajectories.
- demonstrates clear, consistent gains over strong baselines (e.g., SPO, Diffusion-DPO) on large-scale diffusion models like SDXL.

### Weaknesses
- the two-stage training process (learning a value-distribution function before VRPO) adds computational and implementation overhead compared to simpler DPO methods.
- experiments focus mainly on image diffusion models, leaving it unclear how well the approach generalizes to other domains or more diverse reward settings. It would be interesting to compare this to conventional sequential decision-making tasks

### Questions
- do the authors plan to evaluate their method on sequential decision-making tasks? e.g. robotics, atari, and continuous control settings.

### Soundness
3

### Presentation
3

### Contribution
3
