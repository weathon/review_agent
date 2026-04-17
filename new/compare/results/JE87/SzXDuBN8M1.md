# Review

## Summary
The paper presents TD-JEPA, an unsupervised RL algorithm that leverages temporal-difference (TD) learning to create latent-predictive representations for zero-shot policy optimization across multiple tasks. TD-JEPA trains a state encoder, a policy-conditioned multi-step predictor, and a task encoder to enable zero-shot optimization of any reward function at test time. The paper provides theoretical guarantees that TD-JEPA avoids representation collapse and recovers a low-rank factorization of long-term policy dynamics. Empirically, TD-JEPA matches or outperforms state-of-the-art baselines on various tasks in ExoRL and OGBench, especially in zero-shot RL from pixels.

## Soundness
4

## Presentation
4

## Contribution
4

## Strengths
- The paper introduces a novel approach by combining TD learning with latent-predictive representations for unsupervised RL, extending the scope beyond single-task or single-step learning.
- TD-JEPA’s ability to learn from offline, reward-free transitions makes it highly practical for real-world applications where labeled data is scarce.
- The paper provides robust theoretical analysis, proving that TD-JEPA avoids representation collapse and effectively captures long-term policy dynamics.
- Empirical results show that TD-JEPA consistently matches or outperforms state-of-the-art baselines across a wide range of tasks in ExoRL and OGBench, especially in challenging zero-shot RL from pixels.
- TD-JEPA’s architecture allows for zero-shot policy optimization for any downstream reward function, making it highly adaptable for various tasks and environments.

## Weaknesses
- The theoretical guarantees rely on some symmetry assumptions, which may not hold in all practical scenarios.
- While TD-JEPA performs well on the evaluated tasks, it is unclear how it would scale to more complex, large-scale environments or real-world applications.

## Questions
- How does TD-JEPA handle environments with rapidly changing or non-stationary dynamics? 
- Could the approach be extended to online or semi-supervised learning settings? 
- How sensitive is TD-JEPA to the choice of hyperparameters, especially in high-dimensional settings?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
8

## Confidence
4