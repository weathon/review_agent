# Review

## Summary
The paper studies curriculum learning in goal-conditioned RL by biasing the sampling towards underachieved goals. They use potential based reward shaping and UVFAs in a gridworld environment. The results show that the curricula improve the success on difficult edge goals.

## Soundness
2

## Presentation
2

## Contribution
1

## Strengths
The paper is well-written and easy to follow.

## Weaknesses
1. The novelty of the work is limited. The idea of using a density of states as a curriculum is not new. The authors should discuss the related work and highlight how their work differs from prior research.
2. The experiments are limited. The experiments are conducted only in a simple grid-world environment. Even within the scope of grid-world environments, the experiments are limited to a single environment. The authors should consider using more environments, such as those available in the Minigrid benchmark.
3. The results are not significant. In the main experiment, the proposed method shows only marginal improvements over uniform sampling (0.297 vs 0.276). 
4. The proposed method is not generalizable. The authors manually design the curriculum for each environment. A good curriculum should be generalizable and applicable without extensive manual design.

## Questions
1. The authors should discuss the related work and highlight how their work differs from prior research.
2. The authors should consider using more environments, such as those available in the Minigrid benchmark.
3. The authors should try to make the proposed method more generalizable.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
4