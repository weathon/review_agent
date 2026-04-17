# Review

## Summary
This paper proposes a new state representation learning framework that learns the Minimum Action Distance (MAD). MAD is defined as the minimum number of actions required to transition between states. The authors propose two algorithms MadDist and TDMadDist to learn MAD. MadDist minimizes the difference between the learned distance and the step difference, while TDMadDist uses bootstrapped targets. The authors also propose a new quasimetric that is asymmetric. The authors evaluate the methods in several environments with known MAD values, including grid world, CliffWalking, and PointMaze. The proposed method outperforms the baselines.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper is well-written and easy to follow. The authors clearly define the problem and the proposed method.
2. The idea of learning MAD is interesting. It provides a new perspective on state representation learning.
3. The proposed quasimetric is novel and addresses the limitation of previous methods that rely on symmetric distance metrics.

## Weaknesses
1. The motivation of learning MAD is not very clear. The authors should provide more examples of how MAD can be used in downstream tasks.
2. The experiments are limited to environments with known MAD values. It would be more convincing to see the results in more complex environments, such as Atari games.
3. The proposed method requires a dataset of trajectories collected by a behavior policy. The quality of the dataset may affect the performance of the proposed method.
4. The authors should provide more details on how the planning task is set up in the experiments.
5. The authors should discuss the limitations of the proposed method and potential directions for future research.

## Questions
1. How can MAD be used in downstream tasks, such as reinforcement learning and transfer learning?
2. How does the quality of the dataset affect the performance of the proposed method?
3. Can the proposed method be applied to environments without explicit state representations, such as image-based environments?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4