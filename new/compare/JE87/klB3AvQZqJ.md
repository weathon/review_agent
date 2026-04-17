# Review

## Summary
The paper presents a simple and effective method, Constraint-aware Reward (Re)Labeling (CARL), for offline safe reinforcement learning (OSRL). CARL addresses the challenge of learning reward-maximizing policies while satisfying safety constraints in the offline setting. It iteratively updates the cost evaluation function and the policy using relabeled rewards, assigning large penalties to unsafe state-action pairs. The paper demonstrates that CARL consistently enforces safety constraints and achieves high rewards across various benchmark tasks, outperforming prior methods, especially in scenarios with strict cost budgets.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
1. The paper is well-written and easy to follow.
2. The proposed CARL method is simple and intuitive.
3. The empirical results are promising.

## Weaknesses
1. The proposed method is intuitive but lacks theoretical support. In other words, it is unclear why the proposed method works.
2. The proposed method is somewhat incremental, as it is essentially a simple modification of the reward.
3. The proposed method may be sensitive to the hyperparameter $V_{\max}$.

## Questions
1. Why does the proposed method work? Why is it necessary to use $V_{\max}$ instead of $r_{\max}$? Can the authors provide some insights or theoretical analysis?
2. The proposed method is somewhat incremental. It would be better to highlight the novelty and contributions of the paper.
3. The proposed method may be sensitive to the hyperparameter $V_{\max}$. It would be better to include some ablation studies.
4. The authors claim that the proposed method can be integrated with any batch-update offline RL algorithm. It would be better to include more experiments to verify this claim.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4