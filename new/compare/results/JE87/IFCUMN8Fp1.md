# Review

## Summary
The authors study the problem of learning POMDPs from action-observation data. They propose a method to learn the parameters of a POMDP up to a partition of the state space. Their method uses spectral techniques to learn the number of hidden states, and tensor decomposition to estimate the observation and transition matrices. The authors show that their method can learn the parameters of a POMDP even when the observation distribution varies across states but not across actions, which is a more general assumption than in previous work. Finally, the authors evaluate their method on a few small POMDP domains and show that it can recover the POMDP parameters and perform planning with a learned model.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
- The authors study an important problem of learning POMDPs from data
- The paper is well-written and easy to follow
- The proposed method is novel and can learn POMDPs under a more general assumption than previous work
- The experimental results are promising

## Weaknesses
- The proposed method assumes that the agent can sample actions uniformly at random, which may not be realistic in many applications
- The proposed method assumes that the observation distribution varies across states but not across actions, which is still a strong assumption and limits the class of POMDPs that can be learned
- The experimental domains are very small and toy
- The authors do not compare their method to any previous baselines for learning POMDPs

## Questions
- Can the proposed method be extended to work with non-uniform exploration?
- How does the proposed method compare to previous work on learning POMDPs?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4