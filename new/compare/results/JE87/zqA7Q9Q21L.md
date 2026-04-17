# Review

## Summary
This paper focuses on the problem of computing worst-case robust strategies in pursuit-evasion games (PEGs) under partial observability. The authors introduce a dynamic programming (DP) algorithm for solving Markov PEGs and prove its optimality in asynchronous-move settings. They further extend the DP policies to partially observable settings through a belief preservation mechanism. The authors also embed this mechanism into the Equilibrium Policy Generalization (EPG) framework, leading to real-time pursuit strategies. Experiments demonstrate the zero-shot generalization of the learned policies to unseen graphs and their superiority over existing methods.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper addresses an important problem in PEGs and proposes a novel approach to solving it. The introduction of a DP algorithm and its extension to partially observable settings is a significant contribution.
2. The theoretical analysis provided in the paper is rigorous and well-supported. The authors prove the optimality of their methods and demonstrate their effectiveness in various settings.
3. The experimental results are comprehensive and convincing. The authors evaluate their methods on a diverse set of graphs and compare them with state-of-the-art techniques, showing their superiority.

## Weaknesses
1. The paper does not provide a detailed analysis of the computational complexity of their methods. While they mention the time complexity of the DP algorithm and the inference time of the RL policies, a more in-depth analysis of the scalability of the methods would be beneficial.
2. The paper does not discuss the sensitivity of their methods to various hyperparameters, such as the observation range, the number of pursuers, and the graph structures. A more thorough analysis of these factors would provide a better understanding of the limitations and strengths of the methods.

## Questions
1. How does the observation range affect the performance of the methods? What is the minimum observation range required for the methods to be effective?
2. How does the number of pursuers affect the performance of the methods? Is there an optimal number of pursuers for a given graph structure?
3. How does the graph structure affect the performance of the methods? Are there certain graph structures where the methods are more effective or less effective?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
8

## Confidence
4