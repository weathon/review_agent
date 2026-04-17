# Review

## Summary
The paper introduces a novel approach to interventional causal discovery that explicitly models post-treatment selection and latent confounders. The authors propose a new causal formulation, the Fine-grained Interventional Markov equivalence class (FI-Markov equivalence), and a corresponding graphical representation, the F-PAG. They also develop a sound and complete algorithm, F-FCI, to identify causal relations, latent confounders, and post-treatment selection using both observational and interventional data. The paper demonstrates the effectiveness of their approach on synthetic and real-world datasets, showing that it outperforms existing methods in recovering true causal structures.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
The paper addresses a critical and often overlooked problem in causal discovery, post-treatment selection, which can distort causal discovery results and challenge existing causal formulations. The authors provide a clear and rigorous theoretical foundation for their approach, including the characterization of Markov properties and the development of a sound and complete algorithm. The paper is well-organized and clearly written, with detailed explanations of the proposed methods and comprehensive experimental results.

## Weaknesses
The paper does not provide a detailed discussion of the computational complexity of the proposed algorithm or the scalability of the approach to large-scale datasets. While the authors mention the use of augmented DAGs and MAGs, they do not provide a clear explanation of how these representations are used in the algorithm or how they impact its scalability. The paper could benefit from a more detailed discussion of the limitations of the proposed approach, such as the assumptions made and the types of data or scenarios where it may not perform well.

## Questions
1. How does the proposed F-FCI algorithm scale to large-scale datasets, and what is its computational complexity?
2. Can the authors provide more details on the assumptions made by the proposed approach and discuss potential scenarios where it may not perform well?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
8

## Confidence
4