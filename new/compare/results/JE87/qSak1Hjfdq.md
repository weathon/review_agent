# Review

## Summary
This paper presents a novel approach to the problem of lifelong learning for vision-and-language navigation (VLN) agents. The authors propose Tucker Adaptation (TuKA), which represents multi-hierarchical knowledge as a high-order tensor and uses Tucker decomposition to decouple this knowledge into shared subspaces and scenario-specific experts. They also introduce a decoupled knowledge incremental learning strategy to consolidate shared subspaces while constraining specific experts for decoupled lifelong learning. Based on TuKA, they develop a VLN agent named All-dayWalker, which can continually learn across multiple navigation scenarios and achieve all-day multi-scenes navigation.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper is well-organized and easy to follow. The motivation is clear, and the problem is well-defined.
2. The introduction of Tucker Adaptation (TuKA) and the decoupled knowledge incremental learning strategy is innovative and well-justified.
3. The authors provide a thorough analysis of existing parameter-efficient adapters and clearly explain how TuKA addresses their limitations.

## Weaknesses
1. The experiments are conducted in simulated environments. While the authors extend the Habitat simulator, the paper would be strengthened by including real-world experiments or demonstrations to validate the proposed methods in practical settings.
2. The authors mention that TuKA selects specific experts from row-wise factor matrices, reducing their dimensions to vectors. However, the paper lacks a detailed explanation of how these vector representations capture and leverage the multi-hierarchical navigation knowledge.
3. The proposed TuKA and the decoupled knowledge incremental learning strategy involve several hyperparameters. The paper could benefit from a more detailed analysis of how sensitive the model is to these hyperparameters and any guidelines for tuning them.

## Questions
1. How does the computational complexity of TuKA compare to other parameter-efficient adaptation methods like LoRA and its variants? Are there any trade-offs in terms of computational efficiency?
2. The paper mentions the development of the All-dayWalker agent. Can the authors provide more details on the architecture and capabilities of this agent?
3. The authors extend the Habitat simulator with diverse degraded environments. Can the authors provide more details on how realistic these simulations are compared to real-world environments and how they enhance the training and evaluation of the VLN agent?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
8

## Confidence
4