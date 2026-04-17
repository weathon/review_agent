# Review

## Summary
This paper proposes a data-centric framework to maximize reasoning in small language models under limited parameters and tokens. The authors introduce benchmark-free, self-evolving data optimization, a principled dataset-level weighting method that leverages cross-domain influences to dynamically tailor the data mixture. This approach enables strong performance on code, math, and knowledge benchmarks without exposing any benchmark data during training or mixture construction.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper is well-written and easy to follow.
2. The authors provide a comprehensive analysis of the impact of different datasets on reasoning capabilities, and propose a data–model co-evolution strategy to adapt to rapid changes in model capacity during mid-training.
3. Experimental results show that the proposed method can achieve good performance on reasoning tasks with less training data.

## Weaknesses
1. The proposed method is mainly focused on data selection and mixing, and the technical innovation is relatively limited.
2. The authors should provide more detailed ablation studies to analyze the contribution of each component in the proposed method.
3. The authors should provide more discussion on the generalization ability of the proposed method, i.e., whether the selected datasets and the data mixing strategy can be applied to larger models.

## Questions
See weaknesses.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4