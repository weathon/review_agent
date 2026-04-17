# Review

## Summary
The paper proposes a benchmark for online map-based motion prediction and a boundary-free baseline. The authors first identify several challenges in online map-based motion prediction, including inappropriate data splits, different considered ranges for online mapping and motion prediction, and non-discriminative metrics. Based on these analyses, the authors propose a benchmark with a new data split, refined metrics, and a new baseline as the solution. The authors evaluate several methods on the proposed benchmark.

## Soundness
2

## Presentation
2

## Contribution
2

## Strengths
1. The paper is well-written and easy to follow.
2. The paper identifies several challenges in online map-based motion prediction and proposes a benchmark with a new data split, refined metrics, and a new baseline as the solution.
3. The authors evaluate several methods on the proposed benchmark.

## Weaknesses
1. The paper's novelty is limited. The paper mainly focuses on online map-based motion prediction, a subarea of autonomous driving. Although the authors identify several challenges in this area and propose a benchmark to address these challenges, the technical contribution of the paper is limited. The proposed benchmark seems to be a small modification of the existing dataset nuScenes.
2. The authors claim that "the current dataset splits are unsuitable for two-stage training, leading to a severe train-validation gap". However, the train-validation gap is a common issue in two-stage training. The proposed data split solution seems to be a common practice in two-stage training and thus is not very novel.
3. The proposed "boundary-free baseline" is not very novel. The proposed method is a simple extension of the existing method by incorporating image features.

## Questions
Please see the weaknesses.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
4