# Review

## Summary
The paper investigates the scaling behavior of language models under the assumption of abundant compute and limited data. The authors demonstrate that, in this setting, ensembling independently trained models achieves a significantly lower loss asymptote compared to traditional methods. By combining epoching, regularization, parameter scaling, and ensemble scaling, the authors achieve a 9% improvement in validation loss, which generalizes to downstream benchmarks. The findings suggest that simple algorithmic improvements can significantly enhance data efficiency in language model pre-training.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper explores an intriguing scenario where data is limited and compute resources are abundant, offering a fresh perspective on pre-training strategies.
2. The authors provide clear explanations of their methods and the underlying intuition, making the paper accessible and easy to follow.
3. The paper demonstrates that ensembling independently trained models can achieve a lower loss asymptote, which is a novel and valuable insight for improving model performance.

## Weaknesses
1. The authors do not provide a clear explanation of why ensembling leads to a better loss asymptote. They refer to the work of Allen-Zhu and Li (2023), which suggests that ensembling helps when data can be well-classified with one of many features but is best classified using all such features. However, this explanation is not very convincing. In the context of the current paper, ensembling appears to improve performance simply because it effectively increases the model size. The authors should either provide a more robust justification for why ensembling leads to better results or consider alternative methods that achieve similar effects without the computational overhead of ensembling.
2. The authors do not explore whether ensembling could be replaced by other techniques that achieve similar effects without the computational overhead. For instance, they could investigate the impact of increasing the depth of a single model, which would be more efficient than ensembling. The authors should consider alternative methods that achieve similar effects without the computational overhead of ensembling.
3. The authors do not provide a clear explanation of how they tune the learning rate, epoch count, and weight decay. They mention using a coordinate descent algorithm but do not provide sufficient details about the algorithm or the criteria used for tuning. The authors should provide a more detailed explanation of their tuning process, including the specific algorithm used and the criteria for selecting hyperparameters.
4. The authors do not provide a clear explanation of how they determine the optimal weight decay value. They mention that the optimal weight decay is 30 times larger than standard practice but do not provide a clear explanation of how they arrived at this conclusion. The authors should provide a more detailed explanation of their methodology for determining the optimal weight decay value.
5. The authors do not provide a clear explanation of why self-distillation helps. They refer to the work of Allen-Zhu and Li (2023), which suggests that self-distillation can be viewed as implicitly ensembling the teacher and freshly initialized student. However, this explanation is not very convincing. The authors should provide a more robust justification for why self-distillation leads to better results or consider alternative methods that achieve similar effects without the computational overhead of self-distillation.

## Questions
See weaknesses.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4