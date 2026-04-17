# Review

## Summary
This paper addresses the problem of machine unlearning in Forward-Forward (FF) models, which are an alternative to backpropagation (BP). The authors propose a framework called FF-Erase, which introduces a goodness-guided strategy for efficient unlearning. They also propose a Goodness-based Membership Inference Attack (G-MIA) for verifying unlearning effectiveness. Experiments demonstrate that FF-Erase effectively removes data influence while preserving model utility, achieving significant speedup over retraining methods.

## Soundness
2

## Presentation
3

## Contribution
2

## Strengths
1. The paper introduces a novel unlearning framework specifically designed for Forward-Forward (FF) models, addressing an important yet unexplored problem.
2. The proposed FF-Erase method achieves faster unlearning compared to retraining from scratch, with only a minor degradation in accuracy.
3. The paper includes extensive experiments on multiple datasets and models, demonstrating the effectiveness of the proposed method.

## Weaknesses
1. The paper lacks theoretical analysis or guarantees on the effectiveness or privacy aspects of the proposed unlearning method.
2. The paper does not provide a comparison with existing unlearning methods on BP-based models, which could help establish a baseline for the performance of FF-Erase.
3. The paper does not provide a detailed analysis of the computational overhead of training the guidance model, which could be significant, especially for large-scale datasets.
4. The paper does not consider the potential for overfitting during the training of the guidance model, which could affect the stability of the unlearning process.

## Questions
1. How does the proposed method handle cases where the forgetting data is so large that the guidance model cannot be effectively trained on the remaining data?
2. What are the specific challenges in applying FF-Erase to convolutional neural networks (CNNs) or other FF models beyond the ones tested in the paper?
3. How does the performance of FF-Erase degrade as the complexity of the dataset increases, and are there any strategies to mitigate this?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4