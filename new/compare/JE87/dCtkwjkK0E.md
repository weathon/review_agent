# Review

## Summary
This paper proposes an active learning method for flow matching models, focusing on shape design tasks. The authors analyze the influence of data points on model diversity and accuracy through a piecewise-linear neural network framework. They propose two query strategies to enhance diversity and accuracy respectively, and a mixed strategy to balance the two. Experiments on multiple datasets demonstrate the effectiveness of the proposed methods.

## Soundness
2

## Presentation
2

## Contribution
2

## Strengths
1. This paper introduces an active learning method specifically designed for flow matching models, addressing an underexplored area in the field.
2. The paper provides a theoretical analysis framework based on piecewise-linear neural networks, elucidating the impact of data points on model diversity and accuracy.

## Weaknesses
1. The experiments are conducted on relatively simple datasets. The paper would benefit from evaluating the proposed methods on more complex, high-dimensional datasets to better demonstrate their effectiveness and scalability.
2. The authors should consider comparing their proposed query strategies with traditional active learning methods designed for discriminative models, as a baseline comparison would help validate the superiority of their approach.
3. The authors should provide more details about the flow matching models used in the experiments, and the hyperparameters involved in the training process.

## Questions
1. Can the proposed active learning method be applied to other types of generative models beyond flow matching models? If so, what adaptations would be necessary?
2. How does the computational complexity of the proposed method compare to traditional active learning approaches, especially for large-scale datasets?
3. How sensitive are the proposed query strategies to the choice of hyperparameters, such as the weights in the mixed strategy?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4