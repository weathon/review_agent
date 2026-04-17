# Review

## Summary
This paper introduces a method for training delays in recurrent spiking neural networks (SNNs) using surrogate gradient learning. The authors propose a differentiable interpolation technique to handle non-integer delays during training, which are then rounded to the nearest integer for inference. The method is evaluated on two temporal datasets, Spiking Speech Command and Permuted Sequential MNIST, where it achieves state-of-the-art performance.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
1. The paper is well-structured and clearly written, with a logical flow from problem motivation to method description and experimental results.
2. The proposed method is tested on two different datasets, demonstrating its versatility and effectiveness.

## Weaknesses
1. The method of learning delays presented in this paper is quite similar to that in [1]. Could you elaborate on the key differences between them?
2. In the experiments, the authors only compared their method with a few baseline models. Could you include more models for comparison, especially those that also learn delays?
3. The paper primarily focuses on the performance of the model on specific datasets. Could you provide more analysis on the learned delays themselves? For example, how do the learned delays change during training, and are there any interesting patterns or insights you can share?

[1] Learning Delays in Spiking Neural Networks using Dilated Convolutions with Learnable Spacings.

## Questions
See weaknesses.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4