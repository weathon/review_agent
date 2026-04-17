# Review

## Summary
The paper introduces TSPulse, a novel family of ultra-light pre-trained models designed for rapid time-series analysis. TSPulse addresses the limitations of existing pre-trained time-series models by introducing a unique pre-training framework that disentangles signals across multiple representation spaces and abstraction levels. This approach allows TSPulse to learn three complementary embedding views: temporal, spectral, and semantic. The model's lightweight nature, with only 1M parameters, makes it highly efficient and suitable for real-time applications. TSPulse demonstrates superior performance across four time-series diagnostic tasks: anomaly detection, similarity search, imputation, and multivariate classification, outperforming models that are 10 to 100 times larger.

## Soundness
3

## Presentation
2

## Contribution
2

## Strengths
1. TSPulse achieves state-of-the-art performance across four time-series diagnostic tasks, outperforming larger models by a significant margin. This demonstrates the effectiveness of its disentangled representation learning and lightweight design.
2. Despite its compact size, TSPulse is capable of efficient fine-tuning and supports GPU-free deployment, making it highly practical for real-time applications and environments with limited computational resources.

## Weaknesses
1. The paper's organization is not very reader-friendly. It is difficult to quickly grasp the key points of the paper.
2. The paper lacks a detailed comparison with existing methods. For example, in the anomaly detection task, the comparison with Chronos is not comprehensive enough.
3. The description of the methodology is not clear enough. For example, in the ablation study of the classification task, what does "virtual channel expansion" refer to?

## Questions
Please refer to the weaknesses.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4