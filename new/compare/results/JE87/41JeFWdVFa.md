# Review

## Summary
This paper proposes a degradation modeling method for single image super-resolution. It consists of a degradation prediction module and a denoiser module. The proposed method can be used as a loss function during training or as a post-processing step during inference. Experiments show that the proposed method can improve the generalization ability of existing SR models.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
1. The proposed method is lightweight and easy to implement.
2. The proposed method can be used in different scenarios.

## Weaknesses
1. The novelty of this paper is limited. The proposed method is similar to the degradation modeling method in [1]. The difference is that the proposed method uses an autoencoder instead of a GAN.
2. The authors claim that the proposed method is lightweight, but there is no comparison of the number of parameters or running time in the experiments.
3. The authors claim that the proposed method can enhance the generalization of existing SR models, but there is no comparison with existing generalization-aware SR methods, such as [2] and [3].

[1] Unsupervised Diffusion-based Degradation Modeling for Real-World Super-Resolution. AAAI 2025.

[2] Improving Generalization for Super-Resolution by Self-Supervised Learning. CVPR 2024.

[3] Real-World Blind Super-Resolution via Feature Matching with Implicit High-Resolution Priors. ACMMM 2022.

## Questions
1. What is the difference between the proposed method and the degradation modeling method in [1]?
2. Can the proposed method be compared with existing generalization-aware SR methods, such as [2] and [3]?

[1] Unsupervised Diffusion-based Degradation Modeling for Real-World Super-Resolution. AAAI 2025.

[2] Improving Generalization for Super-Resolution by Self-Supervised Learning. CVPR 2024.

[3] Real-World Blind Super-Resolution via Feature Matching with Implicit High-Resolution Priors. ACMMM 2022.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
4