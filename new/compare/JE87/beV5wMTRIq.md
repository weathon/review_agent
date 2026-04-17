# Review

## Summary
The paper presents a framework that embeds rotation-equivariant tensor-field neural operators directly on the sphere, couples them with a numerically rigorous gradient operator based on spherical transforms and physically consistent boundary treatment, and augments the learned dynamics with diffusion terms derived from the atmospheric primitive equations.

## Soundness
2

## Presentation
2

## Contribution
2

## Strengths
The paper is easy to follow. The proposed method outperforms existing baselines on several benchmarks.

## Weaknesses
1. The paper does not provide a comprehensive comparison with existing state-of-the-art methods, such as the neural GCM [1] and Aurora [2]. Including these comparisons would help to better understand the advantages and limitations of the proposed approach.
2. The paper could benefit from a more detailed analysis of the computational efficiency and scalability of the proposed method. It is unclear how the method performs in terms of computational cost and how it scales with increasing resolution or length of the forecast horizon.
3. The paper does not provide a detailed analysis of the sensitivity of the proposed method to its hyperparameters. Understanding how the performance of the method changes with different hyperparameters would provide valuable insights into its robustness and generalizability.

[1] Kochkov, Dmitrii, et al. "Neural general circulation models for weather and climate." Nature 632.8027 (2024): 1060-1066.

[2] Bodnar, Cristian, et al. "Aurora: A foundation model of the atmosphere." arXiv preprint arXiv:2405.13063 (2024).

## Questions
1. How does the proposed method compare to the neural GCM and Aurora in terms of accuracy and computational efficiency?
2. Can you provide more details on the computational efficiency and scalability of the proposed method? How does the computational cost scale with increasing resolution and forecast horizon?
3. Can you provide more details on the sensitivity of the proposed method to its hyperparameters? How does the performance change with different hyperparameters?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
4