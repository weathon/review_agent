# Review

## Summary
This paper studies the theory of separable neural networks (SepNNs) and proposes an efficient training algorithm, SepPGD, to improve SepNN training. For theory, the authors first establish the universal approximation theory for SepNNs. Then, they derive the neural tangent kernel (NTK) regimes for SepNNs under different asymptotic conditions. For the training algorithm, the authors propose SepPGD that provably adjusts the eigenvalue distribution of the NTK matrix. Experiments on various tasks validate the effectiveness of the proposed method.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper is well-written and easy to follow.
2. The paper provides a comprehensive theoretical analysis of SepNNs, including universal approximation theory and NTK regimes. These results contribute to a deeper understanding of SepNNs' representation capacity and training dynamics.
3. The proposed SepPGD method is well-motivated and theoretically justified. SepPGD addresses the spectral bias issue in SepNNs, potentially improving training efficiency.
4. The experiments are extensive and cover various applications such as kernel ridge regression, image representation, surface representation, and PINNs. The results demonstrate the effectiveness of SepPGD in improving convergence and accuracy.

## Weaknesses
1. The paper lacks a discussion on the practical implementation of SepPGD and its scalability to large-scale datasets.
2. The paper does not provide a detailed analysis of the computational complexity of SepPGD compared to other training methods.
3. The paper does not thoroughly discuss the potential limitations or failure cases of SepPGD.

## Questions
1. How does the choice of the modulation function $g$ affect the performance of SepPGD? Are there any guidelines for selecting this function?
2. How does SepPGD scale to large-scale datasets with higher dimensions? Is there a limit on the dimensionality where SepPGD starts to underperform compared to other methods?
3. Have you tested SepPGD on non-grid inputs? How does it perform compared to the classical NTK-based PGD in this case?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4