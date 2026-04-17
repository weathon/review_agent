# Review

## Summary
The paper proposes a method to perform covariance-adjusted support vector classification in non-Euclidean spaces. It addresses the limitations of traditional SVMs, which are designed for Euclidean spaces, by incorporating data covariance into the optimization process. The authors propose an algorithm that iteratively estimates a population covariance-adjusted SVM classifier, demonstrating its effectiveness on multiple datasets compared to traditional SVMs.

## Soundness
2

## Presentation
1

## Contribution
1

## Strengths
The paper introduces a novel approach to performing SVMs in non-Euclidean spaces by incorporating data covariance, which is a significant deviation from traditional methods that rely solely on Euclidean distances.

## Weaknesses
1. The paper's presentation is subpar, with numerous grammatical errors, unclear explanations of the methodology, and lack of detailed descriptions of the experimental setup. The figures are also poorly presented, with insufficient captions and unclear labels, making it difficult for readers to understand the results. The paper would benefit greatly from a thorough rewrite, improved figure quality, and more detailed explanations of the methodology and experiments.
2. The paper lacks a rigorous theoretical foundation for the proposed method. The authors do not provide a detailed mathematical justification for why their approach should work better than traditional SVMs in non-Euclidean spaces. Additionally, there is no theoretical analysis of the algorithm's convergence properties or its performance guarantees.
3. The experimental evaluation is limited in scope and lacks depth. The authors do not compare their method against a wide range of relevant baselines, making it difficult to assess the true effectiveness of their approach. Furthermore, there is no discussion of the computational complexity of their method or how it scales with the size of the dataset.

## Questions
1. Can you provide a more detailed theoretical analysis of your method, including mathematical proofs and convergence properties?
2. How does your method perform compared to a wider range of relevant baselines, including more recent approaches to covariance-adjusted SVMs?
3. What is the computational complexity of your method, and how does it scale with the size of the dataset?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
1

## Confidence
5