# Review

## Summary
The paper introduces a method to estimate information-theoretic measures, such as Mutual Information (MI), for high-dimensional discrete data. The authors propose to use a diffusion-based approach that leverages Continuous Time Markov Chains (CTMCs) to compute KL divergences, from which MI can be derived. The method is lightweight, scalable, and can integrate with pre-trained models. The authors demonstrate the effectiveness of their approach through experiments on synthetic and real-world data, showcasing its superior performance compared to existing methods.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
- The paper is well-written and easy to follow.
- The authors provide a detailed explanation of the theoretical foundations of their method, including the connection between CTMCs and the computation of KL divergences.
- The method is scalable and can handle high-dimensional discrete data, which is a significant advantage over existing methods that often struggle with such data.
- The experimental results are impressive, showing that the proposed method outperforms existing approaches in terms of accuracy and efficiency.

## Weaknesses
- The proposed method is an adaptation of the diffusion-based estimator for information-theoretic measures, which was originally proposed for continuous data. The authors extend this work to the discrete case, but the core idea remains the same, limiting the novelty of the contribution.
- The paper does not provide a detailed analysis of the computational complexity of the proposed method, which is an important consideration for large-scale applications.
- The paper does not compare the proposed method with other state-of-the-art methods for MI estimation on high-dimensional discrete data. While the authors mention the limitations of existing methods, a direct comparison would strengthen the paper's claims.
- The paper does not provide a detailed discussion of the assumptions and limitations of the proposed method. It would be beneficial to address any potential constraints or scenarios where the method may not perform well.

## Questions
- Can you provide a more detailed analysis of the computational complexity of your method, including its scalability to high-dimensional data?
- How does your method perform compared to other state-of-the-art methods for MI estimation on high-dimensional discrete data? Can you provide a direct comparison or an explanation of why such a comparison was not included?
- What are the assumptions and limitations of your method? Can you provide a detailed discussion of potential constraints or scenarios where your method may not perform well?
- How does your method handle missing data or incomplete datasets? Can you provide an explanation of how your method deals with such cases, if applicable?
- How does your method perform on non-standard discrete data, such as images or audio? Can you provide an explanation of why your method is limited to high-dimensional discrete data, if applicable?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4