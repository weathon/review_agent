# Review

## Summary
This paper proposes a method to scale inference time compute in transformers by enabling parallel adaptive computation in the latent space. The authors introduce a forking mechanism that allows tokens requiring more computation to create "bubbles" of cloned residuals within the network, which are then processed in parallel. This approach aims to reduce the computational overhead associated with sequential generation of chain-of-thought reasoning steps. The method is evaluated on zero-shot datasets like HellaSwag and LAMBADA, demonstrating improved performance over standard decoder LMs and non-adaptive parallel computation methods.

## Soundness
2

## Presentation
3

## Contribution
2

## Strengths
1. The paper introduces a novel method for parallel computation in transformers, addressing the limitations of existing approaches that rely on sequential chain-of-thought generation.
2. The proposed method is evaluated across multiple datasets, showing improvements in performance metrics such as perplexity and zero-shot evaluation scores.

## Weaknesses
1. The proposed method is only evaluated on zero-shot datasets, lacking evidence of effectiveness on more challenging tasks such as GSM8K and MATH.
2. The paper lacks a detailed analysis of the computational overhead introduced by the forking mechanism, particularly regarding memory usage and inference time.
3. The method's reliance on a specific forking mechanism that requires manual design for different tasks limits its general applicability.
4. The paper does not provide a thorough comparison with existing methods that enhance transformer computation through adaptive mechanisms, such as the Universal Transformer.

## Questions
1. How does the proposed method perform on more challenging tasks, such as GSM8K and MATH, compared to standard decoder LMs?
2. What is the computational overhead introduced by the forking mechanism in terms of memory usage and inference time?
3. How does the method compare with other adaptive computation methods, such as the Universal Transformer, in terms of performance and efficiency?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4