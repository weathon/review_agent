# Review

## Summary
This paper proposes a novel method for unlearning in LLMs by leveraging the model's own high-confidence predictions (model beliefs) to better suppress both target responses and semantically related rephrasings. The authors introduce a bootstrapping (BS) framework with two implementations: BS-T (token) and BS-S (sequence), which directly counteract the squeezing effect by penalizing high-likelihood tokens or sequences. Extensive experiments demonstrate that this approach outperforms existing methods in achieving more reliable forgetting while preserving model utility.

## Soundness
4

## Presentation
4

## Contribution
4

## Strengths
- The paper addresses a critical limitation in current unlearning methods by targeting the squeezing effect, which is a significant contribution to the field.
- The paper is well-written and clearly structured, with thorough explanations and effective visual aids that enhance understanding.
- The proposed bootstrapping framework is versatile and can be integrated with various unlearning objectives and regularizations, making it a valuable tool for researchers and practitioners.
- The experiments are comprehensive, covering multiple benchmarks and model families, providing strong empirical evidence for the effectiveness of the proposed methods.

## Weaknesses
- The paper lacks a detailed analysis of the computational overhead introduced by the bootstrapping framework, particularly for large-scale applications.
- While the method shows strong performance, the paper could benefit from a more detailed exploration of scenarios where the approach may not be as effective, such as for highly specialized or domain-specific knowledge.

## Questions
- Can you provide more insights into the computational costs associated with the bootstrapping framework, especially for large-scale models or datasets?
- How does the performance of the bootstrapping framework vary with different types of knowledge or content that are more domain-specific or nuanced?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
8

## Confidence
4