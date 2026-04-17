# Review

## Summary
This paper presents an approach to language-based audio retrieval, which is the task of finding audio recordings that match a given text query. The authors propose a dual encoder architecture that leverages contrastive learning, soft-label distillation, and cluster-based classification to improve the alignment between audio and text representations. They also employ an LLM-based augmentation pipeline to enhance the diversity of the training data. The method is evaluated on the CLOTHO dataset and achieves a weighted ensemble performance of 48.8 on the development test split.

## Soundness
2

## Presentation
2

## Contribution
2

## Strengths
1. The paper addresses an important problem in audio retrieval, which has applications in multimedia search, audio annotation, and cross-modal understanding.
2. The proposed method combines several advanced techniques, such as LLM-driven caption augmentation and cluster-guided auxiliary classification, which show promise in improving the robustness of the audio retrieval system.
3. The authors provide some ablation studies to demonstrate the effectiveness of their approach.

## Weaknesses
1. The paper lacks a clear motivation for why the proposed techniques are necessary or how they specifically address the challenges of language-based audio retrieval. For instance, it is not immediately clear why soft-label distillation or cluster-based classification would be particularly beneficial in this context.
2. The novelty of the approach is limited. The use of dual encoders, contrastive learning, and distillation loss is not new and has been explored in previous works. The cluster-based classification could also be seen as a straightforward extension of existing methods.
3. The paper does not provide a detailed analysis of how the proposed techniques interact with each other or how they contribute to the overall performance. For example, it would be helpful to understand the impact of the LLM-based augmentation on the quality of the soft-label distillation.
4. The experimental evaluation is limited to a single dataset (CLOTHO). This makes it difficult to assess the generalizability of the approach to other audio retrieval tasks or datasets.
5. The paper does not compare its results with any baseline methods or state-of-the-art approaches. This makes it hard to evaluate the effectiveness of the proposed method relative to existing solutions.

## Questions
1. What are the main challenges in language-based audio retrieval that the proposed techniques are designed to address?
2. How do the individual components (contrastive learning, distillation loss, cluster-based classification) contribute to the overall performance of the system?
3. What is the impact of the LLM-based augmentation on the quality of the soft-label distillation?
4. How does the proposed method compare to existing approaches in terms of computational efficiency and scalability?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
4