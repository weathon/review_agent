# Review

## Summary
This paper addresses a new problem in multi-modal entity alignment (MMEA), termed Dual-level Noisy Correspondence (DNC), which refers to misalignments in both intra-entity (entity-attribute) and inter-graph (entity-entity and attribute-attribute) correspondences. To tackle the DNC problem, the authors propose a robust MMEA framework called RULE. RULE first estimates the reliability of both intra-entity and inter-graph correspondences using a two-fold principle. Based on the estimated reliabilities, RULE mitigates the negative impact of intra-entity noise during attribute fusion and prevents overfitting to noisy inter-graph correspondences during inter-graph discrepancy elimination. Additionally, RULE incorporates a correspondence reasoning module that uncovers the underlying attribute-attribute connection across graphs, ensuring more accurate equivalent entity identification. Extensive experiments on five benchmarks demonstrate the effectiveness of the proposed method in addressing DNC compared to seven state-of-the-art methods.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper introduces a novel problem in MMEA, termed Dual-level Noisy Correspondence (DNC), which addresses a significant challenge in the field. The authors provide a thorough analysis of the DNC problem and its impact on MMEA, offering valuable insights into this under-explored area.
2. The proposed RULE framework is a robust and innovative approach to tackling the DNC problem. The authors present a comprehensive methodology that estimates the reliability of correspondences and mitigates the negative effects of noise during attribute fusion and inter-graph alignment.
3. The paper includes extensive experiments on five benchmarks, comparing the proposed method with seven state-of-the-art methods. The results demonstrate the effectiveness of the proposed method in addressing DNC, showcasing its superiority over existing approaches.
4. The paper is well-structured and clearly presents the problem, methodology, and experimental results. The authors provide detailed explanations and insights, making the paper accessible to readers with varying levels of expertise in the field.

## Weaknesses
1. The paper could benefit from a more detailed discussion on the limitations of the proposed method and potential directions for future research. This would provide a clearer perspective on the contributions of the current work and its impact on the field.
2. While the paper includes extensive experiments, it would be valuable to have more insights into the computational complexity of the proposed method. Understanding the scalability and efficiency of the approach is crucial for practical applications.

## Questions
1. How does the proposed RULE framework handle extremely large-scale MMKGs, and what are the computational challenges associated with such datasets?
2. Can the proposed method be extended to other types of entity alignment tasks beyond MMEA, such as cross-modal entity alignment or multi-modal relation extraction?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
8

## Confidence
4