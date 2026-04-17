# Review

## Summary
This paper proposes a personalized context-aware tokenizer for generative recommendation, which can tokenize the same item into different semantic IDs under different user contexts, thereby capturing diverse user interpretations and enhancing the model's generative capability. Experiments on three public datasets demonstrate the effectiveness of the proposed method.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
1. This paper is well-organized and easy to follow.

2. This paper focuses on an interesting problem, i.e., the same item may be interpreted differently by different individuals.

3. The authors conduct extensive experiments to demonstrate the effectiveness of the proposed method.

## Weaknesses
1. The motivation of this paper is unclear. The authors claim that the semantic IDs with the same prefixes always receive similar probabilities under the autoregressive paradigm, so a single fixed mapping implicitly enforces a universal item similarity standard across all users. However, in my opinion, the tokenization methods (e.g., TIGER) are able to learn the semantic IDs with diverse probabilities, which can also capture the diverse user intentions and preferences.

2. The proposed method seems to be an incremental improvement based on existing works, i.e., TIGER and ActionPiece. The authors should further clarify the differences and advantages of the proposed method compared with these works.

3. The authors should provide the time complexity analysis and efficiency experiments of the proposed method.

## Questions
Please refer to Weaknesses.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4