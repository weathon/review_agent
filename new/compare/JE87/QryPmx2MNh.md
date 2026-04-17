# Review

## Summary
The paper proposes a method to find a good order for the target sequence in arithmetic tasks, to make it easier for Transformer models to learn. The method is motivated by the fact that the order of the target sequence can make a difference in the learning difficulty for the model. The proposed method works by training on different permutations of the target sequence and identifying the permutation with the fastest loss drop in the early stage of training. To make the method more efficient, the paper also proposes a two-stage hierarchical approach for inter- and intra-block reordering. Experiments on four order-sensitive arithmetic tasks show that the method is able to find a good order out of a few billion candidates.

## Soundness
2

## Presentation
2

## Contribution
2

## Strengths
- The proposed method is able to find a learning-friendly order out of a few billion candidates, which is a significant improvement over brute-force search.
- The hierarchical approach makes the method more efficient, by first finding block-level orders and then refining them at the token level.
- The method is able to rediscover the reverse-digit order for multiplication, which is a learning-friendly order that was reported in prior studies.

## Weaknesses
- The paper only evaluates the method on arithmetic tasks, and it is unclear how well it would generalize to other types of reasoning tasks.
- The method requires training the model multiple times, which can be computationally expensive and inefficient.
- The method relies on the observation that neural networks tend to learn from easy to hard instances during training, which may not always be the case.
- The method may not work well for variable-length target sequences, which is common in many real-world applications.

## Questions
- How well does the method generalize to other types of reasoning tasks beyond arithmetic tasks?
- Is there a way to make the method more efficient, perhaps by reducing the number of training runs needed?
- Are there other ways to initialize the candidate permutation set that might lead to better results?
- How well does the method work for variable-length target sequences?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4