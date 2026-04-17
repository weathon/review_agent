# Review

## Summary
This paper proposes a new transformer architecture that incorporates an external memory. The memory is implemented as a stack of slots where each slot is a d-dimensional vector. The number of slots is fixed and when a new memory entry is added, the oldest entry in the stack is replaced unless the capacity of the stack is not exceeded. Each layer of the transformer has its own stack of memory slots. The transformer also has a special token input that is mapped to a time embedding and is used to query the memory. The memory entries are updated by a cross-attention mechanism. The authors provide some theoretical analysis of the memory mechanism and demonstrate in several experiments that the model is able to solve tasks that require recalling information from a long time in the past.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
The paper is well written and easy to follow. The idea is novel and interesting and the experiments demonstrate that the approach works well.

## Weaknesses
The method is only evaluated on a set of toy tasks. The tasks are designed to test memory but it is unclear how well the method would perform in more complex tasks. For example, in the T-maze task, the agent has to remember a single scalar value. In a more complex task, the agent might have to remember a much more complex data structure. It is unclear how well the method would scale to such tasks. The authors also do not compare their method to other methods that augment transformers with memory such as Memformer [1] or Block-Recurrent Transformers [2].

[1] Wu et al., "Memformer: Transformers with Memory", 2020
[2] Hutchins et al., "Learning to Learn by Gradient Descent by Gradient Descent", 2022

## Questions
* How does the method compare to other memory augmented transformer architectures?
* What is the effect of the LRU replacement strategy on performance? How would the method perform if there was no LRU replacement and all slots had to be retained until capacity was exceeded?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4