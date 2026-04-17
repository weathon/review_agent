# Review

## Summary
This paper introduces a text-to-motion framework with physical constraints. It first constructs a Motion2Motion dataset to capture multi-turn dialog patterns and motion semantics. Then, the authors propose an enhanced GRPO algorithm to generate motion descriptions. Finally, a low-level trajectory refinement is used to translate the descriptions into physically feasible motions. Experimental results show that the proposed method outperforms baselines in both accuracy and interpretability.

## Soundness
2

## Presentation
2

## Contribution
2

## Strengths
1. The paper constructs a Motion2Motion dataset, which contains 7132 annotated human motion samples. The dataset is designed to support downstream tasks by capturing both explicit action sequences and implicit physical constraints within its annotations.
2. The paper conducts comprehensive experiments to validate the proposed method.

## Weaknesses
1. The paper is not well-written. The description of the proposed method is unclear.
2. The paper does not provide videos to demonstrate the effectiveness of the proposed method. It is hard to evaluate the quality of the generated motions with only figures in the paper.
3. The authors do not provide the details of the baseline methods. It is hard to fairly evaluate the proposed method without the details of the compared methods.

## Questions
1. How to use the proposed method to generate motions? What are the inputs and outputs of the proposed framework?
2. How to evaluate the quality of the generated motions? The authors only provide some figures of the generated motions in the paper. However, it is hard to evaluate the motions with only figures. Please provide videos to demonstrate the motions.
3. What are the details of the baseline methods? Please provide the references or implementation details of the compared methods.
4. What is the advantage of using JS divergence in the proposed method? Please provide experimental results to demonstrate the advantage of using JS divergence.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
4