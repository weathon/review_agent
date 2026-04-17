# Review

## Summary
This paper proposes a 1-bit PTQ method for LLMs. This paper first investigates why and under what conditions output-matching fails, in the context of 1-bit LLM quantization. Based on the findings, this paper proposes a novel data-aware PTQ approach for 1-bit LLMs that explicitly accounts for activation error accumulation while keeping optimization efficient. Empirical experiments demonstrate that the solution consistently outperforms existing 1-bit PTQ methods with minimal overhead.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
- The analysis of why output-matching fails in 1-bit PTQ for LLMs is insightful.

- The proposed method is simple and effective.

- The experiments are extensive.

## Weaknesses
- The overhead of the proposed method is not reported, including the training time and GPU memory usage.

- The proposed method is only evaluated on 1-bit PTQ for LLMs. Can this method be applied to 2-bit PTQ for LLMs?

## Questions
See weaknesses.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4