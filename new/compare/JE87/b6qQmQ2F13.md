# Review

## Summary
This paper investigates the memory-accuracy trade-offs in reasoning LLMs under various scaling scenarios. It explores how to allocate memory between model weights and KV cache under different model sizes, weight precisions, token budgets, sampling group sizes, and KV cache compression strategies. The paper provides empirical findings and guidelines for deploying reasoning models effectively.

## Soundness
2

## Presentation
3

## Contribution
2

## Strengths
1. The paper provides comprehensive experimental results, exploring over 1,700 scenarios and offering a detailed analysis of the trade-offs between different factors.
2. The findings and guidelines are useful for practitioners in the field of LLM deployment and optimization.

## Weaknesses
1. The authors primarily focus on the Qwen3 model family and a limited number of benchmarks. Although they claim that their findings generalize to other model families, additional experiments on a wider range of models and tasks would strengthen the validity of their conclusions.
2. The paper concentrates on post-training quantization methods and does not explore other potential compression techniques, such as training-time quantization or pruning methods.
3. The authors do not provide detailed information on the implementation and experimental setup, making it difficult for others to replicate their experiments and verify their findings.

## Questions
1. How do the authors ensure that their findings generalize across different model families and benchmarks? Have they conducted sufficient experiments to validate the robustness of their conclusions?
2. Why did the authors choose to focus on post-training quantization methods? Have they considered the potential benefits of other compression techniques, such as training-time quantization or pruning methods?
3. Can the authors provide more detailed information about the implementation and experimental setup, including the hyperparameters used, the training process, and the evaluation metrics?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4