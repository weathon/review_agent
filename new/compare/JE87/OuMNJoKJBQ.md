# Review

## Summary
The paper proposes a novel method for improving the safety alignment of LLMs through reasoning-aware post-training. The authors first conduct a causal intervention by deactivating reasoning-critical neurons and provide empirical evidence that current alignment is largely independent of deep reasoning. Then, the authors construct and release a novel Chain-of-Thought (CoT) safety fine-tuning dataset that includes both general-purpose utility examples and safety-critical prompts with detailed reasoning traces. Moreover, the authors propose Alignment-Weighted DPO, a new reinforcement learning method that assigns separate weights to reasoning and response components, enabling more fine-grained and targeted preference optimization. Extensive experiments across multiple benchmarks show that the proposed approach consistently improves safety alignment without significantly compromising utility.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper is well-written and easy to follow. The authors provide clear explanations of their methodology and results.
2. The paper addresses an important and timely issue in the field of AI safety. The proposed method has the potential to significantly improve the safety of LLMs, which is crucial for their deployment in real-world applications.
3. The authors conduct extensive experiments across multiple benchmarks to validate their approach. The results show that the proposed method consistently outperforms strong baselines in safety, without significantly compromising utility.

## Weaknesses
1. The paper could benefit from a more detailed discussion of the limitations of the proposed method. It would be helpful to understand the potential drawbacks or scenarios where the approach may not be as effective.
2. The paper could provide more insight into the interpretability of the model's decisions. It would be valuable to understand how the model arrives at its safety alignments and whether there are any biases or patterns in its reasoning.

## Questions
Please see the weaknesses.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4