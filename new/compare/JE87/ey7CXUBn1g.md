# Review

## Summary
This paper proposes AdaSVD, an adaptive SVD-based LLM compression approach. Specifically, it includes adaComp, which adaptively compensates for SVD truncation errors by alternately updating the singular matrices U and V ⊤. Additionally, it includes adaCR, which adaptively assigns layer-specific compression ratios based on the relative importance of each layer. Extensive experiments across multiple LLM/VLM families demonstrate that AdaSVD consistently outperforms state-of-the-art (SOTA) SVD-based methods, achieving superior performance with significantly reduced memory requirements.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper is well-written and easy to follow.
2. The idea of alternately updating U and V to minimize the SVD compression error is novel.
3. The experiments are extensive and demonstrate the effectiveness of the proposed method.

## Weaknesses
1. The idea of layer-wise compression ratio allocation is not new, as it has been proposed in [1]. 
2. The paper lacks experiments on the actual inference speedup of the compressed model. The compression ratio is a theoretical metric that does not necessarily translate to real-world speedup. I suggest the authors report the inference speedup on real hardware (e.g., GPU).
3. The paper lacks a comparison with [2], which is also an SVD-based LLM compression method. I suggest the authors include a comparison with [2] in terms of compression performance and speedup. 

[1] Shortgpt: Layers in large language models are more redundant than you expect.

[2] Eora: Training-free compensation for compressed llm with eigenspace low-rank approximation.

## Questions
1. How does the proposed method perform on larger models (e.g., 13B, 30B, 70B)?
2. What is the actual inference speedup of the compressed model on real hardware (e.g., GPU)?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4