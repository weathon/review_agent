# Review

## Summary
The paper presents a novel interpretation of GPTQ, a popular method for one-shot post-training quantization of large language models (LLMs), through the lens of Babai's nearest plane algorithm for the closest vector problem (CVP) on a lattice defined by the Hessian matrix of the layer's inputs. This mathematical equivalence provides a geometric explanation for GPTQ's error propagation step and introduces an error upper bound under the no-clipping assumption. Leveraging this insight, the authors develop improved post-training quantization methods that avoid clipping and demonstrate superior performance over the original GPTQ. Additionally, they provide efficient GPU inference kernels for the resulting representations.

## Soundness
3

## Presentation
2

## Contribution
3

## Strengths
1. The paper presents an innovative and compelling connection between GPTQ and Babai's nearest plane algorithm, offering a fresh perspective on LLM quantization.
2. The theoretical analysis is thorough and well-supported, with clear mathematical arguments and practical implications.
3. The experimental results demonstrate that the proposed methods outperform the original GPTQ, validating the theoretical findings and showcasing practical benefits.

## Weaknesses
1. The paper lacks a clear and detailed explanation of the practical implementation of the proposed methods, particularly in comparison to the original GPTQ algorithm.
2. The experimental evaluation could be more comprehensive, with additional comparisons to other state-of-the-art quantization methods and a broader range of model architectures and datasets.

## Questions
1. Can you provide a more detailed, step-by-step comparison between your proposed methods and the original GPTQ algorithm? Specifically, how do the error propagation steps differ, and what practical implications does this have for implementation?
2. How does your approach scale to larger models, such as the LLaMA2-70B, and what are the computational complexities involved?
3. Could you include comparisons with other recent advancements in LLM quantization, such as QuIP and LQ-Net, to provide a more comprehensive evaluation of your method's performance?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4