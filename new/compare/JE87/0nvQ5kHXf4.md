# Review

## Summary
The paper introduces Weight-Activation Subspace Iteration (WASI), a method for efficient training of transformer models in resource-constrained environments. The approach is based on the hypothesis that a model's essential information resides in a fixed subspace, which can be leveraged to reduce memory and computational costs during training. The authors demonstrate that WASI can significantly reduce memory usage and computational overhead while maintaining accuracy comparable to traditional training methods.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper proposes a novel method that combines low-rank decomposition of weights and activations, which is a unique contribution to the field.
2. The authors provide a strong theoretical basis for their approach, including a detailed analysis of the stability of parameter subspaces during fine-tuning.
3. The paper includes extensive experiments on multiple datasets and models, demonstrating the effectiveness of WASI in reducing resource requirements while maintaining accuracy.
4. The authors also evaluate WASI in real-world scenarios, such as on a Raspberry Pi, showing its practical applicability.

## Weaknesses
1. The paper does not provide a detailed analysis of the potential impact of WASI on model generalization, especially when applying the method to new, unseen datasets.
2. The authors could provide more insight into the scalability of WASI for extremely large models or datasets.
3. The paper could benefit from a more detailed comparison with other state-of-the-art methods for efficient model training, particularly in terms of accuracy and computational efficiency.

## Questions
1. How does WASI affect the generalization performance of transformer models on new, unseen datasets?
2. What are the potential limitations of WASI when applied to extremely large models or datasets?
3. How does WASI compare to other state-of-the-art methods for efficient model training in terms of accuracy and computational efficiency?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4