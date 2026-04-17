# Review

## Summary
This paper proposes Automatic Complementary Separation Pruning (ACSP), a method that automates the pruning process for convolutional neural networks (CNNs). ACSP combines structured pruning and activation-based pruning to remove redundant neurons and channels, focusing on accelerating inference time. The method constructs a graph space that encodes the separation capabilities of each component across all class pairs, allowing for the selection of diverse and complementary components while maintaining high network performance. ACSP determines the pruning volume automatically, eliminating the need for manual tuning, and achieves significant reductions in FLOPs and inference time without compromising accuracy.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
1. The paper introduces ACSP, a novel approach to neural network pruning that fully automates the process, removing the need for manual intervention and user-defined pruning ratios. This innovation makes the method practical for real-world applications and scalable to larger networks.
2. ACSP combines structured pruning and activation-based pruning, leveraging a graph-based approach to select components with diverse and complementary separation capabilities. This combination ensures the efficient removal of redundant components while maintaining or improving network performance.
3. The method is designed to focus on inference-time efficiency, with the goal of achieving significant speed-ups without compromising accuracy. The experimental results demonstrate substantial improvements in computational efficiency across various architectures and datasets.

## Weaknesses
1. The paper lacks a detailed analysis of the computational overhead introduced by the graph construction and clustering processes. While the method is claimed to be efficient, a more thorough evaluation of the computational cost, especially for large-scale networks and datasets, would strengthen the argument.
2. The experiments are conducted on a limited set of architectures and datasets. To demonstrate the generalizability of the method, it would be beneficial to include more diverse models, such as transformers or more recent CNN architectures, and larger datasets like ImageNet-21K.
3. The paper does not provide a comprehensive comparison with state-of-the-art pruning methods, particularly those that also focus on automated pruning decisions. Including such comparisons would help position ACSP within the current research landscape and highlight its unique contributions.
4. The method relies on the construction of a graph space that encodes the separation capabilities of each component across all class pairs. This approach may become computationally infeasible for networks with millions of parameters and large datasets with hundreds of classes, limiting the scalability of the method.

## Questions
1. Can you provide more details on the computational overhead introduced by the graph construction and clustering processes? How does this overhead scale with larger networks and datasets?
2. Have you considered any approximations or optimizations to reduce the computational cost of graph construction, such as sampling or dimensionality reduction techniques?
3. How does ACSP perform on more recent CNN architectures or transformer-based models? Have you tested the method on any larger-scale datasets beyond ImageNet-1K, such as ImageNet-21K?
4. Can you provide a more comprehensive comparison with state-of-the-art pruning methods, particularly those that also focus on automated pruning decisions? This would help highlight ACSP's unique contributions and limitations.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
5