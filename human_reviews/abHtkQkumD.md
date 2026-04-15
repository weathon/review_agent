# Learnable Context-Aware Attention Mask for Multimodal Transformers

- Decision: Reject
- Scores: 5, 3, 6, 6

## Abstract
The Self-Attention mechanism in Transformer models has shown great success across many domains, but its effectiveness can diminish in complex settings, such as multimodal tasks. This is due to the varying token granularity and the high computational cost of processing long sequences. To overcome these limitations, we propose the Learnable Context-Aware Attention Mask (LCAAM), a novel method that globally adjusts attention maps to prioritize the most important tokens in a sequence. Our approach integrates LCAAM into a BERT-like Transformer network, enhancing the Self-Attention mechanism by capturing token relationships while accounting for their contextual relevance. Additionally, we extend LCAAM to a multi-layer framework, enabling it to capture diverse information across the layers of the Transformer. Extensive experiments on datasets including MADv2, QVHighlights, ImageNet-1K, and MSRVTT demonstrate that LCAAM improves model performance while reducing redundant computations. This innovation offers a significant improvement in tackling complex tasks, such as movie understanding.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper addresses the issues of varying token granularity and the high computational cost of processing long sequences by proposing a learnable context masking mechanism. This mechanism uses a globally learnable attention map to select the most important tokens for subsequent learning. Extensive experiments are provided to support this approach.

The author's detailed response has addressed most of my questions. Utilizing contextual information to dynamically adjust attention maps to enhance the attention mechanism is highly insightful and valuable. From the experimental results, it is evident that LCAAM effectively prunes redundant connections within the attention mechanism. After carefully considering the reviewer's comments in sa9g, I will appropriately increase my score.

### Strengths
1."The paper is logically clear and well-structured."

2."Experiments were conducted on multiple datasets, and additional supplementary materials are provided."

### Weaknesses
1."The idea presented in the paper is very interesting, and the problem addressed is important. However, it is not clear from the paper whether the mask mechanism alone can effectively handle token granularity and the high computational cost. The paper does not provide a detailed analysis. Please provide your motivation and explain how the mask tokens address these issues."

2."The paper implements the mask tokens using a few layers of MLP without any additional constraints. Is the output of a few MLP layers directly suitable as mask features? This part lacks a detailed analysis. Why is the output of a few MLP layers effective for the desired result? This approach seems more like an adapter. Please provide a detailed explanation of this aspect."

### Questions
1.Is the output of a few MLP layers directly suitable as mask features? 
2.Why is the output of a few MLP layers effective for the desired result?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper presents a method that dynamically generates attention score map. The authors claim that in multi-modal settings where inputs of diverse modalities and long sequence lengths make transformers' self attention mechanism hard to perform well. From this motivation, the authors propose an additional approach on top of self-attention that generates attention maps that eventually get incorporated to QK similarity scores by either addition or multiplication.

### Strengths
The algorithm is simple enough for readers to follow. The authors present results in multiple benchmarks, achieving compelling accuracies. This paper tackles a problem inherent in multi-modality, which the community has recently shown significant interests.

### Weaknesses
I have multiple concerns regarding the paper, as listed below:
- The method part is not explaining the concept correctly. In Sec 3.2, which is supposed to be the core contribution of this paper, the final output of the stack of layers is in shape of $\mathbb{R}^{B \times N \times D_{out_L}}$. Do the authors mean that the output dimension of the final projection layer is $N$? Then, how is this method going to handle dynamic numbers of tokens? This is one of the main problems that the authors had to clearly present.
- The presentation must be improved. There are too many unnecessary details that are redundant. For example, most of the equations and the definition (Sec 3.1) can be simplified in a single equation or a few sentences.
- This paper probably does not compare to FlashAttention. Unless there is a solution that the authors put hardware algorithms in consideration, having an explicit attention score mask cannot be efficiently integrated into the FlashAttention [A] framework. Although FlexAttention [B] can handle arbitrary mask values, the explicit materialization of attention scores as did in this paper cannot bypass memory bandwidth overhead. FlashAttention is now being used everywhere, thus the comparisons must be based on FlashAttention.
- I cannot understand how the element-wise multiplication of M to the attention score works. Since M is not constrained to be non-negative, M can completely change attention values.

[A] Dao et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness\
[B] https://pytorch.org/blog/flexattention/

### Questions
- How exactly is M being generated?
- How can it handle arbitrary input lengths?
- In the explanation about Table 3, the authors mention that they augment vanilla Transformer with additional linear layers. How are these layers being added? I cannot agree with the finding that the number of parameters do not correlate with performance gains. I suspect these additional linear layers are being added in a wrong manner such as without residual connections. Instead of adding linear layers, how does the performance change if simply stacking more transformer encoder layers?
- How is this approach better than vanilla self-attention given that both dynamically generate scores? Self attention should be much precise given that it actually visits other tokens' keys, whereas the proposed method just uses linear projections per each token.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes a learnable attention mask to soften the attention map for cross-modal understanding. The authors claim that due to semantic density, the original attention mechanism may not fully capture the relation between different modalities.  Therefore, they propose the learnable context-aware attention mask (LCAAM) to regulate and prioritize the token importance of the long sequence. The implementation of LCAAM is a stack of MLP layers. The authors provide extensive experiments on various task settings and benchmarks.

### Strengths
1. Due to the distinction of semantic density, there is a mismatch of token granularity between different input modalities. Hence, the motivation of this work, regulating the token importance of different modalities, makes sense to me.
2. I believe the implementation of LCAAM (line 224-226) is easy to follow.
3. The extensive experiments demonstrate the effectiveness of this work. The ablation study is comprehensive.

### Weaknesses
1. Although the visualization in the A.4.3 illustrates the learned context, it is not intuitive that the proposed module can learn the context along the sequence.
2. I think LCAAM is related to STN [1], which produces a matrix to regularize the convolution feature.


[1] Spatial Transformer Networks. NIPS 2015

### Questions
1. What is the benefit of LCAAM on language dataset? It seems like it cannot be generalized to NLP since the semantic density of language is quite high and the context mask may not help.
2. I believe this technique would be helpful for text2vedio generation. I hope the author can discuss it.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the Learnable Context-Aware Attention Mask (LCAAM), a new mechanism designed to enhance Transformer models when dealing with complex multimodal tasks involving data like video, audio, and text. Traditional Self-Attention struggles with varying token lengths and high computational costs in such settings. LCAAM tackles this by dynamically generating a global mask that adjusts attention maps to prioritize important tokens based on their context. The authors integrate LCAAM into a BERT-like Transformer and extend it to a multi-layer setup to capture diverse information across layers. They test their approach on datasets like MADv2, QVHighlights, ImageNet-1K, and MSRVTT, showing that LCAAM improves performance and reduces redundant computations in multimodal tasks.

### Strengths
1. LCAAM offers a fresh take on improving attention mechanisms by dynamically adjusting attention maps based on context, particularly for multimodal data, which hasn't been extensively explored before.
2. The experiments are thorough, covering a range of datasets and tasks, from audio description generation to image classification. The ablation studies are helpful in understanding the impact of different components.
3. Addressing the challenges in multimodal data processing is important, and LCAAM shows promising improvements, suggesting it could be beneficial for advancing models in this area.

### Weaknesses
1. The mathematical formulation of LCAAM is a bit light. A more rigorous explanation of how the mask is generated and interacts with the attention mechanism would strengthen the paper.
2. LCAAM shows limited gains in single-modality tasks. It would be useful to investigate why this is the case and whether adjustments could improve its effectiveness in these scenarios.

### Questions
1. Can you delve deeper into why LCAAM improves performance? Specifically, how does the mask influence the attention distribution, and what's the theoretical basis for its effectiveness?
2. How does LCAAM stack up against other dynamic attention mechanisms like sparse attention or adaptive attention spans? Including direct comparisons would clarify its advantages.
3. Could you provide more insights into the computational overhead, such as actual latency measurements or memory usage in practical settings?

### Soundness
3

### Presentation
2

### Contribution
2
