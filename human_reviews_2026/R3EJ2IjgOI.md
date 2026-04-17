# Memory Caching: RNNs with Growing Memory

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Transformers have been established as the de-facto backbones for most recent advances in sequence modeling, mainly due to their growing memory capacity that scales with the context length. While plausible for retrieval tasks, it causes quadratic complexity and so has motivated recent studies to explore viable subquadratic recurrent alternatives. Despite showing promising preliminary results in diverse tasks, such recurrent architectures underperform Transformers in recall-intensive tasks, 
{often attributed to their fixed-size memory. In this paper, we introduce Memory Caching (\mc), a simple yet effective technique that enhances recurrent models by caching checkpoints of their memory states (a.k.a. hidden states). \mc{} allows the effective memory capacity of RNNs to grow with sequence length, offering a flexible trade-off that interpolates between the fixed memory ($\mathcal{O}(L)$ complexity) of RNNs and the growing memory ($\mathcal{O}(L^2)$ complexity) of Transformers. We propose four variants of MC, including gated aggregation and sparse selective mechanisms, and discuss their implications on both linear and deep memory modules.}
Our experimental results on language modeling, and long-context understanding tasks show that \mc{} enhances the performance of recurrent models, supporting its effectiveness. In in-context recall tasks, our results indicate that while Transformers still achieve the best performance, our MC variants show competitive performance, close the gap with Transformers, and performs better than state-of-the-art recurrent models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Memory Caching (MC), a technique that enhances the memory capacity of RNN by caching checkpoints of memory states, thus allowing memory to grow with the sequence length. The MC framework aims to strike a balance between the fixed memory size of traditional RNNs $(O(L))$ and the growing memory complexity of Transformers $(O(L^2))$. The paper introduces several MC variants, including Gated Residual Memory (GRM), Memory Soup, and Sparse Selective Caching (SSC). Experiments on language modeling and retrieval tasks show that MC improves RNNs’ retrieval performance and can close the gap with Transformer-based models, though Transformers still outperform in recall-intensive tasks.

### Strengths
1. The paper effectively addresses the bottleneck of fixed-size memory in RNNs by proposing a mechanism that allows memory to grow with the sequence length, providing a flexible middle ground between traditional RNNs and Transformer models.
2. The MC framework, especially with Gated Residual Memory and Sparse Selective Caching, significantly enhances the retrieval ability of RNNs, improving performance on tasks where sequence history plays a crucial role.

### Weaknesses
1. The proposed GRM still leads to quadratic complexity in the worst case when applied to long sequences, making it no better than vanilla attention in terms of computational complexity. If SWA (Sliding Window Attention) being treated as an RNN, SWA+GRM essentially reduces to vanilla attention, undermining the claim of significant computational efficiency improvements over attention-based approaches like Transformers.
2. Despite the enhancement in memory efficiency and retrieval performance, the results still show that Transformers outperform MC-enhanced RNNs on recall-intensive tasks. This suggests that simply expanding memory is not enough to fully match the expressive power of Transformer models, particularly in tasks that require broader context retention.

### Questions
1. Given that the GRM design still results in quadratic complexity for long sequences, what specific advantage does this design offer over standard attention-based methods that already deal with growing memory in sequences? Is the claimed advantage over Transformers primarily empirical rather than computational?
2. While expanding the memory size through caching improves retrieval performance, why does it not fully align with Transformer models in recall-intensive tasks? Could this be due to inherent limitations in RNN architectures when it comes to capturing long-term dependencies, even with larger memory?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors introduce Memory Caching (MC), a technique designed to enhance Recurrent Neural Networks (RNNs) by allowing their effective memory capacity to grow with the input sequence length, similar to Transformers. The technique involves caching checkpoints of the RNNs' memory states (hidden states) along the sequence. This effectively allows the model's memory to expand as the sequence grows. The experimental results demonstrate that MC (and the multiple proposed variants) are effective for enhancing recurrent models on multiple recall-intensive tasks and long-context understanding settings relative to existing state-of-the-art recurrent models.

### Strengths
The results are well presented and the method is intuitive to follow. Experimental results show cases where improvements are significant.

### Weaknesses
- The experiments are missing a rather important training and inference efficiency comparison of using the cached memories. In general, if the caching is done at a fixed length (suppose $L$), then the memory will grow linearly with the sequence length, potentially limiting the ability of the memory to train on longer contexts as well as conduct significantly longer inference compared to alternative linear models, which normally operate in $O(1)$ memory at inference time.
- The main area where results improve is in retrieval-based tasks, which is rather intuitive since there is an additional cache of prior information that is being used and incorporated within the model.
- I believe rather than comparing against Samba, Hymba would be a more fair comparison as the information mixing happens between parallel branches in the same layer, thus it acts closer to a memory cache unlike Samba where the interleaving occurs between layers.

[1] Hymba: A Hybrid-head Architecture for Small Language Models

### Questions
- Given the scale of the models, it might be more convincing to incorporate the method to additional linear RNN or linear attention based methods such as Mamba/Mamba2 and GLA/GSA/DeltaNet/GatedDeltaNet.

- As a follow up to the first weakness, I would suggest the authors look into verifying whether the method would be useful as a post-tuning augmentation of already pre-trained models. Given the additional memory requirements, I have hesitations about whether or not the proposed method would be feasible for pre-training in most settings but for additional calibration or tuning of models for longer contexts (which I believe is the primary motivation of the authors) there may be greater space to operate. Have the authors attempted this?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Memory Caching (MC), a simple yet effective technique that enhances RNN models that allows memory states grow with sequence length. Experiments demonstrate the effectiveness of proposed method on language modeling, long context understanding and efficiency.

### Strengths
1. Designing a trade-off architecture between the computational complexity of RNNs and transformers is a valuable research area.
2. The proposed method is simple yet effective, and appears to be applicable to various RNN architectures.
3. The paper is clearly structured and includes sufficient experiments.

### Weaknesses
1. The paper seems to lack analysis on the impact of the number of input sequence segments on model performance. This could help us identify the trade-off between the complexity of RNN and transformer architectures and explore better memory caching lengths when the model performs sequence modeling.
2. A characteristic of RNN model architectures is their potential to generalize to longer sequences, but the paper does not seem to focus on the performance of the proposed method in terms of length extrapolation.

### Questions
1. Why do the proposed method and various other linear RNN models shown in Table 1 significantly outperform Transformers++? Since Transformers++ uses a key-value cache to store all the historical information, shouldn't its performance be the best? I referred to the results of some RNN-related papers [1][2][3], and although RNNs show high efficiency with linear complexity, their performance still seems to lag significantly behind the Transformer.
2. Gated Residual Memory (GRM) seems to outperform other memory caching method variants in all tasks. Can you explain this situation?

[1] Songlin Yang, et al. Gated Linear Attention Transformers with Hardware-Efficient Training. ICML, 2024.
[2] Songlin Yang, et al. Parallelizing Linear Transformers with the Delta Rule over Sequence Length. NeurIPS, 2024.
[3] Jiaxi Hu, et al. Improving Bilinear RNNs with Closed-loop Control. NeurIPS, 2025.

### Soundness
3

### Presentation
3

### Contribution
3
