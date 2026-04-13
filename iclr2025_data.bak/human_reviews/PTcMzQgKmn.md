## Human Reviewer 1

### Summary
The paper presents HiP (Hierarchically Pruned Attention), a novel approach aimed at reducing the time and space complexity of the attention mechanism in Large Language Models (LLMs). HiP leverages the observation that tokens close together tend to have similar attention scores to estimate the top-k key tokens for a given query on the fly. This results in sub-quadratic time complexity (O(T log T)) and linear space complexity (O(T)), where T is the sequence length.  Experimental results show that HiP significantly reduces both prefill and decoding latencies while maintaining high performance on benchmarks such as LongBench.

### Strengths
1. HiP does not require retraining, making it easy to apply to pre-trained models. In addition, by reducing the time complexity to O(T log T) and space complexity to O(T), HiP enables the use of longer context lengths without the associated quadratic cost.
2. Through KV-cache offloading, HiP optimizes GPU memory usage, which is especially beneficial for large models.
3. HiP can dynamically adjust to different sequence lengths, making it suitable for a variety of tasks that involve long contexts.

### Weaknesses
1. The effectiveness of HiP is contingent upon the presence of "attention localities," which might vary across different LLM architectures or tasks. The robustness of this operation deserves more discussions.
2. While HiP reduces latency in the decoding phase, the iterative pruning process might introduce overhead in the initial stages. How does the proposed method balance these two parts.
3. The authors have already shown improvements for certain sequence lengths, while it is expected to thoroughly explore how HiP scales to extremely long sequences or very large models. The current configuration on the sequence length (128k) and the model size (8B) are relatively small for a paper studies on the efficient decoding. Alternatively speaking, the scalability of this method deserves more experiments to support.

### Questions
See the weakness for details.

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper proposes a novel, training-free attention mechanism called Hierarchically Pruned Attention (HiP) to accelerate the serving of pre-trained Transformer-based large language models (LLMs) for long-context tasks. HiP addresses the computational challenges posed by the quadratic time and space complexity of standard attention mechanisms in handling long sequences.

### Strengths
1. HiP's training-free nature eliminates the need for costly retraining of large LLMs, making it readily applicable to existing models.
2. HiP's hierarchical pruning significantly reduces the computational complexity of attention from quadratic to log-linear, enabling efficient handling of long sequences.
3. KV cache offloading effectively addresses the memory limitations of GPUs, allowing HiP to scale to much longer context lengths.
4. The paper provides thorough experimental results on diverse benchmarks, showcasing HiP's effectiveness in terms of speedup, performance preservation, and context length extension.
5. The paper includes a theoretical analysis of HiP's hierarchical pruning algorithm, providing insights into its superior performance compared to random key selection.

### Weaknesses
1. HiP's effectiveness relies on the assumption of attention locality. While this assumption generally holds, there might be cases where it's violated, potentially impacting performance.
2. HiP enforces the same sparsity across all rows of the attention matrix. Exploring mechanisms for dynamic sparsity, where sparsity varies based on the input, could further enhance efficiency and performance.
3. The implementation and optimization of HiP, particularly the KV cache offloading, are tailored for specific hardware platforms (e.g., RTX 4090). Performance and optimal configurations might vary across different hardware accelerators.
4. The paper doesn't specifically address potential LLM alignment issues that might arise from applying HiP. Further investigation is needed to ensure HiP's safety and robustness in practical deployments.

### Questions
1. How does the choice of chunk sizes in HiP's hierarchical pruning affect the trade-off between accuracy and efficiency?
2. What strategies can be employed to further optimize KV cache offloading, such as using different memory tiers or compression techniques?
3. How well does HiP integrate with other efficiency techniques, such as quantization or model pruning, to further improve serving efficiency?
4. Can HiP be effectively applied to other Transformer architectures beyond the specific LLM model used in the paper?

### Soundness
3

### Presentation
4

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper introduces Hierarchically Pruned Attention (HiP) which reduces time and space complexity of the attention mechanism. HiP exploits "attention locality" to estimate top-k key tokens (w/ theoretical justifications as insights) for a given query and does so in a hardware aware way. Moreover, the paper introduces a further KV cache offloading steps which reduces space complexity further.

### Strengths
* The method being training-free means it can be used as a drop-in to already trained models.
* The paper does careful complexity analysis of its claims but strikes a balance on introducing information in a way to aid presentation (thinking of the informal theorem) while still being rigorous later. 
* It is extremely valuable to have code examples and implementation released.
* The paper is extremely comprehensive when understanding its metrics across different hardware and comparing with the many version of flash attention.

### Weaknesses
* It would be great to have the long context benchmarks also for different models -- gemma and mistral are both open source and around similar sizes.

### Questions
* Gemma 2 uses this mix between sliding window and normal attention. it would be great to understand if there are any degradation on non-llama architectures.
* The method does show that MMLU is not degraded when using this method. It would be interesting to see this for a broader set of metrics if possible.

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper proposes Hierarchically Pruned Attention (HiP) to reduce the time complexity of attention to O(T logT) and space complexity to O(T) where T is the sequence length. By exploiting the continuity of token sequence (tokens close together tend to have similar scores), for each query, the HiP use a tree-search like algorithm to approximately search the top k key tokens that yield large attention weights. Further, the author developed a KV cache offloading scheme to offload KV cache to host memory and reduce the GPU memory usage.

### Strengths
1. The proposed method uses iterative refinement to dynamically and approximately locate top k tokens, which is interesting. 
2. HiP shows promising efficiency improvement with only small performance drop. 
3. The appendix provides a lot of ablation study to study the behavior of the proposed method. 
4. The method is training free.

### Weaknesses
see question section

### Questions
1. It would be easier to understand to have a figure illustration showing the tree search (or improve figure 2, the figure 2 step 1 is a bit confusing) for section 3.1. 
2. The algorithm divides the sequence into k segments, and one token will be selected in each segment. Is it correct? If so, then the top k tokens must be distributed in the sequence, and cannot be concentrated on certain regions. Why do you make this design choice? 
3. I am aware of some literatures that also use iterative refinement to dynamically calculate attention for efficiency. Have the authors tried to compare to these literatures?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3