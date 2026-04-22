# Hierarchical Routers for Efficient Top-k Retrieval in Sparse Attention

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 6, 2

## Abstract
Attention mechanisms have achieved remarkable success in deep learning through parallel searching for the most relevant tokens in large-scale data. However, both the memory and computational costs of self-attention scale quadratically with sequence length, making it infeasible for long sequences. Recent sparse top-$k$ attention methods can achieve performance comparable to full attention with much lower memory and computational overhead. Nevertheless, they often rely on graph- or tree-based index structures, which are too slow for batches of token sequences to rebuild across layers or heads, or use partition-based techniques which lack precision. 
To address this issue, we propose a search algorithm for sparse attention: Hierarchical Router Algorithm, HiRouter, which can efficiently construct indexing structures and dynamically retrieve top-k tokens on a per-sequence basis, striking a better balance between speed and accuracy.
HiRouter employs a multi-level routing mechanism that hierarchically partitions tokens into discrete buckets along a learned tree structure with O(T) to the sequence length T. Notably, our dual entropy loss directly regularizes embeddings, using affinity for stronger sample–centroid alignment to improve top-$k$ recall and balanced buckets to ensure efficient GPU parallelism.
HiRouter outperforms FlashAttention in speed on long sequences while matching or surpassing the accuracy of full attention, offering a compelling solution for scalable and efficient attention mechanisms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a trainable hierarchical routing scheme for implementing top-k attention, observing performance improvements in Transformers without sacrificing model quality.

### Strengths
1. The paper tackles an important problem - making Transformers resource efficient while not sacrificing quality. Solutions in this space are undoubtedly influential and impactful in the AI space.
2. The paper proposes a novel mechanism for implementing top-k attention:
    * Each query token is routed throughout a hierarchical tree structure that allows for the computation of the top-k tokens. 
    * Routing is performed by maintaining new weight "centroid" matrices and multiplying the query with those matrices. This results in an empirical distribution over the children of a node in the tree. The distribution is smoothed via the softmax Gumbel trick and the two highest probability buckets are selected. In the end, all the candidate key tokens are selected in the leaves of the tree.
    * Sparse attention is then calculated based on these candidates.
    * To encourage separability of the centroid clusters as well as fair partitioning the authors propose to use an entropy-like regularizer.
Though the ideas in this mechanism are not new to this paper (more details below), their combination in this context definitely constitutes an important contribution.
3. The paper performs a wide variety of experiments to verify that the proposed method works as advertised. The experiments are done on empirical datasets as well as on small and larger Transformers. The results show that in general the proposed mechanism does achieve more efficient inference for large context lengths while not degrading the model's quality. 
4. Theoretical insights are also a positive for this paper. They may not actively show that the learning process distills helpful information about the token's structure, but they support the intuition behind defining entropy-based regularizers.
5. Potential for parallelizability and GPU utilization is also a very important plus for this method.

### Weaknesses
1. I think the paper's main method is not presented as well as it could be. Specifically, Section 3.2 can definitely benefit from a rewrite to make the hierarchical routing method more explicit and easy to parse through. Without Figure 2, for example, it would have been very difficult for me to understand what is actually happening. I think the correct way to present this technique is via an explicit algorithm that lists the inputs, expected outputs and hyperparameters. Then statements about the runtime of the algorithm (Appendix E) can be made in the context of such an algorithm.
2. I think the paper can situate itself better in the context of prior work. Many of the ideas it builds upon were introduced in previous works, which often provide more theoretical justification for why these methods work well with attention. I think such context is especially important for papers doing work in such a crowded research space.
    * Clustering-based algorithms for efficient attention and KV cache compression: See, for example, the work of [1]: they maintain representative clusters of the token embeddings in similar fashion as this paper. Also, the work of [2] is another similar example.
    * Theoretical analyses for top-k attention: The paper of [3] gives explicit theoretical explanations of why top-$k$ attention works. 
    * Hierarchical structures for attention: Works such as [4] (and others) have explored hierarchy-based ideas in the past. 
This is not a comment on the quality of the work itself, but mostly on the necessity to connect itself better with prior work.
3. In the experiments, some additional comparisons would be important to see (please correct me if this is something I'm missing)
    * How does the method compare to top-k attention (without hierarchical structures) on non-synthetic benchmarks? 
4. How does the method behave memory-wise? 
    * It is plausible that maintaining a large number of additional centroid matrices can increase the memory requirements a lot?
    * When the context length gets very large, how should the number of levels and the parameter $C$ increase? If the increase is very large, wouldn't this result in loss of efficiency? Such an investigation would be very helpful also as a guiding tool for the reader.

**References**
[1] Zandieh, A., Han, I., Mirrokni, V., & Karbasi, A. (2024). Subgen: Token generation in sublinear time and memory. arXiv preprint arXiv:2402.06082.
[2] Liu, Guangda, et al. "Clusterkv: Manipulating llm kv cache in semantic space for recallable compression." 2025 62nd ACM/IEEE Design Automation Conference (DAC). IEEE, 2025.
[3] Haris T. $ k $ NN Attention Demystified: A Theoretical Exploration for Scalable Transformers. arXiv preprint arXiv:2411.04013. 2024 Nov 6.
[4] Chalkidis, I., Dai, X., Fergadiotis, M., Malakasiotis, P., & Elliott, D. (2022). An exploration of hierarchical attention transformers for efficient long document classification. arXiv preprint arXiv:2210.05529.

### Questions
1. In Section 3.2, isn't the token ordering procedure described taking $\Theta(T)$ time per query token? Wouldn't it be done $O(T)$ times? Then wouldn't the total time complexity still be $O(T^2)$? This goes back to my previous point about Section 3.2 being a bit confusing.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces an algorithm for implementing a sparse attention module, called HiRouter. Instead of calculating the standard dense attention operation (which scales as $O(T^2)$, where $T$ is the sequence length), the algorithm sorts the queries and keys into buckets using a hierarchical search. For each query, only the closest buckets are looked at (using a beam search), leading to an approximately linear complexity in sequence length of the attention operation, $O(Tk)$. The bucketing is done in a hierarchical way; at each level, the query/key is compared to bucket centroids, and a Gumbel-Softmax operation is used to decide the routing. Furthermore, the authors add auxiliary "dual entropy" loss term to improve the bucketing, consisting of a loss forcing the per-token distributions to be "spiky" (i.e., low entropy), but at the same time balancing the bucket assignments by forcing a flat average distribution across all of the buckets. The authors verify their technique by training models with 410M parameters on 10B tokens, using both the standard attention, as well as prior sparse attention technique, showing how HiRouter can lead to improved accuracy compared to the other sparse techniques, while suffering a small degradation compared to the standard transformer architecture.

### Strengths
- The paper introduces a novel technique for training a sparse attention layer. Compared to prior work (e.g., similarly motivated Reformer), the model seems to lead to better performance.
- The method is well-motivated, and the hierarchical search combined with balancing losses provides a sensible approach. The router is differentiable, thus leading to more balanced assignments (see Figure 1).
- The authors provide customised Triton kernels implementing their method.
- The authors compare to a variety of different model architectures, and test for perplexity, common-sense reasoning, recall, as well as long-context tasks.

### Weaknesses
- The experiments are conducted at a relatively small scale (410M parameters/10B tokens); more empirical evidence at larger transformer scales would give more confidence that the method scales well.
- The 1B experiments (in the Appendix) do not seem to showcase a significant improvement in performance compared to 410M numbers; a comparison with other published results at that scale would be helpful.
- The latency measurements indicate that at smaller sequence lengths (4k-16k), dense FlashAttention can still be significantly faster.

### Questions
- Since the execution is slower at shorter sequence lengths, would there be a way of implementing a hybrid approach that would utilise the FlashAttention speed for shorter sequences?
- Are any long-sequence adaptation techniques applied to use the model for long sequences (while being trained at a fixed shorter length presumably)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper describes HIROUTER. This is a hierarchical routing mechanism for efficient top-k attention in Transformer models. The method basically constructs a multi-level tree to route tokens into discrete buckets. These are given by learned centroids and the assignment is via Gumbel-SM. The method uses a dual entropy loss for the embeddings: you will sharpen routing while maximizing bucket occupancy to ensure balanced load. Experiments are reasonable with some speed advantages at very long sequences.

### Strengths
1. Appendix D describes a custom Triton implementation to handle multi-level hierarchical routing efficiently. This will give parallelism through balanced bucket sizes. This is end-to-end differentiable. This can be a useful contribution to the community. 

2. The paper does cover a variety of tasks incl. commonsense reasoning, recall-intensive retrieval, and long-context. So the empirical validation of the method itself is reasonable. 

3. The plots in Figure 3 provide useful information on the role of various factors like sequence length, dimensionality effects, etc. This is fine.

### Weaknesses
There are three main sources of weaknesses in the paper which I will describe below. 

1. The problem space is crowded. Which means that despite the best efforts one or more baselines will be missed. By itself, this would not impact my assessment of the paper but in this case, several results that could potentially be good starting points are missed. For example, ZETA (ICLR this year) is a directly comparable architecture. Multiple other ICLR papers this year -- while not all focused specifically on top-k address the same general problem space like HeadKV-R2, SqueezeAttention and VL-Cache. Again it is fine to _not_ have these in the list of baselines but the positioning of a different top-k attention mechanism needs to position their design choices in this general context. This brings me to the next point. 

2. Hierarchical tree indexing for retrieval is standard when learning indexes and FAISS (and methods using FAISS). The dual entropy loss combines what may be called textbook clustering objectives. Combining these principles, common in clustering, hashing, and MoE routing, cannot really be novel technical contributions.

3. What was a bit confusing was that the paper is sprinkled with theory but this gives no insight. One proposition restates that a good clustering ensures good retrieval. Another derives that entropy gradients have attraction-repulsion form (is this not direct for any entropy-based loss?). Another restates the fact that uniform distributions maximize entropy. These "propositions" are observations or facts, and it is not obvious why they're there.

4. To me this paper comes across as a retrieval method tacked on to attention. This is fine. But for a method centered on top-k retrieval, there are no direct recall@k metrics reported. How often does the hierarchical routing successfully retrieve the true top-k tokens compared to exact search? But even in this case, it is difficult to put a finger on exactly which piece in the construction will yield non-trivial gains over alternatives that exist right now (not necessarily for top-k attention specifically)

### Questions
please see weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2
