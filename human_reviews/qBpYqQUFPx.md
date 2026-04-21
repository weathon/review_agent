# RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval

- Avg Score: 5.50
- Decision: Reject
- Scores: 3, 3, 8, 8

## Abstract
Transformer-based Large Language Models (LLMs) have become increasingly important. However, due to the quadratic time complexity of attention computation, scaling LLMs to longer contexts incurs extremely slow inference speed and high GPU memory consumption for caching key-value (KV) vectors. This paper proposes RetrievalAttention, a training-free approach to both accelerate attention computation and reduce GPU memory consumption. By leveraging the dynamic sparsity of attention mechanism, RetrievalAttention proposes to build approximate nearest neighbour search (ANNS) indexes for KV vectors in CPU memory and retrieve the most relevant ones through vector search during generation. Unfortunately, we observe that the off-the-shelf ANNS indexes are often ineffective for such retrieval tasks due to the out-of-distribution (OOD) between query vectors and key vectors in the attention mechanism. RetrievalAttention addresses the OOD challenge by designing an attention-aware vector search algorithm that can adapt to the distribution of query vectors. Our evaluation demonstrates that RetrievalAttention achieves near full attention accuracy while only requiring access to 1–3% of the data. This leads to a significant reduction in the inference cost of long-context LLMs, with a much lower GPU memory footprint. In particular, RetrievalAttention only needs a single NVIDIA RTX4090 (24GB) to serve 128K tokens for LLMs with 8B parameters, which is capable of generating one token in 0.188 seconds.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper introduces a new KV-cache optimization method called RetrievalAttention, which accelerates attention computation and reduces GPU memory consumption. It leverages the dynamic sparsity of the attention mechanism. It uses an attention-aware vector search algorithm that can adapt to the distribution of query vectors, to build indexes for KV vectors in CPU memory and retrieve the most important ones through search during the token generation. The authors compare the performance of RetrievalAttention in long-context LLM inference against full attention and other state-of-the-art methods. The comparisons mainly focus on the accuracy and latency of the proposed approach.

### Strengths
1. The task this paper tries to solve is crucial. A longer context window for LLMs allows the model to (1) enhance the consistency and coherence flow of information across a larger span of text; (2) improve understanding of complex text; (3) handle multi-step reasoning.
2. The idea of CPU-GPU offloading for LLM KV-cache optimization is interesting. Although there are several works already contributing in similar directions, they mainly focus on different aspects such as channel-dimension reduction and block-based organization.
3. Identify the challenge of OOD in using the ANNS index for attention computation.

### Weaknesses
1. Some details in the method section are unclear and could benefit from further elaboration, particularly regarding the ANNS algorithm and CPU-GPU co-execution. Please refer to Question (1) for more specific inquiries.
2. Certain baselines are missing, such as QUEST[1].
3. I couldn't find the code implementation of the proposed method, which makes it difficult to assess its performance in practice.

### Questions
1. Will the nearest-keys set for prefilled queries (indexed during the prefill phase) be updated during token generation? If not, will the retrieval recall degrade as more new tokens are generated?
2. Even if the nearest-keys set is updated, in extreme cases like code generation, where there are very few prefilled tokens and many newly generated tokens, how can the method ensure that the prefilled tokens remain representative enough for the query vectors of the newly generated tokens to effectively use them as a bridge to find the nearest key vectors?
3. For CPU-GPU co-execution, it is unclear how to separate the two disjoint sets of KV cache vectors between CPU and GPU. From my understanding, it seems challenging to ensure that the significant keys selected by ANNS are not included in the set of fixed initial tokens or within the last sliding window.
4. Although does not support GQA, I believe it is still worthwhile to include QUEST[1] as one of the baselines for comparison with vanilla attention.

I am willing to raise my score if the authors can address the limitations mentioned above. 

References:
[1] https://github.com/mit-han-lab/Quest/tree/main

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The authors propose a method that offloads KV Cache to the CPU combined with an ANNS index for sparse attention computation, aim for addressing GPU memory burdens during inference and accelerating decoding speed.

### Strengths
1. The approach of using Approximate Vector Search is intuitive and appears reasonable in this offloading attention computation.
2. The authors propose a simple but effective ANNS index that addresses potential OOD issues in this scenario.
3. This method outperforms baseline methods in multiple benchmarks.

### Weaknesses
1. It seems that this work relies heavily on obtaining global attention weights during the prefilling process to build the ANNS index. This may be incompatible with the existing FlashAttention technique, which is essential for efficient long-sequence prefilling computation. FlashAttention reduces prefilling overhead by using computation fusion, which avoids explicitly computing global attention weights, yet provides final outputs directly—making it a commonly used approach in long-sequence inference. Empirically, this issue will significantly increase the computing time and memory usage in the prefilling phase. In evaluation, the authors only provided decoding latency and did not consider potential additional overhead caused by the incompatibility with FlashAttention. Since other baselines generally do not encounter this issue, the authors should include experimental results to address these concerns.
2. The article should adopt a more appropriate baseline. Current baselines, such as SnapKV and StreamingLLM, discard a substantial portion of the KV Cache compared to the fully stored KV Cache method used in this paper, predictably leading to lower accuracy. For this work, a more suitable baseline for comparison with the fully stored Cache and offloading method would be:
(1.) Quest[1]: Quest also fully stores the KV Cache and retrieves several important cache entries during decoding. Although the authors note that Quest’s implementation does not support GQA and thus was excluded, since SnapKV can be integrated with GQA, the authors should consider implementing Quest with GQA support or explain why such an implementation would be challenging. (2.) InfiniGen[2]: InfiniGen is a comparable approach that offloads the Cache to the CPU, retrieving only a subset of the Cache during decoding to achieve approximate decoding, resulting in GPU memory savings and accelerated performance. This would likely be the closest baseline to the method in this paper.
3. It appears that this paper does not account for the overhead of index construction, which also directly impacts overall inference time. The authors should provide results showing the time and memory required for index construction in different input lengths.

[1] Tang, Jiaming, et al. "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference." arXiv preprint arXiv:2406.10774 (2024).

[2] Lee, Wonbeom, et al. "{InfiniGen}: Efficient generative inference of large language models with dynamic {KV} cache management." 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24). 2024.

### Questions
Does sequence length during prefilling impact the effectiveness of the index? Intuitively, longer sequences would result in larger indexes, suggesting that variations in sequence length could affect index performance. Testing the index's effectiveness in needle in a Haystack by locating the 'needle' cache across different sequence lengths could help clarify this effect.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper targets the slow inference speed and high memory consumption issue in long context generation when conducting KV caching. They propose RetrievalAttention. The method is training free as it directly leverage the dynamic sparsity of attention mechanism to retrieve the most relevant KV vectors. One challenge discussed in the paper is the out-of-distribution (OOD) between query vectors and key vectors. To mitigate this problem, the authors have designed an attention-aware vector search algorithm. The paper evaluates the accuracy and efficiency of RetrievalAttention across various long-context benchmarks and both commodity and high-end GPUs.

### Strengths
The problem addressed in this paper is highly meaningful. RetrievalAttention enables the execution of 8-billion-parameter models on a single RTX 4090 GPU, demonstrating the significance and practical impact of this research.

The observation regarding the critical key vectors could inspire further research and is beneficial for advancing the entire field.

The experiments are comprehensive and effectively support the main points.

The paper is well-written and easy to follow.

### Weaknesses
My primary concern is that powerful CPUs (in terms of compute speed and memory) are necessary to ensure this method achieves good performance. Otherwise, it could become a bottleneck. For instance, most KV vectors will be offloaded to the CPU memory. The experiment was conducted using Intel i9-10900X CPU with 20 cores and 128GB of DRAM, which is quite powerful. Therefore, I think this method distributes the large cost(computation and memory) between the CPU and GPU to reduce the strain on the GPU. However, the total cost reduction is less significant than just comparing the GPU cost. Additionally, the communication between the CPU and GPU could also negatively impact the speed. The evaluation results may vary across different server architectures.

### Questions
As the recovery rate is still 71%, does this mean that some tokens are consistently important? Is there any idea about the overlap between the dynamic and static selected critical tokens? 

Does the vocabulary size influence efficiency?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces RetrievalAttention to improve inference speed and reduce memory usage for LLMs with long contexts by using Approximate Nearest Neighbor Search (ANNS) to retrieve only the most relevant KV vectors during token generation. By offloading most KV vectors to CPU memory and leveraging dynamic sparsity in attention, RetrievalAttention efficiently retrieves necessary vectors from a small subset, substantially lowering GPU memory requirements. It addresses the OOD problem between query and key vectors by designing an attention-aware search that adapts to query distribution, enabling retrieval with only 1–3% of data and minimal accuracy loss.

### Strengths
- The paper introduces a training-free approach to accelerate long-context LLM inference by leveraging approximate nearest neighbor search (ANNS) for KV vector retrieval.
- The paper presents a well-thought-out and rigorously tested approach, with substantial empirical evidence demonstrating the performance gains of RetrievalAttention.
- RetrievalAttention addresses a significant limitation in the scalability of LLMs, especially as context lengths continue to grow in practical applications. By enabling long-context inference on a single commodity GPU with minimal accuracy loss, RetrievalAttention makes long-context LLMs more accessible and cost-effective, thus broadening their usability.

### Weaknesses
- While the paper presents a unique approach to address the OOD challenge between query and key vectors, it lacks a comparison with alternative methods for handling OOD issues in high-dimensional vector retrieval.
- The evaluation is limited to relatively small models, so it’s unclear how well RetrievalAttention scales for significantly larger models (e.g., 30B, 70B, or more). Larger models would place greater demands on both memory and compute resources, potentially exposing limitations in RetrievalAttention's design, especially concerning the OOD handling mechanism and ANNS index efficiency at scale.

### Questions
- RetrievalAttention applies the same retrieval sparsity across all layers, but it’s known that different layers and heads in transformers contribute differently to attention computation. Could layer-wise or head-wise retrieval strategies improve performance? 
- Have other ANNS index structures, such as product quantization or hierarchical clustering, been considered?

### Soundness
3

### Presentation
3

### Contribution
3
