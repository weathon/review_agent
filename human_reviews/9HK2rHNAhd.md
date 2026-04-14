# SqueezeAttention: 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget

- Decision: Accept (Poster)
- Scores: 8, 5, 3, 6

## Abstract
Optimizing the Key-Value (KV) cache of the Large Language Model (LLM) has been considered critical to saving the cost of inference. Most of the existing KV-cache compression algorithms attempted to sparsify the sequence of tokens by taking advantage of the different importance of tokens. However, most of these methods treat all layers equally, allocating the same KV budget to each layer. This approach is suboptimal, as some layers may be less sensitive to input tokens yet still receive the same budget as others. In this work, we found that by identifying the importance of attention layers, we could optimize the KV-cache jointly from two dimensions, i.e., sequence-wise and layer-wise. Based on our observations regarding layer-wise importance in inference, we propose \sys to precisely optimize the allocation of KV-cache budget among layers on-the-fly and then incorporate three representative sequence-wise algorithms to compress the KV-cache for each layer with its very own budget. Specifically, we first measure each layer's importance by calculating the cosine similarity of the input prompt differences before and after the self-attention layers. Based on this similarity, we then categorize the layers into two groups and adjust their KV budgets accordingly. By optimizing the KV-cache from both sequence's and layer's dimensions, \sys achieves around 30\% to 70\% of the memory reductions and up to 2.2 $\times$ of throughput improvements in a wide range of LLMs and benchmarks. The code is available at https://github.com/hetailang/SqueezeAttention.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes SqueezeAttention, a novel 2D Key-Value (KV) cache management algorithm designed to optimize memory usage and processing efficiency during Large Language Model (LLM) inference. The motivation behind this work is that existing KV-cache compression strategies handle all attention layers equally, which is suboptimal. Instead, SqueezeAttention dynamically allocates the KV-cache budget based on each layer's importance, determined by the cosine similarity of embeddings before and after each self-attention layer. By combining sequence-wise and layer-wise cache optimization, SqueezeAttention provides substantial memory savings (30%-70%) and throughput improvements (up to 2.2×) across various LLM models, including Mistral-7B, Falcon-7B, and Llama2-70B. The experimental results show significant performance gains in memory efficiency and token generation speed.

### Strengths
Novel Layer-Wise Approach: This paper introduces a layer-wise approach to KV-cache optimization, differentiating it from existing sequence-based compression methods. This work fills a gap in current LLM efficiency research.
Significant Performance Improvement: The proposed method improves memory consumption and throughput by reallocating cache budgets based on layer importance.
Robust Experimental Validation: The authors test their approach on multiple models (ranging from 7B to 70B parameters) and datasets, demonstrating its generalizability and efficiency.
Compatibility with Other Methods: SqueezeAttention integrates smoothly with various sequence-wise compression techniques, enhancing its versatility.
Energy Efficiency: The memory and throughput improvements have practical implications, potentially reducing the environmental impact of LLM deployment.

### Weaknesses
Dependency on Sequence-Wise Algorithms: The effectiveness of SqueezeAttention relies on combining it with existing sequence-wise compression methods, which limits its standalone applicability.
Potential Task-Specific Tuning: Although the layer importance measurement is automated, there may be task-specific variations, suggesting possible limitations in generalizing to unseen tasks without fine-tuning.
Limited Analysis of Computational Overheads: Although the paper claims that SqueezeAttention adds a negligible overhead, more analysis on computation costs, particularly for real-time applications, would strengthen the results.
Fixed Group Clustering: The choice of clustering layers into three fixed groups may oversimplify the optimization for some models or tasks where layer importance does not align neatly with this structure.
Risk of Reduced Accuracy: The method risks performance degradation for certain parameter values by under-allocating cache to less "important" layers, which might be essential for specific tasks or models.
In conclusion, the paper presents a promising contribution to LLM inference optimization with its innovative, adaptive KV-cache management strategy. However, further exploration into standalone performance and task-specific tuning would enhance the robustness of SqueezeAttention.

### Questions
See the discussion of weaknesses and kindly address them.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper identifies the importance of different attention layers, and proposes a layer-wise strategy named as SqueezeAttention to allocate different KV cache size for each layer. However, the proposed method still have several significant issues.

### Strengths
The observations of comparing the inputs and outputs of attention modules are good.

### Weaknesses
1.	The proposed method is designed only for the prefilling stage and does not allow for dynamic adjustment of the KV cache size during the decoding stage. To improve applicability, it would be helpful if the authors discussed potential ways to extend the method to the decoding stage, or provided a rationale explaining why it may not be feasible in that context.
2.	The reduced KV cache size is controlled by the hyperparameter \( p \), with values in the range of 0.3-0.4 based on a single model and task. This approach lacks generality. To improve robustness, the authors could conduct experiments across multiple models and tasks to determine if this \( p \) value range holds more broadly. Alternatively, they could propose a method for automatically selecting \( p \) to adapt to different scenarios.
3.	The method uses a fixed number of clusters, specifically 3, which may limit its generalizability. To strengthen the justification for this choice, the authors could either provide a rationale for using 3 clusters or experiment with different numbers of clusters to determine the optimal setting across various scenarios.
4.	The experiments appear incomplete. While Figure 3 includes four baselines, such as the full KV cache, each experiment only presents one baseline alongside the proposed method for comparison. Including all baselines in each experiment would allow for a more comprehensive evaluation. If certain baselines were omitted, the authors should explain why.

### Questions
It is unclear why the authors use ROUGE-2 for CNN/Daily Mail and XSUM, but ROUGE-L for SAMSUM. ROUGE-L is generally considered a more accurate metric for summarization tasks and could be applied consistently across all datasets. The authors could either evaluate all datasets with ROUGE-L for consistency or provide a rationale for choosing different metrics for each dataset.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This work proposed a layer-wise KV cache compression method that reduce the overhead during decoding stage of LLM inference. The proposed squeezeattention use cosine similarity of embeddings before and after attention block to identify the redundancy of kv cache with respect to specific layer. Then more redundant layers will then be assigned with smaller kv cache budget. For each layer, squeezeattention based on previous methods to remove redundant kv pairs, such as H2O, SteamingLLM and Sliding windows.

### Strengths
- The proposed methods is evaluated with multiple LLMs on various downstream tasks, demonstrates non-trivial improvements against previous baselines.

- The manscript is clearly organized with several illustration figures and equations. It's easy to understand the main method of this work.

- Both perfomance comparison and end-to-end memory/thoughput comparison are reported.

### Weaknesses
- The main observation that the cosine similarity of embeddings changes across layers while the first and last layers tend to have more diverse embeddigns, is not very new. Several works have showed similar results[1-3].

- It would be helpful to consider more recent kv cache compression methods, like SnapKV, PyramidKV, KIVI, etc. As the layer-wise strategy seems can be used in either KV cache pruning/quantization/low-rank decomposition methods, etc.

- In Table 3, it's a little bit unfair to compare the thoughput only with the full cache, since the KV cache evicted method is not the contribution of this work while the part of the thoughput improvements is achieved by the kv eviction, rather than the layer-wise strategy.
 

[1] https://arxiv.org/abs/2312.17276

[2] https://proceedings.neurips.cc/paper_files/paper/2023/file/fde1a69a5b6e554b2f1f727197d2651d-Paper-Conference.pdf

[3] https://arxiv.org/pdf/2202.08625

### Questions
- Do G1,G2,G3 changes frequently across different samples? otherwise we can assign the layer-wise budget through a offline process.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes SqueezeAttention, a KV-Cache management algorithm that can be combined with KV-Cache eviction policies to further reduce memory footprint and improve throughput. SqueezeAttention allocates size budgets for the KV-Cache of different layers by utilizing statistics on the importance of the attention layers. Specifically, SqueezeAttention first computes the cosine similarity between the activations before and after each attention layer.  Based on this similarity, the layers are then categorized into two groups and their KV budgets adjusted accordingly. SqueezeAttention achieves around 30% to 70% memory reductions and up to 2.2 × of throughput improvements in a wide range of LLMs and benchmarks.

### Strengths
1. The method can augment other KV-Cache eviction policies, which will benefit the research community.
2. The algorithm is clearly presented and the method's effectiveness has strong experiment evidence.

### Weaknesses
1. There's little analysis of the reason for performance improvement as shown in Figure 3. Some hypothesis or statistics analyses could give readers a deeper understanding of the algorithm.
2. The memory usage of Figure 4 is not clearly explained. What tensors are counted in the PyTorch Profiler? Besides, why does LLama2-70B consume a similar amount of memory to Mistral-7B?

### Questions
1. What inference framework is used for the memory and throughput experiments? Is SqueezeAttention compatible with current inference memory optimization like vllm[1]?

[1] Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C.H., Gonzalez, J., Zhang, H. and Stoica, I., 2023, October. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles (pp. 611-626).

### Soundness
3

### Presentation
3

### Contribution
3
