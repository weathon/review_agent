# KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 3, 3

## Abstract
The development of large language models (LLMs) has significantly expanded model sizes, resulting in substantial GPU memory requirements during inference. The key and value storage of the attention map in the KV (key-value) cache accounts for more than 80\% of this memory consumption. Nowadays, most existing KV cache compression methods focus on intra-layer compression within a single Transformer layer but few works consider layer-wise compression. In this paper, we propose a plug-and-play method called \textit{KVSharer}, which shares the KV cache between layers to achieve layer-wise compression. Rather than intuitively sharing based on higher similarity, we discover a counterintuitive phenomenon: sharing dissimilar KV caches better preserves the model performance. Experiments show that \textit{KVSharer} can reduce KV cache computation by 30\%, thereby lowering memory consumption without significantly impacting model performance and it can also achieve at least 1.3 times generation acceleration. Additionally, we verify that \textit{KVSharer} is compatible with existing intra-layer KV cache compression methods, and combining both can further save memory.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a novel approach to sharing the key-value (KV) cache across different layers in a new dimension, which can lead to more efficient memory usage and improved performance.

### Strengths
This idea offers new insights into how memory size can be further reduced, potentially leading to more efficient model deployments and optimized hardware utilization.

### Weaknesses
1) The paper lacks a comparison with other cache-sharing methods, which would provide a clearer understanding of its advantages.

2) It should consider the scenario when the KV cache is quantized, as quantization is often used during inference to save energy.

3) The paper also lacks a scalability analysis, which is crucial for evaluating how well the proposed method performs as model size and complexity increase.

### Questions
What is the time scalability of the proposed approach? Will the inference time remain acceptable when scaling up to models with over 400 billion parameters? It would be valuable to provide an estimation or analysis to address this concern.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces KVSharer, a post-training method for layerwise KV cache sharing. Based on the counterintuitive observation that sharing KV caches between layers with dissimilar, rather than similar, KV caches leads to less performance degradation, KVSharer employs a systematic search strategy for KV sharing. As a result, KVSharer reduces GPU memory consumption while maintaining model performance.

### Strengths
* Does not require training
* Provides an interesting and novel insight that sharing dissimilar KV caches yields better performance.
* Offers diverse and insightful evaluation results.

### Weaknesses
* Results show a noticeable performance drop even at low compression rates (e.g., 12.5%, 25%), which may limit the practicality of the method.
* Lacks an explanation for why sharing dissimilar KV caches yields better performance, leaving an essential aspect of the method's effectiveness rather unclear.

### Questions
* Why is it better to share dissimilar KV caches? Since the authors themselves describe this as counterintuitive, providing an explanation for this phenomenon would be highly valuable for the community.

* What happens if KVSharer is unable to find $C$ pairs of layers to share KV caches while satisfying the threshold $T$? It would be helpful to include a guideline on setting this threshold and any evaluation showing its impact on search performance.

* In Table 2, why does the memory usage reduction exceed the compression rate? Additionally, what is the source of the observed increase in generation throughput? Since KV cache sharing reduces memory usage but likely not memory bandwidth, it is unclear how this improves inference throughput.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces KVSharer, a plug-and-play method for compressing the key-value (KV) cache of large language models (LLMs) during inference. Unlike the intuitive approach of sharing similar KV caches, KVSharer is based on a counterintuitive observation: sharing different KV caches across layers does not significantly degrade model performance. KVSharer employs a search strategy to identify the optimal KV cache sharing policy across different layers, substantially reducing GPU memory usage while retaining most of the model’s performance. Additionally, KVSharer is compatible with existing intra-layer KV cache compression methods, offering a complementary approach to memory optimization for LLMs.

### Strengths
1. This paper addresses a good research topic: efficient LLM inference.
    
2. The paper is well-organized.
    
3. The proposed method is clearly presented.

### Weaknesses
1. **Lack of novelty and research depth:** This main technique is to share the dissimilar KV cache for efficient inference, which is quite simple. Although authors claim that this originates from a counterintuitive observation, there is no motivation provided in the methodology section. Therefore, both of the novelty and the research depth of this paper are not qualified for the top AI conference.
    
2. **Unreasonable observation without further analysis:** The observation that the sharing the dissimilar KV cache brings in better accuracy than sharing the similar one sounds unreasonable, the dissimilar KV states output different attention scores, making the LLM attend to different part of the query token. It is more convincing that the obtained conclusion is just a coincidence and varies across the models and datasets, considering that no in-depth analysis has been provided.
    
3. Lack Needle-in-a-Haystack experiment.

### Questions
See Above.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper first presents a counterintuitive phenomenon when attempting to leverage the cross-layer pattern to improve the efficiency of the LLM generative inference computation, where sharing dissimilar KV caches better preserves the model performance. Based on this observation, this paper introduces a method named KVSharer, which integrates this observation to implement efficient cross-layer KV cache sharing. An empirical study has been conducted to verify the effectiveness of the proposed methods.

### Strengths
- S1. This paper explores an important problem of improving the efficiency in utilizing KV cache in LLM generative inference. 

- S2. The related work and research context are well summarized.

### Weaknesses
- W1. Heuristic-based on aggregated information. As enumerated in Section 3.1.2, the proposed method uses the averaged value of the KV-cache to consider the similarity between different layers -- it is a little confusing why such highly integrated information could guide the sharing policy, considering lots of recent work has been exploring the KV-cache utilization at token, layer, and head level jointly. My concern is whether such a highly aggregated metric is informative or not.

- W2. My main concern is about the experimental setup. There is a significant mismatch between the motivation example in the introduction, e.g., "During the LLM inference phase, the KV cache typically accounts for 80% of the total memory usage."  and the benchmarked settings, where the context window is set to just a few thousand, e.g., up to 1024+4096 in Table-2. Unless batching to an extremely large value (not mentioned in the paper), there is a significant gap between the motivation and the experiments. I think it would be critical to evaluate the performance of the proposed method over long-context benchmarks (e.g., Infiniti-bench) where the model's context window should be from 32K to 128K  (or even longer). Otherwise, the truly useful scenario is not evaluated.

### Questions
Please address the corresponding concern listed in the weakness section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces a new inter-layer KV cache compression technique through layer-wise KV cache dis-similarity search and sharing. The layers are ranked pairwise in accordance with their dis-similarity score. For each pair, an earlier layer's KV will be shared and reused by an later layer for efficient pre-filling and generation.

### Strengths
This paper and the technique introduced have the following strengths:

1. Paper writing is easy to follow with good figures and illustrations.
2. The experiment sections demonstrate KVSharer can be used in orthogonal with other intra-layer KV compression techniques like H2O and PyramidInfer to achieve higher memory saving and more significant speedup.
3. The paper brings up a new angle

### Weaknesses
I have several concerns about the paper:

1. Even though layer pairs are ranked from high dis-similarity to low dis-similarity, whether to use the pair still depends on the cosine similarity between the the KV-cache compression model and the original model. There is a possibility that the cosine similarity check, rather than dis-similarity ranking, plays a major role.

2. A major claim in the paper is dis-similarity metrics is better than similarity metrics when it comes to inter-layer KV cache sharing. Empirical evidences are provided in Section 5.1 and Figure 6 when changing the Euclidean-distance based ranking from descending order (dis-similarity) to ascending order (similarity). However, I didn't find any theoretical and empirical evidence that "Euclidean distance for KV cache is a sufficient good metrics" in comparison with the other SOTAs. More specifically, how does KVSharer compare with other layer-wise compression strategies, for example miniCache [1], LCA [2], CLLA [3] and simpleLayerKV [4]? Without the experiment results, I don't think the paper is ready at this stage for publication.

[1] Liu, Akide, et al. "MiniCache: KV Cache Compression in Depth Dimension for Large Language Models." arXiv preprint arXiv:2405.14366 (2024).

[2] Brandon, William, et al. "Reducing Transformer Key-Value Cache Size with Cross-Layer Attention." arXiv preprint arXiv:2405.12981 (2024).

[3] Yang, Zhen, et al. "Lossless KV Cache Compression to 2%." arXiv preprint arXiv:2410.15252 (2024).

[4] Zhang, Xuan, et al. "SimLayerKV: A Simple Framework for Layer-Level KV Cache Reduction." arXiv preprint arXiv:2410.13846 (2024).

### Questions
I think the paper will be much more ready if the authors could address the following questions (from high to low priority):

1. Could the authors provide comparisons with other layer-wise compression strategies in terms of accuracy and system performances?

2. Did the authors investigate the relationship between dis-similarity ranking and the acceptance rate by the thresholding condition? It's possible that the cosine similarity check, rather than dissimilarity ranking, plays a primary role. In principle, if the "higher dis-similarity --> inter-layer KV cache sharing gives better performance" hypothesis holds, then a higher rank should correspond to a higher acceptance rate. Could the authors provide additional results and justification on this point?

3. There is an important threshold in this work: cos-similarity (representation similarity) threshold that determines whether to accept a KV cache pair, Can the authors provide explanations on how the value is determined/searched? Moreover, the number of target shared KV cache layers is also an important hyper-parameter, and this is discussed in the paper an ablation study on in Table 1. But can the authors provide some guidance/calculation on how this number translate to memory saving and inference speedup?

4. For KV cache dissimilarity distance, why did the authors choose Euclidean distance? Could the authors ablate on other distance metrics? Similarly, for cosine similarity from the final layer hidden states, what if some other metrics like angular distance is used (less important, just wondering)?

### Soundness
1

### Presentation
3

### Contribution
2
