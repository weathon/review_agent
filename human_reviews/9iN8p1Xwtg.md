# Discovering the Gems in Early Layers: Accelerating Long-Context LLMs with 1000x Input Token Reduction

- Decision: Reject
- Scores: 6, 5, 5, 5

## Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities in handling long context inputs, but this comes at the cost of increased computational resources and latency. Our research introduces a novel approach for the long context bottleneck to accelerate LLM inference and reduce GPU memory consumption. Our research demonstrates that LLMs can identify relevant tokens in the early layers before generating answers to a query. Leveraging this insight, we propose an algorithm that uses early layers of an LLM as filters to select and compress input tokens, significantly reducing the context length for subsequent processing.
Our method, GemFilter, demonstrates substantial improvements in both speed and memory efficiency compared to existing techniques, such as standard attention and SnapKV/H2O. Notably, it achieves a 2.4$\times$ speedup and 30\% reduction in GPU memory usage compared to SOTA methods. Evaluation on the Needle in a Haystack task shows that GemFilter significantly outperforms standard attention, SnapKV and demonstrates comparable performance on the LongBench challenge.
GemFilter is simple, training-free, and broadly applicable across different LLMs. Crucially, it provides interpretability by allowing humans to inspect the selected input sequence. These findings not only offer practical benefits for LLM deployment, but also enhance our understanding of LLM internal mechanisms, paving the way for further optimizations in LLM design and inference.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Stemming from the observation that only selecting 100 tokens in the pre-filling phase is enough to preserve the performance achieved by kv cache eviction methods, this paper proposes GemFilter that identifies relevant tokens with early layers and pass fewer tokens to the generation phase.

### Strengths
(1) While most KV Cache papers focus on the generation phase, this paper demonstrates the possibility of token eviction in the prefilling phase, which to my knowledge is interesting. Also, I am not actively working on KV cache eviction and might miss relevant references. 

(2) The explanation of Algorithm 1 in section 3.2 is very clear. 

(3) The flexible choice of filter layer is quite useful in practice, avoiding extensive hyperparameter tuning. I suspect this is highly related to the pre-LN used in most LLMs. I am wondering if it is possible to report a similar study of the filter layer for Gemma 2, as it uses a different layer norm.

### Weaknesses
(1) What is the intuition behind selecting tokens with the most attention specifically from the last query? I guess this will likely focus on the initial tokens and the last tokens that are close to the query. I would like to see more ablation studies here. Instead of the last query, can we choose other queries?  Baselines here can be randomly selecting rows of the attention matrix, and selecting rows with the largest l2 norm.

(2) A certain level of performance loss is observed for Phi 3.5 Mini. While for larger models like LLaMA 8B and Mixtral 12B, we somehow observe improved average scores. Perhaps, full tokens are still needed for small models.

(3) I am not sure how important to reduce the memory requirement for the pre-filling phase, as this phase is more computing-intensive than memory-intensive. And the computation here can be done in parallel. In contrast, memory is a big issue in the decoding phase.

(4) A very important baseline is missing: MInference (https://github.com/microsoft/MInference), which also aims to reduce pre-filling memory.

### Questions
1. The flexible choice of filter layer is quite useful in practice, avoiding extensive hyperparameter tuning. I suspect this is highly related to the pre-LN used in most LLMs. I am wondering if it is possible to report a similar study of the filter layer for Gemma 2, as it uses a different layer norm.

2. What is the intuition behind selecting tokens with the most attention specifically from the last query? I guess this will likely focus on the initial tokens and the last tokens that are close to the query. I would like to see more ablation studies here. Instead of the last query, can we choose other queries?  Baselines here can be randomly selecting rows of the attention matrix, and selecting rows with the largest l2 norm.

3. A certain level of performance loss is observed for Phi 3.5 Mini. While for larger models like LLaMA 8B and Mixtral 12B, we somehow observe improved average scores. Perhaps, full tokens are still needed for small models.

4. I am not sure how important to reduce the memory requirement for the pre-filling phase, as this phase is more computing-intensive than memory-intensive. And the computation here can be done in parallel. In contrast, memory is a big issue for the decoding phase.

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
This paper introduces GemFilter for accelerating large language models (LLMs) when processing long-context inputs. The key insight is that LLMs can identify important tokens in their early layers before generating answers, which allows for significant input compression. The proposed GemFilter algorithm uses early layers of an LLM as filters to select and compress input tokens, reducing the context length for subsequent processing. The method achieves a 2.4× speedup and 30% reduction in GPU memory usage compared to state-of-the-art methods like SnapKV/H2O. Results show that  GemFilter outperforms standard attention and SnapKV in the Needle in a Haystack test, while maintaining comparable performance on the LongBench benchmark.

### Strengths
1) This paper focuses on a timely and important problem: Accelerating Long-context LLM inference.

2) The method introduced in GemFilter is simple-yet-effective, according to the experimental results provided in the paper (better accuracy than the dense baseline on the NIAH task, comparable performance on LongBench evaluation).

3) GemFilter achieves a 2.4× speedup and 30% reduction in GPU memory usage compared to state-of-the-art methods like SnapKV/H2O.

### Weaknesses
1) GemFilter only uses the attention score of the last token during the context/prefilling stage to determine the important tokens to keep. And then uses the selected tokens to run the full inference process. The process may lose some information in the context. For instance, for multi-hop Q&A tasks/multi-needle in a haystack evaluation, there will be questions focusing on different parts of the given long context. I am wondering if GemFilter will still be able to keep the model's performance on such tasks.

2) The selection of the filter layer seems to be results-oriented. And there seems to be no specific method to efficiently identify the filter layer. In Table 2, it seems that there is prominent performance loss on some benchmarks when choosing layer-13 as the filter layer (e.g., NrtvQA, Qasper, MF-en). The robustness of the method still needs to be further demonstrated.

3) For Mistral Nemor and Phi 3.5, the filter layer appears in the middle (or even later) of the model, limiting the efficiency gain that could be achieved with this method.

### Questions
1) The LongBench evaluation scores for the dense attention baseline (in Table 1) seems to be lower than expectation. For example, the llama-3-8B model only achieves a score of 13.04 on qasper with full KV. 

2) In Figure 4, GemFilter achieves a much higher score than the dense baseline on the NIAH test. I would appreciate it if the authors could provide some analysis for the improvements above the dense attention.

3) How is the running time of decoding stage (gen time) computed in Figure 3/6? What is the specific setting for the evaluation (i.e., generation length)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper looks into KV cache compression for efficient LLM inference under long context. The key insight is that LLMs can identify important tokens in early layers before generating answers. As such, the paper proposes to use early layers to select important tokens, effectively reducing the context length for subsequent generation. Evaluation shows that the proposed method outperforms SnapKV and H2O with additional memory reduction and speedups.

### Strengths
- Interesting observation that the last row of attention matrices in early layers may serve as a signal for locating important tokens. 
- Promising results with good accuracy and memory savings.

### Weaknesses
- Limited technical novelty. Prior work has identified that top layers of LLMs are not very effective, https://arxiv.org/abs/2403.17887, e.g, by pruning up to half of the layers, the model remain accurate. This seems to be more of an issue of the public LLMs that are not well-trained to leverage those deeper layers. From that perspective, the proposed method is more of a hack that tackles these issues, which may disappear as the pre-training recipe evolves. 

- It may not be easy to robustly choose the layer r for different tasks and models. While using early layers to identify a fixed set of tokens, the method may lose adaptiveness in KV cache compression. As a result, it may suffer from poor generalizability on different tasks, e.g., those that are not tested in the paper. Therefore, for this particular work, more datasets should be included for evaluation.

### Questions
In practice, LLMs are generalists, and therefore need to handle different tasks and requests dynamically. How do you make sure that the proposed method can robustly find a good early-layer signal to get high accuracy for a mixture of tasks and requests.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces GemFilter, which leverages early layers of large language models (LLMs) to compress input tokens and accelerate inference. GemFilter identifies relevant tokens in the early layers before generating answers to a query, significantly reducing the context length for subsequent processing. The proposed method achieves a 2.4 $\times$ speedup and 30% reduction in GPU memory usage compared to SOTA methods.

### Strengths
This paper is targeting a critical issue in large language models (LLMs) by proposing a novel approach to accelerate inference and reduce GPU memory consumption. The proposed method, GemFilter, demonstrates substantial improvements in both speed and memory efficiency compared to existing techniques, such as standard attention and SnapKV/H2O. The paper is well-written and easy to follow, and the experimental results are convincing. The evaluation on the Needle in a Haystack task shows that GemFilter significantly outperforms standard attention, SnapKV and demonstrates comparable performance on the LongBench challenge. The proposed method is simple, training-free, and broadly applicable across different LLMs. The paper also provides interpretability by allowing humans to inspect the selected input sequence.

### Weaknesses
The method presented in the paper lies between prompt compression and KV cache compression. Its token selection approach aligns more closely with KV cache compression techniques like SnapKV and H2O, while its re-computation of selected tokens resembles prompt compression methods such as LLMLingua [1] and LongLLMLingua [2]. Although the authors compare their method with SnapKV and H2O, including a comparison with prompt compression methods, particularly LongLLMLingua, would enhance the analysis.

The performance improvement of GemFilter may stem from two factors: (1) the selection of important tokens, and (2) the re-computation of these tokens, which might mitigate issues like "lost-in-the-middle." An ablation study to isolate the contribution of each factor would be beneficial.

[1] Jiang, Huiqiang, et al. "Llmlingua: Compressing prompts for accelerated inference of large language models." arXiv preprint arXiv:2310.05736 (2023).

[2] Jiang, Huiqiang, et al. "Longllmlingua: Accelerating and enhancing llms in long context scenarios via prompt compression." arXiv preprint arXiv:2310.06839 (2023).

### Questions
Please refer to the weaknesses section for questions. I am willing to adjust the score if the concerns are addressed.

### Soundness
3

### Presentation
3

### Contribution
3
