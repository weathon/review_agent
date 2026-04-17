# Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
Recent large vision-language models (LVLMs) demonstrate remarkable capabilities in processing extended multi-modal sequences, yet the resulting key-value (KV) cache expansion creates a critical memory bottleneck that fundamentally limits deployment scalability.
While existing KV cache compression methods focus on retaining high-importance KV pairs to minimize storage, they often overlook the modality-specific semantic redundancy patterns that emerge distinctively in multi-modal KV caches.
In this work, we first analyze how, beyond simple importance, the KV cache in LVLMs exhibits varying levels of redundancy across attention heads. 
We show that relying solely on importance can only cover a subset of the full KV cache information distribution, leading to potential loss of semantic coverage.
To address this, we propose MixKV, a novel method that mixes importance with diversity for optimized KV cache compression in LVLMs. MixKV adapts to head-wise semantic redundancy, selectively balancing diversity and importance when compressing KV pairs. Extensive experiments demonstrate that MixKV consistently enhances existing methods across multiple LVLMs. Under extreme compression (budget=64), MixKV improves baseline methods by an average of 5.1% across five multi-modal understanding benchmarks and achieves remarkable gains of 8.0% and 9.0% for SnapKV and AdaKV on GUI grounding tasks, all while maintaining comparable inference efficiency. Furthermore, MixKV extends seamlessly to LLMs with comparable performance gains.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces MixKV, a simple, plug-and-play KV-cache compression method for (multi-modal) LLMs that mixes standard importance scores with a diversity metric at the per-head level. It estimates head-level redundancy via the off-diagonal average similarity of normalized keys and uses this statistic to adapt the diversity–importance mix, complementing attention- and value-norm–based importance signals. The approach is lightweight—requiring no architectural changes—and, across multiple LVLMs/LLMs under tight KV budgets, consistently outperforms importance-only baselines. Experiments on both multimodal understanding and language modeling tasks further demonstrate its effectiveness.

### Strengths
The paper is well written and easy to follow. It convincingly diagnoses the limitations of importance-only KV compression and motivates a diversity component, positioning the work clearly against prior baselines. The evaluation is comprehensive—spanning multiple models and benchmarks with budget sweeps and ablations (e.g., choice of intrinsic signal, effect of mixing)—and it substantiates the central claims.

### Weaknesses
The paper notes that LVLMs exhibit much higher key similarity. However, the proposed approach appears general and seemingly applicable to LLMs as well. Which components, if any, are specifically designed for LVLMs, and what prevents a straightforward application to text-only LLMs?

### Questions
1. The study is thorough at the 7B dense scale, but scalability to larger backbones and MoE architectures remains unclear. Results on a larger dense LVLM (e.g., 34B–72B) and on an MoE model such as Qwen/Qwen3-VL-30B-A3B-Instruct would help substantiate generalization. 

2. In Section 4.3 and Table 4, the “online head weights” setup is under-specified. Define what “online” means (e.g., per-token, no lookahead) and detail the procedure for computing and applying these weights—i.e., which signals are used and at what step/layer they are measured. 

3. The authors only report results on image understanding benchmarks. How does the proposed method perform on video understanding tasks, such as Video-MME [1]? Including evaluations on video benchmarks would help demonstrate the method’s generality across temporal visual–language settings. 

4. Table 3 presents results on long-context understanding tasks, but the reported LLMs are not specifically optimized for extended context lengths. To make the comparison more meaningful, it would be valuable to include models that have been explicitly tuned for long-context inference, such as Qwen/Qwen3-4B-Instruct-2507. This would better reflect the effectiveness of the proposed method under realistic long-context settings. 

Reference: 

[1] Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv 2024.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces MixKV, a KV cache compression method for large vision-language models (LVLMs) that combines importance and diversity to optimize memory usage and maintain performance. By adapting to head-wise redundancy, MixKV improves compression efficiency and achieves significant performance gains on multiple benchmarks.

### Strengths
The key advantage of MixKV is its comprehensive benchmarking across various tasks and models. It demonstrates consistent performance improvements across multiple multi-modal and text understanding benchmarks, including DocVQA, TextVQA, and LongBench, as well as GUI Grounding tasks.

### Weaknesses
1. I believe the paper spends unnecessary length discussing **modality-specific redundancy differences** and **head-wise redundancy differences**, as these concepts have already been well-established in previous works, such as **MMinference** and **VisionZip**. For instance, the **head-wise redundancy** can be directly observed in **MMinference's Fig. 1**, making it redundant to claim this in such detail. Additionally, I find it questionable to use **text on Qwen2** and **image on Qwen2-VL** to illustrate **modality-specific redundancy differences**. Normally, one would expect to examine both text and image together on a **Qwen2-VL** model to properly demonstrate this kind of redundancy, as that's where multi-modal data is processed simultaneously.

2. MixKV essentially merges two well-established paradigms: **importance-based** selection (FastV,ZipVL) and **diversity-based** retention (VisionZip). This type of fusion has already been explored in various contexts, such as in **attention mechanisms** and **token selection** techniques. The novelty here lies mainly in **applying this fusion to KV cache compression**, rather than introducing a fundamentally new idea. 

3. The paper mainly tests on **images** or **multi-view data** and doesn't address the challenges of **long-context sequences**, like those found in **long video** tasks. Testing on a **20k context** in long video benchmarks would be essential to evaluate how well **MixKV** handles large, complex data with **long-range dependencies** and more diverse redundancy patterns, which might stress-test its memory compression capabilities.

### Questions
See weaknesses.

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
4

### Summary
This work introduces a novel KV cache compression method tailored for VLMs. The paper begins by presenting two core observations, supported by visualizations: (1) KV pairs in LVLMs exhibit substantially higher semantic redundancy than in text-only LLMs, and (2) KV pairs in VLMs show varying degrees of semantic redundancy across attention heads in the LLM. Motivated by these observations, the authors propose a new compression scheme named MixKV. The central idea is to perform head-wise KV cache compression by simultaneously considering both the semantic redundancy and the diversity of the tokens. The paper is evaluated across multiple distinct tasks and compared against several baselines, demonstrating consistent performance advantages under identical settings."

### Strengths
1.  The paper effectively identifies a key problem through insightful feature analysis and subsequently proposes a targeted solution. The motivation is well-grounded, the methodology is appropriate, and the results are solid and convincing.
2.  Although the proposed KV cache compression scheme is designed for VLMs, its effectiveness is also validated on text-only tasks. This demonstrates the method's strong generalizability and versatility.
3.  The experimental evaluation is thorough. The method demonstrates consistent performance gains across various tasks and when compared against multiple baseline methods.
4.  The problem addressed by this paper is of high practical significance, and the proposed solution has strong potential for real-world application in resource-constrained environments.

### Weaknesses
1.  No significant flaws were identified.
2.  However, the paper could be further strengthened by including an analysis of the persistent performance gap that remains when compared to the full KV cache.
3.  Additionally, a broader discussion that horizontally situates the proposed method among other KV cache (or GPU memory) saving techniques would be valuable. While direct experimental comparisons are not strictly necessary, a qualitative discussion of the trade-offs relative to other approaches would enhance the paper's context.

### Questions
1. What about more text tasks?
2. There is still a noticeable performance degradation when compared to the full KV cache. Can the authors include a discussion comparing the proposed method with other KV cache optimization techniques, such as MLA or similar approaches?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes MixKV, a novel KV cache compression method that adaptively balances importance and diversity at the attention head level. MixKV quantifies semantic redundancy per head and mixes importance with diversity to select KV pairs for retention. Extensive experiments across multiple visual-text tasks and compression baselines show that MixKV consistently improves performance under tight memory budgets.

### Strengths
1. The paper provides a thorough analysis of semantic redundancy in LVLMs, highlighting a previously underexplored limitation of importance-only KV compression. 
2. The adaptive mixing of importance and diversity per head is both intuitive and empirically validated.
3. MixKV can be integrated with existing compression methods without modifying the underlying compression operator, making it practical for real-world deployment.

### Weaknesses
1. The paper does not include a direct comparison with closely related methods such as [HeadKV](https://arxiv.org/abs/2410.19258),  which would strengthen the empirical evaluation.
2. The analysis of Cross-modality Redundancy Differences does not clearly illustrate the "cross-modality" aspect. From my understanding, it's more about *Two different modality*. 
3. While the paper discusses Head-wise Redundancy Differences, similar redundancy patterns may also exist in pure LLMs. The unique characteristics or observations specific to VLMs should be clarified and emphasized.
4. In Section 4.2, it would be valuable to further discuss the observed patterns—specifically, why MixKV significantly improves baselines on Information Aggregation tasks but not on Information Localization tasks. Providing concrete examples would enhance the discussion.
5. The definition of "total latency" in Figure 5 is unclear. Details such as prompt length, number of decoding steps, and how compression overhead is measured should be explicitly disclosed.

### Questions
None

### Soundness
2

### Presentation
3

### Contribution
2
