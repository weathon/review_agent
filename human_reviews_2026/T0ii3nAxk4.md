# SentKVCompress: Sentence-Level Dynamic KVCache Compression for Efficient Long-Context LLM Inference

- Decision: Reject
- Scores: 4, 2, 2, 2

## Abstract
The demand for million-token-scale contexts in the agent era has expanded the KVCache to terabyte levels, creating severe inference bottlenecks due to high storage overhead and frequent memory access. Existing KV compression methods struggle to alleviate this memory pressure efficiently, facing challenges with the accuracy or efficiency of KVCache storage and usage. Challenge 1–At the KV storage level, KV preprocessing mechanisms face accuracy and efficiency challenges: those with information loss suffer from a significant end-to-end accuracy drop of over 30%, while near-lossless methods incur substantial overhead, resulting in a superlinear increase in inference time with context length. Challenge 2–At the usage level, KV selection mechanisms face a similar dilemma. Static Selection (e.g., Attention Sink) fails to capture semantic relationships, yielding low recall (<50% for top-10 tokens). Dynamic selection (e.g., online score calculation) incurs prohibitive overhead, consuming over 60% of KV selection GPU time and 70% of CPU-GPU memory bandwidth with redundant transfers.

Our core insight is that the above challenges arise because existing methods follow an unstructured, token-level compression paradigm. This focus on discrete tokens, which inherently lack semantic structure, forces the model to expend substantial additional computation to implicitly re-extract structural information from long texts during inference. To address this, we observe that attention scores aggregate naturally at the sentence level. Leveraging this finding, we propose SentKVCompress, a novel sentence-level dynamic KV cache management framework that explicitly extracts and utilizes this inherent structural information. At the KVCache storage level, to address the challenge of accuracy loss and high preprocessing overhead, we propose a Sentence-level perceived KVCache preprocess framework, maintaining accuracy while cutting overhead to below 20%. At the KVCache usage level, to address the challenge of imprecise selection and high additional overhead, we propose a sentence semantic-driven KVCache selection strategy, enabling 70% of KVs to be reused. Experiments show that SentKVCompress achieves a maximum speedup of 4.2× with nearly no accuracy loss, and reduces the peak memory by 2.7× in long context scenarios, while also achieving the highest accuracy at equivalent KV usage rates. The code will be open-source in https://github.com/Indexleaf475/ICLR26-SentKVCompress

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SentKVCompress, a sentence-level approximate attention that is integrated with KV cache offloading. SentKVCompress mainly introduces a sentence segmentation algorithm, followed by sentence KV cache compression technique, and a KV cache reuse (caching) technique when offloading is applied. SentKVCompress achieves notable speedups, reaching up to 4.2x acceleration with 2.7x peak memory reduction, without significant performance degredation. This paper benchmarks SentKVCompress on LongBench, compared with classical baselines, such as H2O, StreamingLLM, etc., where SentKVCompress surpasses all the baselines.

### Strengths
1. Segmenting the input context at the sentence level is both efficient and effective, as selecting different sentences leads to better task performance. The overall design of the paper is reasonable.

2. The benchmark results show a significant improvement in inference efficiency without performance degradation. SentKVCompress demonstrates strong effectiveness.

### Weaknesses
1. Limited novelty. KV cache offloading and eviction have been widely studied. There are existing methods that perform KV selection at the sentence level, such as SentenceKV [1]. The paper should clarify the major difference between this work and SentenceKV.

2. The benchmark datasets used are rather limited. The test entries in LongBench are relatively short and easy; therefore, more challenging benchmarks such as RULER [2] and LongBench-v2 [3] should be included for evaluation.

3. More baselines should be discussed or included, such as SnapKV [4] and ShadowKV [5], to better position the proposed method within existing literature.

4. The overall writing should be improved. The current description of the method is vague and confusing. For example, in line 315, the term “KV reuse” appears before being clearly defined. All technical terms should be clearly introduced before use.

---

[1] Zhu, Yuxuan, et al. "SentenceKV: Efficient LLM Inference via Sentence-Level Semantic KV Caching." arXiv preprint arXiv:2504.00970 (2025).

[2] Hsieh, Cheng-Ping, et al. "RULER: What's the Real Context Size of Your Long-Context Language Models?." arXiv preprint arXiv:2404.06654 (2024).

[3] Bai, Yushi, et al. "Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks." arXiv preprint arXiv:2412.15204 (2024).

[4] Li, Yuhong, et al. "Snapkv: Llm knows what you are looking for before generation." Advances in Neural Information Processing Systems 37 (2024): 22947-22970.

[5] Sun, Hanshi, et al. "Shadowkv: Kv cache in shadows for high-throughput long-context llm inference." arXiv preprint arXiv:2410.21465 (2024).

### Questions
See above.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes SentKVCompress, a sentence-level framework for KV cache preprocessing and usage in long-context LLM inference. It first splits text into sentence-like chunks using a scoring rule that balances punctuation strength and a target length within a window, giving an O(n) splitting pipeline. This aims to keep semantic units intact while keeping lengths regular. 

The method then computes sentence scores from attention-derived signals and maps them to token scores so that KV selection happens at the sentence level. To support reuse across steps and heads, it introduces a variable-length block processing strategy that aligns reusable KV across attention heads.

### Strengths
1. Sentence-aware splitting balances punctuation and target length, which reduces fragmentation and redundant KV while staying linear time. This is simple and practical. 
2. Sentence-level selection and reuse reduce on-the-fly scoring and bandwidth pressure; the variable-length block alignment across heads is a useful engineering design.  
3. The framework is modular and could be combined with other storage-side methods.

### Weaknesses
1. Missing very-long-context benchmarks. There is no RULER-style evaluation at 32k–64k and 128k where retrieval and reuse matter most. This is essential to validate scalability and accuracy retention at realistic agentic workloads.  
2. Missing sentence-level baselines. A direct comparison with sentence-level offloading or caching methods such as SentenceKV[1] would test whether the gains come from sentence granularity itself or other design choices.  
3. Limited analysis of the sentence-score construction. The paper maintains per-dimension max–min statistics over keys to normalize and score, but it does not test alternatives such as mean-pooled keys or other low-cost sketches; the choice seems heuristic.
4. Robustness to punctuation-sparse inputs is unclear. For math, code, or log-like text without strong punctuation, it is not clear how the splitter behaves and whether accuracy degrades. 

[1] SentenceKV: Efficient LLM Inference via Sentence-Level Semantic KV Caching.    https://openreview.net/pdf?id=HyPeYU9JR6

### Questions
1. Why use per-dimension extreme values for key statistics rather than mean or robust mean as the sentence representation or normalization basis. Have you tried mean-pooled key vectors per sentence or per bucket?
2. What is the average and variance of the chunk length after your split on your main datasets, and how sensitive are results to the target length L and window.
3. How do you split texts for math or code tasks where punctuation is sparse or non-standard.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces SentKVCompress, a sentence-level KV cache compression framework for efficient long-context LLM inference. The key idea is to leverage sentence-level semantic structure rather than token-level processing. The framework includes: 1. a sentence-level preprocessing stage that splits tokens based on punctuation/length heuristics and maintains per-sentence compressed statistics, and 2. a sentence semantic-guided KV selection scheme with variable length block processing. Experiments on LongBench report improved average performance and efficiency compared to baseline approaches.

### Strengths
1. Problem motivation is clear and motivated to try and capture semantic information to enhance KV cache compression methods. Length-control for semantic boundary splitting offer some noticeable improvements as seen from ablation studies.

2. Ablation studies provide evidence that both length control and punctuation cues matters.

3. Strong performance on QA tasks from LongBench, while being more efficient compared to KV loading baseline approaches tested.

### Weaknesses
1. The main idea of punctuation, sentence-level chunking is already explored in related work (e.g. SentenceKV [1]). The main difference here appears to be a length-control term in the splitter's scoring, which feels incremental in absence of a direct comparison. The paper should explicitly position against these works.

2. Presentation and clarity of the methodology needs to be improved. Some important details are left omitted from Section 4, leaving the reader to fill in the gaps themselves. For instance, Sec. 4.1.2 defines per-dim min/max key vectors but does not specify how these are actually use in the selection computation.

3. Justification for using per-head iteration versus aggregate scoring should be made clear. For re-use, Sec. 4.2.2 iterates per-head and truncates to align variable length blocks, which breaks the parallelism. Some explanation for why per-head choices is theoretically or empirically better is needed to demonstrate the trade-off.

4. The heuristic sentence segmentation may be brittle and it is unclear how the "weights" for the semantic boundaries are instantiated.

5. The experiment task diversity is quite limited as SentKVCompress is only evaluated on a subset of 5 LongBench experiments skewed towards QA, and requires more evidence from additional benchmarks to support its claims. Sentence-level KV approach baselines are also missing.

[1] Zhu, Yuxuan, et al. "SentenceKV: Efficient LLM Inference via Sentence-Level Semantic KV Caching." arXiv preprint arXiv:2504.00970 (2025).

### Questions
1. Provide more complete results on a more comprehensive subset of LongBench.
2. Other experiments such as RULER should be run to better support claims.
3. Additional baselines for related sentence-level segmentation approaches should be added.
4. How does the selection process interact with Flash Attention, is it still compatible?
5. Choice and taxonomy of baselines is imprecise. Why is PQCache part of "loading", as it is mainly a KV compression and quantization work? KV quantization (KIVI) does not seem to be related to SentKVCompress, so inclusion in the results does not add much value.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes SentKVCompress, a sentence-level KV cache compression framework for efficient long-context LLM inference. The authors argue that existing token-level compression methods suffer from either accuracy loss or high computational overhead, and propose to leverage sentence-level semantic structure for both KV storage and usage. The method includes a sentence-aware preprocessing framework and a sentence semantic-driven selection strategy.

### Strengths
1. The idea of performing context selection and compression at the sentence level is intuitive and well-motivated. 
2. The paper presents both performance and efficiency in long-context processing.

### Weaknesses
1. The experimental results only present a very limited subset of tasks from LongBench, making it difficult to assess the method's generalizability.
2. The paper only presents results on LongBench-V1, where the average context length is around 8K tokens. The authors should evaluate on more challenging benchmarks with longer contexts, such as LongBench-V2, RULER.
3. The paper lacks comparisons with several important sparse attention methods, particularly block-sparse attention approaches that are conceptually related to the proposed sentence-level method. Notable missing baselines include MInference, InfLLM, XAttention, and other recent block-based or chunk-based attention mechanisms.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2
