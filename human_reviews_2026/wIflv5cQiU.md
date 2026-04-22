# FRAG: Filtering Noise Using Snippet-Level Query Relevance

- Avg Score: 4.40
- Decision: Reject
- Scores: 4, 4, 6, 2, 6

## Abstract
Retrieval-Augmented Generation (RAG) augments large language models (LLMs) with external retrievals. Typically, expanding the retrieval window can improve RAG performance by retrieving more relevant content. However, it risks increased noise, which distracts the model’s attention and degrades accuracy. To mitigate this, we propose Fine-Grained RAG (FRAG), which identifies key snippets from query and extracts relevant information while filtering noise from retrievals using snippet-level query relevance. Yet, a new challenge arises in addressing complex RAG queries, which require knowledge pieces with implicit multi-hop logical relationships. Failure to identify these relationships may lead to loss of inference-based knowledge during filtering, degrading performance. To address this, we propose Self-Recognition, which extracts inference-based knowledge by leveraging historically extracted knowledge as contextual references. While FRAG notably improves performance, it incurs high computational cost. To alleviate this, we present FRAG-ip, a fine-tuned framework which markedly accelerates FRAG by an order of magnitude. Extensive experiments show that FRAG significantly boosts RAG, yielding average accuracy gains of 4.94%/13.44% on simple/complex tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Fine-Grained RAG (FRAG), an extraction framework designed to solve the "noise" problem in Retrieval-Augmented Generation (RAG). Standard RAG performance often degrades when retrieving large amounts of information, as irrelevant content distracts the model. FRAG addresses this by operating at the snippet-level, identifying and extracting only the most relevant information from retrieved documents while filtering out noise. To handle complex, multi-hop queries, FRAG uses a component called "Self-Recognition." This module leverages historically extracted knowledge as context to preserve crucial inference-based information that might otherwise be filtered out. While FRAG significantly boosts accuracy, it introduces latency. To alleviate this, it proposes FRAG-ip, a fine-tuned framework that accelerates FRAG by approximately 10x.

### Strengths
This paper has the following strengths:

- It effectively establishes its theoretical grounding by demonstrating the negative impact of retrieval noise on RAG performance. This clearly motivates its core objective: filtering noise and extracting relevant information to enhance RAG accuracy.
- The core contribution is FRAG, a novel framework that introduces two key innovations: 1) snippet-level query relevance to effectively filter noise, and 2) Self-Recognition to preserve inference-based knowledge for complex, multi-hop tasks. It practically addresses the method's computational cost by proposing FRAG-ip, a fine-tuned framework that improves efficiency by approximately 10x.
- It provides extensive experimental results demonstrating that FRAG markedly boosts RAG performance, particularly on complex tasks (+13.44% gain). The results convincingly support its claims of achieving state-of-the-art performance.

### Weaknesses
This paper has the following weaknesses:

- The central performance claims in Table 2 are confounded by the use of different base LLMs across methods. As model capability is a significant variable, this makes it difficult to isolate the true performance gains attributable to the FRAG framework itself. A more direct comparison, where FRAG and all baselines utilize the same base LLM, would provide a much fairer and clearer assessment of its advantages.

- The paper reports strong accuracy improvements, but it lacks an analysis of statistical significance. Reporting the mean and variance over multiple experimental runs would be necessary to confirm that the observed gains are consistent and not an artifact of run-to-run variability.

- The ablation experiments, while insightful, are conducted on a single benchmark. To strengthen the paper's conclusions, it would be beneficial to replicate this study on a different benchmark (such as LongBench-v2). This would help confirm if the contributions of individual components, particularly Self-Recognition, generalize across different datasets and task structures.

- The paper introduces FRAG-ip to address the latency of FRAG, but it fails to provide a comparative cost analysis (e.g., runtime, inference latency) against the baseline methods. Without this data, it is difficult for a reader to assess the practical cost-benefit trade-off of FRAG's accuracy gains relative to standard RAG and other baselines.

- The discussion of related work appears to overlook recent and relevant studies on noise filtering and robustness in RAG (e.g., [1][2]). Situating FRAG in relation to these contemporary methods would provide a more complete picture of its novel contributions to the field.

[1] Wang, Fei, et al. "Astute rag: Overcoming imperfect retrieval augmentation and knowledge conflicts for large language models." arXiv preprint arXiv:2410.07176 (2024).

[2] Xiang, Chong, et al. "Certifiably robust rag against retrieval corruption." arXiv preprint arXiv:2405.15556 (2024).

### Questions
- Could the authors provide a comparison where FRAG and the key baselines are evaluated using the same base LLM to ensure a fair and direct comparison?

- To demonstrate the robustness of the results, could the authors provide the mean and standard deviation of the key performance metrics over multiple runs?

- To strengthen the claims about the framework's design, could the authors provide results from an ablation study on a different benchmark (such as LongBench-v2)? This would help confirm that the conclusions about each component's contribution generalize across different datasets.

- Could the authors provide data on the runtime and/or latency of both FRAG and FRAG-ip relative to the baselines?

### Soundness
2

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
3

### Summary
FRAG addresses noise in RAG by doing snippet-level extraction guided by key query fragments, plus a “Self-Recognition” step that reuses previously extracted facts for multi-hop retrieval. An efficiency variant (FRAG-ip) folds several steps into the extractor to cut latency. Empirically, FRAG improves over naïve RAG and several contemporary systems on single- and multi-hop benchmarks while reducing the generation context.

### Strengths
The manuscript is clearly written, with a comprehensive problem decomposition supported by careful analysis and experiments.

The proposed method delivers substantial performance gains.

FRAG-ip further offers a practical route to deployment.

### Weaknesses
- System Complexity

The six-stage pipeline and multiple LLM invocations make the overall system heavyweight. Its multi-stage, decision-driven design makes the full FRAG pipeline conceptually closer to an agentic-style system rather than a conventional RAG pipeline. The resulting complexity and latency poses significant challenges for real-world deployment.

Although the introduction of FRAG-ip and a pipelined execution design alleviates some of the delay, the serving setup still co-hosts at least an extractor and a generator, resulting in potential inflated VRAM usage and additional orchestration overhead. However, the discussion of FRAG-ip appears mainly in the appendix and remains incomplete—there is no evaluation of peak memory consumption, concurrent throughput, or tail-latency (P90/P99) under realistic serving conditions. The primary comparison of FRAG-ip is also limited to naïve RAG; a compute- and context-matched evaluation against other RAG noise-reduction baselines would better illustrate the system’s trade-offs.

- Limited Baseline Coverage

The main results table labels many baselines as “inapplicable” without sufficient justification, which undermines the empirical validity of the reported improvements. Providing clearer reasoning or partial reproductions would strengthen the credibility of the experimental claims.

- Reproducibility

No open-source code or implementation details are provided. Given the multi-module pipeline and the separate FRAG-ip training stage, reproducing the results would be difficult without access to the exact prompts, data-processing scripts, evaluation harness, retriever configurations, and corpus versions used in the experiments.

### Questions
Please refer Weaknesses. Moreover, in the FRAG-ip experiments, what TG values were used? Do the same TG-dependent trends observed in FRAG also appear in FRAG-ip?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper talks about FRAG, a method for making Retrieval-Augmented Generation (RAG) less noisy and more accurate. When you make the retrieval window bigger, you get more relevant stuff, but you also get more junk (noise), which really throws off large language models. FRAG tries to fix this by picking out key snippets from queries and filtering everything at a super fine-grained (snippet-level) scale. For harder queries that need multi-hop logic, the paper adds Self-Recognition, which uses what’s already been extracted to keep inference going. There’s also a faster FRAG-ip framework to make things run quicker. In experiments, FRAG improves average accuracy 5-13 points over naive RAG and does a solid job on a bunch of standard datasets

The theoretical motivation matches what many have found in RAG research: more retrieval isn't always better, since noise quickly drowns out the good stuff. The snippet-level filter is well-motivated, and the experiments provide solid evidence. there are clear comparisons against standard and reranked RAG baselines. The handling of multi-hop/inference-style queries via Self-Recognition is clever and nicely shown in ablation studies. The results and methodology seem robust, though the paper's math is a bit dense.

This work addresses a very active problem in RAG for LLMs, getting high relevance with low noise. The move to snippet-level sorting  seems like a natural next step, and integrating Self-Recognition for multi-hop queries is a clever touch. The FRAG-ip speedup shows some attention to practical deployment. These results are important for anyone working on grounded QA, multi-turn dialogue, or scientific/medical data retrieval.

### Strengths
Snippet-level relevance filtering is a meaningful advance over existing sentenct level approaches.Empirical results demonstrate strong gains in both standard and challenging retrieval nd QA tasks.Handles multi-hop and complex queries better than other approaches using the Self-Recognition.Provides clear ablation studies and baseline comparisons.Addresses both accuracy and speed in RAG settings (via FRAG-ip).

### Weaknesses
Despite FRAG-ip, snippet-extraction adds LLM calls that may still slow things down for very large, complex queries.  Paper is packed with jargon and math, which may hurt accessibility for wider audiences.Most benchmarks are synthetic or curated; I'd like to see more real-world open domain or industry use cases.Relies on strong instruction-following LLM, uncertain results with simpler models. I think the core concept (snippet-level filtering in RAG and multi-hop self-awareness) isn’t fully original. major ideas are also present in FineFilter, SCMRAG, SIM-RAG, and related literature. Authors need to cite these works, clarify their unique contributions, and openly discuss overlap. That said, the specific pipeline (Self-Recognition + FRAG-ip) and some implementation details add value beyond simple reproduction of those other concepts.

### Questions
Can FRAG be extended to noisy, open-source web data (not just synthetic/curated datasets)? Any ides on lighter-weight snippet relevance scoring to reduce inference cost? How resilient is FRAG to misleading/snippet-level adversarial noise? Have you tested FRAG in a conversational (not just QA) setting, e.g., for dialogue or summarization? i can see it is bit different but it would be good if you could explicitly explain whats actually new in your approach versus FineFilter and SCMRAG?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work addresses the fine-grained (snippet-level) associations between the query and the relevant documents in RAG (retrieval-augmented generation). Different from conventional RAG, the proposed FRAG framework extracts key sequential snippets S from the query, and then extracts relevant knowledge K from relevant documents according to S. This approach greatly condenses the context of RAG. In experiments, FRAG consistently improves the RAG performances on four LLMs (i.e., ChatQA-1.5-8B, Llama-3-8B-Instruct, Qwen-2-7B-Instruct, and ChatQA-1.5-70B) on 10 datasets.

### Strengths
* Experimental results show consistent improvements contributed by the proposed method on four LLMs on 10 datasets. 

* The method greatly reduces the length of context information during RAG.

### Weaknesses
* While the proposed FRAG method improves the RAG performances of LLMs in vanilla RAG, the comparison with other context-condensing approaches should be comprehensively analyzed in addition to the statements in B5. In this work, no empirical results with RankRAG are shown, and the results of RankGPT were based on a different LLM (old Llama-7B). Without the comparison with these real baseline models, the benefit of the proposed method is still unclear. 

* This method adds computational cost during the inference stage. The overhead could be measured and compared with other methods.

### Questions
* B.5 compares FRAG with reranking methods conceptionally. Can you provide empirical comparisons with those reranking methods?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In the Retrieval-Augmented Language Model (RALM), expanding the retrieval window can enhance performance by including more relevant knowledge. However, it also introduces noise, which may degrade RALM performance. To mitigate the problem, the authors propose Fine-Grained RAG (FRAG), which performs fine-grained extraction of relevant knowledge and filtering noise from initial retrievals. Furthermore, to alleviate the attention distraction caused by noise, the authors decompose the extraction into some steps. Moreover, to deal with omitted prerequisite knowledge caused by the decomposition, the authors propose Self-Recognition, which leverages
historically extracted knowledge as a reference in extraction to restore missing logical relationships. In addition to the above performance-specific proposals, the authors also propose FRAG-ip, a wrapper framework that employs dual-stage fine-tuning and accelerates FRAG. In the experiments, FRAG actually improves RALM performance on both simple and complex tasks.

### Strengths
- This work aims to solve the issues in the fundamental field of the current large language models (LLMs), the Retrieval-Augmented Language Model (RALM).
- This work theoretically demonstrates the negative impact of noise on RALM.
- The proposed method, FRAG, can deal with the noise issue in RALM.
- In addition to FRAG, the authors propose FRAG-ip, an efficient framework for running FRAG.
- The experiments cover various kinds of LLMs, and the settings with and without RAG baselines are also covered.
- As well as using single-hop QA, the conducted experiments employ multi-hop QA, which is considered a complex task.
- The experimental results show the promising performance gains of FARG in multiple tasks.
- The ablation study actually shows the importance of each component.

### Weaknesses
- The used extractor models are restricted to instruction-based models. The commonly used extractors, like BERT and its variants, and BM25-based methods are not covered.

### Questions
- Why did you choose Llama3-8B-Instruct and Qwen2-7B-Instruct for the extractor models in the experiments? Is there any reason for not using commonly used extractors such as BERT, BM25, etc.?

### Soundness
3

### Presentation
3

### Contribution
3
