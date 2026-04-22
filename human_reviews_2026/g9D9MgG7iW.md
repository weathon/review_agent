# Tools are under-documented: Simple Document Expansion Boosts Tool Retrieval

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Large Language Models (LLMs) have recently demonstrated strong capabilities in tool use, yet progress in tool retrieval remains hindered by incomplete and heterogeneous tool documentation. To address this challenge, we introduce Tool-REX, a new benchmark and framework that systematically enriches tool documentation with structured fields to enable more effective tool retrieval,  together with two dedicated models, Tool-Embed and Tool-Rank. We design a scalable document expansion pipeline that leverages both open- and closed-source LLMs to generate, validate, and refine enriched tool profiles at low cost, producing large-scale corpora with 50k instances for 
embedding-based retrievers and 200k for rerankers. On top of this data, we develop two models specifically tailored for tool retrieval: Tool-Embed, a dense retriever, and Tool-Rank, an LLM-based reranker. Extensive experiments on ToolRet and Tool-REX demonstrate that document expansion substantially improves retrieval performance, with Tool-Embed and Tool-Rank achieving new state-of-the-art results on both benchmarks. We further analyze the contribution of individual fields to retrieval effectiveness, as well as the broader impact of document expansion on both training and evaluation. Overall, our findings highlight both the promise and limitations of LLM-driven document expansion, positioning \textsc{Tool-REX}, along with the proposed Tool-Embed and Tool-Rank, as a foundation for future research in tool retrieval.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles a problem in tool retrieval, real tools often have incomplete descriptions, which hurts both retrievers and rerankers. The authors aggregate multiple sources of tool-use data and propose a four-stage, LLM-assisted document expansion pipeline that adds structured fields (e.g., function, when to use, limitations, tags). They then fine-tune a dense retriever and a reranker on these expanded docs.

### Strengths
1. A practical, scalable pipeline with sensible checks.
2. Consistent gains, plus field-level ablations that give actionable advice.
3. Clear look at when expansion helps (train vs. eval); rerankers seem to benefit most when expanded views are present.

### Weaknesses
1. Training on your own expanded corpus and evaluating on the expanded view makes strong gains somewhat expected. The non-expanded results help, but an additional external/unexpanded test would make the case stronger.
2. Only random negatives during training. Adding hard negative mining would be helpful.
3. Documenting how duplicates/near-duplicates were removed across train/val/test after expansion, and report any impact on scores will be more helpful.
4. For human validation, authors should consider a random sample across all stages, report agreement, and provide an error taxonomy.
5. End-to-end throughput and cost (e.g. tokens, $) for generation, judgment, and refinement would help gain more insights.

### Questions
1. What happens when you train with mined hard negatives?
2. How do you handle deduplication and near-duplicate filtering across splits after expansion?
3. Can you share the end-to-end expansion cost/throughput and per-stage success/failure rates?
4. Beyond the reported non-expanded benchmark, can you add another unexpanded corpus or a held-out unexpanded slice to test robustness to documentation style?
5. You state that rerankers benefit more under expanded views. Can you keep the retrieval top-K fixed and rerank twice, once with original docs and once with expanded docs, to isolate the pure evaluation-time effect of expansion on the reranker?
6. The ablation results and the final released field set don’t seem fully aligned. Could you reconstruct this section and reconcile the ablation findings with the fields you keep/drop, including the supporting numbers?

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
3

### Summary
This paper addresses the problem of under-documentation of tool APIs. Specifically, as the authors mentioned, current API documents often lack standardization and critical information. To solve this problem, the authors propose a pipeline for API documentation augmentation. They apply the proposed pipeline on ToolRet to construct a new dataset, ToolDE. And the authors train one embedding model and one reranking model on ToolDE. The evaluation results demonstrate that augmenting documentation is beneficial. And training with standardized documentation leads to better retreival performance.

### Strengths
1. The studied problem is interesting and valuable. Indeed, the quality of API documents often has high variance. Standardization of them is expected to be valuable.
2. The paper is well written and easy to follow.

### Weaknesses
1. The proposed pipeline for augmenting API documentation involves human annotation, which makes it hard to scale up. 
2. According to Table 1, I find that the improvement of augmenting documentation without training is limited, especially for Qwen3-Embedding series. There is even a performance drop after augmenting the documentation, which makes me doubt the solidity of the motivation of this work.
3. Following the second point, it would be more valuable for applications if direct augmentation without training could lead to performance improvement. In real-world applications, APIs are often evolving. It is costly to augment new API documentations and train a new embedding model each time. Yet, as I mentioned in the second weakness, I find the direct improvement of augmentation is limited, which significantly limits the contribution of this work.
4. To demonstrate the generalization of augmenting documentation, the authors should train embedding models other than Qwen series on ToolDE.

### Questions
Please see my comments above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles an underexplored yet critical problem in tool-augmented LLM systems — the poor quality and inconsistency of tool documentation, which limits accurate tool retrieval. The authors introduce TOOL-DE (Tool-Document Expansion), a benchmark and framework that systematically enriches tool documentation through LLM-based expansion. Their pipeline generates structured fields (e.g., function description, when to use, limitations, and tags) via multi-stage prompting, validation, and human checking, yielding large-scale, standardized tool profiles. On top of this, they build two dedicated models: Tool-Embed (a dense retriever) and Tool-Rank (a reranker), trained on 50k and 200k examples respectively. Experiments across the TOOL-DE and ToolRet benchmarks show significant improvements in retrieval quality, achieving new state-of-the-art results (e.g., NDCG@10 = 56.44, Recall@10 = 67.81). Analysis further confirms that document expansion improves both training and evaluation by reducing semantic gaps, enhancing discriminability, and stabilizing optimization.

### Strengths
1. The benchmark is comprehensive. TOOL-DE is built over 35 datasets with a carefully validated expansion process, combining open and closed models (Qwen3, LLaMA-3.1, GPT-4o) and human checks. 
2. Experiments show solid improvements. Both retriever and reranker consistently outperform strong baselines, demonstrating that simple, well-structured enrichment can yield significant improvements.
3. The paper is well-written and easy to follow.

### Weaknesses
1. The manuscript lacks a clear and comprehensive description of the dataset. While some details are provided in the appendix, it would significantly improve clarity and reproducibility to include a dedicated section in the main text describing the dataset composition (e.g., number and types of tools, instances per tool, data sources, and preprocessing steps).
2. The training and testing splits is insufficiently explained. The paper shows the proposed pipeline works well when train and test on the same set of tools. Without a detailed account of how the splits are defined, it is difficult to assess whether the proposed approach effectively generalizes to unseen tools. This consideration is particularly important, as in real-world scenarios the set of available tools is dynamic and evolves continuously.

A relevant paper that might be included in the related works: [Planning and Editing What You Retrieve for Enhanced Tool Learning](https://aclanthology.org/2024.findings-naacl.61/) (Huang et al., Findings 2024)

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce TOOL-DE, a new framework that systematically enriches tool documentation with structured fields to enable more effective tool retrieval. The framework expands the tool documents by using structured fields like function description, when-to-use, limitations, and trains a dedicated retriever and reranker on top of the expanded documents. The experimental results show the effectiveness of tool expansion as well as the trained retriever and reranker.

### Strengths
- The paper effectively addresses the limitations of current tool learning paradigm where existing tool documents are underspecified for effective tool retrieval by using the idea of tool expansion using LLM which shows strong emperical results with and without dedicated trained retrievers and rerankers.
- It is an end-to-end framework where they create a tool document dataset and train the retriever and reranker. 
- The paper includes extensive ablations studies on impact of each field of expanded document in tool retrieval and how this affects the tool retrieval similarity

### Weaknesses
- While the framework shows strong performance, the idea of revising or augmenting the tool documents has been explored by the previous works [1,2]. 
- To make the data generation more scalable, one might consider replacing human verification to using a strong LLM


[1] Huang et al, Planning and Editing What You Retrieve for Enhanced Tool Learning. NAACL 2024 \
[2] Chen et al, EASYTOOL: Enhancing LLM-based Agents with Concise Tool Instruction. ACL 2025

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3
