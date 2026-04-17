# SWERank: Software Issue Localization with Code Ranking

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Software issue localization, the task of identifying the precise code locations (files, classes, or functions) relevant to a natural language issue description (e.g., bug report, feature request), is a critical yet time-consuming aspect of software development. While recent LLM-based agentic approaches demonstrate promise, they often incur significant latency and cost due to complex multi-step reasoning and relying on closed-source LLMs. Alternatively, traditional code ranking models, typically optimized for query-to-code or code-to-code retrieval, struggle with the verbose and failure-descriptive nature of issue localization queries. To bridge this gap, we introduce SWERank, an efficient and effective retrieve-and-rerank framework for software issue localization. To facilitate training, we construct SWELoc, a large-scale dataset curated from public GitHub repositories, featuring real-world issue descriptions paired with corresponding code modifications. Empirical results on SWE-Bench-Lite and LocBench show that SWERank achieves state-of-the-art performance, outperforming both prior ranking models and costly agent-based systems using closed-source LLMs like Claude-3.5. Further, we demonstrate SWELoc's utility in enhancing various existing retriever and reranker models for issue localization, establishing the dataset as a valuable resource for the community.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes SWERANK, a retrieve-and-rerank framework for software issue localization. It combines  (1) SWERANKEMBED, a
bi-encoder embedding model serving as the code retriever; and (2) SWERANKLLM, an instruction-tuned
LLM serving as a code reranker. To support training, the authors curate SWELOC, a large dataset pairing real GitHub issue descriptions with the functions/files modified in the corresponding fixes. On SWE-Bench-Lite and LocBench, SWERANK achieves new SOTA across file/module/function granularity while being far cheaper than multi-step agent systems using closed LLMs; ablations analyze data quality filters (consistency threshold K), dataset size, and localization method.

### Strengths
The work is useful and practical: it reframes issue localization as a lightweight retrieve→rerank pipeline that is easy to deploy and far cheaper than agentic alternatives, while remaining compatible with standard IR/LLM components. It delivers good performance, achieving strong (often SOTA) results across file/module/function granularities on multiple benchmarks. The paper also provides insightful ablations—covering data consistency filtering, dataset scale, and hard-negative mining

### Weaknesses
1. Missing capacity/design ablations. The paper does not report sensitivity to model capacity or design choices for the retriever/reranker—e.g., embedding dimension, encoder depth/width. A controlled ablation study would strengthen causal claims.
2. Numerical latency study. Could you report latency for SWERank vs. agent baselines. The paper currently mentions multi-round agent latency qualitatively (Line 47, Line 125) but lacks a quantitative latency analysis for your method.
3. First-token objective may underutilize reasoning. The reranker trains only on the first generated token to select the positive, which could limit the model’s ability to use structured reasoning before deciding. A format-constrained “think-then-answer” protocol might improve generalization; loss could be defined over the formatted answer (or via RL with a verifier/reward), rather than solely first-token CE.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The author proposed SWERank of an efficient method for SW issue localization. It fine-tuned two models SWERank-Embed and SWERank-LLM to retrieve and rank the issue localization in just one shot. Comparing to agent based approach, it achieves both cost and accuracy goal, as demonstrated by SWE-bench-lite and LocBench SOTA performance. The author also created a new dataset SWELoc by carefully processing public GitHub PR data, valuable resource for future research works. This work shows a purposely built non-agentic retrieval + ranking models can be more effective and practical for automated software engineering than general-purpose reasoning agents.

### Strengths
1. The author proposed the  retrieval + ranking models fine-tuning with open-source qwen model instead of agent using close source model, demonstrating both SOTA performance and cost efficiency. The accuracy improvement (Acc@1,3,5,10) over existing methods are significant.
2, The idea is intuitive and reasonable, the theoretical analysis is good. The novelty on top of existing work is explained well. The experiments are relatively thorough, including baselines setup, ablation study, different methods and metrics comparison.
3. The SWELoc dataset (query, positive, negatives) is a great contribution to the community.
4. Overall writing is decently good.

### Weaknesses
1. even though agent based approach might be costly or inefficient (i.e. not single step), it can be dynamic and up-to-date knowledge with reasoning on the results. while the embedding + LLM finetuning approach becomes static and without much reasoning support with the author's current implementation. Better tradeoff must be taken in real situation.
2. the embedding model is fine-tuned from qwen2-7b, which seems to be extremely huge for a simple embedding. This significant expense of preprocessing the index should be accounted in the cost comparison instead of retrieval and re-rank only.

### Questions
1. SWE is built from Python repositories (>80%). will it work as effectively for strongly typed language such as C++, Java etc.? Will it work as single model finetuning with all major programming languages?
2. do you think a hybrid approach embedding + LLM finetuning as starter to guide the multi-agent - will that combination achieve better accuracy + reasoning + up-to-date knowledge?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents SWERANK, a two-stage retrieve-and-rerank framework for software issue localization. The authors first use a retriever to quickly identify candidate code snippets, then apply an LLM-based reranker to refine the ranking. They also introduce a large-scale training dataset, SWELOC. Experiments on multiple benchmarks show that SWERANK outperforms existing retrieval and agent-based approaches.

### Strengths
- The paper is well-written and easy to follow.
- The proposed SWELOC dataset is a valuable contribution to the software engineering community.
- The authors provide thorough comparisons with multiple baselines across two benchmarks.

### Weaknesses
- Retrieve-and-rerank is a well-established approach, and prior work (e.g., agentless) has already applied it to software issue localization.
- The effectiveness of retrieve-and-rerank is heavily constrained by the recall of the retrieval stage; compared to agent-based methods, its upper bound may be lower.

### Questions
- In some cases, the bug location may have little semantic overlap with the issue description. Agent-based methods can handle this via multi-step tool-assisted navigation within the repository. How would SWERANK handle such scenarios?
- If I understand correctly, SWERANKEMBED performs retrieval at the function level. How does it handle very long functions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents SWERANK, a retrieve-and-rerank framework for software issue localization, i.e., identifying code locations related to natural language issue descriptions such as bug reports. Experiments on SWE-Bench-Lite and LocBench show that SWERANK achieves state-of-the-art performance, surpassing agent-based systems (e.g., LocAgent with Claude-3.5) at all localization granularities.
It also offers significant cost advantages, at only $0.011–$0.015 per instance, compared to $0.46–$0.66 for agent-based approaches.

### Strengths
- The paper is well-structured.

- Practical significance and efficiency. The paper targets the underexplored yet practically important problem of software issue localization. SWERANK offers a simple, efficient alternative to costly multi-step LLM agents. Its cost-effectiveness (up to 40–60× cheaper than Claude-based agents) enhances its industrial applicability.

### Weaknesses
- **Limited methodological novelty**. The overall framework (retrieve-then-rerank) and component designs largely reuse existing paradigms. SWERANKEMBED is a standard bi-encoder trained with InfoNCE loss, similar to prior works such as CodeRankEmbed (2022–2024). SWERANKLLM simplifies listwise reranking to predicting the positive sample ID, which resembles earlier weakly supervised rankers like RankVicuna.
Given the rapid development of retrieval and reranking methods by 2025, the paper’s contribution is mainly empirical rather than conceptual. The authors should better justify why this design suits issue localization uniquely, beyond cost reduction.

- **Lack of independent component analysis**.
The two modules are evaluated only in combination. For SWERANKEMBED, there is no analysis of standalone retrieval quality (e.g., recall@N). If recall is low (e.g., <60% at Top-20), the reranker’s potential impact becomes limited. For SWERANKLLM, the authors do not report results when reranking the full candidate set independently. Demonstrating that each module performs reasonably well on its own would strengthen the technical soundness of the retrieve-then-rerank pipeline.

- **Insufficient analysis on issue complexity** The differences between Acc@5, Acc@10, and Acc@15 are small. It remains unclear whether issue complexity (e.g., single-function vs. cross-module dependencies) influences localization difficulty.
Grouping issues by dependency complexity or code change scope could reveal where the reranker struggles, providing deeper insight into model limitations.

### Questions
Q1: Could the authors provide independent evaluations of SWERANKEMBED and SWERANKLLM?
For example, what is the standalone retrieval recall of SWERANKEMBED at different Top-N settings, and how well does SWERANKLLM rerank full candidate lists without relying on the retriever’s output?

Q2: Given that both modules adopt well-established designs (bi-encoder retrieval with InfoNCE and listwise reranking), how does SWERANK differ conceptually from prior retrieve-and-rerank systems like CodeRankEmbed or RankVicuna?
What makes its design particularly suited to issue localization beyond efficiency gains?

### Soundness
3

### Presentation
3

### Contribution
2
