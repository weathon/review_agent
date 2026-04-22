# DeepResearchGym: A Free, Transparent, and Reproducible Evaluation Sandbox for Deep Research

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Deep research systems represent an emerging class of agentic information retrieval methods that generate comprehensive and well-supported reports to complex queries. However, most existing frameworks rely on dynamic commercial search APIs, which pose reproducibility and transparency challenges in addition to their cost. To address these limitations, we introduce DeepResearchGym as an open-source sandbox that combines a reproducible search API with a rigorous evaluation protocol for benchmarking deep research systems. The API indexes large-scale public web corpora, namely ClueWeb22 and FineWeb, using a state-of-the-art dense retriever and approximate nearest neighbor search via DiskANN. It achieves lower latency than popular commercial APIs while ensuring stable document rankings across runs, and is free for research use. To evaluate deep research systems' outputs, we extend the Researchy Questions benchmark with automatic metrics through LLM-as-a-judge to measure alignment with users' information needs, retrieval faithfulness, and report quality. Experimental results show that systems integrated with DeepResearchGym achieve performance comparable to those using commercial APIs, with performance rankings remaining consistent across evaluation metrics. A case study on short-answer search agents further demonstrates the sandbox's utility for cost-effective training, showing that models trained within the sandbox can generalize to commercial search.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces DeepResearchGym, a benchmark for evaluating deep research systems. The benchmark provides standardized search and fetch APIs to ensure a fair and transparent environment for comparing various systems. Although the benchmark utilizes an 'LLM-as-a-judge' approach, its metrics are validated by human annotation and high inter-annotator agreement. A comparative analysis of two commercial, three white-box, and two baseline deep research systems was conducted using this benchmark.

### Strengths
* Provide a transparent and fair environment to compare various deep research systems.
* Include extensive metrics during evaluation

### Weaknesses
* Novelty: The current corpus and queries of DeepResearchGym are from existing datasets (ClueWeb22, FineWeb, and Researchy Questions). I am very willing to raise my score if there are any missing innovative details in the methodology.

* A few missing details in the benchmark construction and evaluation. Please see the details in the questions.

### Questions
* Line 145: The authors state that low-quality and spam pages were filtered during sampling. Could you clarify if this filtering was part of the original ClueWeb22 dataset's preprocessing, or if an additional filtering step was applied? If it was an additional step, please provide more details on the process.
* The paper mentions that users can switch between corpora (Lines 200-202), yet all evaluations are conducted on either commercial search engines or the ClueWeb22 dataset. The justification for including the FineWeb corpus is unclear. To demonstrate its value and analyze the system's performance across different retrieval corpora, please consider adding experiments based on FineWeb, perhaps in the appendix.
* Figure 1 indicates high efficiency for the search API/retriever in DeepResearchGym. A discussion of the factors contributing to this efficiency would be a valuable addition. Explaining the underlying design choices could further highlight the innovative aspects of the system.
* The evaluation is limited to a small number of systems. While the benchmark aims for fairness, two of the systems evaluated do not use the provided search and fetch APIs, making direct comparison difficult. To provide a more direct and comprehensive comparison, have the authors considered evaluating other generative models within the same standardized pipeline?
* The design of DeepResearchGym is well-suited for efficiency comparisons between different systems (e.g., measuring the number of searches, fetches, or conversational turns). However, the paper currently lacks a discussion on these efficiency metrics. Including an analysis of efficiency would significantly strengthen the evaluation.
* It's always good to include human evaluation as justification for the LLM judges. However, the authors did not discuss the background of the annotators and the instructions for their annotation. Since the task requires expert-level knowledge, it is necessary to have more details regarding human evaluation.

### Soundness
3

### Presentation
3

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
The paper introduces DEEPRESEARCHGYM, an open, reproducible sandbox for evaluating deep research systems that generate long-form, evidence-grounded reports. It provides a free search API over public web corpora (ClueWeb22 A/B and FineWeb), implemented with a dense retriever (MiniCPM-Embedding-Light) and DiskANN, plus endpoints to search and to fetch archived page snapshots for stable, auditable evidence. The authors pair this with a tri-part evaluation protocol on Researchy Questions that scores (i) coverage via Key-Point Recall/Contradiction, (ii) retrieval faithfulness via citation recall/precision, and (iii) report quality (clarity, insight) using LLM-as-judge prompts. Empirically, systems keep similar rankings when swapping commercial search for this API, and the API achieves sub-0.5s median latency, suggesting it is a practical, research-grade substitute for proprietary search backends.

### Strengths
1. This paper proposes DeepResearchGym, an open-source benchmarking framework specifically designed to enable transparent and reproducible evaluation of deep research systems. Being free and open-source makes this work a valuable resource to the community.

2. Empirical evaluations show that the system achieves strong retrieval quality with minimal loss from approximate search, as well as maintaining response times below those attained by commercial APIs.

3. The paper is well-written and easy to follow.

### Weaknesses
1. Although DeepResearchGym helps the reproduction of deep research systems with a higher response speed and lower cost, using static corpora may under-serve very time-sensitive queries.

2. Despite showing high correlation with human evaluation, the empirical results only on Researchy Questions using GPT-4.1 as a judge might not comprehensively and robustly reflect the system's actual performance to serve as a replacement of search engine. Additionally, more fine-grained comparison between the proposed API vs different commercial API would be much appreciated. What's the fundamental difference between different search APIs, and how would that affect the performance of deep research systems?

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces **DeepResearchGym**, an open-source benchmarking framework for evaluating deep research systems. The framework addresses critical reproducibility and transparency issues in current evaluation practices that rely on proprietary, dynamic commercial search APIs. The main contributions are: (1) A **free search API** built on large-scale public web corpora (ClueWeb22 and FineWeb) with dense retrieval (MiniCPM-Embedding) and approximate nearest neighbor search (DiskANN), achieving lower latency than commercial alternatives while ensuring stable rankings; (2) An **evaluation protocol** extending the Researchy Questions benchmark with LLM-as-a-judge metrics measuring report relevance (Key Point Recall/Contradiction), retrieval faithfulness (citation precision/recall), and report quality (clarity/insightfulness); (3) **Empirical validation** showing systems maintain comparable performance when switching from commercial to DeepResearchGym APIs, with human evaluation confirming metric reliability ($\kappa = 0.72$-$0.89$).

### Strengths
1. **Addresses critical reproducibility gap**: The field urgently needs standardized benchmarking infrastructure. The paper directly tackles cost, transparency, and reproducibility issues with commercial APIs.
2. **Comprehensive multi-dimensional evaluation**: The three-faceted framework (relevance, faithfulness, quality) captures different aspects of report generation quality, going beyond simple surface-form metrics.
3. **Nuanced analysis**: Query-level correlation analysis (Figure 2) and query log analysis (Appendix A) provide insights beyond system-level aggregates.

### Weaknesses
1. **Heavy reliance on a single judge model without robustness analysis**: All automatic evaluations use GPT-4.1-mini exclusively as the judge, with no ablation studies using alternative models such as GPT-4o, Claude Sonnet, or open-source alternatives. This creates potential concerns about judge-specific biases, particularly since some evaluated systems (like OpenAI's deep research) are based on GPT models.
2. **Insufficient examination of static corpus limitations and ground-truth quality**: While the paper demonstrates that systems maintain performance when switching to static corpora, it does not analyze when this approach is adequate versus problematic. There is no quantitative assessment of which query types are temporally sensitive and might suffer from the 2022-2024 snapshot. Additionally, the evaluation assumes that clicked documents from search logs constitute comprehensive ground truth, but this assumption is never validated. Users may click tangential documents or miss important sources, yet the paper provides no analysis of click quality or relevance distribution.

### Questions
1. What is the correlation between report length and KPR? Are higher-scoring systems simply more verbose?
2. What are the most common failure modes? Can you provide qualitative analysis of queries where systems perform poorly vs. well?

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
This work introduces DeepResearchGym, an open-source sandbox for deep research agents. The sandbox features a search API implementation that is reproducible to facilitate future research and agent development. Based on the sandbox, the authors further evaluated several state-of-the-art deep research agents and presents insight into their performance along different axes, including relevance, faithfulness, and quality. Results show that agents can achieve comparable performance when using DeepResearchGym compared to using commercial search APIs.

### Strengths
1. The paper contributes an open-source sandbox that can potentially facilitate reproducible research in deep research agents.
2. The authors deliver some insight into fine-grained performance of popular deep research agents.

### Weaknesses
1. line 154-155: While the authors use recent data sources to construct the sandbox, it is unclear from the paper how to keep the sandbox "up-to-date".
2. The contribution beyond the sandbox is limited, as the authors mostly follow existing work in their evaluation protocol. 
3. While I understand the importance of having a reproducible environment for benchmarking deep research agents, I fail to see the discussion on the actual benefit from this paper. What are the evaluation nuances that DeepResearchGym helps to capture which are infeasible with time-varying search APIs?
4. In section 4.3, the authors use the variability observed in query-level analysis to justify the importance of using a standard retrieval API. I think the logic is flawed. Doesn't this show that the evaluation metrics, which heavily relies on textual overlap, are sensitive the search APIs used?
5. For section 4.4, I don't see the meaning of this pairwise human evaluation when the main results in the paper are pointwise. Why not letting human evaluators follow the same protocol as LLM-as-a-judge and compute score correlations?

### Questions
1. line 144, how do you define "low quality"?
2. line 178-179, how do you define "minimal loss"?
2. Section 4.2, why are subjective report quality metrics (Clarity, Insight) comparable to objective information coverage metrics (KPR)?

### Soundness
2

### Presentation
2

### Contribution
2
