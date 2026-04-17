# IntSR: An Integrated Generative Framework for Search and Recommendation

- Decision: Reject
- Scores: 2, 6, 6

## Abstract
Generative recommendation has emerged as a promising paradigm, demonstrating remarkable results in both academic benchmarks and industrial applications. However, existing systems predominantly focus on unifying retrieval and ranking while neglecting the integration of search and recommendation (S&R) tasks. What makes search and recommendation different is how queries are formed: search uses explicit user requests, while recommendation relies on implicit user interests. As for retrieval versus ranking, the distinction comes down to whether the queries are the target items themselves. Recognizing the query as central element, we propose IntSR, an integrated generative framework for S\&R. IntSR integrates these disparate tasks using distinct query modalities. It also addresses the increased computational complexity associated with integrated S&R behaviors and the erroneous pattern learning introduced by a dynamically changing corpus. IntSR has been successfully deployed across various scenarios on a large internet platform serving hundreds of millions of users, leading to substantial improvements: +9.34% GMV, +2.76% CTR, and +7.04% ACC in three distinct scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes IntSR, an integrated generative framework for search and recommendation (S&R) that unifies both tasks and their sub-problems (retrieval and ranking). The model is evaluated on two public datasets (Amazon, KuaiSAR) and an industrial dataset. However this evaluation contains several flaws.

### Strengths
* The paper is well written and easy to follow.
* The figures help to understand the methodology described.
* Good ablation studies.

### Weaknesses
* The mathematical formulation is unclear because some symbolic notations are not introduced in the main body of the article, but in the appendix. The main body of the article should be self-contained.
* Some related works are missing eg : 
   * https://arxiv.org/pdf/2504.06714
   * https://arxiv.org/pdf/2504.05730 
* The evaluation section is unclear: the reported results appear very similar to those in Table 3 of this paper. While the authors share their own code, they do not provide access to the code of their competitors, which raises the concern that the results might have been copied from that table, with their own model simply added, rather than rerunning all baselines under the same conditions.
* No statistical tests are present. 
* Only two datasets are used.
* The evaluation has been carried out with 99 negatives and one positive, despite the paper “Widespread Flaws in Offline Evaluation of Recommender Systems” shows that it is better to use all the items as negatives.

### Questions
1. Could you please add another dataset and the statistical tests?
2. Could you please add the evaluation using all the items as negatives?
3. Could you clarify how the model addresses the cold start problem?
4.To convert natural language user search queries into embedding, why did you choose exactly the Qwen 0.6B model for the without trying any other?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
his paper proposes IntSR, an integrated generative framework that unifies search and recommendation (S&R) tasks along with their retrieval and ranking sub-tasks. The key innovation is treating queries as the central distinguishing element - using explicit natural language queries for search and implicit behavioral patterns for recommendation. The framework introduces a Query-Driven Block (QDB) with customized masking to reduce computational complexity, and addresses temporal vocabulary misalignment through a candidate alignment strategy. The model has been deployed at scale, serving hundreds of millions of users with reported improvements of +9.34% GMV, +2.76% CTR, and +7.04% ACC across three scenarios.

### Strengths
--Practical Impact: The successful deployment at scale with significant business metrics improvement demonstrates real-world value beyond academic contributions.


--Comprehensive Framework: The paper addresses multiple aspects - unification, efficiency, and temporal dynamics - in a coherent framework rather than treating them as isolated problems.


--Temporal Alignment Innovation: The identification and solution of temporal vocabulary misalignment is an important contribution that has been overlooked in prior work. Table 3 convincingly shows consistent improvements across all negative sampling strategies.


--Thorough Experimental Validation: The paper includes experiments on public datasets, proprietary data, ablation studies, and online A/B tests, providing multiple angles of validation.


--Computational Efficiency: The detailed complexity analysis and optimization in Appendix D shows careful engineering consideration for practical deployment.

### Weaknesses
--Limited Baseline Comparisons: For the temporal alignment experiments (Table 3), only basic negative sampling strategies are compared. No comparison with recent advanced sampling methods or other temporal-aware approaches.


--Unclear Architectural Choices:


  * Why is DSFNet necessary when QDB already handles multi-scenario aspects through query types?
  * The relationship between the 3-layer QDB and DSFNet modules is not well justified
  * The choice of Qwen3-0.6B for query encoding seems arbitrary

--Dataset Limitations:

* KuaiSAR requires separate training for recommendation and search due to sparse search behaviors, suggesting the unified approach may not work well when data is imbalanced
* Amazon dataset uses synthetic search queries, limiting the validity of search task evaluation
* Industrial dataset details are too vague for reproducibility

### Questions
-- Query Generation Strategy: In Section 3.2, you mention using probability β to populate non-search Q positions. How sensitive is the model to this hyperparameter? What values were used in experiments?


-- Temporal Alignment Overhead: Does maintaining temporal availability information (It) for all items introduce significant storage/computation overhead in production? How do you handle items with complex availability patterns?


-- Cold Start Performance: How does IntSR perform for new users or items with limited interaction history? The unified approach might exacerbate cold start issues.


-- Search Query Quality: Table 4 shows a dramatic drop when removing search queries for the search task (HR@1: 0.5678→0.4023). Does this suggest the model is overly dependent on explicit queries rather than learning implicit patterns?

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
This paper presents IntSR, a unified generative framework aiming to jointly handle both search and recommendation tasks, as well as their retrieval and ranking subcomponents. The core idea is to model user-item interactions—both search and recommendation—within an autoregressive generative model, distinguishing the task modality through query representations.

### Strengths
1. IntSR achieves excellent results in both offline experiments and online A/B tests.

2. The authors conducted comprehensive ablation studies that demonstrate the effectiveness of each design of IntSR.

3. The paper is well-written, and the figures are clear and illustrative.

### Weaknesses
1. No direct efficiency experiments. The paper claims computational speed-up but only provides theoretical complexity analysis. It would be much stronger to include a table comparing actual inference latency throughput against baselines.
2. What about scaling? The introduction mentions scalability as a key motivation. How does IntSR's performance scale with model size or data volume? Any experiments or insights on its scaling behavior would be great.
2. Missing discussion of recent work. There's no comparison or discussion with relevant concurrent work, such as [1]. Briefly discussing this would better position the paper.

[1] Shi, Teng, et al. "Unified Generative Search and Recommendation."

### Questions
Given the impressive online performance and successful deployment of IntSR, I'm particularly interested in which scenarios the advantages of a unified search-and-recommendation approach are most pronounced？

### Soundness
2

### Presentation
3

### Contribution
3
