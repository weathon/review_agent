# GraphRAG-Bench: Challenging Domain-specific Reasoning for Evaluating Graph Retrieval-Augmented Generation

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Graph Retrieval Augmented Generation (GraphRAG) has garnered increasing recognition for its potential to enhance large language models (LLMs) by structurally organizing domain-specific corpora and facilitating complex reasoning. However, current evaluations of GraphRAG models predominantly rely on traditional question-answering datasets. Their limited scope in questions and evaluation metrics fails to comprehensively assess the reasoning capacity improvements enabled by GraphRAG models. To address this gap, we introduce GraphRAG-Bench, a large-scale, domain-specific benchmark designed to rigorously evaluate GraphRAG models. Our benchmark offers three key superiorities: \((i)\) Challenging question design. Featuring college-level, domain-specific questions that demand multi-hop reasoning, the benchmark ensures that simple content retrieval is insufficient for problem-solving. For example, some questions require mathematical reasoning or programming. \((ii)\) Diverse task coverage. The dataset includes a broad spectrum of reasoning tasks, multiple-choice, true/false, multi-select, open-ended, and fill-in-the-blank. It spans 16 disciplines in twenty core textbooks. \((iii)\) Holistic evaluation framework. GraphRAG-Bench provides comprehensive assessment across the entire GraphRAG pipeline, including graph construction, knowledge retrieval, and answer generation. Beyond final-answer correctness, it evaluates the logical coherence of the reasoning process. By applying nine contemporary GraphRAG methods to GraphRAG-Bench, we demonstrate its utility in quantifying how graph-based structuring improves model reasoning capabilities. Our analysis reveals critical insights about graph architectures, retrieval efficacy, and reasoning capabilities, offering actionable guidance for the research community.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces GraphRAG-Bench, which is a large-scale, domain-specific benchmark designed to evaluate GraphRAG methods.

### Strengths
1. The experiments are comprehensive and cover a wide range of models and tasks.
2. The evaluation metrics consider multiple dimensions, ensuring fairness and balanced comparison.
3. The benchmark provides new directions and inspiration for future research in the GraphRAG community.

### Weaknesses
1. **The domain of corpus for RAG is too narrow**. GraphRAG-Bench limits its corpus to computer science, and although it includes subfields like CV and NLP, it lacks coverage of broader topics such as medicine, law, and biology. This seriously reduces the benchmark’s real-world applicability. Some popular GraphRAG frameworks, such as LightRAG[1], PathRAG[2], and HyperGraphRAG[3], already include datasets from these diverse domains.
2. **Relying entirely on knowledge from textbooks weakens the incremental advantage of GraphRAG**. Results from Table 4 show that, since much of the relevant knowledge was already exposed during pretraining, GPT-4o-mini without retrieval achieved fairly high scores. In fact, some RAG methods even performed worse due to the additional noise introduced by retrieval. Similar to the previous point, using domain-specific or private corpora could help mitigate this issue and better highlight the performance benefits of retrieval-based frameworks.
3. **Avoiding using retrieval strategies based on semantic vector models as baselines**. Relying only on BM25 and TF-IDF while ignoring popular semantic similarity–based retrieval methods[4] is disappointing. I strongly recommend that the authors include related experiments.
4. **Lacks validation across corpora of different scales**. Constructing graphs from all 20 textbooks is computationally intensive and costly. However, in real industrial scenarios, the impact of corpus size variation on GraphRAG performance remains unpredictable—some GraphRAG frameworks might perform better in smaller-scale settings.
5. **Inconsistency between task design and evaluation standards**. GraphRAG-Bench includes tasks such as mathematical computation and programming that rely on reasoning chains and tool usage, yet the reasoning score in GraphRAG-Bench evaluates the similarity between the model’s reasoning process and the golden rationale. In the LLM community, the former typically only requires the final answer to be verifiable[5], which is misaligned with the knowledge extraction and retrieval focus of GraphRAG applications (for example, the model has another solution to solve a programming problem comparing with golden solution).

[1] Guo Z, Xia L, Yu Y, et al. Lightrag: Simple and fast retrieval-augmented generation[J]. arXiv preprint arXiv:2410.05779, 2024.
[2] Chen B, Guo Z, Yang Z, et al. Pathrag: Pruning graph-based retrieval augmented generation with relational paths[J]. arXiv preprint arXiv:2502.14902, 2025.
[3] Luo H, Chen G, Zheng Y, et al. HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation[J]. arXiv preprint arXiv:2503.21322, 2025.
[4] Karpukhin V, Oguz B, Min S, et al. Dense Passage Retrieval for Open-Domain Question Answering[C]//EMNLP (1). 2020: 6769-6781.
[5] Guo D, Yang D, Zhang H, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning[J]. arXiv preprint arXiv:2501.12948, 2025.

### Questions
1. How about considering more domain-specific corpus in GraphRAG-Bench?
2. How about including embedding-based dense-retrieval methods as baselines?
3. What is the performance of GraphRAG methods in different scales of corpus?
4. How to better align tasks like math reasoning and programming with metrics?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a benchmark GraphRAG, and presents the evaluation results of various GraphRAG solutions. Good benchmark and insightful evaluation results.

### Strengths
S1. GraghRAG is a hot research area and a benchmark is overdued.

S2. The paper presents a set of evaluation metrics to evaluate various aspects of the proposed methods.

S3. Rigorous filtering and refinements of the questions and rationals.

### Weaknesses
W1. It is unclear how the questions are crafted? Are they crafted by people or generated by machines? Are they comprehensive and how does that affect the experimental results?

W2. 1018 questions is not much, especially if we consider different buckets and topics. Are the results statistically significant?

W3. It's unclear why GraphRAG helps for TF and OE but not for MC, why the issues happen for one type does not happen for other types. Also, the differences do not always seem big. Again, is that just random noises, or different types of questions require different types of reasonings?

W4. It is nice that the paper classifies GraphRAG methods into 3 classes. Would be nice to show the classes in each table to make it easy to associate, and give better insights on their strengths and weaknesses.

W5. Would be nice to compare w. some recent RAG benchmarks that also require reasoning, such as CRAG and RAGBench.
CRAG--Comprehensive RAG Benchmark
RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems

### Questions
Answer questions and address concerns in weaknesses

### Soundness
4

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
4

### Summary
This paper introduces *GraphRAG-Bench*, a challenging domain-specific benchmark designed to rigorously evaluate GraphRAG models on dimensions including graph construction, retrieval, and multi-hop answer generation and rationale generation. 

Regarding *“Does graph augmentation truly enhance reasoning capabilities beyond simple retrieval?”*, this bench constructed challenging college-level questions of 5 types from 20 computer science textbooks. This paper implements a multi-stage pipeline to extract accurate content from the original corpus, and supplies expert-crafted rationales that articulate the complete logical progression for each question to evaluate whether GraphRAG models truly generate grounded explanations. 

This paper conducts extensive experiments on 9 SOTA GraphRAG methods, providing valuable insights into their performance across different question types and domains. Though only GPT-4o-mini is used as a base LLM, the experiments basically demonstrate that GraphRAG causes different effects on LLMs' performance across different question types, and overall, GraphRAG enhances LLMs' reasoning ability.

### Strengths
- The benchmark is large, diverse, and carefully constructed from accurate content of textbooks, ensuring high quality and relevance. 
- Compared to existing benchmarks, this benchmark supplies expert-crafted rationales for questions, enabling the evaluation of whether models truly generate grounded explanations or simply exploit surface-level patterns.
- This paper conducted extensive experiments to reveal the performance variations across question types and improvements on LLMs' reasoning ability with GraphRAG methods.

### Weaknesses
- Only GPT-4o-mini is used as the base LLM in experiments. The generalizability of the experiment conclusions may be limited.
- In Section 4. Observation, the analysis of why GraphRAG's performance improvement/decline across different question types lacks sufficient experimental validation. For example, when explaining the accuracy drop of MC questions, this paper attributes it to GraphRAG potentially introducing redundant or loosely related information. But this paper provides no relevant experiments or case studies to substantiate this claim, which appears more like speculation rather than analysis.

### Questions
- Are the experimental conclusions, including those regarding different question types and LLMs' reasoning capabilities, applicable to other LLMs, such as open-source models like LLaMA or other close-source models?
- Refer to Weakness 2,  can you provide the relevant experiments or case studies to surpport the analysis?

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
3

### Summary
The paper introduces GraphRAG-Bench, a new benchmark for evaluating GraphRAG models with 1,018 questions in computer science. The paper also presents an extensive evaluation of 9 GraphRAG methods on this benchmark, showing the significant role of graph integration in improving reasoning and generation performance.

### Strengths
* The paper is well-organized.
* The paper introduces a benchmark comprising 1,018 college-level questions in computer science specifically designed to rigorously evaluate GraphRAG models. This benchmark has the potential to become a valuable resource for the research community.
* The paper presents a comprehensive assessment of 9 state-of-the-art GraphRAG methods on the proposed benchmark.

### Weaknesses
* Although the benchmark covers 16 topics, all are within the field of computer science, which limits its generalizability. It is unclear whether this scope meets the high standards of ICLR or would be of interest to a broader audience.
* The paper does not provide full details or justification regarding how representative the selected questions are.

### Questions
* Ultimately, readers will be interested in whether achieving higher performance on this benchmark has meaningful implications. For example, does it strongly suggest improvements in certain downstream applications? Would it indicate better performance on domain-specific questions in other areas?

### Soundness
2

### Presentation
3

### Contribution
2
