# OKBench: Democratizing LLM Evaluation with Fully Automated, On-Demand Open Knowledge Benchmarking

- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
Knowledge-intensive question answering is central to large language models (LLMs) and is typically assessed using static benchmarks derived from sources like Wikipedia and textbooks. However, these benchmarks fail to capture evolving knowledge in a dynamic world, and centralized curation struggles to keep pace with rapid LLM advancements. To address these drawbacks, we propose OpenKnowledgeBench (OKBench), a fully automated framework for generating high-quality, dynamic knowledge benchmarks on demand. Focusing on the news domain where knowledge updates daily, OKBench is an agentic framework that automates the sourcing, creation, validation, and distribution of benchmarks. Our approach democratizes benchmark creation and facilitates thorough evaluation of retrieval-augmented methods by reducing overlap with pretraining data. We evaluate our framework on multiple open-source and proprietary LLMs of various sizes and configurations, both with and without retrieval over freshly generated knowledge. Our results reveal distinct model behaviors when confronted with new information and highlight how retrieval narrows the performance gap between small and large models. These findings underscore the importance of evaluating LLMs on evolving knowledge benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents an automatic benchmark construction method for evaluating LLMs with multi-choice questions based on recent news that are expected to be not memorized by LLMs. The construction method uses LLM to generate questions and validate questions. 4 PhD students are asked to validate 200 questions with two panels: the answers have a correct rate of 94% in the first panel and 100% in the second panel. 

The evaluate reports results of testing four LLMs, without context, without oracle context and with retrievers of BM25, DPR and ColBERT.

### Strengths
1. Evaluating the performance of LLMs and LLMs with RAG is an interesting research problem. The dataset attempts to bridge this gap.

2. The paper is well written. 

3. A lot of results on different LLMs, with and without context, and different retrievers are reported and analysed.

### Weaknesses
1. The current evaluation focuses on different LLM-based QA models, but ignore the dataset construction method, and the key features of the dataset. 

2. I do not see any results on the efficiency and scalability of the benchmark construction method. High efficiency is required to ensure emerging LLMs. The validation of the question-answer quality depends on human annotation. This is time consuming. I also think this manual validation is required when new samples are generated, as the effectiveness of the LLMs for creating these samples may expire in dealing with new data. 

3. There is a shortage of incremental evaluation, e.g., results of datasets constructed in Jan 2025, Apr 2025, Jul 2025. This can observe the effectiveness of the data construction method in the time line.

### Questions
1. As the questions are new (after the LLM is pre-trained), why some LLMs can still achieve as good performance as 50%? 

2. Is the following paper, which use new knowledge from Wikidata for evaluating LLMs, relevant? https://arxiv.org/abs/2412.17032 

3. How do you check the licenses or copyright of the news data providers? Does the news data include any sensitive personal information?

### Soundness
2

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
5

### Summary
This paper introduces a novel framework for dynamic and decentralized evaluation of LLMs on evolving factual knowledge. The core contribution is OKBench, a fully automated, agentic pipeline that continuously generates, validates, and versions QA benchmarks from daily news streams. OKBench operates entirely autonomously, scraping fresh news articles, generating multiple-choice and open-ended questions with LLM agents, validating them through a second model, and assigning reproducible dataset signatures that ensure transparent version control.

The authors evaluate OKBench on both data quality and benchmark utility. Using OKBench, the paper conducts large-scale experiments across multiple open-source LLM families (e.g., Gemma, LLaMA, Qwen, Phi) and retrieval strategies (BM25, DPR, ColBERT v2).

### Strengths
The paper addresses an important and timely problem: the challenge of evaluating LLMs on factual knowledge without contamination from training data. The authors convincingly argue that testing on newly emerging knowledge is one of the most effective ways to mitigate data leakage, providing a well-motivated and practically relevant setting. 

A key strength lies in the design of a fully autonomous benchmark generation pipeline, which makes OKBench highly scalable and easily reproducible. The inclusion of a data versioning protocol with unique dataset signatures is an especially thoughtful feature, ensuring transparent and repeatable evaluations across time. 

The paper also demonstrates strong attention to data quality, supported by a human validation study confirming that the majority of automatically generated questions are clear and factually correct. Furthermore, the authors provide a detailed cost analysis, showing that daily benchmark generation is affordable, an important consideration for community adoption. 

Finally, the experimental section is comprehensive and insightful, covering a broad range of models and retrieval-augmented generation methods, and yielding meaningful findings on knowledge freshness and retrieval effectiveness.

### Weaknesses
OKBench's novelty claim as “the first fully automated factual QA benchmark” is somewhat overstated. Previous works such as TemporalWiki[1], WikiFactDiff[2], and especially WikiBigEdit[3] have already introduced fully automated benchmark generation pipelines for evaluating factual and temporal knowledge in LLMs. Although OKBench distinguishes itself by focusing on the news domain rather than structured knowledge graph updates, these earlier benchmarks should be explicitly acknowledged and included in Table 2, and the corresponding claim of being the first fully automated benchmark (end of Section 2) should be softened accordingly.

Second, the empirical results in Section 5.1 raise concerns about data contamination, which directly undermines the benchmark’s stated motivation. If the benchmark genuinely captures unseen, post-cutoff knowledge, models should not substantially exceed the 25 % random baseline in the no-context setting. However, the reported results show significantly higher accuracies, suggesting that a nontrivial portion of the data overlaps with the models’ pretraining corpora or with frequently reported background facts. A more thorough contamination analysis or explicit discussion of this limitation would strengthen the paper’s credibility.

Third, the framework’s dependence on the underlying LLM used for question generation and validation is not explored experimentally. Since the agentic pipeline relies heavily on a single base model (GPT-4.1-2025-04-14), an ablation study varying the generation LLM would clarify the benchmark's robustness and reproducibility across different base models. Similarly, while the qualitative validation study is informative, a quantitative summary of the filtering rate (how many generated questions are discarded during validation) would help assess the pipeline’s efficiency and the actual yield of high-quality questions.

Finally, there are a few presentation issues that could improve readability: Figure 2 is introduced early but only discussed in Section 5.2, making its placement suboptimal. Figure 4 would benefit from adopting the same layout as Figure 3 (four subplots in a single row) with a less prominent color for the no-context baseline (this is more of a suggestion than an actual weakness). 

[1] Jang et al. (2022): TemporalWiki: A lifelong benchmark for training and evaluating Ever-Evolving language models.
[2] Khodja et al. (2024): Wikifactdiff: A large, realistic, and temporally adaptable dataset for atomic factual knowledge update in causal language models.
[3] Thede et al. (2024): WikiBigEdit: Understanding the Limits of Lifelong Knowledge Editing in LLMs

### Questions
1) Novelty and Relation to Prior Work: How does OKBench fundamentally differ from WikiBigEdit, which also features a fully automated factual QA generation pipeline? Could the authors clarify whether they view OKBench as complementary to or extending this line of work, and why it was omitted from Table 2 and the related work section?
2) Data Contamination Analysis: Given that models achieve substantially higher than random accuracy in the no-context setting, how do the authors explain this performance?
3) Pipeline Robustness: Since the benchmark generation heavily relies on a specific base LLM (GPT-4.1-2025-04-14), how stable is the pipeline when using different generation or validation models? 
4) Filtering Statistics: Can the authors quantify how many of the initially generated questions are filtered out during the validation stage?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes OKBench as a fully automated framework for generating dynamic benchmarks.  It can be automatically generated and used for the evaluation of retrieval-augmented methods. On experiments with multiple open-source and proprietary LLMs, it finds multiple observation about the model behaviors toward novel information.

### Strengths
1. This work proposes an approach to construct an on-demand knowledge base from the Internet text source. It can be used for evaluation of RAG systems. 

2. This work experiments LLMs’ behaviors on the dynamic corpus and discovers multiple intriguing observations.

### Weaknesses
1. The dataset novelty is limited. 
- As summarized in Table 1, the novelty of the dataset is not clear compared to the existing ones. 
- One key novel feature is “any time” in the update frequency, but making other datasets anytime too is not challenging. 
- As such, no prominent novel features can be found for this dataset.

2. The novelty of the proposed data collection pipeline is limited. 
- As illustrated in Fig. 1 and section 3.1, there is no novelty in the benchmark construction pipeline, as it is quite standard and straightforward. 

3. The usability of the dataset is limited. 
- As shown in Fig. 3, once the oracle document is given, most models attain very high accuracy (near 100%). 
- It may implicate that once the retrieval is successful, the QA part could be very easy to solve. Then, this benchmark may evaluate mostly the retriever’s performance rather than LLMs’ QA capability. 
- As reported in Fig.4, four different LLMs show almost similar performance. It could be a piece of evidence that the choice of an LLM does not matter to solve this benchmark. Only retriever selection matters. 

4. Only three basic retrievers are tested.

### Questions
Please refer to the Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces OKBench, a dynamic and knowledge-intensive benchmark designed to automatically evaluate large language models on their ability to handle factual, up-to-date information.

### Strengths
- This paper proposes OKBench, a dynamic, knowledge-intensive benchmark that is automatically updated.
- The authors test models of various sizes and compare their performance on MMLU Pro and OKBench, analyzing differences in memorization and adaptability to newly introduced information.

### Weaknesses
**Limited novelty:**
- Although the paper claims that OKBench is “the first fully automated benchmark for evaluating factual question answering ability” (L146), similar dynamic benchmarks have already been proposed in prior work [1, 2, 3].
- These existing works also describe pipelines for continual updates, whereas this paper presents only the initial construction pipeline, without demonstrating an actual update process. The lack of comparison to prior dynamic benchmarks may mislead readers about the paper’s contribution.
- The paper asserts that it introduces an agentic framework for benchmark construction (L153). However, the pipeline illustrated in Figure 2 lacks a clear agentic component; each step appears to be manually designed, without autonomous decision-making or agent-based iteration.


**Benchmark difficulty and saturation:**
- The results in Figure 4 show that BM25 Context performance (≈90–95 for Gemma) is nearly identical to Oracle performance (≈95). Even accounting for BM25 being a simple lexical retriever, this narrow gap suggests that the benchmark questions may be too easy.
- Moreover, BM25 Context significantly outperforms DPR Context (90–95 vs. 75–80), indicating that word matching alone suffices to answer most questions.
- This also implies that questions might have been generated from single factual sentences, often reusing phrases directly from the source text.
 Such design choices reduce the benchmark’s ability to evaluate deeper reasoning and may compromise its validity.

[1] Ko et al., "GrowOVER: How Can LLMs Adapt to Growing Real-World Knowledge?", ACL 2024.    
[2] Lin et al., "DynaQuest: A Dynamic Question Answering Dataset Reflecting Real-World Knowledge Updates", ACL 2025 Findings.    
[3] Ouyang et al., "HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation", ArXiv 2025.

### Questions
- The DPR retriever shows much lower performance than BM25 or ColBERTv2, and inconsistent results across the 1-, 5-, and 10-day corpus.
Have the authors considered evaluating stronger retrievers, such as Qwen-3-Embedding or E5, which are known to perform better on factual retrieval tasks? Including such models could clarify whether the issue lies with the retriever’s capability or with the benchmark’s inherent design.

### Soundness
1

### Presentation
3

### Contribution
1
