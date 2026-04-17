# Hybrid Deep Searcher: Scalable Parallel and Sequential Search Reasoning

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 4

## Abstract
Large reasoning models (LRMs) combined with retrieval-augmented generation (RAG) have enabled deep research agents capable of multi-step reasoning with external knowledge retrieval. However, previous methods that extend reasoning with single-query search steps struggle to scale to complex tasks demanding broad document exploration. Meanwhile, approaches that generate multiple independent queries simultaneously may limit deeper, sequential reasoning. To address these limitations, we propose HybridDeepSearcher that dynamically integrates parallel and sequential search strategies to enable effective search scaling. To support training, we introduce HDS-QA, a novel dataset that seamlessly integrates broad parallel search with sequential search reasoning, providing answer trajectories in the form of reasoning-query-retrieval loops with parallel sub-queries. Across all five benchmarks, our approach significantly outperforms the state-of-the-art, improving F1 scores by +15.9 on FanOutQA and +11.5 on a subset of BrowseComp. Further analysis reveals that HybridDeepSearcher effectively scales performance with additional test-time search resources and demonstrates robustness on questions requiring more evidence, achieving higher evidence coverage. We include the code in the supplementary materials and will release the dataset and code publicly.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper focuses on multi-turn search agents. Most existing search agents perform complex information retrieval using sequential querying, whereas approaches that generate independent queries simultaneously may limit deeper, sequential reasoning. The authors construct a dataset that integrates parallel search with sequential search reasoning, collect answer trajectories, and fine-tune an LLM as a search agent via SFT.

Contributions:
* HDS-QA Dataset: An automatically constructed synthetic dataset derived from Natural Questions, containing questions that require both parallel and sequential search reasoning.
* HybridDeepSearcher: A search agent (fine-tuned Qwen3-8B) that adaptively integrates both parallel and sequential search strategies during the reasoning process.

### Strengths
* **Well-Motivated Problem**. The paper identifies a limitation in current search agents: sequential search agents cover limited information, while parallel search agents may sacrifice deeper, sequential reasoning capabilities.
* **Scalable Dataset Construction**. The HDS-QA dataset generation pipeline is fully automatic, eliminating the need for human annotation and enabling scalable data creation.
* **Strong Empirical Results**. The model demonstrates substantial improvements across multiple benchmarks, validating the effectiveness of the proposed approach.

### Weaknesses
1. **Training Methodology Limitations**. The paper collects trajectories using Qwen3-32B and fine-tunes Qwen3-8B via SFT. However, given that competitive baselines (Search-R1, RAG-R1) employ reinforcement learning for training, the absence of an RL-trained version limits understanding of the effectiveness of the method.
2. **Computational Cost and Fairness Concerns**. The use of Qwen3-32B for document summarization adds significant computational overhead without clear justification. Why not use retrieval results directly? More critically, this creates an unfair comparison with RAG-R1, the strongest baseline, which operates without a summarization module. The added computational budget makes it difficult to isolate the true benefits of the proposed approach.
3. **Dataset Construction Validity**. Based on the dataset generation process (Section 3.1) and the prompts in the appendix, the Qwen3-32B receives a bunch of extracted information and constructs the question in Step 3. Since there are no constraints, we can not determine if a generated question is really a parallel question. Furthermore, for hybrid questions structured as A|B->C (where A and B are parallel subquestions and C depends on their answers), the pipeline does not prevent information leakage. Agents may obtain sub-answers from a single branch without genuinely solving the parallel components. This raises concerns about whether the dataset truly improves parallel search capabilities or merely teaches pattern matching.
4. **Incomplete Qualitative Analysis**. The qualitative case studies (Appendix) lack comparisons with RAG-R1, the most competitive multi-query baseline, making it difficult to assess the specific advantages of HybridDeepSearcher over existing parallel query methods.

### Questions
1. **Formatting and Technical Errors**. The manuscript contains multiple formatting issues that affect readability: incorrect citation format at Line 048, improper vspace usage at Line 443, layout inconsistencies on Page 15, and reference errors in Appendix D. These should be addressed for publication quality.
2. **Confusing notation**. In Table 2, the original Qwen3-8B is replaced. Using "+Qwen2.5-7b" here is confusing.
3. **Parallel Query Generation Patterns**. Could the authors clarify the parallel query generation behavior observed across different case studies? In Tables 4 and 10 (MuSiQue and FRAMES examples), the first search turn generates queries that appear to rephrase the original question with minor variations, while Table 7 (BrowseComp example) produces genuinely diverse parallel queries. Since these use the same model and prompts, what factors determine when the model generates reformulations versus diverse parallel queries? Understanding this would help assess whether the model has learned systematic parallel search strategies or if the behavior is primarily question-dependent.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces a dataset and a training method that teaches the model to perform parallel search in addition to the common sequential search. The proposed dataset, HDS-QA, requires both parallel and sequential search, which can serve as a challenging training set that is scarce in the current training community.

### Strengths
1. The paper raised attention to parallel search, which is an under-studied capability of deep research models.
2. Using parallel search might enable the model with extended capability to gather information and more efficient utilization of context length.

### Weaknesses
1. The paper highlights its novelty on doing parallel search in addition to sequential search; however, the main change can be understood as adding a parser on the tool side to parse a list of queries from a single query string, which seems trivial to me. This has been implemented in some open-source projects like MiroMind as a small feature.
2. The construction of the dataset claims to require parallel search, but the ablation failed to accurately demonstrate how training a sequential search model on the same data might not work well as parallel search.

### Questions
1. There are many works from Tongyi (e.g., WebWalker, WebExplorer, etc.) and the fully open-sourced work called ASearcher on constructing synthetic BrowseComp-like QA datasets for deep research training. Could the author clarify the difference of HDS-QA from these works?
2. The choice of the 50 questions from BrowseComp needs a bit more justification. Using a super small test set will make the results high variance. Did the author benchmarked the variance and show what is the CI of this evaluation and how significant is the current gap between different models?
3. Could the author compare their models with baselines that are trained on synthetic data designed for sequential search, such as WebExplorer-8B and ASearcher-8B to show the benefit of parallel search?
4. I'm curious if the main benefit comes from the synthetic data or from developing parallel search. Could the author train the model for sequential search on the same training questions and ablate the effect? In addition, I wonder if the "parallel search" can be viewed as a new search tool that can parse a long string of queries and divide into sub-queries to gather information, because the model inference pipeline seems exactly the same as sequential search.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces HybridDeepSearcher, a deep search agent that integrates parallel and sequential search to perform RAG-based reasoning and question answering. To train it, the paper first synthesizes the HDS-QA dataset, focusing on generating hard, high-quality data points and corresponding parallel search traces, which consist of reason-query-retrieval. By using the HDS-QA dataset, fine-tuning alone is able to result in state-of-the-art performance on multiple challenging multi-hop QA datasets, providing both efficiency and effectiveness.

### Strengths
1. The paper is well-written with clear illustrations. 
2. The model training recipe is clean and systematic. 
3. The dataset is generated using Google’s People Also Ask feature, which seems to have significantly increased the difficulty and quality of the generated questions.
4. The experimental evaluation is comprehensive with a sufficient number of datasets and baselines. The analytical studies on test-time scaling are also insightful.

### Weaknesses
1. The pipeline still uses a bigger model Qwen3-32B to summarize retrieved documents, which may incur unwanted costs. Have the authors examined the performance using a smaller model, like the 8B model, as a summarizer?
2. As the primary contribution is a high-quality data synthesis pipeline, the paper did not explore the scaling behavior with respect to the number of generated data. It would be interesting to explore questions like how much data is needed, and whether more data helps.

### Questions
1. Generating parallel-hop questions is very interesting, but how does sequential search perform on this task compared to parallel search with the same number of queries submitted?
2. The method uses the Jina search API to perform parallel search. Are all baselines, including RL-based ones (e.g., search-R1), also trained on and use the Jina API?
3. The generated dataset can be put to better use. Does the author think that the HDS-QA dataset (without trajectories) can be used for RL training? What are the advantages and disadvantages?
4. Have the author examine the model performance against other baseline methods on one or two simpler general QA or multi-hop QA datasets used by Search-R1?
5. Typos: line 1191, Table numbers are missing.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors of this paper introduce HDS-QA, a new training dataset that is used to teach models how to make multiple parallel and sequential web search calls for answering hard reasoning-intensive questions. The authors fine-tune a qwen3-8b model on this dataset to obtain the HybridDeepSearcher model. HybridDeepSearcher uses parallel and sequential search to obtain state-of-the art performance on multiple hard question-answering datasets.

### Strengths
The strengths of this paper are:
- Authors compare to multiple recent methods that serve as strong baselines. 
- HybridDeepSearcher beats baselines on multiple datasets. Some of these datasets like BrowseComp are very challenging QA datasets. 
- The scaling curves convincingly show that HybridDeepSearcher is able to use extra tool calls and search turns effectively.

### Weaknesses
The weaknesses of this paper are:
- Parallel and multi-hop questions are generated using one fixed pipeline. This might lead to limited diversity and many questions might have very similar patterns. 
- Is HDS-QA only useful for improving search capabilities in 7-8b models or do smaller/larger models also benefit from training on HDS-QA? For larger models you likely need traces from a larger teacher model?
- Experiments are limited to SFT on HDS-QA and there no experiments to see if performance can further be improved by using RL.

### Questions
Here are a couple questions:
- Are all questions constructed by starting from a question in the NQ dataset?
- Does HybridDeepSearcher overthink on easier benchmarks? Figure 3 shows that HybridDeepSearcher is worse compared to other baselines when fewer calls are utilized. Is this a concern for easier datasets?
- How does the performance of HybridDeepSearcher scale with training data?
- What is the distribution of the number of parallel and sequential tool calls required for samples in HDS-QA?

### Soundness
4

### Presentation
3

### Contribution
3
