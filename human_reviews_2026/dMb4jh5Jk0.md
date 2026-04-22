# When should I search more: Adaptive Complex Query Optimization with Reinforcement Learning

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 6

## Abstract
Query optimization is a crucial component for the efficacy of Retrieval-Augmented Generation (RAG) systems. While reinforcement learning (RL)-based agentic and reasoning methods have recently emerged as a promising direction on query optimization, most existing approaches focus on the expansion and abstraction of a single query. However, complex user queries are prevalent in real-world scenarios, often requiring multiple parallel and sequential search strategies to handle disambiguation and decomposition. Directly applying RL to these complex cases introduces significant hurdles. Determining the optimal number of sub-queries and effectively re-ranking and merging retrieved documents vastly expands the search space and complicates reward design, frequently leading to training instability.
To address these challenges, we propose a novel RL framework called Adaptive Complex Query Optimization (ACQO). Our framework is designed to adaptively determine when and how to expand the search process. It features two core components: an Adaptive Query Reformulation (AQR) module that dynamically decides when to decompose a query into multiple sub-queries, and a Rank-Score Fusion (RSF) module that ensures robust result aggregation and provides stable reward signals for the learning agent. To mitigate training instabilities, we adopt a Curriculum Reinforcement Learning (CRL) approach, which stabilizes the training process by progressively introducing more challenging queries through a two-stage strategy.
Our comprehensive experiments demonstrate that ACQO achieves state-of-the-art performance on three complex query benchmarks, significantly outperforming established baselines. The framework also showcases improved computational efficiency and broad compatibility with different retrieval architectures, establishing it as a powerful and generalizable solution for next-generation RAG systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on the query optimization task in RAG systems. Specifically, the authors propose ACQO, a two-stage curriculum Reinforcement Learning framework that trains the LLM to determine when and how to expand the search process. This framework is powered by the unique reward design, where the first stage optimizes the best sub-query combinations and the second stage focuses on the ranking quality. Experiments on TopiOCQA and HotpotQA demonstrate the effectiveness of the proposed method.

### Strengths
1. The paper introduces a novel perspective on curriculum reinforcement learning where different stages focus on different objectives, instead of only focusing on data difficulty.

2. The Rank-Score Fusion module provides an elegant and model-agnostic solution for aggregating multiple retrieval results, which works seamlessly with both sparse and dense retrievers without requiring retriever-specific modifications.

3. The adaptive query decomposition mechanism allows the model to autonomously decide when and how to expand queries based on query complexity, avoiding the limitations of fixed decomposition strategies.

### Weaknesses
1. The experimental evaluation is limited in scope.  Specifically, desipe the main results are evaluated on two tasks (disambiguation and multi-hop task), each task only contains just one dataset. Including additional benchmarks for each task type would strengthen the empirical validation of the proposed method.

2. The performance improvements of ACQO shown in Table 2 and 3 are marginal and the comparisons raise several concerns:
  * As shown in Table 2, the performance of ACQO is even lower than the baseline ConvSearch-R1 on half of the experiment settings.
  * Most baselines in Table 2 use T5 as the backbone, while ACQO employs Qwen2.5-3B, which may benefit from a more powerful pre-trained model rather than the proposed method itself.
  * The baselines in Table 3 are limited to simple prompt-based methods, lacking comparison with other RL-based or advanced query optimization approaches. 
  In summary, these results do not provide sufficient evidence for the effectiveness of the proposed method.

3. The paper lacks clarity in several technical details:
  * The notation is confusing: $\mathcal{R}$ (for the RankScore Fusion module) and $R$ (for reward function) are too similiar and easily confused.
  * The definition of score $s_j$ mentioned in Eq.(1) is not clearly specified.
  * The abbreviation `qd` in Table 3 should be explicitly defined as `query decomposition` either in the caption of Table 3 or in Section 4.1 for better readability.

### Questions
Apart from the questions in Weakness, what is the fundamental motivation for optimizing the query optimization task in isolation, and how does this align with the broader objectives of RAG systems? I have two main concerns:

1. First, the main concern lies in the reward function itself, which relies on the groudtruth documents as supervison. As we all know, it may not be available in many read-world senarios or more challenging tasks, such as browsing competition. Moreover, in real-world dynamic environment, the "groundtruth" documents of each question may evolve over time, potentially leading to misleading reward signals. While I understand this work focuses on static environment for research purpose, how would this approach adapt to more realistic settings?

2. Are there any potential misalignment between query optimization and end-to-end RAG performance? From my perspective, query optimization is a subtask within RAG systems. Optimizing for retrieval metrics (findings groundtruth documents) may not directly translate to better final answer quality. Have you considered end-to-end optimization where the reward is based on answer correctness rather than retrieval metrics? This cound reveal whether better retrieval metrics actually leads to better RAG performance, or if there's a gap between local optimization (query -> documents) and global optimization (query -> answer).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes **Adaptive Complex Query Optimization (ACQO)**, a novel reinforcement learning framework for improving query optimization in Retrieval-Augmented Generation (RAG) systems when handling complex user queries that require disambiguation or decomposition. ACQO features an **Adaptive Query Reformulation (AQR)** module that dynamically decides whether and how to decompose a query into multiple sub-queries, and a **Rank-Score Fusion (RSF)** module that robustly aggregates retrieval results by combining both rank positions and retrieval scores. To stabilize training, ACQO employs a **two-stage Curriculum Reinforcement Learning (CRL)** strategy—first exploring broadly across all queries and then focusing on challenging cases. Experiments on benchmarks like TopiOCQA, HotpotQA, and MultiHop-RAG show that ACQO achieves state-of-the-art performance, generalizes well to unseen domains, and integrates efficiently with both sparse and dense retrievers—all without supervised data.

### Strengths
1. ACQO achieves state-of-the-art results on multiple benchmarks (HotpotQA, MultiHop-RAG), outperforming baselines even with a small 3B-parameter model.
2. The two-stage Curriculum Reinforcement Learning (CRL) strategy—starting with broad exploration and then focusing on hard examples—mitigates reward sparsity and training instability common in RL-based query optimization.
3. The Rank-Score Fusion (RSF) module effectively combines results from multiple sub-queries by leveraging both rank positions and retrieval scores, ensuring compatibility with diverse retrievers (sparse and dense) without added latency.

### Weaknesses
1. The experiments focus primarily on retrieval metrics (e.g., Recall@k, MRR), but do not include downstream end-to-end question answering accuracy or generation quality, leaving open whether retrieval gains consistently translate to better final answers.
2. The reward signal assumes access to ground-truth relevant documents during training, which may not hold in fully unsupervised or open-domain settings—limiting true “zero-supervision” applicability.
3. Although the paper claims efficiency, generating and retrieving for multiple sub-queries inherently increases latency and resource usage compared to single-query baselines—especially in latency-sensitive applications. The trade-off between performance gain and cost is not fully quantified.

### Questions
1. Please provide the precise definition of R in line 265.
2. Please explain the detailed meaning of "unsupervised learning" in line 350.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this work the authors present a method to train an LLM to generate queries for retrieval.

The application is retrieval for ambiguous or multi-hop question answering, where the question should be expanded into up to 3 queries to retrieve documents needed for answering.

The authors propose a method to join the lists of documents retrieved for several queries that uses both ranks and retriever scores.

Furthermore, and this is their main contribution, they use reinforcement learning to train a base LLM for the query expansion task. The LLM is instructed to generate queries from the question or chat context. The reward is based on the retrieved documents.

The RL setup uses curriculum learning starting with the full dataset. Then they select the hard examples for the second stage by running a couple roll-outs and computing the best subset of queries for each, resulting in optimistic rewards. The average optimistic reward over the rollouts is used to select hard examples for the second phase.

Experiments use Qwen-2.5-3B, DAPO for policy optimization, and TopiOCQA and HotpotQA as datasets and show overall strong performance.

### Strengths
The overall method seems sound, incorporating scores when joining lists of documents is a valid approach and using curriculum learning can sometimes improve performance, especially for hard to optimize tasks like RL.

Good results

Overall well written paper

### Weaknesses
The experiment setup does not explain if hyperparameters were selected using a dev set.

While there is no fundamental issue with the proposed method, it is in essence a combination of tricks. From the presented experiments it is not clear to me whether this curriculum method is robust enough for general use.

### Questions
In line 165 it is stated that performance drops sharply between easy and hard subsets of TopiOCQA but the given number can not be found in Table 1.

Please double-check the linear mapping definition in Formula (3)

In Table 3, why is dense retrieval worse than sparse?

Improvements:

There are sometimes added or missing words, e.g. 'which' in line 43, 'are' in 352.


Figure 1 should be changed to a different type of diagram, e.g. bars.

Since you already ran Vanilla RL for Table 1, why not include these results in the main results table

In formula 7 did you mean G instead of script_G?

In Table 2 check the numbers for RETPO Dense retrieval vs. Jang et al (2024) Table 1

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors propose Adaptive Complex Query Optimization (ACQO), a novel, end-to-end RL framework. The framework trains a 3B-parameter Qwen model  to function as an adaptive agent that learns when (whether to decompose) and how (what sub-queries to generate) to expand the search process.   
The ACQO methodology is built on three synergistic components:
An Adaptive Query Reformulation (AQR) module, which serves as the agent's policy to dynamically generate a set of one-to-many sub-queries based on the input.   
A Rank-Score Fusion (RSF) module, a novel, heuristic-based method for robustly aggregating the retrieval results from all sub-queries. Critically, this module provides a stable, dense, intermediate reward signal for training the RL agent.   
A Curriculum Reinforcement Learning (CRL) strategy, a two-stage training process ("Explore" and "Converge") designed to mitigate reward sparsity and stabilize the agent. The curriculum progressively introduces more challenging queries, ensuring the agent first learns a general policy before refining it for precision on difficult cases.   

The authors demonstrate that ACQO achieves state-of-the-art (SOTA) performance on complex query benchmarks, including the conversational TopiOCQA and the multi-hop HotpotQA. It significantly outperforms existing baselines, including prompt-based methods (e.g., DeepSeek-V3.1), Supervised Fine-Tuning (SFT), and "Vanilla" RL. A key finding is that the ACQO agent emergently learns retriever-specific reformulation strategies—generating different optimal queries for a sparse (BM25) versus a dense (ANCE) retriever—purely by optimizing for the retrieval-based reward signal.

### Strengths
1. The paper's strength is the discovery and clear demonstration of retriever-specific policy learning. The case study in Appendix B.1 (Figure 5) provides definitive evidence. It qualitatively shows the agent learning to "keyword stuff" for the sparse BM25 retriever. In contrast, for the dense ANCE retriever, it learns to perform semantic decomposition into distinct sub-queries. This insight—that the agent learns what the retriever considers a "good" query—is a fundamental contribution.
2. Another significant achievement is solving the instability of RL for complex query optimization. It correctly identifies this as the key barrier to progress and provides a complete, 3-part solution (AQR+RSF+CRL). The catastrophic failure of the w/o Stage II ablation (Table 5)  proves that "Vanilla RL" is non-viable for this task and that the proposed CRL strategy is the key to unlocking RL for complex, multi-path RAG.

### Weaknesses
1. A critical reviewer could argue that the individual components lack fundamental novelty. Curriculum Learning is a well-established field, rank fusion methods (the paper itself cites RRF ) are common, and RL for query reformulation has been explored. The paper's primary weakness, from this perspective, is that its contribution could be framed as "superior systems engineering" or a novel integration rather than a singular algorithmic breakthrough. While the emergent adaptation is novel, the building blocks are familiar.
2. The RSF module's final step is critically under-specified. It computes two values for each document p: \(P(p)\) (rank-based) and \(S(p)\) (score-based). The paper then states documents are re-ranked "according to the pair \((P(p), S(p))\) in ascending order" (Eq. 2). This is ambiguous. Is this a lexicographical sort? If so, which value is prioritized? \(P(p)\) is a (harmonic) mean of ranks, while \(S(p)\) is a max of scores, which can be on completely different and un-normalized scales (e.g., BM25's unbounded scores vs. ANCE's cosine similarity). A simple lexicographical sort seems brittle and ill-defined. This crucial detail of the fusion algorithm is unclear and hinders reproducibility.

### Questions
1. Please clarify the exact sorting mechanism for the pair \((P(p), S(p))\). Is this a lexicographical sort, and if so, in what order? Or are \(P(p)\) and \(S(p)\) normalized (e.g., min-max or z-score) and combined via a weighted sum? This is a critical, missing detail of the core methodology.
2. The claim of "improved efficiency" is a major selling point but is under-supported by Table 7 (token count). This claim should be substantiated with a proper analysis of: a) end-to-end inference latency (in milliseconds) versus the baselines, and b) total training cost (in GPU-hours) versus Vanilla RL to demonstrate the cost/benefit of the CRL strategy.

### Soundness
3

### Presentation
3

### Contribution
3
