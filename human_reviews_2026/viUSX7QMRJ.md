# Enhancing Agentic Textual Graph Retrieval with Synthetic Step-wise Supervision

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 2

## Abstract
A significant portion of real-world data is inherently represented as textual graphs, and integrating these graphs into large language models (LLMs) is promising to enable complex graph-based question answering.
However, a key challenge in LLM-based textual graph QA systems lies in graph retrieval, i.e., how to retrieve relevant content from large graphs that is sufficiently informative while remaining compact for the LLM context. 
Existing retrievers suffer from poor performance since they either rely on shallow embedding similarity or employ interactive retrieving policies that demand excessive data labeling and training cost.
To address these issues, we present Graph-S$^3$, an agentic textual graph reasoning framework that employs an LLM-based retriever trained with synthetic stepwise supervision. 
Instead of rewarding the agent based on the final answers, which may lead to sparse and unstable training signals, we propose to closely evaluate each step of the retriever based on offline-extracted golden subgraphs.
Our main techniques include a data synthesis pipeline to extract the golden subgraphs for reward generation and a two-stage training scheme to learn the interactive graph exploration policy based on the synthesized rewards.
Based on extensive experiments on three common datasets in comparison with seven strong baselines, our approach achieves an average improvement of 8.1\% in accuracy and 9.7\% in F$_1$ score. The advantage is even higher in more complicated multi-hop reasoning tasks. Our code will be open-sourced.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents an agentic graph retriever for question answering by training large language models (LLMs) to interact with knowledge graph structures.
The authors develop a data synthesis pipeline that generates exploration trajectories for supervised fine-tuning and reinforcement learning. 
To address the sparse reward problem in reinforcement learning, they assign rewards to individual actions rather than only to final outcomes.
The approach is evaluated on three QA benchmarks and compared against five baseline methods.

### Strengths
1. The proposed method demonstrates gains over baselines across multiple benchmarks, with improvements that appear consistent across diverse task characteristics. This suggests the approach generalizes well rather than overfitting to a single task structure.
2. The stepwise reward assignment directly addresses a known challenge in RL-based retrieval where delayed feedback makes credit assignment difficult, and by rewarding intermediate exploration steps, the model can learn more efficiently which graph traversal actions are productive.
3. The paper is clear and well-written.

### Weaknesses
1. The novelty and technical depth of this paper is very limited. These are very common strategies for training an LLM agent, and this paper simply applies them to this graph retrieval application. (Q1, Q2)
2. The experiments should summarize the insights instead of just reporting the numbers, i.e., what can we learn from them? Additional studies such as error analysis can be very helpful. (Q3, Q4)

### Questions
1. The paper does not adequately explain what is technically novel about the proposed approach, which appears to combine standard techniques applied to graph retrieval. If the contribution is simply combining existing methods, you need to justify why this combination is non-trivial or represents a novel insight. Specifically, what challenges arise when applying these techniques to graph retrieval that required new solutions or adaptations? The related work section should explicitly identify similar works that combine RL with agentic retrieval or that use stepwise supervision for structured exploration, and then clearly articulate what prevented those approaches from being directly applied to your setting.
2. Section 3 introduces design choices without sufficient motivation or justification. For example, in Section 3.1 graph data synthesis, the authors describe generating graph exploration trajectories but do not explain what makes this challenging. It appears to be simple random traversal without any careful strategy, and no specific design considerations are needed when using it as agent training trajectories. Each design choice should include a clear problem statement explaining what doesn't work with existing approaches, followed by your solution and why it is better.
3. Section 4 only reports numbers without deeper analysis. For example, it would be very helpful to know what questions cannot be solved by the proposed method, and what questions can be solved by the proposed method but not by the baselines.
4. Please do not limit the analysis to only the provided examples. The more analysis the authors can provide, the better reviewers can understand the contributions. The paper would largely benefit from deeper analysis beyond the quantitative results.

### Soundness
2

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
This paper presents Graph-S3, an agentic textual graph retrieval framework that enhances large language models’ reasoning ability over textual graphs via synthetic stepwise supervision. The key novelty lies in introducing a data synthesis pipeline to automatically generate “golden subgraphs” for fine-grained reward signals, addressing the sparsity and instability of outcome-based RL in graph reasoning. A two-stage training paradigm—supervised fine-tuning followed by reinforcement learning (GRPO) with process-level rewards—is adopted to improve reasoning robustness and exploration efficiency. Experiments on WebQSP, CWQ, and MetaQA show an average +8.1% accuracy and +9.7% F1 improvement over seven strong baselines, including LightRAG, G-Retriever, and KG-Agent. The ablation studies convincingly demonstrate the necessity of stepwise supervision, interactive retrieval, and trajectory refinement.

### Strengths
S1. The paper identifies and addresses the practical challenge of sparse supervision and expensive expert annotation in interactive graph retrieval, providing an automated solution that reduces annotation costs. S2. Well-motivated two-stage training: The combination of SFT for basic navigation followed by RL with stepwise rewards is conceptually sound and empirically validated through ablations showing both stages contribute meaningfully. 

S3. The paper evaluates across three datasets (WebQSP, CWQ, MetaQA) with multiple generator backbones (Qwen3-8B, Llama3.1-8B, Finetuned-8B), demonstrating consistency. The ablation studies are complete, the structure is clear, and the effectiveness of the proposed method is demonstrated. 

S4. The paper clearly provides implementation details and training hyperparameters, and promises to open-source code. The appendix includes prompt templates and reasoning examples, ensuring good reproducibility.

### Weaknesses
**W1.** Insufficient comparison with baselines. The comparison with existing baselines is insufficient. Specifically, the paper lacks comparison with several advanced graph-based RAG baselines such as SubgraphRAG and GNN-RAG, which are state-of-the-art methods in graph-based question answering. Furthermore, to clearly isolate the benefits of the proposed approach, the comparison with the G-retriever baseline should include a version that has been fine-tuned using LoRA, which would provide a stronger and more relevant comparison point for methods involving model adaptation.

**W2.** The novelty of the proposed method appears slightly limited. While the paper introduces improvements in the generation of training signal data and the reinforcement learning strategy, the core idea closely resembles the Process-Reward Modeling (PRM) research from recent years, such as the work conducted by OpenAI-o1. The authors should more clearly articulate the fundamental difference and the unique technical contributions that distinguish this work from existing PRM and process-supervision techniques applied in complex reasoning settings.

**W3.** Lack of analysis on deployment feasibility and cost. The paper fails to provide a comparison of computational cost and time performance during the live question-answering phase. It leads to a lack of assessment regarding the method's feasibility in actual deployment scenarios. Additionally, the paper does not report the total number of tokens spent on the initial data synthesis pipeline, which raises concerns about the overall resource intensity.

**W4.** The lack of analysis of performance drop on multi-hop reasoning. Intuitively, this approach should be well-suited for complex multi-hop reasoning tasks due to the nature of stepwise supervision. However, the accuracy results on MetaQA 3-hop drop significantly (though still outperforming other reported baselines). It is better for the authors to analyze the reasons for this sharp decline in performance when scaling to very long reasoning chains. This analysis should specifically investigate the types of challenges encountered, and a thorough examination of the causes of failure samples is needed to pinpoint the specific limitations of the current mechanism in highly complex, multi-step scenarios.

### Questions
Q1. In the ablation study, the performance when the interactive inference module is removed is slightly lower than the RAG/1hop and G-retriever methods. If these two methods were combined with interactive inference, would they also achieve a good level of accuracy?

Q2. Regarding the trajectory refinement algorithm, could you provide algorithmic details for trajectory refinement (Section 3.1.3)? Specifically, could you show examples of trajectories before and after refinement to illustrate exactly what is being removed?

Q3. What is the cost of data synthesis for each dataset? Could you analyze potential biases in the GPT-4o exploration strategy (and how the synthesis quality compares to human-annotated trajectories, if available)?

Q4. Regarding the scalability and generalization ability of the fine-tuned model, how does its performance fare when migrating to an untrained domain dataset? The 3-hop MetaQA results (14.70% accuracy) suggest challenges—what causes this performance degradation? What is the maximum reasoning depth your method can effectively handle?

### Soundness
3

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
4

### Summary
This paper proposes Graph-S3, an agentic textual graph retrieval framework that enables LLMs to
perform interactive, multi-step graph exploration for question answering. The core idea is to train the
retriever with synthetic stepwise supervision generated from automatically extracted golden subgraphs.
This paper proposes an automatic pipeline for synthesizing the stepwise supervision data. Then the
trained is trained using a two-stage training paradigm: (1) supervised fine-tuning (SFT) on synthetic
exploratory trajectories and (2) reinforcement learning (RL) with stepwise rewards derived from refined
subgraphs.

### Strengths
The paper addresses the underexplored area of agentic textual graph retrieval, moving from one-shot
similarity retrieval to structured, interactive reasoning. The proposed synthetic stepwise data synthesis
pipeline effectively solves the issue of sparse and unstable outcome-based rewards in RL relevant
methods. The paper conducted comprehensive experiments over multiple datasets and baselines and
conducted detailed ablation studies to analyze the properties of the proposed method.

### Weaknesses
1. While the synthetic stepwise reward concept is valuable, the reward design is relatively heuristic (rule-based
correctness scoring). The RL setup does not introduce fundamentally new algorithmic insights beyond
adapting GRPO to this domain. The method relies on GPT-4o-generated trajectories, which might limit
reproducibility and scalability for open-source or resource-constrained settings. And there is no detailed
statistics about the synthesized dataset introduced in the paper. 
2. Lack of novelty. This paper mainly introduces the fine-tuning of LLM to serve as retriever for graph QA. Existing
domain has explored similar approaches of training LLMs as retrievers. This paper simply leverage 
similar ideas to address the problem of graph QA.
3. The experiments are based on weak base models such as llama 3.1 8b. More advanced and larger models
can help strength the quality of paper. 
4. The authors only compare the efficiency of retrieved graph, while training a retriever and use it for inference
can introduce a lot of time complexity. The authors should conduct experiments to validate this. In addition, SFT 
and RL can introduce a lot of complexity.
1. Line 101 - 104 is inconsistent with your contribution summary. You only mentioned how to build the
stepwise reward but didn't mentioned specifically your second step in this two-step training pipeline.
2. The citation for RL algorithms are missing, e.g., GRPO.
3. In section 3.1.3, there is only the formal definition and the general goal of this trajectory refinement
process without the detailed content about how to achieve it. Specifically, how to find the shortest
feasible subsequence, and whether the incoherence of the trajectory results from removing
redundant detours will cause problems, or do you aggregate the sub-trajectories after removing.

### Questions
see above.

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
3

### Summary
This paper tackles a trending topic on how to leverage graph data to effectively improve the QA system. The main bottleneck lies on the mis-alignment between the relational data stucture and unstructured pre-training corpus. This paper proposes the mitigate this gap by synthesizing an agentic graph retrieval trajectory data and leverage them by first supervised fine-tuning then stepwise RL with process reward. It adopts a verifiable reward for both outcome and process reward, the experimental performance shows improvements on multiple QA benchmarks.

### Strengths
1. The obtained $\text{Graph-S}^3$ achieves better performance with just 10% of the retrieved triples, which demonstrates the effectiveness on selecting the correction actions.

### Weaknesses
1. Although the paper discusses the comparison of the retrieved graph size between different baselines, it's also important to illustrate the token efficiency and reasoning steps of the model. If the proposed model does not yield great token efficiency, the overall efficiency and cost of the system could still be higher. 

2. The novel is fairly limited, this paper targets at the scalability issue of the agentic graphrag system by synthesizing stepwise training data. First of all, it has been a popular method to use RL especially RLVR on KGQA. This paper didn't differ with the existing algorithms with clear motivation or mitigating the challenge in a smart enough way. I am still concerned about the generalization capability across different KGs?

### Questions
1. Why is the reported performance on WebQSP and CWQ lower than previous arts? Existing work already achieves 80%+ accuracy on WebQSP, can author elaborate on this? 

2. Does $\text{Graph-S}^3$ generalize to out-of-distribution data? E.g. the relation schema of the graph has never been seen in the training data

### Soundness
2

### Presentation
3

### Contribution
2
