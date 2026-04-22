# Graph of Agents: Principled Long Context Modeling by Emergent Multi-Agent Collaboration

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
As a model-agnostic approach to long context modeling, multi-agent systems can process inputs longer than a large language model's context window without retraining or architectural modifications. However, their performance often heavily relies on hand-crafted multi-agent collaboration strategies and prompt engineering, which limit generalizability. In this work, we introduce a principled framework that formalizes the model-agnostic long context modeling problem as a compression problem, yielding an information-theoretic compression objective. Building on this framework, we propose Graph of Agents (GoA), which dynamically constructs an input-dependent collaboration structure that maximizes this objective. For Llama 3.1 8B and Qwen3 8B across six document question answering benchmarks, GoA improves the average $F_1$ score of retrieval-augmented generation by 5.7\% and a strong multi-agent baseline using a fixed collaboration structure by 16.35\%, respectively. Even with only a 2K context window, GoA surpasses the 128K context window Llama 3.1 8B on LongBench, showing a dramatic increase in effective context length.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Graph-of-Agents (GoA), a method that begins by semantically clustering text chunks. For each cluster, it then constructs a path graph by assessing the mutual information between the chunks and the specific input question. An agent component sequentially summarizes the documents along this path graph. Finally, the aggregated summary is passed to a Large Language Model to generate the answer.

### Strengths
The core strength of this work lies in its input-dependent graph construction. From a compression perspective, this approach dynamically measures the relevance between the question and the incrementally built summary, aiming to achieve efficient information compression and long-context question answering.

### Weaknesses
1. Despite the "Graph" nomenclature, the resulting structure is a "path graph"—a linear list. This architecture is structurally similar to a "Chain of Agents," which significantly diminishes the novelty and the claimed advantages over simpler, chain-based methods.

2. The path graph construction process appears computationally intensive. Even with a greedy algorithm, adding each new node incurs a computational complexity of o(N), where N is the total number of chunks. This overhead could be very high for long contexts.

3. The algorithm's objective function is based on the similarity between the LLM's summary and the question. However, the underlying assumptions seem overly idealized, casting doubt on the method's practical feasibility. For instance, LLM may inevitably omit critical, question-relevant details during its summarization pass. Furthermore, GoA also does not account for cases requiring the synthesis of information from multiple, non-sequential chunks.

4. The entire graph is question-dependent and offers no reusability. A new graph must be constructed de novo for every single query, which severely limits the method's practicality and scalability in real-world applications.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper reframes model-agnostic long-context modeling as a compression problem and derives a principled objective via mutual information, then instantiates it with GoA—a dynamic, input-dependent linear forest of collaborating worker LLMs plus a manager. The graph is built by clustering semantically related chunks and greedily ordering within each cluster to maximize semantic alignment between the evolving summary and the query, motivated by InfoNCE/PMI connections. Empirically, on six LongBench QA tasks with Llama-3.1-8B and Qwen3-8B, GoA improves averaged F1 over strong RAG and Chain-of-Agents baselines and, notably, with a 2K window surpasses Llama-3.1-8B with a 128K window. Figures and Table 1 support these gains and ablations show the importance of semantic, contextual next-chunk selection.

### Strengths
* This paper is clear writing and easy to follow.

* Across two backbones and six datasets, GoA consistently matches or beats RAG/CoA at the same context length; with 2K context it even edges the 128K vanilla model

### Weaknesses
* Experiments focus on LongBench QA; claims about “long-context modeling” could be tested on repo-level code understanding, or agentic tasks to assess generality of the compressor.

* While linear forests allow parallelism, the paper does not quantify total token usage, wall-clock latency, or budget vs. RAG/CoA under equal compute constraints. (Algorithm suggests multiple LLM calls.)

* Appendix shows k-medoids underperforms k-means/spectral; the main results fix 𝑘=4 and a particular clustering, leaving open how robust performance is under automatic selection of 𝑘/method.

### Questions
* Did you try simple look-ahead (beam) or submodularity analyses to bound the greedy gap? Even small beams could test whether myopia matters in multi-hop cases.

* Under a fixed token/latency budget, how do GoA, RAG (with reranking), and CoA compare in end-to-end wall-clock and dollar cost, given multiple worker calls + manager? A cost-efficiency plot would be valuable.

### Soundness
3

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
3

### Summary
In this paper, the authors propose Graph-of-Agents (GoA), a framework for long context question answer tasks. In GoA, the input text is first split into chunks, then clustered according to their embedding. Next, each worker agent working on each cluster is given these chunks and is asked to summarize them in order of similarity of the chunk and the current summary and query. Finally, a manager agent reads the results and generates the answer. Results on 6 datasets with 2 models show that the proposed method surpasses RAG and CoA baselines on average.

### Strengths
1. Long context task is an important and urgent issue in LLMs.
2. The results show that the average performance is better than baseline.

### Weaknesses
1. The biggest concern is novelty and contribution. Compared with the baseline, the new component is basically clustering the chunks of document into different groups and assigning them to multiple Worker agents. However, such a method is widely used in various papers (document clustering works). Besides, cluster-then-process only changes the reading order of the baseline method, which it difficult to recognize it as a significant contribution of the method.

2. The improvement is unjustified. Since many results are lower than the baseline, it is important to carefully conduct an analysis of why it is lower. On the other hand, some case studies on why the proposed method is better than the baseline are also needed. Without these analyses, it is not very convincing to believe the proposed method works well.

3. The experiments are not solid. Some baselines can be added to enhance the results. Such as LongAgents, which is also a tree structure. Also, it is not very clear about the effectiveness of each component in the framework, such as an ablation study for random order GoA or random clustered GoA. Furthermore, the motivation for using clustering of documents can be tested, for instance, whether clustering can break the natural order of the document, and it is not certain if there really exist some topics for each cluster, or just a mixture of things in each chunk.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Graph of Agents (GoA), a multi-agent framework for long-context reasoning in LLMs. Instead of a fixed Chain of Agents (CoA)—a serial pipeline where each agent summarizes one chunk—GoA dynamically constructs a graph (specifically, a linear forest) of agents based on semantic similarity among chunks. Each subgraph processes a semantically coherent subset of text, and all run in parallel; their outputs are later combined by a manager agent. Theoretical grounding is provided: for a Bayes-optimal predictor, maximizing expected log-likelihood equals maximizing mutual information between compressed input and answer, motivating semantic information preservation as the key compression criterion. Empirically, GoA outperforms baselines (vanilla, RAG, CoA) on LongBench QA tasks using small-context LLMs, suggesting improved effective context length. However, dynamic graph construction introduces non-trivial overhead not measured in latency terms, leaving open the practical speed-accuracy trade-off.

### Strengths
+ principled design: it derives the agent architecture from an information-theoretic formulation rather than heuristics. The proposed GoA generalizes prior CoA structures, introducing dynamic, input-dependent collaboration among agents. This enables better modeling of non-linear information dependencies in long documents. 
+ Empirical results demonstrate consistent accuracy improvements across six QA datasets, even outperforming models with much larger context windows. The authors also provide meaningful ablations (embedding models, clustering strategies, contextual vs. lexical search) that validate design choices. The method’s modularity makes it model-agnostic—applicable to any base LLM without fine-tuning—and it is theoretically elegant in unifying mutual-information maximization with graph-structured collaboration.

### Weaknesses
- GoA’s performance depends heavily on the semantic embedding and clustering quality; if embeddings fail to capture query relevance, summarization paths degrade. 

- The dynamic graph construction—computing embeddings, clustering, building subgraphs, and launching multiple agent calls—introduces computational overhead. Now in section 4.1, the authors claim that Parallelization "significantly
reducing latency". I find this hard to believe for the reason mentioned. For this claim, I would encourage the authors to to report latency or time-to-first-token measurements.

### Questions
Proposition 1 Motivation: The Bayes-optimal predictor result equating log-likelihood and mutual information seems mainly illustrative. Is there a deeper analogy to be interpreted?

Clarification: Section 5.1 datasets mentions "The datasets range from 3.6K to 18K tokens,..", but table-1 references 128-K?

### Soundness
3

### Presentation
3

### Contribution
3
