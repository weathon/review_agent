# Beyond Chunks and Graphs: Retrieval-Augmented Generation through Triplet-Driven Thinking

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 4

## Abstract
Retrieval-augmented generation (RAG) is critical for reducing hallucinations and incorporating external knowledge into Large Language Models (LLMs). However, advanced RAG systems face a trade-off between performance and efficiency. Multi-round RAG approaches achieve strong reasoning but incur excessive LLM calls and token costs, while Graph RAG methods suffer from computationally expensive, error-prone graph construction and retrieval redundancy. To address these challenges, we propose T$^2$RAG, a novel framework that operates on a simple, graph-free knowledge base of atomic triplets. T$^2$RAG leverages an LLM to decompose questions into searchable triplets with placeholders, which it then iteratively resolves by retrieving evidence from the triplet database. Empirical results show that T$^2$RAG significantly outperforms state-of-the-art multi-round and Graph RAG methods, achieving an average performance gain of up to 11% across six datasets while reducing retrieval costs by up to 45%.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes $T^2RAG$, a lightweight and efficient retrieval-augmented generation framework that uses a graph-free knowledge base of atomic triplets. $T^2RAG$ decomposes questions into searchable triplets with placeholders and iteratively resolves them via evidence retrieval, reducing reliance on costly multi-round interactions or complex graph construction. Experiments show it outperforms state-of-the-art RAG methods, offering a more effective and efficient solution for knowledge-intensive reasoning.

### Strengths
* Using triplet with placeholder to represent the required information is new and interesting to me
* The experiments are extensive

### Weaknesses
* The triplet update itself is not complicated, but in Step 2.3, the use of mathematical notation makes the process significantly harder to follow.
* T^2RAG is actually a multi rould RAG, but only one multi round baseline is compared, maybe some more multi round baselines should be added.

### Questions
* In section 4.3, line 262, the authors convert the triplet into query by simply concatenating the elements without the placeholder, but this query will be quite different from the normal ones, will the dense retriever perform suboptimally due to out of domain?
* I do not quite understand why using triplet can reduce the number of llm calls. For each retrieval, the method still requires an LLM call to update the triplet, and other approaches can similarly update the next subquery after each retrieval.
* Instructing the LLM to generate triplet seems to be difficult, how does small models like Llama 8B and Qwen 7B performs

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes T2RAG, a retrieval-augmented generation framework that replaces chunk-level retrieval with a graph-free store of atomic triplets. An offline stage distills a corpus into triplets and embeds them with FAISS for fast lookup. At inference time, an LLM decomposes the user question into placeholder triplets, iteratively resolves missing arguments by retrieving candidate triplets, and finally answers the question using the resolved evidence. Across six datasets, the authors report performance gains over selected multi-round and graph-based RAG baselines while reducing retrieval costs.

### Strengths
1.The work proposes triplet-driven thinking to reduce token footprint and simplify graph dependencies in the RAG pipeline.

2.The split between offline distillation (triplet extraction/embedding) and online iterative resolution is clean and easy to reproduce, with a plausible path to system engineering and deployment. The placeholder-resolution idea encourages explicit evidence grounding rather than opaque, multi-turn prompts; this has the potential to reduce uncontrolled drift.

3.The paper claims consistent improvements on multiple datasets along with cost reductions, suggesting that lighter-weight semantic atoms can be competitive with heavier chunk or graph pipelines.

### Weaknesses
1.The settings of experiments appear to resolve to entity answers or facts naturally expressible as single triplets. This inherently benefits triplet retrieval and makes direct comparisons to multi-round RAG and GraphRAG less fair. The paper should include tasks where answers are compositional, multi-hop, or non-entity (rationales, procedural steps) to validate generality.

2.The placeholder triplet decomposition and iterative resolution are described verbosely, but the novelty claim would be stronger with: (a) ablations comparing against simpler query rewriting + re-ranking loops; (b) a version that uses slot-filling over chunks without pre-extracted triplets; and (c) error-driven analyses showing why placeholders materially outperform strong query planners.

3.Ablation results indicate that raw chunks still matter during retrieval, and that intermediate LLM calls over these chunks increase token usage and hallucination risk. This undercuts the central claim that “replacing chunks with triplets” consistently reduces cost and risk. 

4.There is no robustness study for incorrect or incomplete query-triplet decomposition. Since downstream retrieval depends on these slots, small decomposition errors could derail performance on complex questions. 

5.Only a few representatives of multi-round and GraphRAG are included. More representative baselines should be added to support the “state-of-the-art” claim. Also report exact prompting/cost budgets for each baseline to ensure fairness.

6.Triplet-first retrieval may struggle on compositional, multi-hop, or explanatory tasks where answers are not atomic entities, which limits the practical value of T2RAG. 

7.Minor typos: Line 280: “and and.” Line 415: “Figure 5” should be “Table 2”.

### Questions
Please refer to Weaknesses.

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
4

### Summary
The paper proposes T2RAG, a novel framework for Retrieval-Augmented Generation (RAG) that improves multi-hop question answering by leveraging atomic triplets as the fundamental unit of retrieval and reasoning. Unlike traditional chunk-based or graph-based methods, T2RAG avoids costly graph construction and reduces token overhead by focusing on triplets with placeholders, which are resolved iteratively through an adaptive retrieval process. The paper demonstrates that T2RAG outperforms state-of-the-art RAG systems, achieving an 11% performance gain on various QA datasets while reducing retrieval costs by up to 45%.

### Strengths
1.	The paper proposes T2RAG, an innovative framework that leverages atomic knowledge triplets to go beyond traditional chunk- or graph-based retrieval-augmented generation (RAG) approaches.

2.	The authors demonstrate that T2RAG yields strong empirical gains across multiple QA datasets, improving average performance by 11% while reducing retrieval cost by up to 45%, which shows both effectiveness and efficiency.

### Weaknesses
1.	The related-work section could be strengthened. Although GraphRAG and multi-round RAG methods are discussed, the paper lacks comparison with other triplet-based retrieval methods such as SubgraphRAG, which also uses triplets for retrieval.

2.	Experimental parameter choices are not always fair. When comparing multi-round methods to direct chunk-retrieval baselines there is an information imbalance: some settings allow more total retrieved content than others. 

3.	Several implementation details are unclear and need to be spelled out — for example, the paper does not explicitly state how the system behaves when there are no searchable triplets available.

### Questions
1.	The related-work section would benefit from a broader discussion of other triplet-based retrieval approaches. In particular, SubgraphRAG (Li et al., 2025) also employs triplet structures for retrieval and could serve as a valuable point of comparison. Including a discussion of such methods, as well as adding SubgraphRAG as an experimental baseline for both accuracy and retrieval efficiency, would help position T2RAG’s contribution more clearly within the current RAG landscape.

**Ref:** Simple is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation

2.	Although the paper highlights avoiding graph-construction overhead, it still depends on LLMs to extract triplets, which is conceptually similar to GraphRAG’s reliance on LLMs. Could the authors provide a more detailed analysis of the triplet-extraction step? In particular, do different extraction prompts significantly affect performance?
3.	How does the system handle queries whose triplets are all fuzzy (i.e., contain two or more placeholders)? If there are no searchable triplets, retrieval may fail — is there a fallback or mitigation strategy for this case?
4.	Regarding the fairness of comparisons: multi-round settings use N = 3, k = 5, which can retrieve up to 15 chunks for answering, while the direct chunk-retrieval baseline sets k = 5. This creates unequal information budgets. The authors should consider reporting the average number of chunks T2RAG actually uses and using that as a cap for other methods to ensure parity.
5.	When reporting token and time overheads for T2RAG, does the measurement include the cost of repairing or refining triplets, or does it only account for the final answer-generation phase? Please clarify.
6.	Typo: A space is missing in line 145 after "into multi-round".

If the authors can appropriately address the above issues, I would consider raising my score.

### Soundness
3

### Presentation
2

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
This paper proposes T2RAG, a framework to conduct GraphRAG without actually building knowledge graphs (KGs), which may help avoid the costly, time-consuming, and error-prone process of offline KG construction. This is done by extracting triplets from the corpus and then storing these triplets directly.
To store these triplets, T2RAG converts each triplet into natural language, embeds them as dense vectors, and stores them in vector databases. Given a question, the LLMs would be asked to decompose the query into triplets, and these triplets are then embedded and thus used to search relevant triplets in the triplet vector datasets to resolve the questions. Experiments show that T2RAG can improve some QA datasets significantly.

### Strengths
- The idea of using purely triplet-based databases for multi-hop reasoning with RAG is interesting.
- The proposed online retrieval method is also interesting. With those placeholders, the query triplets can be semantically closer to the relevant triplets in the vector database.
- For QA datasets that do not require that many hops, the proposed method may improve accuracy significantly.

### Weaknesses
- One reason to use graph data is that it can be more friendly when dealing with multi-hop reasoning. Building databases without entity links but with only triplets essentially gives up graph structures during offline building. So, during the online process, the method would, in principle, need to somehow reconstruct the multi-hop links. 
    - This can be especially hard for long-hop reasoning, and this may also be the reason why T2RAG mostly improves datasets with fewer hops, e.g., PopQA, 2Wiki.
    - So, it would be more interesting to see whether T2RAG can still perform decently for reasoning with even longer hops. And in such cases, what the costs would be.
- In the meantime, even though offline building of KGs can have some issues, storing things in triplet embedding format would also cause issues, especially when dealing with super large corpora.
    - For example, the number of triplets can grow fast, and the embedding quality would then affect the retrieval quality more significantly. With methods like FAISS for efficiency, more noise would also be introduced.
    - Therefore, it should also be required to test T2RAG on super large corpora.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2
