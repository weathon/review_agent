# From Single to Multi-Granularity: Toward Long-Term Memory Association and Selection of Conversational Agents

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
Large Language Models (LLMs) have recently been widely adopted in conversational agents. However, the increasingly long interactions between users and agents accumulate extensive dialogue records, making it difficult for LLMs with limited context windows to maintain a coherent long-term dialogue memory and deliver personalized responses. While retrieval-augmented memory systems have emerged to address this issue, existing methods often depend on single-granularity memory segmentation and retrieval. This approach falls short in capturing deep memory connections, leading to partial retrieval of useful information or substantial noise, resulting in suboptimal performance. To tackle these limits, we propose MemGAS, a framework that enhances memory consolidation by constructing multi-granularity association, adaptive selection, and retrieval. MemGAS is based on multi-granularity memory units and employs Gaussian Mixture Models to cluster and associate new memories with historical ones. An entropy-based router adaptively selects optimal granularity by evaluating query relevance distributions and balancing information completeness and noise. Retrieved memories are further refined via LLM-based filtering. Experiments on four long-term memory benchmarks demonstrate that MemGAS outperforms state-of-the-art methods on both question answer and retrieval tasks, achieving superior performance across different query types and top-K settings\footnote{https://github.com/Applied-Machine-Learning-Lab/ICLR2026\_MemGAS}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents **MemGAS**, a framework for **multi-granularity association and adaptive selection** in long-term conversational memory. It constructs a multi-granularity memory graph via **GMM-based accept/reject association**, routes queries with an **entropy-driven router**, performs **PPR** on the association graph, and applies **LLM-based filtering**. 

Experiments on four benchmarks show consistent gains over single-granularity and structured baselines

### Strengths
1. **Clear motivation**: this paper identifies the limitations of single-granularity memory and fixed-granularity retrieval; frames a noise–information trade-off handled by adaptive routing. 

2. **Association construction**: Form multi-granularity units via LLM-generated summaries/keywords and turn-level splits; use **GMM** to accept/reject similarity vectors and update the association graph.
   - **Granularity routing**: Assign weights by the **Shannon entropy** of similarity distributions at each granularity (lower entropy → higher weight), removing ad-hoc choices.
   - **Retrieval and filtering**: Run **PPR** from the routed prior, take top-K, then apply **LLM filtering** to reduce redundancy. 
3. **Strong empirical results**: Consistent gains on QA and retrieval metrics over baselines (e.g., Full-History, Contriever/MPNet, SeCom, HippoRAG2, RAPTOR, A-Mem), with comparable token/latency budgets.
4. **Ablations and fine-grained analyses**: Removing **GMM**, **PPR**, **routing**, or **associations** leads to clear drops; provides trend analyses over different **Top-K** and **Q types**.

### Weaknesses
1. **Stability of GMM and routing**:  
   - Since GMM is essentially a binary (accept/reject) clustering, please provide a more systematic sensitivity analysis and confidence intervals on how the choice/adaptation of the number of components and the temperature λ affect entropy and the routing weights.  
   - Report robustness under different encoders (beyond Contriever) and alternative similarity measures.

2. **Fairness and comparability**:  
   The implementation unifies the generator as **gpt-4o-mini** and fixes a top-3 session pre-retrieval, which may bias comparability; please add **controlled experiments with multiple generators/retrievers** and learning curves.  
   **New baseline suggestion**: include **COMEDY** (*COLING 2025 Oral*) as a baseline, and revise the corresponding description concering **COMEDY** in the paper.

3. **Cost and scalability**:  
   Although average latency is comparable, as the memory store grows, the **complexity of PPR on large graphs**, the cost of frequent GMM updates, and the inference overhead of LLM filtering should be presented as **data-scaling curves** (number of samples/edges/graph diameter → latency and token cost), along with incremental/approximate strategies.

4. **Evaluation protocol**:  
   For some structured methods the retrieval evaluation is “undeterminable,” and retrieval on LongMTBench+ is missing. Please clarify in the appendix the sources of non-comparability (implementation differences, labeling criteria, task incompatibilities), and provide alternatives (e.g., human spot-checking on a sampled subset, or rerunning a unified comparable subset).

5. **Adaptive Top-K**:  
   Since large K introduces noise (F1 drops), consider making K part of the routing/uncertainty-driven **dynamic selection** (e.g., entropy/CI-based adaptive K coupled with redundancy-removal thresholds), and report the gains and stability.

### Questions
1. **Temperature λ**: how selected; any adaptive scheme or theoretical bounds? 
2. **PPR hyper-parameters**: restart prob/seed selection and convergence on large graphs? 
3. **Graph maintenance**: pruning/forgetting policies to avoid graph bloat and stale memory?  
4. **Judging protocol**: variance control and human calibration for GPT4o-as-Judge?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposed MemGAS, a novel framework for long-term memory construction and retrieval that integrates multi-granular memory units and enables adaptive selection and retrieval. It is motivated by two flaws of existing methods: 1) insufficient multi-granulartiy memory connection; and 2) lack of adaptive multi-granular memory selection. To solve (1), it complements session-level memory with turns, summarize and keywords, and connect these granularities across different sessions based on Gaussian Mixture Models. For (2), it uses similarity between query and each granularity, and run pagerank algorithm to find top-k relevant memories, then feed refined memories by LLM to generate the response. Generally, the paper is well-motivated, well-supported, easy-to-follow. The experimental results are solid and comprehensive in terms of effectiveness and efficiency.

### Strengths
1. the paper is well-motivated, well-supported, easy-to-follow.
2. the experimental results are solid and comprehensive in terms of effectiveness and efficiency.

### Weaknesses
no significant weakness identified

### Questions
1. it is better to analyze efficiency fators for each module, i.e., which module cost most time?
2. it is better to have some in-depth analysis about each module, i.e., how many samples gets different top-k results before and after you run page rank algorithm, and any common patterns?

### Soundness
4

### Presentation
4

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
he paper introduces MemGAS, a new framework for managing long-term memory in conversational agents. The proposed system addresses limitations of previous models by using multi-granularity memory units and an adaptive granularity selection mechanism. MemGAS enhances the memory retrieval process through Gaussian Mixture Models (GMM) to cluster and associate new memories with historical ones. Additionally, an entropy-based router is employed to select the optimal granularity for each query, balancing the trade-off between information completeness and retrieval noise. The framework is evaluated across several long-term memory benchmarks, showing significant performance improvements over existing methods in both question answering (QA) and retrieval tasks.

### Strengths
- MemGAS introduces a novel multi-granularity memory framework with Gaussian Mixture Models for memory association and an entropy-driven router for adaptive granularity selection, surpassing existing single-granularity methods.
- The entropy-based router allows MemGAS to dynamically select the most relevant granularity for each query, enhancing retrieval efficiency and reducing noise. This adaptability is a clear improvement over fixed-granularity methods, which can fail to balance information completeness with noise.
- The use of GMM for clustering historical memories into relevant and irrelevant sets improves the consolidation of new and old memories, ensuring more coherent long-term memory management.
- MemGAS addresses key challenges in long-term memory management for conversational agents, offering improvements in both retrieval accuracy and efficiency, with strong potential for real-world applications.

### Weaknesses
- The level of edge construction is inconsistent throughout the paper. Line 184-185 and the upper-right part of Figure 2 suggest that each granularity of each memory is treated as a node. Still, Equation 5 computes the weight on each memory across different granularities, which indicates that the PPR is run on a memory association graph where each memory $M_i$ is a node. Additionally, the word "vectors" in line 157 is also ambiguous, without pointing out whether the entire memory bank $\mathcal{M}_{\text{cur}}$ or each element of $\mathcal{M}_{\text{cur}}$, or even each granularity of each element is encoded as a vector.
- Several substantial experiments lack sufficient explanation. In Sec. 3.4, the classification of "query types" and the exact module and task for "top-K retrieval settings" seem ambiguous to me. The "Combination" and "Optimal Selection" in Tables 6 and 7 in Appendix C lack explanation. The experimental setting of error analysis in Appendix G is also unexplained.
- The conclusion drawn in Line 368 that "we keep average tokens and latency close to lightweight retrievers and below HippoRAG2 and A-Mem" does not align with the experimental results. The number of average tokens used by MemGAS is larger than that of HippoRAG2 in LongMemEval-s, and not significantly smaller in other datasets.
- There are a few typos and grammatical errors in the paper: the last $\alpha_s$ should be $\alpha_u$ in Line 135; "RAPOTR" -> "RAPTOR" in Line 269, and the sentence in Line 474 is incomplete (or removing "which"?), etc..
- The theoretical analysis in Appendix H is a bit coarse in writing, with quite a few writing hints (sketch) not removed.

### Questions
1. Why do session $S_i$ and the segmented turns $T_i$ appear together in each memory bank? Are they encoded to the same vector by Contriever?
2. All metrics are automatic (F1, GPT-4o-as-judge...). Were any human evaluations conducted to assess response coherence, personalization, or perceived helpfulness? As automatic metrics can be misleading in dialogue settings.

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
3

### Summary
The paper takes the multi-granularity into consideration for long-term memory construction and retrieval for the first time. The memory of each session is partitioned into 4 granularities: sessions, turns, keywords, and summary. The paper proposes a training-free method, MemGAS, which consists of a GMM clustering model to construct edges for the memory association graph, an entropy-based router to weight the granularities for the entire memory bank, a PPR method to select the candidate context, and an LLM-based filter to extract the related content. Experiments show that MemGAS outperforms several recent baselines in QA and retrieval tasks with a similar number of tokens used, and a very small trade-off in latency.

### Strengths
1. The paper presents a novel idea by utilizing the multi-granularity of each session to learn a structured long-term memory bank and retrieve relevant information to queries. 
2. The GMM distribution assumption of the similarity scores in dynamical memory association is concise and flexible, compared to other oversimplified classification models. 
3. The core algorithms, GMM clustering and PPR, are model-based approaches that ensure the entire model is training-free and preserves interpretability.
4. The trade-off between performance and latency is impressive, making the model competitive or superior to the baselines.
5. The presentation is elegant and effectively showcases the paper's methodology and results.

### Weaknesses
1. The paper focuses on four fixed granularities (session, turn, summary, keyword) created from a session chunk. This design feels a bit rigid. The core weakness identified in prior work is often the choice of fixed segmentation (e.g., semantic-based clustering/segmentation). Why were only these four chosen? Other approaches integrate hierarchical structures (RAPTOR, MemTree). It remains unclear that whether the simple summary/keyword levels fully address the limitations of hierarchical knowledge organization. The comparison to other structured RAG models like RAPTOR and HippoRAG 2 is strong on end-task performance, but a deeper discussion and ablation of why this specific set of granularities is better than a structured tree/graph is missing.
2. The authors reviewed a number of studies on memory segmentation in the introduction (Sec. 1, Paragraph 2), but a recent hierarchical memory architecture [1] is not included and compared in experiments.
3. The paper did not provide an ablation study on the LLM information filter. The relevance of the retrieved sessions and the question is not evaluated; thus, whether the LLM filter or the PPR algorithm contributes to the better performance is unclear. 
4. Minor typos:  In Line 269, "RAPOTR" should be "RAPTOR".
5. It seems the anonymous code link provided in the article cannot display the file content ("The requested file is not found."), only the file structure can be seen.

Reference:

[1] Sun, H., & Zeng, S. (2025). Hierarchical memory for high-efficiency long-term reasoning in llm agents. *arXiv preprint arXiv:2507.22925*.

### Questions
1. The metric "Avg. Tokens" in Table 1 seems ambiguous regarding token consumption. Since MemGAS leverages LLM calls for multi-granularity memory generation, redundancy filtering, and the final QA generation. The authors could clarify: Which specific stage does this token count refer to? How is this "Avg. Tokens" measure normalized to be directly comparable to the token counts of single-granularity baselines that may not utilize LLM filtering? Can the authors provide an analysis on token consumption breakdown? 
2. The importance of each granularity could vary for each memory bank $M_i$. Why does Equation 4 assign an overall weight to each granularity for the entire memory bank instead of each $M_i$?

### Soundness
2

### Presentation
3

### Contribution
2
