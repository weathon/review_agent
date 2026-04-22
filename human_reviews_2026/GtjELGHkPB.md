# Weak-to-Strong GraphRAG: Aligning Weak Retrievers with Large Language Models for Graph-based Retrieval Augmented Generation

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 2, 8

## Abstract
Graph-based retrieval-augmented generation (RAG) enables large language models (LLMs) to ground responses with structured external knowledge from up-to-date knowledge graphs (KGs) and reduce hallucinations. However, LLMs often rely on a weak retriever in graph-based RAG: I) Due to the lack of ground truth, the retriever is often trained on weak supervision, which often introduces spurious signals to the LLMs. II) Due to the abstraction of graph data, the retrieved knowledge is often presented in unorganized forms. To mitigate the issue, we present Refined Graph-based RAG (ReG) to align weak retrievers to LLMs for graph-based RAG. Specifically, ReG incorporates LLM feedback to get rid of spurious signals and improve the quality of the supervision. Meanwhile, ReG introduces a structure-aware reorganization module to refactor the retrieval results into logically coherent evidence chains. Experiments on prominent benchmarks demonstrate that ReG significantly and consistently brings improvements across different LLM backbones by up to 10%. The improved supervision quality enables ReG to match the state-of-the-art performance with 5% training data and to transfer to out-of-distribution KGs. Notably, when adopted to reasoning-based LLMs, ReG reduces the reasoning token cost by up to 30% and improves the performance by up to 4%.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes Refined GraphRAG, which leverage refined retrieval graph to train a retriever for GraphRAG. Specifically, the authors first generate a multi-faced candidate with the shortest path, query neighbors and answer neigbors. Then, LLMs would refine the candidate graph to reduce the size. Finally, a retriever is trained based on the refined graph. Experimental results demonstrate the effectiveness of the proposed ReG.

### Strengths
1. It is reasonable that using better retrieval graph leads to better  performance.
2. The paper is well-written and easy to follow.

### Weaknesses
1. The proposed approach appears ad-hoc for multi-faceted candidate generation. The authors argue that previous methods relying on shortest paths suffer from a lack of reasoning signal. However, simply including the neighbors of the query and answer nodes does not adequately address this issues, particularly for multi-hop QA.

2. The LLM-Guided Candidate Refinement introduces unfairness in the comparison. This refinement step effectively performs part of the generation process, as the LLM may directly identify the correct answer for the query. As a result, comparing this approach with baselines that lack such refinement is not entirely fair.

3. The experiments do not convincingly demonstrate the generalizability of the proposed method to OOD KGs. Both CWQ and GrailQA are based on Freebase. I suggest that the authors include experiments on datasets with different underlying KGs, such as MetaQA, to strengthen their claims.

### Questions
Please refer to the weaknesses.

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
3

### Summary
This paper addresses the challenge of aligning weak retrievers with LLMs in graph-based RAG systems. The authors identify two key problems: (1) weak supervision signals from heuristic methods (e.g., shortest paths) that introduce spurious connections or miss critical evidence, and (2) misorganized representation of retrieved graph information. They propose ReG (Refined graph-based RAG), which uses LLM feedback to refine supervision signals and employs structure-aware reorganization to present retrieved information in logically coherent chains.

### Strengths
The paper clearly articulates the limitations of current graph-based RAG approaches. The method also achieves state-of-the-art performance across benchmarks.

### Weaknesses
1. Limited novelty in core techniques: While the combination is effective, the individual components are relatively standard. Using LLM feedback to filter/refine candidates is not new (acknowledged in related work). BFS-based chain expansion is a straightforward graph traversal technique. The main contribution appears to be the specific application to graph-based RAG rather than methodological innovation
2. No analysis of retrieval quality independent of QA performance (e.g., precision/recall of retrieved triples vs. oracle)

### Questions
Same as above

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
5

### Summary
This paper proposes ReG to improve traditional KGQA by using LLMs to refine weak supervision signals and train the retriever and to reorganize retrieved knowledge into logical chains. The core claim is that this aligns "weak" retrievers with "strong" LLMs. While the experimental results on KGQA benchmarks are solid, the paper suffers from fundamental conceptual and methodological problems that make it low-quality.

### Strengths
- It solves a practical problem since heuristic-based supervision like shortest paths for graph retrievers is noisy and misaligned with LLM reasoning.
- The results are promising.​​ The method demonstrates strong performance gains on traditional KGQA datasets (WebQSP, CWQ) and shows impressive data efficiency, matching SOTA performance with only 5% of training data.

### Weaknesses
- The title and claims are misleading. This is indeed an LLM-based KGQA paper, not GraphRAG. The community refers GraphRAG as a complete pipeline, similar to but more than RAG, that involves ​​constructing a graph from raw documents​​ and then retrieving from it. This work operates purely on ​​existing KGs​​ in a traditional LLM-based KGQA setting, let lone not comparing against real GraphRAG baselines, the paper addresses a much narrower problem than it claims.
- It is a very important prerequisite that we use KGs to enhance LLMs for unseen or difficult domain-specific scenarios. This arouses two major problems of this paper:
    - There lacks the zero-shot performance of LLMs. If LLMs could already achieve good accuracy, what is the advantage of this paper? This is also a concern to the entire KGQA task, it is not generalizable and applicable for LLMs nowadays. Therefore, the contribution of this paper is not enough.
    - Weak-to-strong is a good hypothesis but should not be static. LLMs are treated as oracle, but the feedback should be used to iteratively refine the signal for better loop. The design lacks enough consideration to make it a real 'weak-to-strong'.

### Questions
- What is the advantage of training a specialized retriever compared to directly using the powerful zero-shot LLM? also any comparisons?
- If the alignment is not iteratively achieved, how do you ensure LLM is a reliable oracle since we often aim to solve the hallucination and domain knowledge lacking problem in GraphRAG?

### Soundness
2

### Presentation
4

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper uses LLMs to refine input graphs and then trains retrievers via supervised learning. Experiments show that ReG achieves state of the art performance using only 5% of the training data and transfers well to out of distribution KGs.

### Strengths
1.	Tackles an important and timely problem.
2.	Thorough experimental evaluation, including an analysis of the proposed “overthinking” problem.
3.	Maintains high performance even when trained on just 5% of the data.

### Weaknesses
1.	Some parts of the text are hard to follow and would benefit from rewriting for clarity.
2.	Key methodological details are omitted and should be added.
3.     The use of LLM to evaluate each candidate P is expensive.

### Questions
1.	Figure 1 does not make the ReG workflow clear. Please redraw the diagram to more explicitly show the end to end pipeline and the role of each component.
2.	The motivation for introducing Definition 3.1 is unclear, since prior work has already formalized graph based RAG. Please clarify how Definition 3.1 differs from or complements existing formalisms (e.g., Peng et al., 2024, "Graph Retrieval Augmented Generation: A Survey").
3.	For graph refinement, the query centric neighborhood construction is well described, but the paper does not explain how answer centric neighborhoods are built in practice—only that they enable comparisons across answer candidates using numeric or categorical attributes. How are answer centric neighborhoods generated when entities lack attributes in the KG? Please provide concrete procedures or fallback strategies for such cases.

### Soundness
3

### Presentation
3

### Contribution
3
