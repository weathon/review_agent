# Beyond Static Retrieval: Opportunities and Pitfalls of Iterative Retrieval in GraphRAG

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 8, 4

## Abstract
Retrieval-augmented generation (RAG) is a powerful paradigm for improving large language models (LLMs) on knowledge-intensive question answering. Graph-based RAG (GraphRAG) leverages entity–relation graphs to support multi-hop reasoning, but most systems still rely on static retrieval. When crucial evidence, especially bridge documents that connect disjoint entities, is absent, reasoning collapses and hallucinations persist. Iterative retrieval, which performs multiple rounds of evidence selection, has emerged as a promising alternative, yet its role within GraphRAG remains poorly understood.  We present the first systematic study of iterative retrieval in GraphRAG, analyzing how different strategies interact with graph-based backbones and under what conditions they succeed or fail. Our findings reveal clear opportunities: iteration improves complex multi-hop questions, helps promote bridge documents into leading ranks, and different strategies offer complementary strengths. At the same time, pitfalls remain: naive expansion often introduces noise that reduces precision, gains are limited on single-hop or simple comparison questions, and several bridge evidences still be buried too deep to be effectively used. Together, these results highlight a central bottleneck, namely that GraphRAG’s effectiveness depends not only on recall but also on whether bridge evidence is consistently promoted into leading positions where it can support reasoning chains.  
To address this challenge, we propose Bridge-Guided Dual-Thought-based Retrieval (BDTR), a simple yet effective framework that generates complementary thoughts and leverages reasoning chains to recalibrate rankings and bring bridge evidence into leading positions. BDTR achieves consistent improvements across diverse GraphRAG settings and provides guidance for the design of future GraphRAG systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper reveals that iterative retrieval can significantly enhance multi-hop reasoning in GraphRAG by uncovering bridging evidence, and introduces BDTR, a dual-thought and bridge-guided framework that makes this process more accurate and efficient.

### Strengths
1. This paper conducts a comprehensive preliminary study that systematically examines the opportunities and challenges of iterative retrieval in GraphRAG, providing strong empirical motivation and clear research directions for the subsequent method design and experiments.

2. The proposed BDTR framework effectively enhances GraphRAG by combining dual-thought retrieval and bridge-guided evidence calibration, which improves both retrieval and reasoning in multi-hop QA tasks.

### Weaknesses
1.	**Narrow motivation scope.** In Section 2 (Preliminary Study), the identified opportunities and pitfalls are derived from empirical studies on multi-hop QA datasets. However, since these datasets are deliberately constructed and highly structured, the motivation and findings are confined to a narrow task setting. This restricts the generalizability of BDTR to broader retrieval-augmented tasks such as multi-turn dialogue, contextual summarization, fact checking and other open-ended querying, where multiple pieces of evidence from a corpus must also be aggregated to support a response. These tasks are not predefined with fine-grained question types like "bridging" or "comparison" but worth handling.

2.	**Lack of case studies.** The paper does not provide case studies or qualitative examples to illustrate how iterative retrieval operates in practice, or to demonstrate why BDTR outperforms other iterative retrieval methods in a more interpretable manner. This omission weakens readers’ intuition about the mechanism behind BDTR’s improvements.

3.	**Sparse integration with GraphRAG.** In Section 3.1 (DTR), the backbone retriever merely returns documents from the candidate pool, with GraphRAG’s sole contribution being the provision of retrieval scores. Consequently, BDTR lacks explicit interaction with entities or relations within the underlying graph. Furthermore, the baseline implementation of ToG, which should involve graph-based search and pruning, is simplified into a document-level retrieval process. This oversimplification of graph integration in the experimental setup arguably dilutes the contribution of BDTR within the GraphRAG framework.

4.	**Omission of embedding-based dense retrievers.** The paper does not explore BDTR’s performance when paired with embedding-based dense retrievers. It remains unclear what concrete advantages a GraphRAG-based retriever provides over a dense retriever, given that both can supply the retrieval scores required by BDTR, while the latter is typically more efficient and computationally cheaper.

### Questions
As discussed in Weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates iterative retrieval strategies within the GraphRAG framework for multi-hop question answering. The authors conduct preliminary studies to identify opportunities and pitfalls of iterative retrieval, and compare different strategies.

### Strengths
**Clear motivation and empirical investigation:** The paper systematically analyzes how iterative retrieval affects performance on different question types (Bridge vs. Comparison), highlighting both benefits and limitations.

**Comparative analysis of strategies:** Evaluating multiple iterative retrieval strategies and suggesting possible combination or adaptive selection could be valuable.

### Weaknesses
Shown in **Questions**.

### Questions
**1. Lack of in-depth analysis**
FT and ST are both generated by the model through prompts. This process is highly dependent on the model's capabilities and has low controllability. The experiments in the paper do not clearly demonstrate the specific reasoning behaviors under the BDTR strategy. I suspect that such a simple expansion strategy could introduce unexpected noise information, which may reduce retrieval and reasoning precision. **Quantitative results on retrieval accuracy are needed to substantiate the claimed improvements in retrieval. In addition, could the authors provide several examples of both successful and failure cases to illustrate how BDTR behaves in practice?**

**2. The conclusions in the "Preliminary Studies" section are not surprising**
The conclusions presented here are widely recognized phenomena in the field, and they do not provide novel insights. It would be helpful if the authors could provide more original analysis or novel observations based on your work.

**3. Missing important details**
> Retrieval scores are generated by GraphRAG backbones.

The pipeline heavily relies on these scores, but the paper does not explain how these scores are computed. The authors should provide more details on the score generation process and their implementations.

**4. Static iterations**
Although the authors introduce the "fast" and "slow" thought process in multi-round RAG to address the problem of overthinking when switching between complex and simple tasks,
The method in this paper fixes the number of iterations in advance. For simple problems that require just one step, additional iterations still introduce unnecessary noise. This contradicts the goal of "Beyond Static Retrieval," as it does not adapt dynamically to different cases.

**5. Limited innovation**
Apart from the points mentioned above, the innovation of the approach is rather limited. The overall process still revolves around prompt-based multi-round query rewriting followed by LLM self-validation.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper systematically analyzes the iterative retrieval mechanism within GraphRAG. The authors identify a key bottleneck in GraphRAG: while graph structures support multi-hop reasoning, static retrieval often fails to surface critical *bridge documents*, the intermediate evidence that connects disjoint entities. 

Through thorough experiments across multiple dimensions, such as effectiveness, question types,  recall, number of rounds, and complementarity, this paper demonstrates that iterative retrieval can improve performance in complex multi-hop questions, help promote bridge documents into leading ranks, and different strategies offer complementary strengths. The experiments also show the pitfalls that naive expansion often introduces noise, iterations offer little benefit on simple comparison questions, and many evidences still fail to appear within the leading positions.

Based on these observations, this paper proposes Bridge-Guided Dual-Thought-based Retrieval (BDTR), which combines dual-thought generation with bridge-aware reranking to better promote bridge evidence into leading positions. BDTR shows consistent improvements across multiple GraphRAG backbones and datasets, offering both empirical insights and a practical solution for enhancing GraphRAG systems.

### Strengths
- This paper systematically studies the iterative retrieval mechanism in GraphRAG, offering a empirical insights for enhancing GraphRAG systems.
- The proposed BDTR framework is novel, solidly grounded in the experiments and observations on iterative retrieval mechanisms, and is clearly described in the paper.
- Both in the investigation of mechanisms and the validation of BDTR, the experiments are thoroughly conducted and robust, with conclusions sufficiently supported by evidences.

### Weaknesses
BDTR exhibits a strong reliance on the LLM, so the LLM's performance may impact the final results and computation costs. However, the paper lacks relevant experiments across different LLMs.

### Questions
Refer to Weaknesses, will the performance of LLMs used in BDTR significantly affect the experimental results?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Bridge-Guided Dual-Thought-based Retrieval (BDTR), a novel iterative framework for GraphRAG that systematically promotes bridge evidence into leading retrieval ranks, thereby overcoming the limitations of static retrieval and significantly improving multi-hop reasoning performance.

### Strengths
This paper effectively addresses the long-standing challenge of missing or under-ranked bridge evidence in GraphRAG, which often breaks multi-hop reasoning chains. Its observations are strongly supported by extensive and systematic experiments across multiple datasets and GraphRAG backbones, demonstrating consistent and significant performance gains.

### Weaknesses
1. The experimental coverage and task diversity are limited. Since all the test datasets are derived from Wikipedia, the paper provides insufficient evidence of generalization and robustness to open-domain, multi-source, or non-encyclopedic corpora such as enterprise knowledge bases.

2. The paper’s practical value for broader complex reasoning tasks remains uncertain. While BDTR effectively targets the “bridge evidence” problem in multi-hop QA, its design is highly specialized for this scenario and may not generalize well to other reasoning settings—such as temporal, causal, or commonsense reasoning—where evidence relationships are more abstract and cannot be explicitly modeled through bridge documents.

3. The proposed approach underutilizes the structured nature of GraphRAG. Although GraphRAG is designed to leverage entity–relation graphs for structured reasoning, BDTR primarily operates at the text retrieval and reranking level, without explicitly incorporating or reasoning over the underlying graph topology, edge semantics, or relational paths, missing opportunities to exploit the full potential of graph-based representations.

4. The paper lacks a detailed analysis of failure cases where the Dual-Thought mechanism fails to retrieve the bridge documents. Without examining such negative examples, it remains unclear how robust the overall retrieval system is under challenging scenarios—e.g., when bridge evidence is implicitly expressed, sparsely connected, or missing from the corpus.

### Questions
As discussed above in weakness.

### Soundness
3

### Presentation
3

### Contribution
2
