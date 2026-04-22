# KG-Infused RAG: Augmenting Corpus-Based RAG with External Knowledge Graphs

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Retrieval-Augmented Generation (RAG) improves factual accuracy by grounding responses in external knowledge. However, existing RAG methods either rely solely on text corpora and neglect structural knowledge, or build ad-hoc knowledge graphs (KGs) at high cost and low reliability. To address these issues, we propose **KG-Infused RAG**, a framework that incorporates pre-existing large-scale KGs into RAG and applies *spreading activation* to enhance both retrieval and generation. 
KG-Infused RAG directly performs spreading activation over external KGs to retrieve relevant structured knowledge, which is then used to expand queries and integrated with corpus passages, enabling interpretable and semantically grounded multi-source retrieval. We further improve KG-Infused RAG through preference learning on sampled key stages of the pipeline. 
Experiments on five QA benchmarks show that KG-Infused RAG consistently outperforms vanilla RAG (by 3.9% to 17.8%). Compared with KG-based approaches such as GraphRAG and LightRAG, our method obtains structured knowledge at lower cost while achieving superior performance. Additionally, integrating KG-Infused RAG with Self-RAG and DeepNote yields further gains, demonstrating its effectiveness and versatility as a plug-and-play enhancement module for corpus-based RAG methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces KG-Infused RAG, that applies an innovative multi-round activation spreading technique to link semantically relevant knowledge, and through query expansion and seed retrievals, incorporates sturctured knowledge to help LLMs solve complex QA tasks.
Results demonstrate the superiority of this technique over vanilla baselines, and also offer efficacy gains as it avoids generating KGs.

### Strengths
- The paper is well motivated and clearly written. All experimental configurations along with LLM prompts have been revealed, making it easy to reporduce results.
- The plug-and-play ability demonstrated further strengthens the paper towards deploying this in practical scenarios.

### Weaknesses
The framework avoids the upfront cost of generating a Knowledge Graph, but this resource saving is largely offset by the increased latency and computational expense of multiple, sequential LLM inferences during the query process.

### Questions
What is the proportion of passages from q and q' are kept in the mixed passages? Can you share some results on how much they overall for q and q'? This would help understand the extent of additional information that is helping with the overall improvement of KG-Infused RAG over its baselines.

### Soundness
3

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
3

### Summary
This paper proposes how to leverage KGs to expand queries and augment generation for QA. The paper is well written and the proposed method is mostly interesting. The paper lacks comprehensive comparison w. existing work on KG-based RAG, to show the novelty of the proposed method.

### Strengths
S1. Paper very well written, with clear explanations and formalisms. 

S2. The method is comprehensive, covering various aspects of RAG.

S3. Good experimental study showing quality improvements.

### Weaknesses
W1. The paper is not comprehensive in comparison w. sota solutions regarding RAG on KGs:
[1] KERAG: Knowledge-Enhanced Retrieval-Augmented Generation for Advanced Question Answering
[2] Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph

Section 3.3 is very similar to [1], reducing novelty of the work. 

W2. Using KG for query extension can be interesting. However, the retrieved KG subgraph can be huge and the query extension can thus bring in a lot of retrieval noises, in turn reduce QA accuracy. It's unclear how the results of KG-infused RAG compare w. no query expansion (just use KG retrieval results and corpus retrieval results), and w. normal query expansion. Table 4 shows comparison, but it is unclear what (w/o KG-based QE) refers to. 

W3. Related to W2, the text corpus is either just wiki, or a small corpus, not showing potential retrieval noises from much larger corpus (e.g., web). 

Also, unclear for each benchmark, what's the retrieval recall and QA accuracy for KG only,  Corpus only, and KG+Corpus w/o query extension? 

I also encourage the authors to report hallucinations. Oftentimes accuracy is achieved w. higher hallucinations (vs saying "I don't know"), which may not be worthwhile. 

W4. The paper shall show experimental results on CRAG, the latest RAG benchmark, which contains various types of complex questions, and provides KG and text corpus for retrieval. That will also make better comparison w. sota solutions.
[3] CRAG--Comprehensive RAG Benchmark

W5. Clear defn of the metrics should be given. How is Accu and F1 computed? Is it based on Exact-match? What if we do LLM-as-a-judge?

### Questions
Please answer questions and address concerns in weaknesses

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents KG-Infused RAG, a framework that augments standard retrieval-augmented generation by integrating external knowledge graphs through a spreading-activation mechanism. It enriches query retrieval with structured entity relations from existing KGs (e.g., Wikidata5M) without requiring costly graph construction. Experiments on five QA benchmarks show consistent accuracy and efficiency improvements over baseline RAG and prior KG-based systems.

### Strengths
This paper addresses a key gap between corpus-based and KG-based RAG approaches, namely, how to combine structured and unstructured knowledge for retrieval and reasoning efficiently.

Demonstrates gains on multiple QA datasets and provides comparisons with both corpus-only and KG-constructed baselines.

The framework can potentially be applied to enhance other RAG systems.

### Weaknesses
1. The writing of the paper is not very clear: The spreading-activation process (e.g., weighting, termination, or entity-selection heuristics) is only partially described. The use of spreading activation is intuitively motivated but lacks rigorous analysis or ablation exploring why it outperforms simpler entity expansion methods.

2. All benchmarks are QA-centric; there is no evidence the approach generalizes to other tasks (e.g., reasoning, summarization, scientific retrieval). Also, ther paper does not compare with existing retrieval-based methods such as Search-O1, IRCOT, HippoRAG, etc.

3. The paper occasionally positions the framework as a general RAG paradigm, but the results are confined to standard QA datasets.

4. The proposed method is somehow straightforward and heuristic in nature. The technical depth is limited.

### Questions
## Minor suggestions:

Figures and formulas (e.g., Eq. 2–5) could be condensed; some prompts and hyperparameters are deferred to appendices without summary.

## Questions:

How are activation weights propagated through multi-hop entities—uniformly or via learned relevance scores?

How sensitive is performance to the choice of KG (e.g., Wikidata vs. domain-specific graphs)?

Could this approach generalize to non-QA tasks, or is it specialized to entity-centric reasoning?

### Soundness
2

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
2

### Summary
The paper presents a framework designed to enhance RAG systems by integrating a large, pre-existing KG. The paper demonstrate promising experimental results across 5 QA benchmarks.

### Strengths
* The pipeline is clearly articulated and easy to follow.
* The proposed method is evaluated using a variety of LLMs.
* The paper provides comprehensive experimental results, including a clear ablation study that highlights the contributions of different model components.

### Weaknesses
* The novelty appears somewhat incremental, as previous research has also explored the use of KGs for QA and retrieval enhancement. The authors should more strongly emphasis what is new vs what is incremental. In particular, since the paper claims novelty in leveraging a pre-existing KG to enhance RAG, it should include comparisons with existing methods that also utilize pre-existing KGs (e.g., [1]). Additionally, it would be valuable to compare with the extensive body of work on QA systems that directly query existing KGs (e.g., [2]).
* All evaluation benchmarks are derived from Wikipedia, which overlaps significantly with Wikidata. As a result, the generalization and robustness of the proposed approach remain untested on more diverse datasets and realistic scenarios.
* The results for KG-based and non-KG-based methods are not directly comparable, as they use different corpora.

[1]. KERAG: Knowledge-Enhanced Retrieval-Augmented Generation for Advanced Question Answering. 

[2]. QALD-10 — The 10th Challenge on Question Answering over Linked Data.

### Questions
none

### Soundness
2

### Presentation
3

### Contribution
2
