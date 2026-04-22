# PolicyRAG: Prompt-Guided Symbolic Graph Memory for Interpretable Multi-Hop Retrieval

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2

## Abstract
Retrieval augmented generation is a powerful way to ground large language models in external knowledge, yet most pipelines still treat the prompt as instructions for text production rather than as a control surface for retrieval. We introduce PolicyRAG, a framework that recasts retrieval as an explicit, auditable policy operating over a symbolic graph memory. Text is organized into lightweight typed links between entities and passages, enabling transparent search and controllable evidence selection. Beyond accuracy, the policy is compact and human editable, supporting governance, domain adaptation, and safety review without retraining. At query time, the policy seeds candidate entities, invokes brief LLM calls only for disambiguation and local gating, and performs symbolic traversal with Personalized PageRank (PPR). The resulting scores are projected to passages and finalized with a small, transparent re-ranker, producing a per-query trace of seeds, paths, and scores for explainable evidence selection. Compared with long-context expansion, the policy keeps test-time compute modest while preserving answer quality. On multi-hop question answering benchmarks, PolicyRAG achieves state-of-the-art results on HotpotQA (F1 80.7), 2WikiMultiHopQA (F1 78.9), and MuSiQue (F1 55.9) while remaining fully auditable and training free. We also assess domain adaptability on domain-specific datasets. By coupling symbolic structure with prompt level control, PolicyRAG provides a practical route from question to verifiable evidence and advances accuracy, efficiency, and trust in retrieval augmented generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes PolicyRAG, which extracts entities and builds a KG at offline time, and leverages the KG for retrieval and QA in RAG, especially for multi-hop questions. My main concern for this paper is that I don't see how it is different from past work like HippoRAG and why the quality is better than HippoRAG

### Strengths
S1. The paper solves an important problem--how to answer multi-hop questions using RAG.

S2. Extensive experiments on 5 benchmarks showing improvement of PolicyRAG over existing methods.

### Weaknesses
W1. The description of how to construct the symbolic graph is not super clear.
- How to discover salient entities? At what granularity?
- How to identify factual assertions (triples)?
- How are the typed relations defined? 
- How are the extractions from different passages connected?
- How are dedups done?

W2. Related to W1, it is unclear what's the quality of the extracted symbolic graph and how does that affect QA quality? What's the design choices in graph building and granularity selections?

W3. For Composite transitions (Sec 3.1), what exactly is the goal and method? What's B and how is it computed? What's A_r and A_s and how to compute?

W4. Similarly, a lot of description for run-time QA is unclear. In (2), how is spec(e) computed? Why compact set F_s and how does that affect retrieval recall and e2e results?

W5. What's the latency impact for runtime PPR? 

W6. More importantly, the paper discussed HippoGraph, but it's unclear how it goes beyond HippoGraph and significantly improves the results? The methods sound very similar. 

W7. How much is the method constrained by corpus size? How well does it scale up?

W8. What are limitations of the method?

### Questions
Please answer all questions in the weakness session.

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
3

### Summary
This paper proposes to utilize prompt-based policies to improve the performance of LLMs on multi-hop reasoning tasks. Several experiments were conducted to verify the effectiveness of the proposed method.

### Strengths
1. The experiments in this paper are relatively abundant, and many auxiliary experiments are conducted to further explore the issues.

2. The proposed method has certain insights.

### Weaknesses
1. The presentation should be improved. The Introduction section is somewhat redundant and does not clearly and concisely state the motivation for the proposed method, which is very confusing. I suggest that the paper use examples to illustrate the specific motivation, including the problems encountered in previous work and the motivation for the proposed method.

2. Non-sentence-initial citations should use \citep instead of \citet.

3. The Methods section, Section 3, is also difficult to understand. The descriptions of the pipeline are not clear enough. I suggest adding a few more examples to help clarify the details.

4. No code is provided.

### Questions
See weaknesses above. Please respond to these cons.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper introduces PolicyRAG, a framework for multi-hop retrieval that enables transparent, training-free, and interpretable multi-hop evidence selection. The paper reports experimental results on multiple benchmark datasets and claims the proposed framework achieves state-of-the-art performance.

### Strengths
* The paper addresses a relevant problem, and introduces a framework tailored to it.
* The authors present experimental results across multiple benchmark datasets.

### Weaknesses
* The paper asserts that the proposed approach achieves strong performance; however, the comparison with existing works appears to be unfair. For instance, in Table 2, the underlying base LLMs used for comparison differ, making it unclear whether the reported superiority of the proposed method is solely attributable to the strength of GPT-4.1 mini.

* The paper only reports results based on GPT-4.1 mini, leaving the generalizability of the method unclear.

* The presentation of the paper requires significant improvement. For example, the cited "ACL" paper "Hipporag 2: Enhanced retrieval-augmented generation with neurobiological memory principles." does not exist. It appears that minimal effort was made to properly reformat the manuscript according to the ICLR template.

### Questions
* Where can we find the paper "Hipporag 2: Enhanced retrieval-augmented generation with neurobiological memory principles." in the references?

### Soundness
2

### Presentation
1

### Contribution
2
