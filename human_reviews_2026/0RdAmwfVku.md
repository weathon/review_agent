# VoG: Enhancing LLM Reasoning through Stepwise Verification on Knowledge Graphs

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Large Language Models (LLMs) excel at various reasoning tasks but still encounter challenges such as hallucination and factual inconsistency in knowledge-intensive tasks, primarily due to a lack of external knowledge and factual verification. These challenges could be mitigated by leveraging knowledge graphs (KGs) to support more reliable LLM reasoning. However, existing KG-augmented LLM frameworks still rely on static integration mechanisms that cannot adjust reasoning in response to evolving context and retrieved evidence, resulting in error propagation and incomplete reasoning. To alleviate these issues, we propose  **V**erify-**o**n-**G**raph (**VoG**), a scalable and model-agnostic framework to enhance LLM reasoning via iterative retrieval, stepwise verification, and adaptive revision. Besides performing KG retrieval guided by an initially generated reasoning plan, VoG iteratively verifies and revises the reasoning plan, correcting intermediate errors in consideration of the varying contextual conditions. During plan revision, VoG leverages a context-aware multi-armed bandit strategy, guided by reward signals that capture uncertainty and semantic consistency, to enhance the alignment between the reasoning plan and retrieved evidence in a more adaptive and reliable way. Experimental results across three benchmark datasets show that VoG consistently improves both reasoning accuracy and efficiency. Our code is available at https://github.com/WenxinAZhao/VoG.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Verify-on-Graph (VoG), a framework that enhances Large Language Models' reasoning by integrating knowledge graphs for reliable inference. VoG features iterative retrieval, verification, and adaptive planning refinement to boost accuracy and efficiency. Experiments on three benchmark datasets demonstrate its effectiveness in improving reasoning performance.

### Strengths
1. VoG introduces a novel framework that enables stepwise verification and planning refinement on KGs to mitigate error propagation during multi-hop reasoning.
2. VoG proposes a KG-aware multi-armed bandit (MAB) mechanism for adaptive context selection, which is a valuable contribution for dynamically determining context information for refining reasoning plans.
3. Experiment on three benchmark datasets shows that VoG significantly outperforms existing methods, demonstrating its effectiveness in enhancing the reasoning capabilities of LLMs.

### Weaknesses
1. The architecture of VoG seems complex with multiple components. While each component is motivated and performance improvements are shown, the overall complexity may hinder practical adoption. The authors should provide more analysis on the trade-offs between complexity and performance gains, and possibly explore simplifications.
2. UCB hyperparameter sensitivity is not deeply analyzed — how robust is the method to different scaling of entropy or exploration bonuses?
3. No evaluation beyond Freebase / KGQA — unclear whether VoG generalizes to non-Freebase or domain-specialized KGs (e.g., ConceptNet , biomedical UMLS).
4. The proposed method relies on retrieving evidence from the KG, but it is unclear how VoG handles cases where valid evidence/fact are missing from the KG.

### Questions
1. How does VoG handle valid but missing KG facts during reasoning?
2. How sensitive is the performance to the UCB hyperparameters?
3. Could VoG generalize beyond Freebase, e.g., ConceptNet or domain-specific KGs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes VoG (Verification-over-Generation), a reasoning framework that enhances large language models (LLMs) with structured, knowledge-grounded verification and revision. Unlike previous reasoning or retrieval-augmented methods that rely solely on planning or evidence retrieval, VoG organizes reasoning into a Plan–Retrieve–Verify–Revise loop.

### Strengths
The paper introduces a unified Plan–Retrieve–Verify–Revise reasoning framework, emphasizing step-wise knowledge-grounded verification. This design effectively connects planning with factual correction and hallucination suppression, offering a more systematic approach than prior single-stage (planning-only or retrieval-only) methods. Compared to ToG/PoG agents, VoG achieves lower average token consumption and fewer reasoning turns while maintaining higher accuracy, supporting the paper’s claim of being lightweight and efficient.

### Weaknesses
The author’s efficiency analysis section is very convincing; however, I am still curious about the end-to-end latency, as its framework introduces more steps compared to ToG and PoG.

### Questions
See weakness above.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a novel Verify-on-Graph (VoG) framework, which supports dynamic and context-aware LLM reasoning over KG through iterative retrieval, verification, and adaptive refinement. To be specific, the proposed method firstly employs a plan agent to generate reasoning chains, serving as guidance for multi-hop retrieval process. Then, stepwise verification and adaptive refinement is adopted to detect reasoning inconsistencies and make sure the correctness of subsequent reasoning steps. Finally, a confidence-based reward is designed to capture uncertain information for revision. Extensive experimental results demonstrate the effectiveness of the proposed method.

### Strengths
1.	This paper is well-organized and easy to follow.
2.	This paper presents a VoG framework consisting of three specialized LLM agents for retrieval, verification, and revision, which address the challenges of inflexible reasoning and limited utilization of information.
3.	This paper provides the source code to ensure the reproducibility of the proposed method.

### Weaknesses
1.	The figures could be further refined to enhance readability. In particular, the font size in Figures 1 and 3 is quite small, and Figure 7 appears to have low resolution.
2.	The paper may lack some baseline methods for comparison, such as GNN-RAG [1], SubgraphRAG [2].
3.	The core idea may not be highly novel, since the retrieval–plan–verify pipeline has been adopted in prior studies.

[1] Mavromatis, Costas, and George Karypis. "Gnn-rag: Graph neural retrieval for large language model reasoning." arXiv preprint arXiv:2405.20139 (2024).
[2] Li, Mufei, Siqi Miao, and Pan Li. "Simple is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation." The Thirteenth International Conference on Learning Representations.

### Questions
Please refer to Section Weaknesses.

### Soundness
2

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
5

### Summary
This paper proposes Verify-on-Graph (VoG), a model-agnostic framework that enhances reasoning reliability of LLMs on knowledge-intensive tasks by couping iterative retrieval, stepwise verification and revision, based on knowledge triplets retrieved from knowledge graphs. Unlike existing KG-augmented reasoning systems that follow fixed reasoning plans, VoG introduces (1) Stepwise KG verification that detects factual inconsistencies at each reasoning step. (2) Plan revision via multi-armed bandit context selection that adaptively decides which contextual scope to use for revising reasoning. Experiments on KG reasoning benchmarks show consistent improvements across backbone models of different sizes, compared to the strong agentic baselines with reduced token cost.

### Strengths
1. Novel verification framework for reasoning. The proposed step verification has the potential of mitigating error/hallucination propagation and improve faithful reasoning.

2. Adaptive context selection.  The idea of using UCB to balance the information window in the decision making process is innovative, and has the potential to be extended to other long-reasoning tasks as well. 

3. Strong empirical results. Experiments show consistent gains across three benchmarks and multiple LLM sizes, which validate the approach's generality.

### Weaknesses
Some key components, such as reward design details for UCB are located in the Appendix. They should stand out in the body of the methodology. See Questions for other technical weakness.

### Questions
1. What is the motivation to revise the reasoning plan given the retrieved knowledge triplets? Why does the factual knowledge retrieved play a role in evaluating(verify) the reasoning sub-step? A more natural way could be to populate the retrieved knowledge when it is not sufficient to solve the reasoning sub-step. 

2. There is no guarantee that the prompted based verification agent would return a faithful and correct response. In Table 2, the authors conducted ablation studies to show the effect of verify/revision. However, it is not clear whether the improvement is due to the verifier, or it's a effect of parallel thinking introduced by the revision itself. A more rigorous way is to test on a dataset that has ground truth reasoning([1] for example) steps and show these models can indeed distinguish the correct reasonings from the incorrect. 

3. What is the benefit of having a plan first and conduct step-with verify and revise, instead of iteratively reasoning and retrieve as in [2]?

4.  VoG retrieves relation and entity based on some semantic similarity score, which is similar to the retrieval approach introduced in [3], it is beneficial to include a discussion or comparison on that. 

5. Why the semantic similarity between the "predicted observation" and the input question can serve as a quality measure of reasoning, as introduced in Appendix B.2. Not every reasoning step needs to share similar semantic meaning with the question.

6. All experiments in Table 1 are conducted on Freebase knowledge graph, it remains unclear whether VoG generalizes to other domains, like scientific or biomedical KGs (such as UMLS).

[1] MINT-CoT: Enabling Interleaved Visual Tokens in Mathematical Chain-of-Thought Reasoning

[2] Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning

[3] GIVE: Structured Reasoning of Large Language Models with Knowledge Graph Inspired Veracity Extrapolation

### Soundness
2

### Presentation
3

### Contribution
2
