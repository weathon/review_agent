# REL-RAG: Relation-Aware Retrieval-Augmented Generation for Generalizable Knowledge Graph Question Answering

- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
Large Language Models (LLMs) augmented with knowledge graphs (KGs) have been widely studied for knowledge graph question answering (KGQA). Graph-based retrievers exhibit strong empirical performance, but their generalization ability remains limited.  In this work, we show that applying a *line graph transformation* to the KG provably enhances the generalizability of GNN-based retrievers. By elevating relations to first-class objects, line graphs encode relation transitions explicitly, and the resulting inductive bias aligns naturally with relational reasoning in KGs. This alignment makes multi-hop reasoning substantially easier to learn and improves generalizability across different types of distribution shifts.  Building upon this representation, we propose $\texttt{REL-RAG}$, a framework that emphasizes relational reasoning for graph retrievers and is equipped with two complementary training objectives for flexible integration with LLMs. Path-based learning achieves higher precision with fewer tokens, making it especially suitable for smaller LLMs with limited context capacity. Triple-based learning encourages richer evidence diversity, which stronger LLMs can exploit more effectively with larger token budgets.  Empirically, $\texttt{REL-RAG}$ establishes new state-of-the-art results on KGQA benchmarks, surpassing prior graph retrievers by up to $20.3\\%$ with Llama3.1-8B and $10.3\\%$ with GPT-4o-mini.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes REL-RAG, a GNN-based RAG solution for knowledge graph question answering. 
The main contribution of REL-RAG lies in follows. First, REL-RAG proposes to transform KG into a line graph, which each triple is a node. The REL-RAG then devises a message passing mechanism on this line graph. From the theoretical perspective, the authors prove that owing to line-graph, GNN retriever has better generalization. The experimental results also empirically show the effectiveness of the proposed method.

### Strengths
1.	The proposed method shows to be effective, achieving better performance than baseline method Sub-graph RAG and GNN-RAG. 

2.	The authors theoretically proves the better generalization of the proposed method, making the main claims theoretically grounded. 

3.	The manuscript shows to have comprehensive experimental design, which empirically reconfirms the conclusion by testing OOD performance.

### Weaknesses
1.	This paper contains several confusions need to clearly defined or explained.

2.	The paper states how to perform message passing and path encoding in section 3, but does not introduce how to select out the best relation path that contributes to the question. To be specific, there exists a clear logic gap between Section 3 and Section 4. This reviewer cannot even make an educatable guess that how z_t^{(1)}  in equation (5) contributes in equation (10) and (11). 

3.	In Appendix E.1, the manuscript mentions “when multiple valid paths exist”. However, in Section 4 and Appendix B (Inference.), how to select valid path in path-based learning is not explicitly mentioned. This reviewer can only make an educatable guess that we can calculate the product of each triple within the path and sort them. In addition, the term “predicted probabilities” mentioned in line 1216/1217, lack clear definition. This reviewer can only make an educatable guess that the probability of a triple is the softmax term exp(z_q, z_q(i))/ \sum … in equation (75).

4.	Based on the available information provided in this manuscript, it can be hard for an educatable researcher to construct the system and replicate the results. Due to the logical gaps and the absence of precise definitions, this reviewer believes that researchers outside the KBQA-related areas may be completely lost when reading this paper.

5.	This paper lacks discussion and/or comparison with several baseline methods, namely DoG [1], FastToG [2], GoG [3], PoG [4], KARPA [5], SRP [6], RAR [7], and READS [8], where [1-5] are accepted to top-tier conferences. 

[1] Li et al., Decoding on Graphs: Faithful and Sound Reasoning on Knowledge Graphs through Generation of Well-Formed Chains (ACL2025)

[2] Liang and Gu., Fast Think-on-Graph: Wider, Deeper and Faster Reasoning of Large Language Model on Knowledge Graph (AAAI2025)

[3] Xu et al., Generate-on-Graph: Treat LLM as both Agent and KG for Incomplete Knowledge Graph Question Answering (EMNLP2024)

[4] Chen et al., Plan-on-Graph: Self-Correcting Adaptive Planning of Large Language Model on Knowledge Graphs (NeurIPS 2024)

[5] Fang et al., KARPA: A Training-free Method of Adapting Knowledge Graph as References for Large Language Model’s Reasoning Path Aggregation (ACL 2025)

[6] Zhu et al., Self-Reflective Planning with Knowledge Graphs: Enhancing LLM Reasoning Reliability for Question Answering

[7] Shen et al., Reason-Align-Respond: Aligning LLM Reasoning with Knowledge Graphs for KGQA

[8] Xu et al., LLM-based Discriminative Reasoning for Knowledge Graph Question Answering

### Questions
1.	What does “First-class objects” means needs to be clearly defined in the manuscript. It appears 4 times in the manuscript but neither of its occurrences is associated with clear definitions / citation to other paper. 

2.	Although theorems and proofs can lend rigor and credibility to the main claims, ***their inclusion should not come at the expense of the main sections’ overall logical coherence***. According to W2, this reviewer strongly suggests the authors to move equation (74)-(76) to the main sections and provide more detailed elaboration. 

3.	Which specific work does “prior work” in line 354 means? Are embeddings of entities and relations learnable? For a “representation learning” conference, these are highly important. This reviewer sincerely requests the authors to add relevant information, including explanations and citations. 

4.	This reviewer sincerely requests the authors to discuss related works [1]-[8] mentioned before, and compare [1]-[5] accordingly.

In view of the manuscript’s current organization, this reviewer recommends rejection. ***Nonetheless, given the paper’s findings and contributions, this reviewer would consider raising the score contingent on a comprehensive improvement of the overall organization***.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
REL-RAG proposes a line graph transformation that better adapts knowledge graphs for message passing in KGQA tasks. By treating triplets as nodes in this transformation, REL-RAG enables two learning strategies: sequential path-based learning and classic node classification (equivalent to triplet retrieval). Experimental results on WebQSP and CWQ benchmarks demonstrate that REL-RAG achieves competitive performance against other graph retrievers, with the line graph showing better generalization compared to the original graph structure.

### Strengths
- S1) REL-RAG operates on a line graph that treats triplets as nodes, better suiting triplet retrieval with GNNs. By compacting triplets into graph nodes, the underlying GNN requires fewer layers to work effectively and the line graph transformation enables REL-RAG to employ GNNs for both path-based learning (sequential predictions) and triplet retrieval tasks (Section 4).

- S2) Table 2 demonstrates that the line graph achieves better generalization compared to the raw graph. This improvement likely stems from triplets having richer semantics than individual nodes, making unseen triplets easier to handle than unseen nodes.

### Weaknesses
- W1) REL-RAG's theoretical analysis (Section 3) assumes GCN-style MPNNs that aggregate information from all relations (Eq. 1). However, practical GNN/MPNN implementations use query-conditioned message passing based on semantic similarity between the query and relations/subgraphs (NSM, GNN-RAG), which the current theory doesn't capture. This limits the insights to a narrow case; the authors should also demonstrate that line graph transformation benefits other GNN architectures beyond GCN.

- W2) REL-RAG requires storing embeddings for all triplets (edges), making it computationally impractical for billion-scale or real-world graphs. While REL-RAG's performance may benefit from triplets 
generalizing better than graph nodes (e.g., handling unseen entity names), its practical application is limited its storage requirements.


-  W3) The paper makes some overclaims on performance results. Line 369 states "REL-RAG achieves improvements of up to 20.3% on CWQ," but the actual results show GNN-RAG at 66.8% versus REL-RAG at 67.2%, which is only a 0.4 percentage point difference. Given that REL-RAG retrieves more triplets than GNN-RAG (Table 5), these improvements appear marginal.

### Questions
- Q1) How is the question triplet $v_{q(0)}$ obtained in Section 4? This appears to be a crucial component but I could not find the explanation.

- Q2) Can the line graph transformation benefit other KGQA GNNs beyond vanilla GCN?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces REL-RAG, a relation-aware retrieval-augmented generation framework for knowledge graph question answering (KGQA). The key innovation is to transfer KG into line graph, elevating relations to first-class objects and explicitly encoding relation transitions. This paper provides a thorough theoretical analysis showing that this transformation leads to tighter generalization bounds compared to standard entity-graph representations. The model offers two training regimes: path-based learning for token-efficient reasoning with smaller LLM and triple-based learning for evidence-diverse reasoning with larger LLMs, which makes the contribution even larger and model more flexible. Experiments on WebQSP, CWQ, and GraiLQA benchmarks demonstrate improvements over existing baselines.

### Strengths
1. The paper is well written, with clear motivation and problem formulation, and provides a compelling intuition for why line graph transformation helps - by eliminating "relational mixing" at entity nodes and making relation transitions explicit in the graph structure. The visual illustrations effectively communicate this idea. The paper goes beyond empirical results to provide formal analysis, including proofs that line graph models admit tighter generalization bounds under various distribution shifts (ID, compositional, and OOD). This theoretical foundation strengthens the contribution.

2. Transfer KG into line-graph sounds interesting and has been proven to have good results.

3. The dual training regime (path-based vs. triple-based) shows consideration for real-world deployment with different LLM capacities and token budgets.

4. The experiments cover multiple datasets and different types of generalization scenarios.

### Weaknesses
1. As for theoretical analysis, the proofs rely on several strong assumptions (e.g., sub-Gaussian concentration, specific Lipschitz constants) that may not hold in practice. The connection between the theoretical guarantees and empirical performance is not clearly established.

2. While the method achieves SOTA results, the improvements are often modest (e.g., 3.1% on WebQSP with 500 triples). Given the additional complexity of line graph transformation, the cost-benefit trade-off is questionable.

3. The paper mentions $O(|E|d_{max})$ preprocessing time but doesn't provide wall-clock comparisons or memory usage analysis. For large-scale KGs, this could be a significant limitation. 

4. Some design decisions lack justification (e.g., why specifically 50 vs 500 triples? Why 2-layer GCN?). The reliance on GPT-4o for path annotation introduces a circular dependency that isn't adequately addressed.

### Questions
1. Have you evaluated the scalability of line graph transformation on industrial-scale KGs (e.g., full Freebase or Wikidata)? What are the practical memory and time constraints?
2. Can you provide examples where line graph transformation actually hurts performance? Are there specific types of questions or graph structures where the approach fails?
3. The theoretical analysis assumes the number of relations R ≤ d (embedding dimension). How realistic is this assumption for real-world KGs, and what happens when it's violated? (In ultra-large KG, the number of relations may be larger than embedding dimension)
4. Why 50 triples are used? And why 2-layer GCN used? Can you provide some sensitivity analysis to justify your choice?
5. The improvement on GrailQA is relatively small compared to other datasets. However, GrailQA has the smallest value, suggesting significant room for improvement. Can you provide some analysis to explain why this is the case?
6. How sensitive is the method to the quality of the initial shortest path extraction? Have you experimented with alternative path selection strategies?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes an RAG approach by applying a line graph transformation to knowledge graphs.

### Strengths
Flexible Training Objectives: The introduction of two complementary training objectives (path-based learning and triple-based learning) is a solid contribution. It offers flexibility to accommodate different capacities of downstream LLMs.

### Weaknesses
**Redundancy and Complex Notation**

 The paper uses several mathematical definitions and formulas that may seem redundant and overly complicated, such as the introduction of ID, OOD, message-passing updates as well as the. The theorems and proofs presented in the paper provide valuable theoretical insights and enhance the depth of the work. However, they largely abstract away from the uncertainties present in real-world graph retrieval and RAG systems. From a practical or system-oriented perspective, empirical results demonstrating actual performance are more critical than the theoretical analysis.

**Handling of Ambiguous Queries** 

The paper does not sufficiently address how the method handles ambiguous or vague queries that require more exhaustive reasoning, such as "Who is the largest shareholder of Tencent?" In such cases, the system must evaluate potentially hundreds of candidates, and path-based retrieval alone would likely be insufficient. The method might be prone to "memorizing" the correct answers when no further context is available, especially for questions with a large number of possible answers. A more robust mechanism for handling such cases would strengthen the proposed approach.

**Limited Innovation** 

While the paper claims a novel approach with the line graph transformation, the overall reasoning and training methodologies are relatively standard. 

The line graph proposed in the paper actually only adds an extra layer of abstraction on top of the triples.  The method essentially wraps the triples in a new "package" without introducing fundamentally new reasoning mechanisms. Therefore, the use of the line graph is a redefinition of existing concepts rather than a truly groundbreaking innovation. 

**Insufficient Experimental Validation of Generalization:**

Although the paper claims enhanced generalization capabilities, the experimental validation is limited to only three datasets (WebQSP, CWQ, GrailQA), which are relatively simple and traditional. The claim of generalization across distribution shifts would be more convincing with experiments on more diverse and challenging datasets. Additionally, testing on more complex and large-scale problems would demonstrate the true strength of the proposed method.

### Questions
The bidirectional GCN on the directed and reversed graphs is mathematically equivalent to using a single GCN on an undirected graph. The authors should clarify the motivation for this design choice.

### Soundness
1

### Presentation
2

### Contribution
1
