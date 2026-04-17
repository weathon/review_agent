# SLogic: Subgraph-Informed Logical Rule Learning for Knowledge Graph Completion

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Logical rule-based methods offer an interpretable approach to knowledge graph completion (KGC) by capturing compositional relationships in the form of human-readable inference rules. While existing logical rule-based methods learn rule confidence scores, they typically assign a global weight to each rule schema, applied uniformly across the graph. This is a significant limitation, as a rule’s importance often varies depending on the specific query instance. To address this, we introduce SLogic (Subgraph-Informed Logical Rule learning), a novel framework that assigns query-dependent scores to logical rules. The core of SLogic is a context-aware scoring function. This function determines the importance of a rule by analyzing the subgraph locally defined by the query’s head entity, thereby enabling a differentiated weighting of rules specific to their local query contexts. Extensive experiments on benchmark datasets show that SLogic outperforms existing rule-based methods and achieves competitive performance against state-of-the-art baselines. It also generates query-dependent, human-readable logical rules that serve as explicit explanations for its inferences.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces SLogic, a framework for knowledge graph completion (KGC) that aims to improve upon traditional logical rule-based methods. The core idea is to move beyond static, global rule confidences by learning a dynamic, query-dependent scoring function. This function leverages the local subgraph context of a query's head entity, extracted and encoded by a Relational Graph Convolutional Network (R-GCN). The model combines this subgraph representation with a GRU-based rule embedding and static rule features to predict a context-specific score for each rule. Experiments on WN18RR, FB15k-237, and YAGO3-10 show that SLogic outperforms baselines on two of the three datasets.

### Strengths
- Intuitive Core Idea: The central premise of using the local subgraph context around a query entity to dynamically re-weight the importance of logical rules is sensible and intuitively appealing.

- Detailed Methodology: The paper provides a comprehensive description of its framework, detailing the offline "instance creation" pipeline (rule mining, subgraph extraction, feature engineering) and the hybrid neural architecture (R-GCN + GRU + MLP) used for scoring.

- Clear Case Study: The case study in Section 5.3 effectively illustrates the model's intended mechanism, showing how SLogic's rule preferences change for the same relation (isLocatedIn) based on two different query entities (one film-related, one-geography-related), demonstrating its context-aware capability.

### Weaknesses
- Reliance on Heuristic Engineering: A very large part of the proposed contribution (Section 4.1) is a complex, multi-stage pipeline of heuristic-driven data and feature engineering. This includes DFS for path finding, k-hop BFS with neighbor sampling for subgraph extraction, and a hand-crafted set of topological node features (e.g., distance to head, global degree). This makes the framework feel less like a novel learning paradigm and more like a complicated feature engineering effort.

- Misleading Novelty Claims: The paper's primary motivation, stated in the abstract, is that "current approaches typically treat logical rules as universal, assigning each rule a fixed confidence score that ignores query-specific context." This statement is factually incorrect. A vast body of prior work (e.g., Markov Logic Networks, pLogicNet, and many others) has focused on learning weights or confidences for logical rules for decades. The paper fails to properly differentiate its specific contribution (using subgraph context) from this extensive literature, making its novelty unclear. The related work section is insufficient in this regard.

- Limited Scope of Rules: The method only handles chain-like, compositional rules (i.e., relational paths). This is explicitly stated in the definition in Section 3 ($r_h(X,Y) \leftarrow r_1(X,Z_1) \wedge \cdot\cdot\cdot \wedge r_L(Z_{L-1},Y)$) and confirmed by the DFS-based rule mining process (Section 4.1.1). This ignores all other, more complex rule structures (e.g., rules with multiple atoms, rules with constants) that are crucial for reasoning. This severe limitation on the form of logic used makes the framework's practical utility and generality questionable.

- Mixed Experimental Results: The method does not consistently outperform baselines. It notably performs worse than several baselines, including RLogic and RotatE, on the FB15K-237 dataset. The paper offers no substantive explanation for this failure, which undermines the general applicability of the SLogic framework.

### Questions
- Could the authors please justify the novelty claim from the abstract ("current approaches typically treat logical rules as universal...") against prior work like pLogicNet, Markov Logic Networks, or other methods that learn rule weights/confidences? How is the subgraph-based context proposed here fundamentally different from other forms of contextual or dynamic rule scoring in the literature?

- Why does the SLogic model perform poorly on FB15K-237 compared to other methods? What specific properties of this dataset (e.g., graph density, rule types) might cause a subgraph-informed approach to fail or underperform?

- The method is strictly limited to relational paths. What are the main conceptual or technical barriers to extending this framework to support more complex rule structures, such as rules with multiple branches (e.g., $r_h(X,Y) \leftarrow r_1(X,Z_1) \wedge r_2(X,Z_2)$)? How would the subgraph-based scoring and grounding mechanisms need to be adapted?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses a key limitation in existing logical rule-based methods for Knowledge Graph Completion (KGC)—the reliance on static, query-agnostic rule confidence scores—by proposing SLogic, a framework introducing query-dependent, context-aware rule scoring. SLogic uses a GNN-based subgraph encoder to capture local structural context around the query head entity, enabling more precise assessment of candidate rule importance. The framework integrates symbolic rules with neural representations, combining offline rule mining with a contrastive learning-based scoring model. Experiments on WN18RR, FB15k-237, and YAGO3-10 show state-of-the-art performance on WN18RR and YAGO3-10.

### Strengths
The paper clearly identifies a fundamental limitation of static rule confidences and proposes a principled solution. The core concept of query-dependent, context-aware rule scoring represents a meaningful paradigm shift from global prevalence to local relevance in rule learning.

The proposed SLogic framework effectively integrates symbolic and neural approaches. It maintains the interpretability of symbolic rules by building upon a mined rule base while harnessing the representational power of GNNs to encode rich local subgraph context. This hybrid design successfully balances transparency and predictive performance.

### Weaknesses
The authors argue that "query-dependent scoring is more reasonable than static rules," but they lack a clear delineation of the specific categories of KGs or relation types for which this conclusion holds (e.g., scenarios with high/low relation counts, strong/weak hub structures, high/low rule coverage). Currently, this is supported only by post-hoc experimental observations (effectiveness on certain datasets), and there is no theoretical guidance or well-defined quantitative metrics (e.g., rule coverage thresholds, relation-sparsity indicators) to guide when to adopt this method.

The paper employs k-hop BFS with a fixed neighbor sampling threshold (α), but it lacks a theoretical or empirical analysis justifying this specific choice (e.g., why random sampling is preferable to degree-weighted or importance sampling). The method's first step selects only "locally applicable rules ranked top-N by Wilson score" as candidates. However, the Wilson score itself is influenced by body-count, and the definition of "local applicability" is contingent upon the subgraph extraction parameters (α, k). If different methods were to use different subgraph extraction strategies, the resulting "comparable candidate sets" could be biased, thereby affecting the fairness of the re-ranking comparison.

The results tables report only single-run values (point estimates), without means ± standard deviations across multiple random seeds or significance tests. This is particularly crucial when performance improvements are marginal or mixed across datasets, and confidence intervals are necessary for robust evaluation.

Despite the use of LLMs for polishing, formatting, and grammatical errors remains. For instance: "The instances generated in this step... comprise rule-enriched triplets, ..." Here, the subscript i is inconsistent and should be 1. The paper exhibits inconsistent referencing of equations.

### Questions
1. Can the authors provide quantitative or theoretical analysis clarifying which KG types (e.g., relation count, node degree, rule coverage) benefit most from SLogic? Any correlation analyses between KG structure and 
performance?

2. Were alternative neighbor sampling strategies (degree-weighted, importance sampling) evaluated? 

3 .NCRL shows discrepancies between reported scores (NCRL†) and scores under the authors’ protocol (NCRL*). What explains this? Were all baselines evaluated under identical candidate rule sets and inference procedures?

4. Table 1 and the ablation study present results from a single run. Did the authors conduct multiple runs with different random seeds? If not, please supplement the results with the mean ± standard deviation from 3–5 independent runs.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SLogic, a subgraph-informed logical rule learning framework for Knowledge Graph Completion (KGC). SLogic could  incorporate query-dependent rule scoring, where the significance of each rule is dynamically recalculated using a subgraph centered on the query’s head entity. Extensive experiments on three datasets demonstrate that SLogic outperforms several embedding-based and rule-based baselines, while maintaining interpretability through explicit reasoning paths.

### Strengths
1. The paper’s key contribution of assigning dynamic rule weights conditioned on query subgraphs is well-motivated. It effectively bridges symbolic and neural reasoning.

2. The overall pipeline, including rule mining, subgraph extraction, and query-specific scoring, is clearly explained and internally coherent. The use of the Wilson confidence score and contextual GNN embeddings reflects thoughtful design choices that strengthen robustness.

3. SLogic demonstrates competitive or superior results to state-of-the-art rule-based and embedding methods on two of three benchmark datasets. The case study convincingly illustrates that the model adapts rule importance to distinct local contexts.

### Weaknesses
1. The performance improvement is inconsistent, strong on WN18RR and YAGO3-10, but weaker on FB15k-237. The discussion attributes this to graph density and relation diversity, but a deeper diagnostic such as rule quality distribution, subgraph connectivity analysis would better substantiate the explanation.

2. Training and negative sampling costs are significantly higher than baselines. While the cause is identified, potential optimizations such as subgraph pruning, caching, mini-batch rule evaluation are not explored.

3. The ablation studies focus on inference components, but ignore analyses of relation embedding, subgraph encoder, rule encoder, and negative sampling strategy. Demonstrating their individual contributions would clarify the necessity of each module.

4. Minor stylistic and typographic inconsistencies (such as some equations are numbered while others are not) slightly detract from readability, especially in the methodology section.

5. Some significant and typical related works are neglected, such as joint rule and embedding-based models IterE [1] and RPJE [2]. These models are suggested to be added and compared.  
[1] Iteratively Learning Embeddings and Rules for Knowledge Graph Reasoning. WWW 2019.  
[2] Rule-Guided Compositional Representation Learning on Knowledge Graphs. AAAI 2020.

### Questions
1. How does subgraph size (hop number) interact with rule length L in determining performance? Is there a trade-off between local and global reasoning depth?
2. Could the authors elaborate on why SLogic underperforms on FB15k-237 despite its relatively rich relational structure?

### Soundness
3

### Presentation
3

### Contribution
3
