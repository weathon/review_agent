# Atomic HINs: Entity-Attribute Duality for Heterogeneous Graph Modeling

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Heterogeneous Information Networks (HINs) provide a powerful framework for modeling multi-typed entities and relations, typically defined under a fixed schema. Yet, most research assumes this structure is given, overlooking the fact that alternative designs can emphasize different aspects of the data and substantially influence downstream performance.
As a theoretical foundation for such designs, we introduce the principle of entity-attribute duality: attributes can be atomized as entities with their associated relations, while entities can, in turn, serve as attributes of others. This principle motivates atomic HIN, a canonical representation that makes all modeling choices explicit and achieves maximal expressiveness.
Building on this foundation, we propose a systematic framework for task-specific schema refinement.
Within this framework, we demonstrate that widely used benchmarks correspond to heuristic refinements of the atomic HIN—often far from optimal.
Across eight datasets, refinement alone enables a simplified Relational GCN (sRGCN) to achieve state-of-the-art performance on node- and link-level tasks, with further gains from advanced HGNNs. These results highlight schema design as a key dimension in heterogeneous graph modeling.
By releasing the atomic HINs, searched schemas, and refinement framework, we enable principled benchmarking and open the way for future work on schema-aware learning, automated structure discovery, and next-generation HGNNs.
Our code is available at: https://github.com/ntuidssplab/AtomHIN.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the principle of entity-attribute duality, motivating an 'atomic HIN' where all attributes become entities to maximize the graph's expressiveness. On this foundation, the authors propose a framework for task-specific schema refinement, which systematically selects the most informative nodes and relations for a given task. This allows a simple model to achieve state-of-the-art performance and stronger models to improve further, demonstrating that schema design is a critical and overlooked component in heterogeneous graph modelling.

### Strengths
1. The papers validates its framework across eight heterogeneous benchmark datasets from diverse domains (bibliometrics, e-commerce, knowledge graphs, biomedicine).
2. By canonicalizing schema design and turning it into a first-class, searchable object, the work shifts attention from only designing graph neural layers to co-designing data schemas + models.
3. The finding that benchmark schemas are often far from optimal (under equal model capacity) is important for the community’s evaluation practices; the release of atomic graphs, refined schemas, and a refinement framework directly supports more principled future benchmarking.

### Weaknesses
1. The paper formalizes attribute atomization and 'atomic HINs,' but it also acknowledges that constructing structure from attributes is a long-standing preprocessing technique (e.g., one-hot/multi-hot attributes turned into nodes/edges, as in IMDB). This makes the core idea feel more like a principled unification than a new representational primitive. It is recommended to add a small related-work ablation to show where each piece of the proposed appproach adds value beyond prior metapath/feature-node practices.
2. Schema refinement is cast as binary search with a genetic algorithm and a sizeable budget (1024 schema trials), followed by 256 model-HP trials. This raises fairness and overfitting concerns if competitor baselines do not receive commensurate tuning. 
3. Several methods effectively learn metapaths/subgraphs via soft weights (e.g., GTN, MHGCN, RE-GNN). Since these are differentiable schema search to a degree, they’re natural baselines/foils for the discrete approach. If excluded due to scale or incompatibility, justify; otherwise is would be better to add them.

### Questions
1. What specific components constitute the core novelty beyond formalizing attribute atomization (e.g., canonical atomic HIN, schema refinement/search, pre-propagation), and how does each component independently contribute to performance?
2. What are the per-dataset costs of atomization in terms of graph expansion (nodes/edges/types), density, training time, and peak memory?
3. What are the time and memory complexities of pre-propagation as a function of relation/type counts, and how do those complexities manifest on OGBN-MAG-scale graphs?
4. How does the proposed discrete search compare empirically to differentiable schema/subgraph selection methods (e.g., GTN/MHGCN/RE-GNN) on small/medium benchmarks?
5. How transferable are refined schemas across heterogeneous GNN architectures and tasks, and what failure cases or negative transfers were observed?

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
2

### Summary
The paper proposes an atomic view of heterogeneous information networks (HINs) by atomizing attributes into explicit nodes and relations, so that all information becomes structural. On top of this representation, the authors introduce schema refinement via a genetic algorithm that searches over which attribute-derived node/edge types to include for a downstream task.

### Strengths
- Elevating attributes to first-class graph elements clarifies a long-standing, often ad-hoc HIN design step. The discussion is genuinely interesting and important for the community.
- Using GA to navigate the combinatorial schema space is a neat and pragmatic idea.
- Framing schema choice as an optimization target (rather than a one-off manual decision) is valuable and underexplored.
- Proofs and experiments are commendable

### Weaknesses
- Baseline: How does the proposed method compare with the tabular learning baseline?
- Additional ablation study and details over the GA could be added, i.e., different variants of the fitness function
- Scalability analysis is missing: what is the atomization cost, searching cost, and runtime/memory of the model added up, and how do it compare with standard GNN methods?

### Questions
- What is the connection between atomic representation and normalization form in database theory? 
- Does this method extend to relational databases? Could you apply the same idea to solve the relational deep learning problem like in RelBench?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The work explores the new structure of HIN, which transforms certain feature values into new node types.

To this end, the authors propose an atomic HIN, which selectively transforms certain features into nodes via an evolutionary algorithm.

The authors demonstrate the effectiveness of the atomic HIN in various heterogeneous graph benchmark datasets.

### Strengths
S1. The motivation and presentation are clear and intriguing.

S2. Key arguments are supported by theoretical analysis.

S3. Various datasets have been used for evaluations.

### Weaknesses
I have several questions regarding this work:

**W1. [Complexity]** While the authors use an evolutionary algorithm to avoid the exhaustive exponential search regarding which features to be transformed into nodes, I still think the search process requires heavy computations. Can the authors analyze the time consumption for this search process?

**W2. [Backbone HIN]** It seems the proposed method is coupled with SHGC. Can the method be coupled with other types of HINs?

**W3. [Feature constraint]** I think for binary features, the transformation is natural. However, for numeric features, how is the transform being performed? Are the values assigned by edge types (or weights)? If so, then wouldn't the graph be very dense?

My initial score is below the acceptance threshold, but I’m willing to raise it pending the authors’ clarifications.

### Questions
See Weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper works on heterogeneous information networks for which many different machine learning techniques are known. These models start with the data and their structure as given. The authors argue that there is in fact a lot of different ways the data can be structured into HINs and this can have a high impact on the results. They define a formal framework to show the entity - attribute duality which leads to the possibility to do this. In two propositions the show that two existing methods (from 2018 and 2019) can be expressed within this framework. The results indicate that indeed the results vary with the choice of entities vs attributes and that the revised schemas proposed by the authors have good performance.

### Strengths
- Good theoretical foundation for the entity - attribute duality and the following atomization step.
- Unification of existing methods based on the framework. 
- Good experimental evaluation

### Weaknesses
- This paper is more a database / KDD type of paper, for me that would be the more appropriate venues. The authors are correct by writing that ICLR typically start from the given structure. This is also reflected in the references where the only refs to ICLR and related venues are for the tools used not as comparisons to the proposed methods.
- The reformulation of two methods into the new setting is interesting but the methods are from 2018 and 2019

### Questions
- How about more recent methods compared to the 2018 and 2019 ones. Why only consider those two?

### Soundness
4

### Presentation
3

### Contribution
2
