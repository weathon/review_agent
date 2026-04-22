# RelDiff: Relational Data Generative Modeling with Graph-Based Diffusion Models

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Real-world databases are predominantly relational, comprising multiple interlinked tables that contain complex structural and statistical dependencies. 
Learning generative models on relational data has shown great promise for producing synthetic data, which can unlock access to previously underutilized information and support the training of powerful foundation models.
However, existing methods often struggle to capture their complexity, typically reducing relational data to conditionally generated flat tables and imposing limiting structural assumptions. 
To address these limitations, we introduce RelDiff, a novel diffusion generative model that synthesizes relational databases by explicitly modeling their foreign key graph structure. 
RelDiff combines a joint graph-conditioned diffusion process across all tables for attribute synthesis and a $D2K+$SBM graph generator based on the stochastic block model for structure generation. 
The decomposition of graph structure and relational attributes ensures both high fidelity and referential integrity, both of which are crucial aspects of synthetic relational database generation. 
RelDiff achieves state-of-the-art performance in generating synthetic relational databases on 11 benchmark datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces RELDIFF, a generative framework for synthesizing complex relational databases. Unlike prior methods that flatten schemas or assume conditional independence, RELDIFF explicitly models database structures as graphs and uses a graph-based diffusion model to generate mixed-type attributes across interconnected tables. The framework ensures referential integrity via a D2K + SBM graph generator and captures both inter- and intra-table dependencies using GNNs. Experiments are conducted on 11 datasets.

### Strengths
1.The paper is generally well-written and easy to follow.

2.The use of the D2K + SBM graph generator to preserve foreign key cardinality and hierarchical dependencies is novel and technically interesting.

3.The ablation study is comprehensive.

### Weaknesses
1. The decomposition $p(\mathcal{V},\mathcal{E})$ = $p(\mathcal{E})p(\mathcal{V}|\mathcal{E})$ is assumed without theoretical support.

2. The proposed joint diffusion model is not clearly novel compared with existing tabular diffusion approaches such as TabDDPM, TABSYN, and TabDiff.

3. The high training cost of RelDiff raises scalability concerns, and memory usage is not reported.

### Questions
1. The statement “tabular data includes complex and varied distributions” (lines 41-42) appears somewhat vague. Image and text datasets can also exhibit diverse and complex distributions due to varying sources and contexts. Could the authors clarify in what specific sense tabular data distributions are considered more complex or varied?

2. The decomposition $p(\mathcal{V},\mathcal{E})$ = $p(\mathcal{E})p(\mathcal{V}|\mathcal{E})$ seems to be taken as a modeling assumption without sufficient justification. It is unclear why the generative process is assumed to first sample the relational structure and then the attributes. In practice, foreign-key relationships (edges) may be influenced by attribute distributions (e.g., business logic or temporal constraints), while attribute distributions can also be constrained by the structure (e.g., table hierarchy and connection density). Therefore, this factorization implicitly assumes a unidirectional dependency from structure generation to attribute generation, yet the paper provides neither theoretical justification nor empirical evidence to support this assumption.

3. The proposed joint diffusion model seems not an innovative design. The use of diffusion models for generating heterogeneous tabular features (i.e., numerical and categorical) has been extensively studied in prior works such as TabDDPM [1], TABSYN [2], and TabDiff [3]. The authors are encouraged to clarify what makes their proposed hybrid generation method novel beyond existing tabular diffusion approaches and to provide stronger empirical evidence demonstrating its superior effectiveness.

4. While quantitative metrics are provided, the quality of the generated tabular data should be further demonstrated through visualization to offer more intuitive and interpretable evidence of the model’s effectiveness.

5. As shown in Table 10, the training cost of RelDiff is substantially higher than that of ClavaDDPM, raising concerns about the method’s scalability and practicality on large-scale datasets. Moreover, the paper does not report the memory cost across different datasets, which is important for assessing the overall efficiency and deployability of the proposed framework.

[1] TabDDPM: Modeling tabular data with diffusion models. ICML2023

[2]  Mixed-type tabular data synthesis with score-based diffusion in latent space. ICLR2024

[3] TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation. ICLR2025

### Soundness
3

### Presentation
3

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
The paper tackles relational data generation via graph diffusion that first resamples a D2K+ SBM foreign-key graph to preserve referential integrity, and then jointly denoises numerical and categorical attributes using a heterogeneous GNN, achieving state-of-the-art multi-table fidelity and up to an 80\% improvement in k-hop correlations across 11 real-world databases.

### Strengths
**Quality**. The paper uses a combination of different techniques. First, it generates a graph via their D2K + SBM generator. Their generator is comprised of Bayesian SBM as a model of graphs + D2K graph generator to preserve relationships between nodes. Subsequently, they define a conditional hybrid diffusion process which generates categorical and numerical samples conditioned on the generated graph.

**Clarity**. Paper is easy to follow.

**Significance**. Paper looks at tabular data generation for relational databases.

**Originality**. A conditional generation framework of integrating graphs could be interesting to the community.

### Weaknesses
Overall, experiments and ablation studies are comprehensive, comprising of performance, runtime, computation and privacy.  However, a concern I have is its novelty. Its a combination of existing well-known methods which I believe for the current standards of conferences like NeurIPS, ICLR and ICML, it may be insufficient. The main takeaway that the framework provides is that integrating graph based generators into diffusion models help provide extra signal to improve generative performance.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper tackles synthetic relational database generation. Rather than flattening multi-table schemas or generating tables in a pre-set order, the authors decompose the task into sampling a relational entity graph that respects foreign-key cardinalities and hierarchy using a microcanonical, degree-corrected Stochastic Block Model, and a joint, graph-conditioned diffusion model that denoises mixed-type attributes across all tables with a heterogeneous GNN. Training uses subgraph neighbor sampling and a hybrid continuous + categorical masking diffusion objective; sampling first draws a new entity graph with the SBM module, then jointly denoises attributes conditioned on the graph. On two benchmarks covering 11 real-world databases, the method reports stronger multi-table fidelity and good downstream RDL utility compared to ClavaDDPM, RCTGAN, SDV, RealTabFormer, TabularARGN, and PrivLava.

### Strengths
1. The modeling choices are well-motivated: microcanonical SBM gives hard constraints for referential integrity; hybrid diffusion aligns with mixed continuous/categorical columns; heterogeneous GNNs with subgraph sampling are a sensible scalability strategy.
2. Joint graph-conditioned diffusion over the entire entity graph, coupled with a microcanonical, nested SBM to preserve relational cardinalities and hierarchy, is a clean and compelling synthesis.

### Weaknesses
1. The baselines omit recent joint modeling approaches like GRDM (Graph-Conditional Relational Diffusion Model), which also performs joint denoising over relational graphs and reports strong k-hop performance. The paper positions prior work mainly as sequential/conditional (ClavaDDPM, etc.), but the landscape now includes joint graph-conditioned diffusion and flow-matching variants.
2. The nested SBM is well-motivated for modular schemas, but the paper preprocesses two-parent/no-child tables to many-to-many edges and then learns blocks and degrees under hard constraints. That can bias structure when schemas are weakly modular and may interact with functional dependencies in ways the SBM cannot express.

### Questions
1. How sensitive are results to the neighborhood radius and the subgraph neighbor-sampling scheme during training and inference?
2. Many databases encode temporal edges. Can RelDiff model time-stamped relations and reproduce temporal integrity?

### Soundness
2

### Presentation
2

### Contribution
2
