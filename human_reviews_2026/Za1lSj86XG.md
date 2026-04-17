# Analog Circuit Topology Design and Sizing with Flow Matching Graph Learning

- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
The soaring demand for electronic devices calls for novel and more efficient analog circuits design.
Deep generative models have shown promise in assisting topology, parameter sizing, and layout design process, but existing approaches treat these tasks separately and lack generalizability across diverse problem settings.
In this work we introduce a flow matching model for automatic analog circuit design, which achieves high-quality sampling across a variety of topologies and representations.
Our model showcases state-of-the-art performances on end-to-end topology design and sizing on the Open Circuit Benchmark (OCB) dataset, and on transistor-level topology generation on the AnalogGenie dataset.
Code and models are provided as external supplementary files to this submission.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work present CircuitFlow, for end-to-end generation of analog circuit topology and device sizing.
This work adopt a multimodal flow matching model and show strong empirical performance on OCB and AnalogGenie dataset.

### Strengths
1. One of the first works tackles topology generation and circuit sizing at the same time. 
2. Detailed method presentation
3. Using flow-matching on analog circuit front-end design is an underexplored but interesting topic

### Weaknesses
1. Lack of strong motivation for unified topology and sizing designs
As the authors point out, there are a lot of dedicated existing works focus on topology or sizing. Why do we need a unified framework? What kind of benefit can it bring?
2. Lack of strong motivation for flow matching
Flow matching is a popular method in image generation. But why do we need flow matching for analog circuit designs, particularly? Does it enable us to generate topology and sizing at the same time compared to the early approach?
3. The current paper describes the above two points as an interesting direction to pursue, but does not show a strong motivation and results on the necessity. It's very hard for me to strongly recommend this paper based on the current material presented. 
4. More circuit examples can help the current paper presentation with just graphs

### Questions
1. How do you generate different circuits with different sizes (number of devices) with a fixed number of denoising steps?
2. How do you make sure your generated graph is always a single connected component rather than several disconnected components?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a flow-based method for both topology generation and device sizing. Extensive experiments show the state-of-the-art performance of the proposed method compared to existing work.

### Strengths
The proposed multimodal flow matching framework attempts to unify topology and device generation, which is an interesting direction for analog design automation.

### Weaknesses
1. The paper claims (L414–415) to be “the first architecture capable of jointly generating both circuit topologies and device features.”
This is an overclaim, as prior work such as DiffCkt and CktGen [1] has already demonstrated a diffusion-based framework for joint topology-sizing generation.
It seems that the key difference of this work compared to CktGen is that CktGen is a variational autoencoder (VAE) model, while CircuitFlow is based on flow matching.
Section 5.2 does not compare against such state-of-the-art methods, which weakens the experimental credibility and makes it difficult to assess the claimed advantage of the proposed model. 

[1] CktGen: Specification-Conditioned Analog Circuit Generation, https://arxiv.org/pdf/2410.00995


2. In Section 5.2, the sizing experiment is underspecified. The training objective maximizes the log-likelihood of device sizes (i.e., matches the dataset distribution), which does not imply performance optimization (gain/pm/ugf) unless the dataset distribution is itself optimal. Evaluation only compares against “data values” and uses a t-SNE plot without defining baseline sampling ranges or reporting quantitative metrics (e.g., validity rate, pass rate with thresholds, Wasserstein/MMD distance, or confidence intervals).
The claim that generated sizes serve as strong BO/GA initialization points is not supported by controlled downstream experiments.

3. Lack of sufficient evidence in the transistor-level experiment. Although Section 5.3 is described as “transistor-level topology generation,” the experiment does not include device sizing, despite the paper’s end-to-end claim.
The reported metrics (V.U.N., validity similarity, uniqueness, novelty, etc.) primarily measure structural diversity and do not demonstrate the functional quality of the generated circuits after sizing or simulation. Thus, the results cannot substantiate the practical utility of the generated designs.

4. Across all experiments, the generation process is not conditioned on circuit performance or design specifications.
For analog front-end design, this significantly limits real-world relevance; generating structurally valid circuits does not imply generating high-quality, spec-satisfying circuits. 
The absence of performance-driven evaluation weakens the overall contribution and restricts the framework’s applicability to practical analog design tasks.

5. The comparison with AnalogCoder and AnalogGenie is unfair. How many types of circuits can CircuitFlow generate?

### Questions
How does the proposed model handle device sizing or performance constraints during generation?

Can the authors provide post-sizing simulation results (e.g., gain, pm, ugf) to validate circuit quality, especially for the transistor-level generation?

Is it possible to condition the generation process on performance specifications?

How does the method compare with joint topology-sizing models such as CktGen?

What determines the upper limit of the number of devices that CircuitFlow generates?

### Soundness
2

### Presentation
1

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
This paper presents CircuitFlow, a multimodal flow-matching model for joint analog circuit topology generation and device sizing. It introduces dimension- and modality-wise time sampling to handle heterogeneous variables and uses a graph transformer backbone. With a unified graph representation and a separate pin-prediction module, the method reports state-of-the-art results on OCB and AnalogGenie, and performance can be further improved via a simple continued-denoising post-process.

### Strengths
+ This paper proposes a unified multimodal flow-matching formulation for analog circuit topology and sizing, which is novel and well-motivated.
+ The CircuitFlow framework supports flexible inference tasks (e.g., generating new topologies or editing partial structures), which is practically useful.
+ Experiments show state-of-the-art results on OCB and AnalogGenie, with additional gains from a lightweight post-processing step.

### Weaknesses
- The paper appears to overclaim on sizing. In Table 2, comparisons are limited to data vs. CircuitFlow vs. random, where CircuitFlow’s gain and phase margin are below data, which cannot support the claim that the model has “indirectly learned to maximize all three properties.” And it does not compare the difference between the predicted sizes and the real optimized sizes.
- Although the method section emphasizes the dimension- and modality-dependent time sampling scheme design choice, there is no ablation study demonstrating its specific contribution. 
- Some important experiments are missing. The influence of different circuit graph representations (with or without preprocessing). For the two-stage pin assignment, it is good to have an ablation study for unified v.s. separate processing. 
- The paper uses substantial space to restate the known formulas and preliminaries, but provides insufficient implementation detail for the proposed method itself. For example, a concrete algorithm for the dimension- and modality-  during training and inference would be helpful.

### Questions
For the AnalogGenie (no pins), there are only ~38 nodes and ~60 edges per sample, but you use 1000 denoising Euler steps. Why use so many steps for inference? What is the impact of fewer steps on quality and runtime? Please report inference latency for different inference step settings and compare with the baseline.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a generative model for analog circuits using multimodal flow matching. It jointly generates (1) circuit topology (discrete graph: devices, nets, connections) and (2) device sizing parameters (continuous values). The key claimed contribution is a per-dimension time schedule: each node, edge, and sizing parameter is assigned its own noise time, which the authors say enables conditional tasks like partial completion and repair. The model is evaluated on two circuit benchmarks and reports high validity / uniqueness / novelty / simulability. The problem is important. Automatically generating spec-worthy analog circuits (both topology and sizing) is a major goal in analog EDA.

### Strengths
(1) Tackles a high-impact task (automatic analog circuit synthesis).

(2) Attempts joint topology + sizing generation with one model, which is practically valuable.

(3) Claims ability to “edit/repair” subcircuits or size a fixed topology by selectively denoising only parts of the graph.

(4) Shows promising validity/simulability numbers.

### Weaknesses
(1) The model is largely an application of known multimodal flow matching (discrete jump process for structure + continuous rectified flow for sizing). The only new mechanism claimed is the per-dimension time scheduling. This looks like an incremental extension of the standard factorized noise assumption, not a fundamentally new objective, and it is currently oversold.

(2) **No ablation for the claimed contribution.**
There is no experiment showing that per-dimension time scheduling actually matters. We need a direct comparison against:

(A) single global time for all dimensions,

(B) per-modality time (one for topology, one for sizing),

(C) the proposed per-dimension time (one per node/edge/parameter).


(3) **Architecture underspecified:** 
The paper does not clearly explain how the model enforces circuit legality:

(3.1) how illegal device types / pin assignments are prevented,

(3.2) how symmetric structures (current mirrors, differential pairs) are handled,

(3.3) how the second-stage pin assignment module works at inference.

These constraints are central to analog design and must be described mathematically to make the work reproducible.

(4) **No design-yield / success rate.**
Analog design is a constrained optimization problem, not just “generate something simulatable.” The paper does not report the core metric an analog designer cares about: if you sample N times (e.g. 5–10), what fraction of generated circuits are (i) topologically valid, (ii) fully sized, and (iii) satisfy all required specs (gain, bandwidth/UGBW, phase margin, power) with no post hoc tuning?
Without this “yield,” it’s unclear if the method is actually designing usable circuits or just producing plausible sketches.

(5) **Baselines are weak.**
Apart from CktGNN, most baselines are generic generative models or author-modified references not built for analog circuit synthesis. This makes the reported gains less convincing. The paper does not compare against realistic circuit design pipelines such as:

topology generation + RL/BO/SPICE sizing loops,

retrieval + constraint-guided GNN repair,

LLM-generated SPICE + filtering.
Claiming state of the art without these comparisons is premature. (refer the following work for baselines or related works):

[Related Works]:
- Graph of circuits with GNNs for exploring optimal design space
- Learning to Design Analog Circuits to Meet Threshold Specifications
- GANA: Graph Convolutional Network Based Automated Netlist Annotation for Analog Circuits 


(6) **Related work is incomplete.**
The paper under-cites prior work that already treats circuits as graphs and uses GNNs / RL / optimization to generate or refine analog topologies and size them to meet specs. These should be cited and, where possible, used as baselines.

(7) **Metrics are poorly defined**:
“Validity,” “uniqueness,” “novelty,” and “max node count” are reported but not rigorously defined or tied to actual analog design criteria. It’s unclear:
whether “validity” means “SPICE runs,” or “meets spec,”
how “uniqueness” handles isomorphic netlists,
what level of change counts as “novel,”
what “max node count” means in terms of real design complexity.
As written, these look like generic graph-generation metrics, not design quality metrics.

### Questions
Refer to weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
