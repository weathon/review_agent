# Generating Directed Graphs with Dual Attention and Asymmetric Encoding

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Directed graphs naturally model systems with asymmetric, ordered relationships, essential to applications in biology, transportation, social networks, or visual understanding. Generating such graphs enables simulation, data augmentation and novel instance discovery; however, this task remains underexplored. We identify two key reasons: first, modeling edge directionality introduces a substantially larger dependency space, making the underlying distribution harder to learn; second, the absence of standardized benchmarks hinders rigorous evaluation. Addressing the former limitation requires more expressive models that are sensitive to directional topologies. Thus, we propose Directo, the first generative model for directed graphs built upon the discrete flow matching framework. Our approach combines: (i) a dual-attention mechanism distinctly capturing incoming and outgoing dependencies, (ii) a robust, discrete generative framework, and (iii) principled positional encodings tailored to asymmetric pairwise relations. To address the second limitation and support evaluation, we introduce a novel and extensive benchmark suite covering synthetic and real-world datasets. Experiments show that our method outperforms existing directed graph generation approaches across diverse settings and competes with specialized models for particular classes, such as directed acyclic graphs. These results highlight the effectiveness and generality of our approach, establishing a solid foundation for future research in directed graph generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
DIRECTO is a discrete-state, iterative-refinement generator specifically designed for directed graphs. It introduces two key enhancements: (1) direction-aware positional encodings, and (2) a dual-attention transformer that explicitly combines source-to-target and target-to-source channels. The training process employs discrete flow matching, and the authors also present a discrete-diffusion variant. The generated graphs are evaluated on various benchmarks, including synthetic directed/DAG distributions, TPU compute DAGs, and Visual Genome scene graphs. The evaluation metrics focus on validity, uniqueness, and novelty, as well as a normalized MMD Ratio and label-aware distances on real-world data.

### Strengths
Treating directionality as a first-class citizen (both in PEs and attention) is long overdue. Dual attention is a clean, architecture-level bias that addresses asymmetry.

The comparison between “dual vs double depth” demonstrates a genuine architectural advantage, not merely an increase in capacity.

Evaluating typed constraints on Visual Genome and acyclicity on DAGs is the appropriate approach. The V.U.N. effectively penalizes memorization.

The same directional concepts also apply to discrete diffusion (DIRECTO-DD), indicating that they are not exclusive to DFM-specific techniques.

### Weaknesses
1. Directed attention doubles attention maps, while MagLap/Multi-q PEs are expensive. Additionally, CTMC sampling requires numerous steps to achieve quality. The paper claims decent scaling, but we need to compare it to strong autoregressive digraph models on larger graphs, considering wall-clock time and VRAM usage versus.

2. The benchmarks are relatively small and close to the training support. We need out-of-distribution (OOD) tests, such as altered degree exponents, flipped community asymmetry, or different label marginals. This will help us understand how brittle dual attention is when arrow statistics shift.

### Questions
How much of the performance improvement is attributed to (a) splitting the roles (S/T) versus (b) employing the aggregation trick (concat two maps and then apply one softmax)? Please provide an ablation study that distinguishes between (i) role-splitting alone, (ii) role-splitting with FiLM edge modulation, and (iii) role-splitting with a unified softmax.

Were the sampling steps, time, and VRAM used in each model (including DIRECTO-DD) consistent? Please include a fairness table.

The authors consider every ordered pair as a categorical “edge (including absent).” How is the graph size N determined at the generation time, and do the metrics account for N?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Directo, a method for generating directed graphs based on discrete flow matching. Great care is given to the denoising model arch. To enhance the capability of the method to handle edge directionality, 1) a tailored attention mechanism called "dual attention" is proposed; and 2) asymmetric position encodings, such as those based on directional Laplacians, are used. Other GNN components from prior work, FiLM and PNA, are also incorporated into the arch. Besides the method, the paper also creates a benchmark for directed graph generation, with several synthetic and real-world datasets. Directo is shown to have strong performance on this benchmark relative to other methods, based on evaluation metrics adapted to the setting. Finally, further experiments are discussed that explore the impact of ablating the major model arch components, as well as the scalability of the method.

### Strengths
- The area of directed graph generation seems important but relatively unexplored given the many works for undirected graph generation.
- The writing is clear, grammatical, and well-organized.
- The paper introduces not only a new method, but also a benchmark for the area of directed graph generation. A code repository is included for reproducibility.
- Ablations are included to justify major new components of the model arch.

### Weaknesses
- The proposed arch is complex, and while there are ablation studies for some important components, this is not true of all components, e.g., use of FiLM.
- As the authors note, the proposed method can fail to maintain validity when scaling up (e.g., failing to maintain strict acyclicity beyond 200 nodes).
- Ideally the broad setting of the paper in terms of the scale of generated graphs would be clarified earlier in the paper. Graph generation papers and algorithms roughly cluster on two categories, those for 10s-100s of nodes (this paper), and others for 1000s-10000s and up.
- (nit) Some notation could be easier to read, e.g., the use of Kronecker delta in Eq 1 could be replaced with piecewise notation.

### Questions
- Much of the main paper describes a complex GNN arch tailored for directed graphs, which could be applied outside of the graph generation setting, e.g., to link prediction or node classification. Has this been attempted? It seems worthwhile to evaluate the arch on the more standardized benchmarks for those tasks. If the results are weak on other tasks but strong for generation, that is also an interesting finding.
- Relatedly, there is little discussion of the flow matching part in the main paper. Were there any interesting findings related to the flow matching as opposed to the arch that have not been discussed in prior work?
- Regarding the time distortion function, it is stated that "these functions are selected based on dataset-specific properties to improve fidelity and structural constraints without need for retraining." Could you please clarify the selection procedure, given that there is not a unique objective unlike in typical hyperparameter selection via cross-validation, e.g., for node classification?
- For graph generation papers in general, a core theme is the trade-off between generating diverse graphs (U.N. from V.U.N.) and graphs that match the training distribution ("ratio" in this paper). Is there some way to tune the proposed method to favor one or the other? How does the ease of such tuning compare to other methods?

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
This work proposes a model for generating directed graphs. Its main contributions are in (i) the use of direction-aware positional encodings, (ii) a dual attention mechanism that again takes edge directions into account and (iii) the introduction of new benchmarks for evaluating directed graph generative models. The performance is measured across different synthetic and true graphs and seemingly outperforms other undirected GGM or specialized in generating DAG models. An ablation study shows that the dual attention mechanism is critical in the increased performance.

### Strengths
* The model proposed in this work succeeds in generating graphs with specific constraints (acyclic, or edges only across certain types) even if those constrains are not explicitly stated (e.g., through some regularization parameter) during the training process. 
* The architecture is quite generic in that it can be used to generate directed graphs of virtually any type as edges are encoding through categorical variables, while it can also accomodate node and graph features.

### Weaknesses
*  The model itself, at least as presented, is not self-contained. Section 2 which provides a background on diffusion models and past work seems disconnected to Section 3 that describes the attention mechanism employed. For example, it is not specified how edges are modeled through categorical variables (e.g, is it representing one of the 4 possible classes -- edges in both directions, in one of the two (x2), or absent?), how the rate matrix is parameterized though the proposed architecture, while fig. 2 employs global features that are not part of the loss function in eq. 4.
* The complexity of the proposed model seems to make it a bit susceptible to choice of hyperparameters (see fluctuation in performance in fig 4).
* The novelty of the work is on the dual attention mechanism, but it does not extend to other parts of it (e.g., new positional encoding, diffusion process is of DFM, evaluation metrics is MMD).

### Questions
Q1: are global features utilized/ part of the training loss? what is the categorical modeling for edges for the datasets the model is evaluated against?
Q2: What is the benefit of MagLap encoding vs the directed laplacian of chung?
Q3: Ratio comes from averaging the MMD for different descriptors. Is it though a good practice to average MMD for different descriptors when it is unbounded and lacks scale comparison?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors present DIRECTO, a new method for generating directed graphs, an area that has been underexplored in graph generative modeling. The approach builds upon the discrete flow matching framework and introduces two key architectural components: (1) a dual attention mechanism that captures both source-to-target and target-to-source representations, and (2) direction-aware positional encodings. The authors indicate that the dual attention component is more critical to performance than the positional encodings. Additionally, they contribute a new benchmark suite containing synthetic and real-world datasets, along with metrics tailored for evaluating directed graph generation quality. The paper demonstrates strong empirical results across diverse settings. While the work provides a solid foundation for directed graph generation, conditional generation remains relatively underexplored and is identified as an important direction for future work.

### Strengths
- Novel and Important Problem: The paper addresses a significantly underexplored area in graph generation. The focus on directed graphs is well-motivated, with clear applications in biology, transportation networks, and scene understanding.
- Comprehensive Technical Approach: The dual attention mechanism is presented as a solution to capture bidirectional dependencies inherent in directed graphs. The integration with discrete flow matching provides a principled generative framework with theoretical grounding.
- Extensive Empirical Evaluation: The paper presents thorough experiments across multiple datasets, baselines, and ablation studies. The results consistently demonstrate the effectiveness of the proposed approach.
- Benchmark Contribution: The introduction of standardized benchmarks for directed graph generation is a valuable contribution that will facilitate future research in this area. This work can provide a solid evaluation suite to later directions in the field.
- Clear Presentation: The paper is well-written with clear motivation, technical exposition, and comprehensive experimental analysis. All visualizations seem relevant and convey information clearly.
- Thorough Ablations: The systematic ablation studies effectively isolate the contributions of different components.

### Weaknesses
- Unclear Claims on Expressiveness: While the authors claim their method is "expressive" and "robust," these claims lack formal theoretical justification or empirical evidence. The notion of expressiveness in the context of directed graph generation needs clearer definition and supporting arguments.

- Scalability Concerns vs. Efficiency Claims: The paper mentions "efficient generation" but simultaneously acknowledges scalability limitations. This contradiction needs clarification. Table 20 shows significant performance degradation for larger graphs (200-250 nodes), which undermines efficiency claims.

- Conditional Generation Underexplored: While acknowledged as future work, the conditional generation experiments (Section H.7) are limited. Given the practical importance of conditional generation for real-world applications, this might deserve more attention.

Limited Architectural Justification: Several design choices appear ad-hoc:
- Why is independent interpolation used in the noising process (Eq. 1). It is this unclear if this is a simplifying assumption or theoretically motivated.
- The gated residual connection for node features (Eq. 23-24) lacks justification compared to standard residual connections used elsewhere.

### Questions
- Positional Encoding Design Choice: The authors concatenate positional encodings to node and edge features rather than using more integrated approaches common in modern transformers (e.g., addition, rotational embeddings like RoPE, or ALiBi). What is the rationale for choosing concatenation? Has the impact on parameter efficiency been considered, given that concatenation increases the input dimensionality? Were alternative integration methods explored?

- Figure 1b Clarification: What do the node colors represent in Figure 1b? This visual element should be explained or removed if purely aesthetic.

- Independent Interpolation Assumption: Why is independent interpolation used in the noising process (Eq. 1)? Is this a simplifying assumption for computational tractability, or is there theoretical justification for treating nodes and edges independently during the noising process?

- Architectural Choices: What is the motivation for the gated residual connection in the node feature update (Eq. 23-24) when standard residual connections are used elsewhere? Has ablation been performed on this component?

- Section 3.2 Reference: The last paragraph mentions ablation studies but doesn't specify where results are presented. Could you add a forward reference?

- TPU Tiles Dataset: For the TPU Tiles dataset, was the preprocessing identical to Li et al. (2025), or were there modifications? This should be clarified.

### Soundness
3

### Presentation
4

### Contribution
3
