# Disco: Densely-overlapping Cell Instance Segmentation via Adjacency-aware Collaborative Coloring

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 2

## Abstract
Accurate cell instance segmentation is foundational for digital pathology analysis. Existing methods based on contour detection and distance mapping still face significant challenges in processing complex and dense cellular regions. Graph coloring-based methods provide a new paradigm for this task, yet the effectiveness of this paradigm in real-world scenarios with dense overlaps and complex topologies has not been verified. Addressing this issue, we release a large-scale dataset GBC-FS 2025, which contains highly complex and dense sub-cellular nuclear arrangements. We conduct the first systematic analysis of the chromatic properties of cell adjacency graphs across four diverse datasets and reveal an important discovery: most real-world cell graphs are non-bipartite, with a high prevalence of odd-length cycles (predominantly triangles). This makes simple 2-coloring theory insufficient for handling complex tissues, while higher-chromaticity models would cause representational redundancy and optimization difficulties. Building on this observation of complex real-world contexts, we propose Disco (Densely-overlapping Cell Instance Segmentation via Adjacency-aware Collaborative Coloring), an adjacency-aware framework based on the “divide and conquer” principle. It uniquely combines a data-driven topological labeling strategy with a constrained deep learning system to resolve complex adjacency conflicts. First, “Explicit Marking” strategy transforms the topological challenge into a learnable classification task by recursively decomposing the cell graph and isolating a “conflict set.” Second, “Implicit Disambiguation” mechanism resolves ambiguities in conflict regions by enforcing feature dissimilarity between different instances, enabling the model to learn separable feature representations. Disco achieves a significant 7.08\% improvement in the PQ metric on the GBC-FS 2025 dataset and an average improvement of 2.72% across all datasets. Furthermore, the predicted “Conflict Map” serves as a novel tool for interpreting topological complexity, offering new potential for data-driven pathology research.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents DISCO, an effective framework for dense cell instance segmentation with detailed theoretically analysis. The authors identify the lack of global topological awareness in existing methods, and provide solution by graph coloring theory. 
Through the introduction of a new GBC-FS 2025 dataset and a systematic cross-dataset topological analysis, the authors show that cell adjacency graphs are predominantly non-bipartite with a high density of odd-length cycles.  DISCO addresses the challenge by the proposed explicit marking and implicit disambiguation mechanism.
Comprehensive experiments across four datasets demonstrate consistent and significant gains, with qualitative visualizations and detailed ablations further verify Disco's effectiveness.

### Strengths
1.The authors construct a large-scale GBC-FS 2025 benchmark which contains highly complex and dense sub cellular structures. The also report the comparisons of recent models on the benchmark. 

2.The paper conduct systematic, quantitative analysis of the complex topology of cell adjacency graphs and reveal their inherent non-bipartite nature. They also establish a clear conceptual shift from local geometric modeling to global topological reasoning for instance segmentation.

3.The “divide and conquer” strategy handles bipartite regions efficiently while explicitly modeling non-bipartite conflict clusters. The integration of “Explicit Marking” and “Implicit Disambiguation” aligns well align with the theoretical analysis.

### Weaknesses
1. The method introduces multiple loss components and graph-based computations. Although training settings are reported, there is limited discussion or empirical evidence regarding runtime and memory overhead compared to baselines such as FCIS or HoverNet. This omission is particularly notable since the paper highlights DISCO’s efficiency advantage over FCIS.

2. While the topological observations are empirically sound, the paper stops short of providing formal theoretical bounds or proofs. Incorporating theoretical justification such as convergence analysis or expected bounds would enhance the rigor and clarity of the claims.

3. It would be beneficial to discuss or empirically compare against recent GNN-based segmentation or relational reasoning approaches. This would help contextualize the novelty and positioning of the proposed framework within the broader landscape of graph-based learning methods.

4. Although the paper provides an anonymous repository, the released materials currently lack full access to the proposed GBC-FS dataset and the complete implementation of DISCO. This incomplete release raises concerns about reproducibility. It would be helpful to clarify whether the dataset and code will be fully released upon publication, as stated in the reproducibility section.

### Questions
1. Could the authors provide quantitative evidence (e.g., runtime, GPU memory usage) to support the claim that DISCO is more efficient than FCIS, particularly given its additional loss terms and graph computations?

2. Do the authors plan to fully open-source the GBC-FS dataset and the complete DISCO implementation after the review phase? 

3. Can the authors offer theoretical insights or proofs regarding the efficiency or convergence of DISCO?

4. Have the authors considered comparing DISCO to GNN-based instance segmentation or relational reasoning methods?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the instance segmentation of dense, overlapping cells in pathology images. The authors present three main contributions: 1) A systematic, quantitative topological analysis of Cell Adjacency Graphs (CAGs) across four datasets, finding that real-world cell graphs are non-bipartite, dominated by 3-cycles, and empirically have a chromatic number $\chi(G)$ that is almost always 3. 2) The release of a new, large-scale, high-density dataset, GBC-FS 2025, which features unique "sub-nuclear" level annotations. 3) A proposed segmentation framework, Disco, which uses a "divide and conquer" (2+1) coloring strategy ("Explicit Marking") and an "Implicit Disambiguation" mechanism ($\mathcal{L}_{adj}$ loss) to handle topological conflicts.

### Strengths
1. The paper provides the first systematic quantitative topological analysis of CAGs. Its key finding (that real-world cell graphs are non-bipartite, 3-cycle dominant, and empirically have $\chi(G)=3$) provides a solid empirical foundation for the graph-coloring paradigm in this field. This analysis moves beyond simple theoretical assumptions (e.g., bipartiteness or the 4-color theorem) and points toward designing more efficient, targeted models.

2. The release of the GBC-FS 2025 dataset is a contribution. Its extremely high instance density (avg. 304.44/patch) and unique "sub-nuclear" annotation granularity provide a novel, highly challenging benchmark for the community, filling a gap left by existing datasets in extremely dense scenarios.

### Weaknesses
1. The (2+1) "Explicit Marking" strategy of Disco is logically flawed. The authors' own analysis (Appendix A.4.2) clearly states that an empirical chromatic number of $\chi(G)=3$ is almost always sufficient. However, instead of generating an unambiguous 3-color label (as shown in Appendix A.1(d)), the authors' strategy lumps all conflicting nodes—regardless of adjacency—into the same conflict class ($c=t=3$). This artificially creates "secondary conflicts" at the label level (i.e., two adjacent conflict nodes are assigned the same ground-truth label). Consequently, the "Implicit Disambiguation" mechanism (especially the $\mathcal{L}_{adj}$ loss) appears to be a complex solution to an avoidable problem introduced by the labeling strategy itself. This is an unnecessarily convoluted design.

2. The method's most significant performance gain (a 7.08% PQ improvement) is reported on the authors' own GBC-FS 2025 dataset. However, as described in Appendix A.6.2, this dataset uses "sub-nuclear instances" for annotation. This is a fundamentally different task from the "nucleus segmentation" task in benchmarks like PanNuke and DSB2018. This "apples-to-oranges" comparison invalidates the claim of superior performance. Is Disco's superiority on GBC-FS 2025 attributable to its superior handling of complex topology, or is it simply better at learning this specific and unusual "sub-nuclear segmentation" task? This ambiguity in task definition severely weakens the paper's central claims about its topological robustness.

3. The abstract claims that the effectiveness of the graph coloring paradigm "has not been verified" in real-world scenarios. This is inaccurate and contradicts Section 2.2, which acknowledges the "pioneering work of FCIS (2025c)" demonstrating the "potential of a universal 4-coloring model." The paper's topological contribution is the quantification of 3-cycle prevalence, not the novel discovery of non-bipartite structures (which is already implied by FCIS's 4-coloring). This framing exaggerates the paper's novelty.

### Questions
1. Given your own empirical evidence (Appendix A.4.2) that $\chi(G)=3$ is almost always sufficient, why did you opt for the (2+1) "Explicit Marking" strategy, which creates label ambiguity, instead of using a standard, unambiguous greedy 3-coloring algorithm (as shown in your Appendix A.1(d)) as the supervisory signal?

2. How can you decouple the performance gains on GBC-FS 2025 from the unique "sub-nuclear" task definition to prove that Disco's advantage truly stems from its topological handling capabilities? Did you consider an ablation study where you merge the "sub-nuclear" annotations into standard "nucleus" annotations to conduct a fair comparison?

3. The ablation study (Table 7) shows that $\mathcal{L}_{adj}$ provides a massive ~6% PQ boost. In light of "Weakness 1," does this not simply prove that $\mathcal{L}_{adj}$ is highly effective at resolving the specific label ambiguity you introduced, rather than proving it is a generally superior mechanism for handling topological conflicts compared to a clean 3-color supervision?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses cell instance segmentation in dense, overlapping regions by reformulating the problem through graph coloring theory. The authors present three main contributions:

Dataset: GBC-FS 2025 (Gallbladder Cancer Frozen Section), comprising 2,839 H&E-stained frozen section images with 864,204 manually annotated sub-cellular nuclei instances. This represents a 40× scale increase over CryoNuSeg, and addresses a critical gap for intraoperative diagnosis algorithms.

Topological Analysis: A systematic cross-dataset study of cell adjacency graph (CAG) properties across four benchmarks is presented. The authors construct CAGs where nodes represent cell instances and edges represent 8-connected spatial adjacency. Key findings include: (1) PanNuke is 100% bipartite while DSB2018 (1.99%), CryoNuSeg (5.64%), and GBC-FS 2025 (30.49%) contain increasing proportions of conflict nodes that violate bipartite structure; (2) Among non-bipartite components, 90.51-98.12% of odd-length cycles are 3-cycles (triangles), with GBC-FS 2025 showing 24.64% secondary conflict nodes (adjacent conflict nodes). 

Method: DISCO employs a "divide and conquer" strategy with two core mechanisms. "Explicit Marking" uses BFS-based graph decomposition to extract the maximal bipartite subgraph, partitioning cells into two primary colors (V₁, V₂) and consolidating remaining non-bipartite nodes into a conflict set (Vconf) assigned a third color. This generates a (t+1)-value ground truth map where t=3. "Implicit Disambiguation" introduces an adjacency constraint loss (L_adj) that minimizes cosine similarity between mean probability vectors of adjacent instances, forcing the model to learn angularly separated representations in probability space. The total loss combines semantic segmentation (L_sem), weighted coloring (L_color), bipartite consistency (L_cons), conflict resolution (L_conf), and adjacency constraints (L_adj). At inference, instances are reconstructed by grouping connected components with identical color predictions.

Results: DISCO achieves PQ of 62.71% on PanNuke (+1.62% vs FCIS), 77.81% on DSB2018 (+0.42%), 59.70% on CryoNuSeg (+1.77%), and 50.87% on GBC-FS 2025 (+7.08%), representing an average 2.72% improvement across benchmarks. Ablations demonstrate the adjacency constraint loss contributes ~6% absolute PQ improvement (42.57% → 48.26%) on GBC-FS 2025.

### Strengths
Compelling topology evidence. Cross‑dataset analysis shows frequent odd cycles with >90% of odd cycles being triangles in non‑bipartite graphs; conflict/secondary‑conflict ratios quantify difficulty (Table 1, p. 2; Fig. 3b, p. 5).

Clean “2 + 1” design. Explicit Marking extracts a large bipartite backbone and pools the rest into a conflict set; Implicit Disambiguation resolves label ambiguity via 𝐿_adj over CAG edges (Sec. 4.2–4.3; Eqs. 1–3, pp. 6–7).

Graph‑aware loss that works. Ablations show 𝐿_adj alone adds ~+6 PQ on GBC‑FS 2025; the full system achieves PQ = 0.5087 (Table 7; Table 5, p. 9).

State‑of‑the‑art on dense regime. Disco surpasses FCIS by +7.08 PQ on GBC‑FS 2025 and improves AJI/DQ/SQ concurrently (Table 5, p. 9); visuals highlight separation in dense clusters (Fig. 7, p. 8).

High‑value dataset. GBC‑FS 2025: 2,839 patches, ~304 instances/patch on average, 864k+ sub‑nuclear instances; orders‑of‑magnitude denser than public sets (Sec. 5.1; App. A.6; Table 8, p. 20).

Reproducibility intent. Code and dataset release promised (Reproducibility Statement, p. 10).

### Weaknesses
1. Is there a formal definition of "dense" and "complex" cells for segmentation analysis? If no, can the authors quantify this aspect of the data to define "highly dense" cell segmentation?

2. "First Topological Analysis" claim is overstated: The repeated assertion of conducting the "first systematic topological analysis" of cell adjacency graphs appears throughout the paper: This claim is incorrect given extensive prior work: Topological Tumor Graphs (Failmezger et al., Cancer Research 2020), Ceograph (Wang et al., Nature Communications 2023), HistoCartography (Pati et al., Medical Image Analysis 2021), CellSpatialGraph (Chen et al., 2022, Software Impacts Journal), SpaGCN (Hu et al., Nature Methods 2021), GraphST (Long et al., Nature Communications 2023), etc. Thus, there are several  graph-based topological analysis papers on cell spatial arrangements.
	
However, the proposed specific analysis of chromatic numbers, bipartite versus non-bipartite classification, odd-cycle length distributions, conflict node ratios, and secondary conflict prevalence seem novel to this manuscript..Thus, the contribution seems "first systematic analysis of chromatic properties and graph coloring characteristics in cell adjacency graphs" not "first topological analysis."

The authors should consider adding a new paragraph in Introduction acknowledging: "While graph-based analysis of cell spatial arrangements is well-established in computational pathology [cite TTG, Ceograph, HistoCartography] and spatial transcriptomics [cite SpaGCN, GraphST], prior work has focused on node centrality, clustering, and GNN-based prediction rather than graph coloring characteristics. To our knowledge, this is the first systematic study of chromatic properties…”. Also, the abstract calls the predicted “Conflict Map” an “unsupervised tool”, though conflict labels for training are produced from GT via Explicit Marking; wording may mislead (Abstract, p. 1).

3.  Missing Critical Citation: The adjacency constraint loss (Section 4.3, Equation 3, L_adj) has very close conceptual precedent that must be cited and differentiated: NonAdjLoss (Ganaye et al., "Removing segmentation inconsistencies with semi-supervised non-adjacency constraint", in Medical Image Analysis 2019). Both NonAdjLoss and the proposed method use adjacency graphs to penalize unwanted relationships. The paper should provide explicit technical comparison, with NonAdjLoss, discussing this conceptual similarity and explaining how instance disambiguation loss differs from anatomical consistency loss beyond just the application domain.

4. Insufficient Comparison with FCIS (ICML 2025): FCIS (Zhang et al., "The Four Color Theorem for Cell Instance Segmentation," ICML 2025, arXiv:2506.09724) is the most directly comparable concurrent work and requires deep technical comparison.

Why is divide-and-conquer superior to direct 4-color prediction? The paper shows DISCO outperforms FCIS on GBC-FS 2025 (+6.91% AJI) but provides no explanation of why the proposed divide-and-conquer is superior to direct 4-color prediction in FCIS?

Computational efficiency comparison is missing. FCIS claims efficiency advantages by reducing to semantic segmentation. Does DISCO maintain or sacrifice efficiency for accuracy? Need: training time, inference speed (FPS), memory requirements, FLOPs comparison.

Would FCIS-style encoding work on GBC-FS 2025? A critical ablation would be testing uniform 4-color prediction (FCIS approach) with DISCO's backbone and training procedure on GBC-FS 2025. This would isolate whether improvements come from divide-and-conquer versus other factors (architecture choices, training hyperparameters, etc.).

5. Incomplete Dataset Documentation: GBC-FS 2025 is a strong contribution but lacks essential documentation that limits its utility.

Tissue Composition: What proportions are:
* Tumor tissue (epithelial cancer cells)?
* Stroma (fibroblasts, extracellular matrix)?
* Inflammatory infiltrate (lymphocytes, macrophages)?
* Necrotic regions?
* Normal tissue?

This matters because algorithm performance varies dramatically across tissue types. Methods optimized for dense tumor nests may fail on sparse inflammatory regions.

Source Data:
* How many whole slide images (WSIs) produced the 2,839 patches?
* How many patients?
* What is patient-level diversity (age range, disease stage, tumor grade)?
Annotation Methodology:
* Manual pixel-wise annotation by pathologists? How many pathologists?
* Crowdsourcing platform (like NuCLS using novel protocols for 220K+ annotations)?
* Semi-automated pipeline (like SNOW using HoVer-Net on StyleGAN2-ADA synthetic images)?
* What annotation tools/software?
* What is annotation time per image?
Quality Control:
* Inter-annotator agreement metrics (Dice scores, IoU between multiple annotators)?
* How was consensus achieved for disagreements?
* What validation ensured correct capture of sub-cellular nuclei versus artifacts?
* What percentage of annotations required adjudication?
Quantitative Density Metrics:
* Cells per mm² (accounting for magnification)?
* Percentage of overlapping/touching cells?
* Distribution of neighbor counts quantitatively?
* Comparison to other datasets' density metrics?
The paper reports 304.44 cells per 256×256 patch but doesn't translate to physical density or overlap percentage.

6. No Computational Cost Analysis Prevents Efficiency Assessment, the paper reports no computational metrics:
* Training time (hours to convergence)?
* Inference speed (FPS, seconds per image)?
* Memory requirements (peak GPU memory during training/inference)?
* Model size (number of parameters)?
* FLOPs per forward pass?

7. Missing Biological Interpretation of Triangle Dominance Finding: The 90.51-98.12% triangle dominance in odd cycles is potentially the most interesting finding but lacks biological grounding. Why would cell packing favour triangles? Why not hexagonal packing for epithelial cells? Are these findings consistent across FFPE and frozen sections?

8. Limited Baseline Coverage Missing Recent Methods: The baseline selection is good but misses most recent methods representing 2024-2025 state-of-the-art such as CellSAM (Israel et al., bioRxiv 2023.11.17.567630), Cell-TRACTR (O'Connor & Dunlop, PLOS Computational Biology 2025, 21(5), e1013071), and SwinCell (Zhang et al., Communications Biology 2025, 8, 962).

9. Missing Statistical Analysis: No error bars, confidence intervals, or standard deviations and no significance testing (t-tests, Wilcoxon tests across images) reported.

10. Missing Failure Case Analysis: Are there cases where graph colouring fails? May be for FFPE → frozen section transfer? H&E → fluorescence transfer? Trained on one organ, tested on another? Different scanners/staining protocols?

11. Decoding unspecified: “Topological decoding” is mentioned but not detailed: how per‑pixel colors are converted into final instances (connectivity, merging/splitting, conflict assignment) is unclear (Fig. 1d, p. 2)

12. Explicit Marking under‑specified: The BFS‑based conflict labeling heuristic lacks tie‑breaking rules for overlapping odd cycles and complexity bounds, though supervision hinges on it (Sec. 4.2, p. 6; Fig. 5).

13. Adjacency sensitivity: CAG edges rely on 3×3 dilation (8‑connectivity) (Def. 1, p. 3); there’s no ablation for kernel size/shape or alternative proximity rules, which can alter odd‑cycle counts and conflict sets.

14. Figure 4 captions is very abstract and does not provide an overview of the proposed method. Could that be revised?

### Questions
See weaknesses.

### Soundness
2

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
4

### Summary
In this work, the authors deal with dense cell instance segmentation in histopathology images. They introduce a new dataset, analyze bipartite nature of 4 datasets, and propose DISCO, a divide and conquer approach for the segmentation task.

### Strengths
- This is the first study revealing that non-bipartite is actually more prevalent in cell datasets, and hence the need for more sophisticated methods aside from 2-coloring or 4-coloring approaches. The authors provide statistics on percentage of bipartite nodes, conflict nodes, and that 3-cycles are more prevalent among the odd-length cycles.
- The authors perform adequate experiments, in terms of 4 datasets, several recent and relevant baselines and ablation studies. Their method is usually either the best or second-best in performance.

### Weaknesses
- The actual method doesn't seem very novel, as BFS exists in literature, and forcing adjacent nodes to have different feature representation is common
- The authors do not provide standard deviation in the results. The mean numbers seem very close to baselines. t-test needs to be done to determine if the results are statistically significant or not.
- Authors need to provide inference run-times. Because constructing the graph and performing BFS can be time-consuming.
- Hyperparameter tuning: Authors need to provide results of different hyperparameter values (such as loss weights) to understand sensitivity of each term.

### Questions
Please see the weakness section. I am willing to increase the score after seeing authors' rebuttal and discussing with other reviewers.

### Soundness
3

### Presentation
3

### Contribution
3
