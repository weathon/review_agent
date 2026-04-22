# Bento: Benchmarking Classical and AI Docking on Drug Design–Relevant Data

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Recent advances in artificial intelligence have introduced deep learning and co-folding approaches for predicting protein-ligand complexes, raising the question of their applicability and how they compare with classical docking methods. In this work, we present a thorough benchmarking study of eleven tools for protein-ligand interaction prediction, spanning classical molecular docking methods, deep learning-based models, and co-folding algorithms. While most related benchmarking efforts primarily assess the generalization capacity, we extend the analysis to also evaluate the performance on drug design-relevant data and across different classes of protein-ligand complexes. Here, we introduce \textsc{Bento}, a comprehensive benchmark that evaluates 11 tools for protein-ligand interaction prediction -- both established and recently developed -- across four test datasets and multiple derived subsets in a pocket-aware setup. We show that 1) careful dataset curation is essential -- filtering by pocket structural similarity and controlling ligand complexity exposes generalization failures that are obscured in conventional benchmarks; 2) classical and deep learning-based docking tools perform similarly well on drug-like ligands, making them comparably useful for virtual screening, with physics-based methods offering a clear advantage in speed; 3) co-folding tools outperform other approaches on structurally complex ligands, whereas most methods achieve similar accuracy on regular small molecules; and 4) all methods struggle to generalize to unseen pockets, with deep learning models being the most prone to overfitting. Overall, our results show that while current docking and DL-based approaches are reliable for many drug-design-relevant scenarios, genuine pocket-level generalization remains an open challenge. \textsc{Bento} provides a rigorous and transparent framework for diagnosing these limitations and guiding the development of more robust protein-ligand prediction models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors introduce Bento, a new comprehensive benchmark for protein-ligand interaction prediction. The design and scope of this benchmark are compelling, and the authors' results and conclusions raise important questions for future work in developing deep learning models for protein-ligand interaction modeling.

### Strengths
1. The authors introduce the (now) largest benchmark for protein-ligand interaction prediction, which covers a wide variety of prediction use cases.
2. The authors' results and discussions highlight the importance of training new deep learning models with stronger generalization to novel protein binding pockets/interacting ligands.
3. The design of the benchmark's evaluation procedures follows best practices in the field.

### Weaknesses
1. As far as I can see, the authors haven't analyzed whether or how much deep learning methods' performance can be affected by the application of post-hoc structural relaxation via molecular dynamics software. This would be an interesting experiment to explore.
2. I did not find the benchmark's source code attached in the supplementary materials, which raises the question of whether the authors plan to open-source such code in the future. I would strongly encourage them to do so (if possible) and to prepare an online leaderboard as well.

### Questions
1. Is "This progress highlighting" a typo in Line 034?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces BENTO, a benchmark of 11 protein–ligand pose prediction tools (classical docking, DL docking, and co-folding) across four public test sets, with curated subsets stratified by ligand class/complexity and by ligand/pocket similarity to "training sets" (PDBBind train and Binding MOAD). Authors found that AlphaFold3 performs best on structurally complex ligands. For typical small-molecule cases, several methods are comparable. While DL docking tends to overfit, Gnina appears most robust under pocket shift. By analyzing results of the models with various kinds of splitted benchmark datasets, the authors emphasize the need to evaluate not only on challenging OOD sets but also on homogeneous, drug-design–relevant subsets.

### Strengths
- **Well-scoped tool taxonomy and coverage.** The paper clearly separates DL docking (protein structure as input) from co-folding (protein sequence + ligand) and evaluates 11 tools spanning both categories.
- **Thoughtful dataset stratification.** Besides Timesplit/PoseBusters/Astex/DockGen, BENTO introduces subsets by ligand class, ligand complexity (size, rotatable bonds, burial), drug-like criteria, and pocket/ligand similarity using G-LoSA GA-score and Tanimoto thresholds.
- **Several interesting findings.** The highest ligand similarity in DockGen benchmark, good performance of Matcha (which predicts conformation based on translation, rotation, torsion Riemannian flow matching) in highly complex ligand molecules and etc.

### Weaknesses
- **Key experimental details are unspecified.** The paper does not elaborate on per-tool training dataset; "training sets" are treated at the benchmark level (PDBBind train/Binding MOAD/time-split criteria) for similarity analysis, making it hard to interpret generalization claims for each method. Also, details of running each tool are missing.
- **Pocket-conditioning realism and consistency.** Authors used P2Rank centers for Uni-Mol V2. However, when evaluating the performance of a pose prediction model, introducing P2Rank may create unintended biases unrelated to the model itself. Therefore, since models like Boltz-2 support pocket specification, it would be better to compare performance with and without pocket conditioning.
- **Lack of novelty.** Given that the paper synthesizes prior benchmarks rather than introducing a new one, the novelty is modest.

### Questions
1. **NeuralPLexer (NP2) categorization.** Table 1 places NeuralPLexer under DL docking rather than co-folding. As far as I know, NP2 can also generate the structure from sequence, so I've thought that this can be classified into co-folding model. Why did authors classified NP2 differently with the other co-folding models?
2. **Multi-ligand complexes.** How were cases with cofactors/ions/secondary ligands handled during inference and evaluation—were additional ligands kept, and if not, how might their absence affect PB-valid? (Only a single reference ligand and true-center usage are described.)
3. **Boltz-2 PB-validity.** PB-validity appears low for Boltz-2 in several subsets. This is directly opposite result compared to the Boltz-1x utilizing FK-steering with PB potential. Can the authors provide more details about running Boltz-2?
4. **Claim about Gnina’s training.** The text states Gnina "was not trained on PDBBind or Binding MOAD.", but they actually used their own CrossDocked dataset, which curated from PDBBind dataset. Please clarify the exact training dataset and provide a citation.
5. **Why Matcha excels on highly flexible ligands?** Can the authors provide an ablation/hypothesis (architecture, sampling, scoring) explaining Matcha's favorable behavior on flexible ligands? (Currently reported empirically without mechanism.)
6. **Details about experiment for different ligand classes.** How did authors predict the cofactor ligands? PoseBusters dataset contains many multi-ligand systems with cofactor and small molecule ligand. In this system, did authors only used cofactor, which consists of a single heavy atom, rather than using all non-polymer entities in the system?

## Typos
- Odd inequality symbols in the ligand-similarity thresholds in Appendix B.3.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents BENTO, a comprehensive benchmark that evaluates 11 protein-ligand interaction prediction tools, spanning classical physics-based methods, various deep learning (DL) models, and co-folding approaches. The work does not introduce new raw data but instead curates and re-partitions existing, well-known datasets (PDBBind Timesplit, PoseBusters, Astex, DockGen). The core contribution is the systematic analysis of tool performance across multiple subsets stratified by ligand class, ligand complexity, drug-likeness, and, crucially, the similarity of both protein pockets and ligands to common training sets. The study aims to disentangle these factors to provide a more nuanced understanding of tool performance, particularly concerning generalization to novel targets and applicability in real-world drug design scenarios.

### Strengths
1. The benchmark is extensive, evaluating a wide and timely range of 11 different tools, including very recent methods like AlphaFold3, FlowDock, and Matcha, which provides a valuable, up-to-date comparison.
2. The primary strength lies in the meticulous curation of datasets into focused subsets. This stratification allows the authors to move beyond simplistic overall performance metrics and provide deeper insights into how factors like ligand complexity or pocket similarity specifically impact the performance of different classes of tools.
3. The paper provides practical takeaways for the community. For example, it highlights AlphaFold3's strength on complex molecules, Gnina's impressive robustness and performance on drug-like tasks, and the persistent generalization challenges for many pure DL-based methods. These findings are useful for guiding researchers in selecting the appropriate tool for their specific task.

### Weaknesses
1.  The benchmark is constructed entirely from previously published datasets. While the curation and analysis are valuable, the work does not contribute new primary data to the field. Furthermore, some of the conclusions, while systematically demonstrated here across a broader set of tools, reinforce trends already known in the community. For instance, the observation that DL models can struggle to generalize to proteins with binding pockets dissimilar to their training set has been a key takeaway from prior benchmarks like DockGen and PoseBusters. Similarly, the finding that larger and more flexible ligands are inherently more difficult to dock is a well-established principle.
2.  The paper mentions other benchmarks in the introduction, but it lacks a formal "Related Work" section. Such a section would be crucial for formally positioning BENTO within the existing landscape of benchmarking efforts. It would allow the authors to more clearly contrast their methodology and objectives with preceding work and better articulate the specific gaps their benchmark aims to fill.
3.  The visual presentation of results needs improvement. In several key figures (e.g., Figures 2, 4, 5, and others), the text labels, numbers, and data points are overlapping and crowded. This makes the plots difficult to read and interpret, detracting from the paper's ability to clearly communicate its quantitative findings. Given that a benchmark paper relies heavily on graphical comparisons, ensuring the clarity of all figures is essential.

### Questions
See the weaknesses detailed above. My main suggestions are to more explicitly frame the novelty of the findings against previous work (perhaps in a dedicated Related Work section) and to remake the figures to ensure all text and data points are clearly legible.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces BENTO, a comprehensive benchmark that evaluates 11 diverse docking and co-folding methods for protein–ligand interaction prediction. The benchmark combines classical physics-based tools, deep learning-based docking methods (Gnina, DiffDock, Matcha), and co-folding approaches. Unlike prior benchmarks, BENTO emphasizes curated subsets stratified by ligand physicochemical properties, binding pocket similarity, and protein–ligand pair diversity.

The results are clearly analyzed: physics-augmented and hybrid methods perform most robustly on unseen pockets, while deep learning methods tend to overfit. The analysis also identifies distinct behavior between protein-similar and pocket-dissimilar cases, providing valuable insight into model generalization.

Overall, this paper is a valuable addition to the docking-benchmarking landscape, though some conceptual gaps and missing discussions limit its completeness.

### Strengths
**Originality:** The integration of pocket- and ligand-based stratification into a unified benchmark is novel. The combination of classical, deep learning, and co-folding methods in a single framework is also valuable.

**Quality:** The methodological design is thorough — dataset preprocessing, G-LoSA pocket analysis, and ligand categorization (rotatable bonds, molecular size, etc.) are well-documented.

**Clarity:** The paper is written clearly and logically. Figures and tables are informative, particularly the comparative results.

**Significance:** The study contributes to the evaluation of generalization in docking models and provides an extensible foundation for future benchmarks. Its insights are useful for both ML researchers and computational chemists developing next-generation docking systems.

### Weaknesses
**Originality:** While comprehensive, the benchmark overlaps heavily with prior datasets already used in [1,2,3]. The novelty mainly lies in combining existing testbeds under one evaluation protocol. No new metric or dataset is introduced.

**Quality:** The main methodological limitation is the single-pocket assumption. All evaluations assume a known binding site, whereas realistic docking and screening pipeline should consider multiple potential pockets[3].

A second issue concerns partial dataset coverage, particularly the Timesplit test set (reported as “332/363” docked complexes).
It remains unclear whether the exclusion of 31 complexes was due to dataset preprocessing (e.g., missing coordinates, cofactors, or large ligands) or tool-specific failures. 

If the filtering was not uniform across all methods, this may unintentionally bias the benchmark toward easier systems and underestimate performance variability across tools.

Clarifying the criteria for these exclusions and ensuring consistent evaluation coverage would enhance the fairness and transparency of the benchmark.

Additionally, the inclusion of Matcha (an unreleased method under concurrent review) raises questions of reproducibility and fairness. It is unclear whether the model weights or code were publicly available at the time of evaluation.

**Clarity:** Some terms are used too broadly (e.g., “co-folding” to describe distinct architectures like AlphaFold3 and Boltz-2). Figures are occasionally dense and could benefit from simplified legends.

**Significance:** While valuable as a standardized benchmark, the study’s impact is incremental. It reinforces known findings — physics-based methods remain robust [3]; DL models overfit — without deeply analyzing why certain architectures generalize better.

**Suggestions:**

I think the authors can improve the paper by: 

- Extending the benchmark to multi-pocket or blind docking scenarios, which are closer to real-world drug discovery.
- Providing rationale for coefficient choices in RMSD/PB-valid scoring and thresholds.
- Simplifying figure layouts by summarizing core results in one overview chart.
- Including future directions on integrating pocket ensembles or multiple binding sites.

**References:**

[1] Cao, D., Chen, M., Zhang, R. et al. SurfDock is a surface-informed diffusion generative model for reliable and accurate protein–ligand complex prediction. Nat Methods 22, 310–322 (2025). https://doi.org/10.1038/s41592-024-02516-y

[2] Jiang Z. et al., PoseX: AI Defeats Physics Approaches on Protein-Ligand Cross Docking. arXiv 2025, arXiv:2505.01700

[3] Sarigun A. et al., PocketVina Enables Scalable and Highly Accurate Physically Valid Docking through Multi-Pocket Conditioning. arXiv 2025, arXiv:2506.20043

### Questions
- How was Matcha evaluated given that it is “under review” at ICLR 2026? Were pre-release weights provided by the authors or independently reproduced?

- Could the authors discuss how results might change in multi-pocket or blind docking settings where the binding site is not known a priori?

- Are the observed performance differences statistically significant (e.g., using paired bootstrap or t-tests)?

- Given the overlap with previous benchmarks, what does BENTO uniquely reveal about generalization that prior work (e.g., PocketVina or DockGen) did not?

- Would the authors consider releasing per-target breakdowns or including docking time/performance trade-offs in supplementary material?

### Soundness
3

### Presentation
4

### Contribution
3
