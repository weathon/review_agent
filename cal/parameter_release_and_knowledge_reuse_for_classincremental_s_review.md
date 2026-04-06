=== CALIBRATION EXAMPLE 33 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title "Parameter Release and Knowledge Reuse for Class-Incremental Semantic Segmentation" clearly indicates the core contributions. The abstract succinctly states the problem (parameter competition and underutilized knowledge in existing KD for CISS), proposes DKD with a minimization–maximization strategy, and summarizes results (near joint-training performance on multiple benchmarks). Claims are supported by evidence later in the paper. No major issues.

### Introduction & Motivation
Well-structured. The problem of CISS is motivated by real-world constraints (privacy, storage). The analysis of limitations in current KD methods (parameter competition, wasted knowledge) is clear and leads to a focused research question: *"Can we release parameters and simultaneously reuse the acquired knowledge when learning new knowledge without introducing an additional inference burden?"* The three contributions are listed explicitly and align with the paper's content.

### Method / Approach
This is the core of the paper and requires careful scrutiny.

1. **Problem Definition (Sec. 3.1):** Standard and clear.

2. **Parameter-driven Minimization (Sec. 3.2(a), L_Min):**
   - The idea of releasing "low-sensitivity parameters" from the old model via pruning (threshold τ on filter L2 norm) to free capacity is intuitive. However, the justification for this specific pruning criterion is thin. Why is filter norm a good proxy for "sensitivity" to old classes? Pruning is done once per step, which is efficient, but its impact on the old model's representation is not analyzed beyond a small table (Appendix Table 11). A more rigorous analysis (e.g., showing the actual parameter redistribution or change in activation patterns) would strengthen the claim.
   - Equation (2) updates the label \(y_c^*\) using the prediction after parameter release. This seems to create a moving target for distillation. How does this interact with the background shift? The explanation is brief.

3. **Laplacian-based Projection Estimation (Sec. 3.2(b), L_Esti):**
   - The construction of the position map \(P_t(h,w)\) using second-order gradients (Eq. 4) is novel but poorly motivated. The link between "low-curvature regions" and "coexistence" of old and new knowledge is not explained intuitively. The derivation appears in Appendix A.2, but the main text should provide a clearer geometric or information-theoretic rationale.
   - The confidence map \(C_t(h,w)\) (Eq. 5) is a normalized dot product, measuring alignment. This is reasonable but feels ad-hoc. Why is this particular form effective for reusing knowledge?
   - The loss \(L_{Esti}\) combines a Laplacian consistency term \(L_{lap}\) and a projection term \(L_{pro}\) with a weight γ. The ablation studies show their importance, but the underlying principle—how these two maps collectively "estimate the new knowledge distribution"—is conceptually fuzzy.

4. **Entropy-induced Optimization (Sec. 3.2(c), L_Max):**
   - **Major Issue:** The definition of conditional entropy \(H(Y_t | Y_{t-1})\) in Equation (9) is incorrect. The formula given is actually the entropy of \(Y_t\) averaged over samples, *not* the conditional entropy given \(Y_{t-1}\). True conditional entropy requires the conditional distribution \(P(Y_t|Y_{t-1})\), which is not defined here. This misdefinition undermines the theoretical interpretation of \(L_{Max}\) as maximizing mutual information. The subsequent analysis in Appendix A.3 builds on this flawed definition and may need revision. The empirical effectiveness of \(L_{Max}\) is shown in ablations, but the theoretical claim is unsound.

5. **Overall Formulation (Eq. 11):**
   - The total loss combines four terms. The balancing of these terms is not discussed (except for γ in \(L_{Esti}\)). Are there any weighting hyperparameters? The training stability (Fig. 5) suggests good convergence, but the optimization dynamics with potentially competing objectives deserve more analysis.

6. **Reproducibility:** The method description is sufficiently detailed for implementation, and code is promised. Some minor clarifications would help: e.g., which layers are pruned (all conv/FC?), and is pruning applied to the backbone, decoder, or both?

### Experiments & Results
Extensive and generally strong.

1. **Setups (Sec. 4.1):** Standard benchmarks, ViT backbone, multiple incremental settings. Training details are clear.

2. **Quantitative Comparisons (Tables 1 & 2):**
   - Comprehensive comparison with many SOTA methods across 5 VOC and 4 ADE20K settings. The results are convincing, showing consistent improvements, especially in challenging many-step scenarios (e.g., 10-1, 2-2). The use of both ResNet and ViT baselines is appropriate.
   - The average gains (e.g., 1.7% on VOC) are meaningful, though not always dramatic. The claim that performance "approaches that of joint learning" is supported (e.g., ADE20K average 48.1 vs joint 49.2).

3. **Ablation Studies (Sec. 4.3 & Appendices):**
   - Component ablation (Table 3,12) clearly shows the contribution of each loss.
   - Hyperparameter studies for τ (Table 10) and γ (Table 4) are thorough.
   - Error analysis (Tables 5,13) with repeated runs shows low variance, supporting robustness.
   - Integration experiments (CoinSeg + DKD, Tables 8,9) demonstrate plug-and-play effectiveness across backbones.

4. **Qualitative Results (Fig. 4, 7-10):**
   - Visual segmentation comparisons and t-SNE plots illustrate improved accuracy and reduced confusion. These are effective.

5. **Potential Gaps:**
   - While forgetting is measured via mIoU on old classes, a more detailed analysis of *which* old classes are forgotten could be insightful.
   - The computational overhead is mentioned (7s/epoch), but memory footprint and inference time comparisons are missing (though claimed to be no extra inference cost).
   - The comparison to dynamic architecture methods is limited; a discussion on the trade-offs (parameter efficiency vs performance) would be valuable.

### Writing & Clarity
The paper is well-organized and mostly clear. However, the method section, particularly the derivation and explanation of the Laplacian-based projection and the entropy loss, is dense and lacks intuitive grounding. The incorrect definition of conditional entropy (Eq. 9) is a significant clarity issue that must be fixed. Figures and tables are helpful, though some suffer from OCR formatting artifacts (not the authors' fault).

### Limitations & Broader Impact
- **Limitations:** Not explicitly discussed in the main paper. Important limitations include: (1) reliance on a simple pruning heuristic; (2) the complexity and somewhat opaque design of the Laplacian and entropy losses; (3) the method is evaluated only on standard benchmarks with ViT/ResNet—its behavior on other architectures or extreme class-imbalanced scenarios is unknown; (4) the theoretical grounding needs correction and strengthening. A dedicated limitations section is needed.
- **Broader Impact & Ethics:** The ethics statement is comprehensive and addresses relevant concerns (data use, potential misuse, fairness). Appendix E provides a brief broader impact statement. These are adequate for ICLR.

### Overall Assessment
This paper presents a novel and empirically effective method for CISS. The core ideas—parameter release via pruning and knowledge reuse via a designed distillation loss—are interesting and address genuine limitations in prior work. The experimental evaluation is extensive and demonstrates state-of-the-art performance across diverse settings. However, the method's theoretical underpinnings are currently flawed (specifically the conditional entropy definition) and some components (Laplacian-based projection) are not well-motivated. The empirical gains, while consistent, are sometimes incremental over strong baselines. For ICLR, where both novelty and rigorous foundations are valued, the paper would be significantly strengthened by correcting the theoretical errors and providing clearer justifications for the design choices. If the authors can address these issues in a revision, the contribution would be solid. As it stands, the empirical results are promising but the conceptual clarity needs improvement.

**Recommendation:** Borderline Accept. The paper has strong empirical results and a novel approach, but requires major revisions to fix the theoretical errors (especially Eq. 9 and associated analysis) and improve the motivation/explanation of key components (Laplacian projection, pruning criterion). A clear limitations section should also be added.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes Distribution-based Knowledge Distillation (DKD), a novel method for class-incremental semantic segmentation (CISS). DKD addresses two identified limitations of standard knowledge distillation: parameter competition between old and new classes, and underutilization of previously learned knowledge. The method employs a minimization-maximization strategy: minimizing the old knowledge distribution by releasing low-sensitivity parameters, and maximizing the shared knowledge distribution via Laplacian-based projection estimation and entropy-induced optimization. Extensive experiments on Pascal VOC and ADE20K across nine incremental settings demonstrate state-of-the-art performance, closely approaching the joint training upper bound.

### Strengths
1. **Well-Motivated and Novel Approach**: The paper clearly identifies two underexplored issues in CISS—parameter competition and knowledge underutilization—and proposes a novel, theoretically grounded solution. The combination of parameter release (via pruning and L_Min) and knowledge reuse (via L_Esti and L_Max) introduces a fresh perspective to knowledge distillation in incremental learning.
2. **Extensive and Rigorous Evaluation**: The method is evaluated on two standard benchmarks (Pascal VOC, ADE20K) across nine diverse incremental settings, including challenging configurations with many steps (e.g., 10-1, 2-2). Results consistently outperform prior art, with average performance nearing the joint training upper bound. The inclusion of both overlapping and disjoint settings (Appendix C.1) further demonstrates robustness.
3. **Thorough Ablation and Analysis**: Comprehensive ablation studies validate each component of DKD (Table 3, 12). Hyperparameter sensitivity (Table 4, 10), error analysis (Table 5, 13), convergence curves (Figure 5), and qualitative visualizations (Figures 4, 7-10) provide strong evidence for the method's effectiveness. The t-SNE analysis (Figure 4b) convincingly shows reduced class confusion.
4. **Theoretical Grounding**: The appendix provides detailed theoretical analysis for each loss component, linking L_Min to soft target alignment, L_Esti to distillation gap reduction, and L_Max to information maximization. This strengthens the methodological foundation.
5. **Reproducibility and Ethics**: The paper includes a detailed reproducibility statement, commits to releasing code, and provides an ethics statement addressing data use and potential misuse. Training times are reported, and the method adds no inference overhead.

### Weaknesses
1. **Clarity of Method Description**: The technical description, particularly for Laplacian-based projection estimation (Section 3.2(b)), is dense and difficult to follow. Key equations (e.g., Eq. 4, 5) are not sufficiently explained, and the intuition behind the position and confidence maps is unclear. The loss L_Esti (Eq. 6) is presented without clear motivation, hindering understanding.
2. **Incomplete Comparison with Recent Work**: While comparisons are extensive, they omit several very recent CISS methods (e.g., from CVPR/ECCV 2024 beyond those cited). This could affect the claim of state-of-the-art, especially given the fast-moving field.
3. **Training Overhead and Hyperparameter Sensitivity**: The method introduces a non-trivial training overhead (∼7 seconds/epoch over CKD). While inference cost is unchanged, this may be a concern for large-scale applications. Additionally, the pruning threshold τ and balance parameter γ require per-dataset tuning (τ=0.1 chosen empirically, γ set to 1 or 0.4), indicating sensitivity that could affect generalizability.
4. **Limited Analysis on Parameter Release Impact**: The pruning strategy (based on L2 norm) is justified only by a small performance drop on old classes (Table 11). A deeper analysis—e.g., showing the fraction of parameters released per step, or correlating released capacity with new class performance—would strengthen the claim of alleviating parameter competition.
5. **Grammatical and Presentation Issues**: The writing contains grammatical errors and awkward phrasing that occasionally impede readability. Some figures (e.g., Figure 2) are referenced but not fully explained in the text.

### Novelty & Significance
**Novelty**: The paper introduces a novel distribution-based distillation paradigm for CISS. The core ideas—dynamically releasing old parameters and explicitly reusing old knowledge via projection and entropy maximization—are distinct from prior KD techniques that typically freeze the old model. The minimization-maximization framework is a fresh and theoretically sound contribution.

**Significance**: The method achieves compelling empirical gains across a wide range of settings, effectively balancing stability and plasticity. By addressing parameter competition and knowledge reuse, it advances the state-of-the-art in CISS and provides a new direction for incremental learning research. The approach is architecture-agnostic (shown to work with both ViT and ResNet) and could inspire applications in other continual learning tasks.

### Suggestions for Improvement
1. **Improve Method Clarity**: Rewrite Section 3.2(b) to intuitively explain the Laplacian-based projection, position map, and confidence map. Use a schematic or step-by-step example to illustrate how L_Esti guides knowledge reuse. Clarify the variables and operations in Eqs. 4-6.
2. **Update Comparisons and Related Work**: Include comparisons with the most recent CISS methods (e.g., from 2024 conferences/workshops) to ensure the state-of-the-art claim is current. Briefly discuss how DKD differs from concurrent work in the related work section.
3. **Deeper Analysis of Parameter Release**: Add an analysis quantifying the amount of parameters released per incremental step and correlate this with new class performance. Consider discussing adaptive threshold mechanisms or comparing with other pruning criteria.
4. **Discuss Limitations Explicitly**: Add a dedicated limitations paragraph discussing the training overhead, hyperparameter sensitivity, and any assumptions (e.g., static architecture). Suggest directions to mitigate these in future work.
5. **Enhance Readability**: Perform thorough proofreading to correct grammatical errors and improve sentence flow. Ensure all figures are clearly referenced and explained in the main text.
6. **Expand Broader Impacts**: The ethics statement is adequate, but the broader impacts section could be expanded to discuss potential negative societal implications (e.g., surveillance misuse) and mitigation strategies more concretely.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Comparison with pruning-based continual learning methods.** The paper introduces parameter release via pruning but does not compare with existing pruning or sparse training methods for continual learning. Without this, it is unclear if the performance gains stem from the specific DKD strategy or simply from parameter reduction.
2. **Comprehensive sensitivity analysis of the pruning threshold τ.** The choice of τ=0.1 is only briefly examined. A thorough ablation across all settings is needed to show robustness and provide guidance, as an arbitrary threshold undermines the claim of a principled parameter release.
3. **Experiments with additional backbones (e.g., ResNet50, Swin Transformer).** The paper primarily uses ViT, with only a single ResNet101 experiment. To claim generality, the method must be validated on at least one more modern backbone across multiple incremental settings.
4. **Evaluation on a third benchmark dataset (e.g., Cityscapes).** The claims of effectiveness and near-upper-bound performance are based on Pascal VOC and ADE20K. Testing on a more complex dataset like Cityscapes is necessary to demonstrate broad applicability and robustness.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantitative analysis of parameter release impact.** The paper lacks metrics showing how many parameters are pruned per step and how this correlates with performance on old and new classes. This is critical to substantiate the claim of alleviating parameter competition.
2. **Quantitative measurement of stability-plasticity trade-off.** The paper claims to balance stability and plasticity but does not report standard continual learning metrics (e.g., forgetting rate, learning accuracy). These metrics are essential for a rigorous evaluation in an incremental learning paper.
3. **Analysis of the position and confidence maps.** The paper does not visualize or statistically analyze the generated position and confidence maps. Without this, it is impossible to verify if they indeed identify reusable knowledge and how they guide the learning process.
4. **More thorough analysis of class confusion reduction.** The similarity matrix is shown only for the 10-1 setting. A quantitative analysis (e.g., inter-class distance, confusion matrix entropy) across all settings is needed to solidly support the claim of reduced confusion.

### Visualizations & Case Studies
1. **Visualization of failure cases.** The paper only shows successful qualitative results. Displaying failure cases (e.g., where DKD forgets old classes or fails to learn new ones) is crucial to understand the method's limitations and boundary conditions.
2. **Visualization of the position map (Pt) and confidence map (Ct).** Showing these maps for sample images would reveal what spatial regions and features are considered reusable, making the knowledge reuse process interpretable and verifiable.
3. **Feature distribution evolution over steps.** The t-SNE visualization is only for a single step (15-1). Showing how feature distributions change across incremental steps would illustrate how DKD mitigates catastrophic forgetting and maintains separation.

### Obvious Next Steps
1. **Incorporate standard continual learning metrics.** The paper must report metrics like Average Accuracy, Forgetting, and Intransigence to properly situate its performance within the continual learning literature.
2. **Compare with contemporary knowledge distillation variants that address capacity.** The paper should compare with methods that adapt distillation losses to free up capacity (e.g., via adaptive weighting or sparse distillation) to better justify the novelty of DKD.
3. **Provide a principled method for setting τ and γ.** The hyperparameters are set empirically (τ=0.1, γ=1 or 0.4). A more data-driven or theoretical justification for their selection would strengthen the method's design.

# Final Consolidated Review
## Summary
This paper proposes Distribution-based Knowledge Distillation (DKD), a method for class-incremental semantic segmentation (CISS). DKD addresses two identified limitations of standard knowledge distillation—parameter competition between old and new classes and the underutilization of previously learned knowledge—via a minimization-maximization strategy. This involves releasing low-sensitivity parameters from the old model and reusing old knowledge through Laplacian-based projection estimation and entropy-based optimization. The method achieves state-of-the-art performance across nine incremental settings on Pascal VOC and ADE20K, closely approaching the joint-training upper bound without increasing inference cost.

## Strengths
- **Novel and well-motivated approach:** The paper clearly identifies underexplored issues (parameter competition and knowledge underutilization) in CISS and introduces a fresh distillation paradigm with a minimization-maximization strategy. The core ideas of dynamic parameter release and explicit knowledge reuse are distinct from prior work that typically freezes the old model.
- **Extensive and rigorous evaluation:** The method is evaluated on two standard benchmarks (Pascal VOC and ADE20K) across nine diverse incremental settings, including challenging many-step scenarios (e.g., 10-1, 2-2). Results consistently outperform prior art, with average performance nearing the joint-training upper bound. Additional experiments in disjoint settings (Appendix) and integration with other backbones demonstrate robustness.
- **Thorough ablation and analysis:** Comprehensive ablation studies validate each component of DKD. Hyperparameter sensitivity, error analysis with repeated runs, convergence curves, and qualitative visualizations (including t-SNE plots showing reduced class confusion) provide strong empirical support for the method's effectiveness.

## Weaknesses
- **Flawed theoretical foundation for L_Max:** The definition of conditional entropy \(H(Y_t | Y_{t-1})\) in Equation (9) is mathematically incorrect. The formula provided is the sample-averaged entropy of \(Y_t\), not the conditional entropy given \(Y_{t-1}\). This error undermines the theoretical interpretation of \(L_{Max}\) as maximizing mutual information and the related analysis in Appendix A.3. The empirical utility of the loss term may remain, but the theoretical claim is unsound.
- **Dense and insufficiently motivated method description:** The technical description of the Laplacian-based projection estimation (Section 3.2(b)) is dense and lacks intuitive grounding. The link between low-curvature regions (position map) and "coexistence" of knowledge, and the design of the confidence map, are not clearly explained. This opacity hinders understanding and reproducibility of the core novel mechanism.

## Nice-to-Haves
- Deeper quantitative analysis of the parameter release impact (e.g., fraction of parameters pruned per step and correlation with new class performance) to further substantiate the claim of alleviating parameter competition.
- Visualization and statistical analysis of the generated position and confidence maps to make the knowledge reuse process more interpretable and verifiable.
- Exploration of a more principled or adaptive method for setting the pruning threshold \(\tau\) and the balance parameter \(\gamma\), though the provided ablation studies demonstrate reasonable empirical selections.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Correct the definition of conditional entropy in Equation (9) and the subsequent theoretical analysis in Appendix A.3 to provide a sound information-theoretic foundation for \(L_{Max}\).
- Rewrite Section 3.2(b) to provide a clearer, more intuitive explanation of the Laplacian-based projection estimation, the role of the position and confidence maps, and how \(L_{Esti}\) guides knowledge reuse. A schematic or step-by-step illustration could be helpful.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 4.0]
Average score: 3.3
Binary outcome: Reject
