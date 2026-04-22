# stUAI: Uncertainty-Aware Clustering of Spatially Resolved Transcriptomics Data

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Spatially resolved transcriptomics (SRT) technologies enable high-resolution investigation of gene expression patterns across tissue sections, providing unprecedented insights into the molecular architecture of tissues. However, the inherent sparsity and over-dispersion of gene expression across spots, together with view-specific heterogeneity, jointly complicate modeling and present formidable challenges to reliable spatially informed downstream analyses. To address these issues, we propose stUAI, an Uncertainty-AwareIntegration framework for spatially transcriptomics data. stUAI first learns spatial- and expression-view embeddings by two separate graph-based encoders to capture multi-scale information. Then stUAI exploits the distributional representation of spots to quantify the view-specific uncertainty caused by technology limitations or noise in SRT, which is further leveraged for intra-view contrastive learning, cross-view information alignment, and representation integration. Furthermore, stUAI incorporates a zero-inflated negative binomial decoder to handle expression sparsity and imposes spatial structural constraints to preserve spatial continuity. Extensive experimental results on multiple benchmark datasets validate the effectiveness of our proposed stUAI in spatial clustering and several downstream applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces probabilistic embedding approach for spatial transcriptomics data. They propose intra-view contrastive learning and cross-view information alignment for embedding, and evaluated in spatial transcripotmics data.

### Strengths
This work first employ probabilistic embedding for spatial transcriptomics.

### Weaknesses
- Limited Novelty: Both probabilistic embedding and multiple modality are existing methodologies. This paper has merely applied them to the field of spatial transcriptomics.

- Source of Uncertainty: It is unclear where the uncertainty originates. In this paper, the distribution is primarily learned with a focus on the intra-view data. What does the variance in this context signify? For example, if variance were learned in the cross-view, one could attribute it to the difference between the two modalities. However, what causes this variance in the intra-view setting?

- Utility of Uncertainty: How can the uncertainty be utilized? The case in Section 3.5 is a post-hoc analysis. What practical action can be taken if the model determines the uncertainty is high?

- Alternative Divergences (Section 2.3): What would happen to the model if it were trained using KL Divergence or JS Divergence instead of the current formulation in Section 2.3?

- Benefit to Uni-modal Data: Can Spatial + Single Cell multimodal training also benefit single cell unimodal data?

- Missing Reference: The following reference is missing: Global Context-aware Representation Learning for Spatially Resolved Transcriptomics, ICML 2025.

### Questions
See Weakness Section

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors note that most existing methods struggle with effective denoising and suffer from model uncertainty, which reflects inherent semantic ambiguity and limited knowledge sharing between spatial and expression modalities. To address these challenges, they propose stUAI, an Uncertainty-Aware Integration framework for clustering spatially resolved transcriptomics data. The framework quantifies uncertainty in both modalities and performs cross-view distributional representation alignment using an optimal transport algorithm. In addition, the authors introduce Cluster-Guided Intra-View Contrastive Learning (CGCL), which defines confidence based on the distance to the cluster center and treats a pair as positive if both samples are confident and belong to the same cluster. The fusion of the two modalities is then carried out based on the learned uncertainty and optimized through reconstruction and spatial regularization objectives. The proposed method is evaluated on the HBC, DLPFC, and MBA tissue datasets.

### Strengths
- The authors propose a well-founded method that effectively addresses an important challenge in spatially resolved transcriptomics—quantifying uncertainty and learning a fused representation from two distinct modalities.

- The presentation of the work is clear and easy to follow.

### Weaknesses
- The dataset used in this study is limited, which is a major weakness and undermines the sufficiency of the experimental validation.

- The ablation study on the fusion strategy presented in Table 3 only evaluates a naïve averaging approach; additional comparisons with more advanced fusion strategies would strengthen the analysis.

- Although the authors attempt to demonstrate the reliability of the learned uncertainty in Section 3.5, the provided evidence is insufficient to fully support their claims.

### Questions
- Please provide spatial domain identification results on additional datasets, such as MERFISH, to further validate the generalizability of the proposed method.

- Include a comparison with attention-based fusion approaches to better demonstrate the effectiveness of the proposed fusion strategy.

- The current validation of the learned uncertainty is insufficient. The authors are encouraged to test whether spots with lower uncertainty are more likely to be well-clustered, for example, by evaluating this relationship using cluster accuracy (CA) metrics.

- Please provide more detailed information about the preprocessing steps applied to the DLPFC dataset.

### Soundness
3

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
The paper introduces stUAI, a novel framework for representation learning in Spatially Resolved Transcriptomics (SRT) data. It aims to overcome (1) the spasity and over-dispersion in SRT data, (2) sampling bias in contrative learning, and (3) shallow guidance and knowledge sharing between modality-specific representation (spatial and gene expression).

To address these issue, stUAI introduces three components:

1. Cluster-guided Intra-view Contrastive Learning: Creates positive pairs without augmentation using distributional representations and the reparameterization trick. Contrastive learning is then performed using reliable positive pairs, which must satisfy two conditions: (1) both spots are high-confidence (close to their cluster center), and (2) both spots are predicted to be in the same cluster.

2. Cross-view Distributional Representation Alignment: Employs Optimal Transport (OT) to ensure that the two different views—spatial (position) information and feature (gene) information—produce consistent predictions.

3. Representation Fusion & ZINB Reconstruction: Fuses spatial and gene information by weighting them based on uncertainty (derived from the inverse of their variances), normalizing these weights via softmax, and then performing a ZINB-based reconstruction.

To verify its effectiveness, the model was evaluated on clustering, DEG analysis, trajectory inference, gene imputation, and gene pathway enrichment, in addition to uncertainty analysis, demonstrating strong performance.

### Strengths
* The paper clearly explains its key motivations. Additionally, it proposes a methodology that aligns well with these motivations.
* The method appropriately incorporates uncertainty and validates this inclusion through experiments.
* The paper is well-written and easy-to-follow.

### Weaknesses
* The time complexity is a concern, as the model appears to perform k-means clustering and Optimal Transport (OT) in every epoch.
* The justification for the architecture is not fully convincing. The model uses CGCL on each modality before fusing them and uses OT for alignment. Why not apply CGCL directly to the final fused representation? Alternatively, why not integrate the graphs at the input level rather than aligning the representations via OT?
* The motivation should be more focused on the target domain. The ZINB loss is a common component in single-cell analysis and not a novel motivation in itself. The paper motivates its contrastive learning approach by citing general limitations (e.g., sampling bias). A stronger motivation would be to directly link this to specific SRT challenges. For instance, the high sparsity and over-dispersion in SRT data make traditional augmentation strategies ineffective for generating reliable positive/negative pairs.

### Questions
* What is the time complexity of stUAI compared to the baselines?
* Performance in unsupervised settings can often be unstable.
    * How many times were the experiments run (e.g., with different random seeds)?
    * Please provide statistical tests (e.g., p-values) to demonstrate the robustness of stUAI over baselines.
    * In conventional approaches, preprocessing steps and hyperparameters are typically fixed due to the lack of a validation set. Is there a specific reason for using different HVG counts (2000 vs. 3000) and k-nearest neighbor counts (14 vs. 15)2? Could you provide results where these are fixed to the same values across datasets?
* In the ablation study (Table 2), does "stUAI w/o $\mathcal{L}_{ZINB_{reg}}$" remove the reconstruction loss entirely? What is the performance if the ZINB loss is replaced with a standard reconstruction loss, such as MSE?
* In the fusion strategy comparison (Table 3), how does the proposed uncertainty-aware fusion ($M_4$) compare to a more common approach using learnable gating parameters?
* The sensitivity analysis (Figure 4) explores the $k$ for the feature graph neighbors. However, since k-means clustering is used during training, what is the model's sensitivity to the number of clusters ($K$) used in k-means?
* How are the "uncertainty scores" calculated in Section 3.5 (Figure 6b)?
* The pseudotime trajectory inference results (Figure 7) appear unusual. Typically, the trajectory in the DLPFC dataset is expected to originate from White Matter (WM) and proceed toward Layer 1. It would also be beneficial to compare this trajectory with those generated by other baselines, as in [1].
* For the gene imputation task (Figure 8), how does the model performance vary with different random mask ratios 5, as analyzed in [2]?
---
[1] Identifying multicellular spatiotemporal organization of cells with SpaceFlow. Nature communications. 2022.

[2] Global Context-aware Representation Learning for Spatially Resolved Transcriptomics. ICML. 2025.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper works on the clustering of spatial transcriptomics sites. In addition to fusing the spatial coordinates and the gene readouts, the authors add an uncertainty quantification part where embeddings from each modality are passed through or trains a parametrized gaussian distribution which introduces four (mean and std x 2) heads to represent each set of embeddings as a gaussian distribution. With such modeling of each modality’s embeddings as a distribution, the authors also propose to guide the learning of such distribution (which will backprop to embeddings representations) using pseudo-label (clustered) guided contrastive learning. In order to make sure each modality’s distribution truly associates with the other, the authors also apply an optimal transport method to align two distributions. Then the two embeddings are fused together and guided by ZINB reconstruction with a spatial regularizer.
The authors perform experiments on three popular datasets and report the ARI and NMI for different spatial clustering methods.

### Strengths
1. This paper incorporates uncertainty or distribution modelling in the method and also utilized the newly introduced data distribution network to perform contrastive learning supervision and optimal transport to force/further fuse the two modalities.
2. The authors compared an abundance of prior methods that conduct clustering with spatial coordinates and the gene count table.
3. The authors conducted experiments on recent and popular public datasets.

### Weaknesses
1. The reviewer is worried that associating/clustering only the spatial coordinates (not the H&E content and spatial coordinates) with Gene might be a problem that has not much usefulness to study. The reviewer understands that this is an important problem but it also always shows up in the first few samples/tutorial when processing a newly harvested ST dataset but it does not provide much further insights other than some visualizations.
2. The reviewer is worried that in order to show a method works well on such ST dataset, we need a more robust or complete or much more samples of ST slides than the ones being compared in the paper. The reviewer does understand that these are the commonly used samples but concerning these samples and their results won’t draw meaningful conclusions.
3. The ablation study is too short to justify each choice made in designing the method.
4. Figure 2, 3, and 5 are not legible enough when printed the paper out and reading in arms length.

### Questions
1. The reviewer is not too familiar with all the prior works, how is the training data scope, do the authors train all the spots from several ST together or each training would only contain one single ST’s all spots (which ranges from few hundreds to few thousands)?
2. In the CGCL part, the authors use clusters to create positive and negative labels; there are methods that do not require positive or negative pairs to be defined. Would those methods work here?
3. Could the authors please provide a more detailed ablation study?
4. With all these distribution modeling and guiding bits added, how much does the computation cost or runtime change compared with prior methods?
5. There has been quite a few ST foundation models proposed recently. How do those models perform on such tasks, what is the gap between those zero-shot representations with this single-slide learned representation?


The reviewer is giving a borderline reject recommendation at this stage but happy to modify the score during the discussion phase.

### Soundness
2

### Presentation
2

### Contribution
2
