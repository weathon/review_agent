# SPATIA: Multimodal Model for Prediction and Generation of Spatial Cell Phenotypes

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Understanding how cellular morphology, gene expression, and spatial organization jointly shape tissue function is a central challenge in biology. Image-based spatial transcriptomics technologies now provide high-resolution measurements of cell images and gene expression profiles, but existing methods typically analyze these modalities in isolation or at limited resolution. 
We address the problem by introducing SPATIA, a multi-scale generative and predictive model that learns unified, spatially aware representations by fusing morphology, gene expression, and spatial context from single-cell to tissue level. SPATIA incorporates a spatially conditioned image-to-image generation module that predicts cell morphologies under perturbations, enabling the study of microenvironment-dependent morphological changes such as tumor progression, immune remodeling, and subtype transitions.
We assembled a multi-scale dataset consisting of $17$ million cell-gene pairs, $1$ million niche-gene pairs, and $10,000$ tissue-gene pairs across diverse tissues and disease states. We benchmark SPATIA against $16$ existing models across $12$ individual tasks, which span several categories including cell annotation, cell clustering, gene imputation, cross-modal prediction, and image generation. SPATIA achieves improved performance over baselines and generates realistic cell morphologies that reflect transcriptomic perturbations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes Spatia, a multi-scale, multimodal model designed to integrate cellular morphology, gene expression, and spatial location information in spatial transcriptomics. The model employs a hierarchical Transformer architecture to fuse features across three scales: cell, niche, and tissue. It also introduces a conditional generation method based on flow matching to predict morphological changes in cells under spatial microenvironment perturbations. The authors construct and release the MIST dataset, containing over 17 million single-cell samples.

### Strengths
1.It is the first work to achieve cross-attention fusion of morphology and gene expression at the single-cell level and introduces the spatially conditioned generation task, demonstrating certain innovation.

2.The paper proposes a multi-scale modeling framework from cell to tissue level and constructs a large-scale multimodal dataset (MIST).

### Weaknesses
1.Some key hyperparameters (e.g the contrastive loss weight ρ in flow matching) are not explicitly stated in the main text, leading to insufficient reproducibility.

2.The paper lacks an in-depth analysis of the model's computational burden and memory requirements. applying Transformer at the tissue level to whole-slide images, particularly when processing millions of cells, could be computationally prohibitive.

3.The gene expression vectors are treated as an unordered set and processed through a simple encoder (scPRINT). But this approach overlooks the inherent, known relationships between genes, such as those derived from gene pathways or co-expression networks.

4.The paper claims to model spatial context at the "niche" and "tissue" levels, but the "niche" is simplistically defined as a fixed 256×256 pixel grid. This division is mechanistic and biologically irrelevant, failing to adaptively capture the irregular and functionally diverse microenvironments in real tissues, such as tumor margins or perivascular regions.

### Questions
1.Beyond image similarity metrics, is there plans to evaluate the realism of generated cell morphology through pathologist assessment or functional experiments (such as immunofluorescence validation)?

2.In the MIST dataset, could cells from the same donor appear in both training and test sets? how is evaluation fairness ensured?

3.In the multi-scale modeling, was the actual contribution of the "tissue-level" representation to single-cell tasks verified? Is there a risk of over-parameterization?

### Soundness
2

### Presentation
2

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
The paper proposes SPATIA, a multi-resolution multimodal framework for spatially resolved transcriptomics pretrained on a large scale dataset MIST. It fuses cell morphology, gene expression, and spatial coordinates across cell, niche, and tissue levels via hierarchical transformers. SPATIA claims state-of-the-art performance in cell annotation, clustering, gene expression prediction, biomarker status prediction, and morphology generation.

### Strengths
- SPATIA unifies morphology, gene expression, and spatial structure at multiple scales. The idea is novel.
- The MIST dataset is a substantial contribution, providing multimodal data across scales with one-to-one mapping between images and transcriptomics.

### Weaknesses
- The methodology section is overly dense, with many moving components (encoders, fusion, niche/tissue transformers, pseudo-perturbation embeddings, flow matching) introduced in quick succession.The motivation for some design choices is underexplained.
- MIST is assembled from Xenium datasets, but details of preprocessing, normalization, and gene panel harmonization are sparse in the main text. It is unclear how batch effects are handled across donors and tissue types, which could inflate cross-task performance.
- The paper positions SPATIA as a “foundation model” for spatial omics, but pretraining/fine-tuning strategy and scaling laws are not thoroughly evaluated.

### Questions
- How sensitive are results to niche size and tissue grid definition?

- How does SPATIA handle batch effects in MIST, given data are from 49 donors with varied gene panels?

- How does SPATIA address the issue of varying cell sizes in the input images? Are all cropped cell images resized to a fixed dimension, or are smaller cells padded with blank space? Additionally, how does the model deal with cases where cells are tightly packed and boundaries are unclear?

- Does the pseudo-perturbation embedding (∆g) risk leaking target information? Since ∆g is derived from matched control–target pairs, this seems infeasible for real-world generation, where target states are not available at inference. Could the authors clarify how this issue is mitigated?

- Why is a 3-layer MLP used for biomarker status prediction, while XGBoost is used for gene expression prediction? What motivates these different choices, and are the results robust to alternative predictors?

- How crucial are pretrained weights for SPATIA’s performance? What is the performance gap when training from scratch compared to using pretrained components?	

- For biomarker status prediction, could improvements simply reflect leakage from tissue morphology rather than gene–morphology integration?

### Soundness
3

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
This paper proposes SPATIA, a hierarchical multimodal model integrating cell morphology, gene expression, and spatial context for spatial transcriptomics analysis. It also introduces a conditional flow-matching module for perturbation-aware morphological change prediction. The model is trained on the newly assembled MIST dataset (17M cell-gene pairs, 1M niche-gene pairs, 10K tissue-gene entries) and evaluated across 12 tasks including generation, annotation, clustering, biomarker prediction, and gene-expression regression

### Strengths
- Strong multimodal fusion with cross-attention at single-cell resolution.

- Novel hierarchical design: combines cell, niche, and tissue transformers for spatial dependency modeling.

### Weaknesses
-  Algorithmic / Methodological Concerns

While the overall hierarchical design is interesting, several components remain underspecified or potentially brittle:

(i) The conditional generation module critically depends on weakly paired control/perturbed cells matched via optimal transport in gene-expression space. No analysis is provided on how pairing errors affect the learned flows, which is crucial in spatially heterogeneous niches.
 (ii) Because niche context and perturbation signatures are correlated in the dataset, the conditional flow may learn dataset co-occurrence rather than truly spatially grounded perturbation effects.
 (iii) The paper does not clearly describe the sampling or batching strategy for training multi-level transformers on slides with very large numbers of cells, leaving computational feasibility and scalability unclear.
 (iv) Since cell-level embeddings attend to niche/tissue features, there is a risk of information leakage across scales, which should be controlled when comparing to single-cell baselines.

-  Misleading or Incomplete Claim of Novelty

The paper claims that existing methods fail to integrate spatial, molecular, and morphological information at single-cell resolution.
 However, recent multimodal spatial models already achieve this integration, such as SpaGCN, STAligner, SpaOTsc and so on, which align histology and transcriptomics at the single-cell or subcellular level using transformer-based architectures.
 These models are neither discussed nor compared, making it difficult to evaluate SPATIA’s incremental contribution.

- Limited Platform Generalization

All datasets used in this study appear to come from Xenium, which restricts evaluation to a single spatial transcriptomics platform.
 Given the diversity of spatial technologies—such as MERFISH, seqFISH, Stereo-seq, and Slide-seq—it remains unclear whether SPATIA’s design generalizes across imaging and transcriptomic modalities

### Questions
- On the ablation design (Table 4):

 The current ablation seems incremental (Cell → +MAE → +Multi-level → +Fusion), showing monotonic improvement.
 However, this design does not disentangle the individual contribution of each component.
 Could the authors provide or discuss a factorial or pairwise ablation — for instance, MAE without multi-level, or fusion without multi-level — to verify whether each module independently improves performance or only works in combination?

- On algorithmic clarity:
 How sensitive is the conditional generation module to errors in the weak OT-based pairing of control/perturbed cells?
 Have the authors tried perturbing or noising the matching to evaluate robustness?

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
The manuscript presents SPATIA, a multimodal multiscale model for spatial transcriptomics and morphological imaging at cell, niche, and tissue levels.

### Strengths
* Tackles a real problem in spatial transcriptomics
* Contribution in multimodality and multiscaling

### Weaknesses
1. Soundness concern regarding summation of expression vectors
2. Optimal tranport use is not new
3. The contribution is workmanlike
4. Lack of clarity regarding sigle cell data

### Questions
1. Expression vectors are summed across cells (188, 262). This makes an assumption of linearity which I believe is false - the authors are required to justify making it, in particular, as it invalidates the motivaiton for the approach to combining spots into cells by concatenation + embedding (221-234).

2. Optimal transport is intensively used in this field for this exact purpose. The authors should cite and use an existing framework or justify any changes they make to the formulae.

3. Unclear how partitions to niches and tissues are defined

4. Single cell data typically measures thousands of gene per cell with numenr of cells now reaching millions. The authors' use of 17M pairs is thus questionable. Relatedly the noise and sparsity level in single cell data is an issue the authors ignor

### Soundness
2

### Presentation
3

### Contribution
2
