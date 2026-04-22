# Geo-Mamba: Dual-Path Riemannian State-Space Models for Functional Dynamics

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6, 4

## Abstract
Functional magnetic resonance imaging (fMRI)–derived functional connectivity (FC) is represented as graphs and as correlation/covariance matrices that live on non-Euclidean spaces—cortical graphs and the Riemannian manifold of symmetric positive-definite (SPD) matrices—so conventional Euclidean sequence models are misspecified. To this end, we introduce *Geo-Mamba*, a geometric variant of Mamba formulated on Riemannian manifolds. *Geo-Mamba* employs a dual-path selective state-space design: *a stacked path* performs hierarchical spatial modeling by aggregating pyramid multi-scale features to capture local and global dependencies, while *a embedding path* combats redundancy in high-dimensional SPD inputs via progressive, geometry-aware dimensionality reduction (operating in the appropriate manifold spaces) to produce compact states without violating Riemannian constraints. Their complementary outputs are fused through the tailored *GeoMix* operator to yield a compact, discriminative SPD representation. *Geo-Mamba* is evaluated on six public fMRI datasets—ADNI, OASIS, PPMI, Taowu, Neurocon, and Mātai—spanning Alzheimer’s and Parkinson’s cohorts as well as multi-site normative populations with diverse acquisition protocols. Across these benchmarks, it delivers consistently competitive accuracy and robustness, supporting the value of dual-path manifold modeling for neuroimaging and its potential for clinical translation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes an extension of the SPDNet architecture to the recently emerging Mamba state-space modeling framework, referred to as Geo-Mamba. The method aims to model brain functional connectivity matrices on the SPD manifold while leveraging Mamba’s efficient sequence modeling capacity. The authors conduct experiments on six public fMRI datasets and report consistent performance gains, suggesting that the proposed model effectively captures multi-scale geometric and temporal dependencies.

While the idea of integrating geometric deep learning on SPD manifolds with structured state-space models is interesting and timely, the paper lacks clarity in several conceptual, mathematical, and experimental aspects, which significantly weaken its overall rigor and reproducibility.

### Strengths
The paper addresses a clear architectural gap by bridging SPD-based geometric representation learning and the latest Mamba sequence model, combining two active but previously disconnected research domains (geometric deep learning and state-space modeling).

The motivation to extend SPDNet to temporal modeling via Mamba is intuitively appealing and potentially impactful for neural signal and fMRI analysis, where both geometric structure and dynamic dependencies are crucial.

The empirical study across multiple datasets demonstrates that the proposed approach can generalize across diverse fMRI sources, suggesting practical relevance if rigorously validated.

### Weaknesses
The paper does not clearly articulate why the SPD manifold is the most appropriate representation for fMRI data in this context. What type of statistical or functional information is being preserved by the SPD structure? Why is the log-Euclidean metric chosen over alternatives such as AIRM or Stein divergence? Without such justification, the methodological grounding feels ad hoc.

The implementation (as observed in MambaSPD_Attn_then_SSM) employs Riemannian Batch Normalization, which is derived for the AIRM metric, while the paper consistently claims to use the log-Euclidean metric. Mixing these two incompatible metrics within the same model undermines mathematical consistency and raises questions about the validity of the results.

Experimental design lacks important information: the number of subjects used in the experiments does not correspond to any known public version. The category definitions (e.g., diagnostic groups) and cross-validation protocols (5-fold vs. 10-fold) are inconsistent across datasets without explanation. Preprocessing steps, such as confound regression, normalization, or atlas selection, are insufficiently described, making reproducibility difficult, even with access to the code.

It is not clear how the SPD matrices are represented within the Mamba pipeline in your baseline comparison. From the description, the “Geo-Mamba” variant seems to apply a logarithmic mapping only at the final stage. Conceptually, this should yield behavior similar to standard Mamba, yet large performance differences are reported. Can you explain why?

### Questions
1. Could you elaborate on the motivation for using SPD representations in this task? What specific information from fMRI connectivity are you preserving that would be lost in a Euclidean embedding?

2. How do you reconcile the use of Riemannian Batch Normalization (AIRM-based) with your log-Euclidean metric formulation?

3. Please clarify the exact dataset versions, subject selection criteria, ROI definitions, and cross-validation strategies.

4. How are SPD matrices encoded as tensors in the Mamba architecture? Are manifold operations actually preserved throughout the sequence modeling pipeline? What explains the large performance differences between Geo-Mamba and vanilla Mamba across datasets if the only modification is a logarithmic mapping?

5. Figure 2 appears to show extremely sparse connectivity patterns. Was any thresholding or sparsification applied to produce the visualizations in Figure 2?

### Soundness
2

### Presentation
3

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
The paper presents Geo-Mamba, an approach to build classification models for fMRI data using the ideas of Riemannian manifolds and state-space models (Mamba). First, the input (SPD) fMRI correlation matrix is translated into a sequence of smaller SPD matrices obtained through windows (that capture dynamics at different scales) on the input matrix and log-attention-exp transformations (stacked path). This path results in a final SPD matrix. Another SPD matrix (distillation path) is obtained using compression. The two are combined together using GeoMix for the classification task.

### Strengths
The paper does a good job of integrating Riemannian manifolds with fMRI signals. It can serve as a baseline for other efforts.

The performance results are quite good across a variety of datasets.

### Weaknesses
1. The approach is quite complex and it is difficult to get an idea of how much of the complexity is actually needed.
2. The training and inference time are not given. 
3. The cortical regions define a graph with strong/weak connectivity across regions. It is not clear what is the impact of linearizing them,(used in the definition of windows).
4. The paper keeps referring to the sequence as a "temporal sequence" and each step as "time step". This is not quite correct.
5. Table 2 seems like a half-hearted attempt to compare with a temporal windowing scheme. It is not clear if the best temporal baseline is being used.

### Questions
1. Can you explain how the matrix W in Eq 3 is defined/learnt? 
2. What are the P matrices in Eq 10? Are they the same P matrices in the stacked path (Eq 5)?
3, Does the order matter in the construction of the S_{all}? Would one get the same results if the input matrices were shuffled?
4. What is the contribution of the two paths and their combination to the training and inference overheads?
5. Can you explain why the performance jump over GIN is so much higher for Matai?
6. What if the cortical windows were linearized differently? Does the linearization obey cortical neighborhood? What happens to the performance?
7. How is matrix Q (the geometrical anchor) learned? Can you report ablation studies if a random Q was used?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents a geometric extension of the Mamba selective state space model and applies it for the analysis of fMRI functional connectivity (FC) data. The connectivity measures are represented as symmetric positive definite (SPD) matrices on Riemannian manifolds due to its natural correlational structure. Additionally, the method uses a pyramid-based multi-scale geometric feature extraction from the original SPD matrices and also ensures that the sub-blocks of matrices at multiple levels themselves satisfy the SPD constraint. The paper uses a dual-path architecture (proposed before) that uses a a stacked path for hierarchical temporal modeling and a distillation path for progressive, manifold-preserving dimensionality reduction. They further use the geomix operator (which also has been proposed before) for manifold-consistent fusion. The authors show experimental results on six public fMRI datasets (ADNI, OASIS, PPMI, Taowu, Neurocon, Matai) and also show comparisons and performance improvement over existing methods including the original Mamba model, SPDNet, and BrainGNN.

### Strengths
The main novelty of the paper is to ensure that the SPD manifold constraints at each stage of the selective state space model evolution are satisfied. 

It extends the selective state-space models to SPD manifolds with explicit log-exp mapping and orthogonality constraints.


The authors perform comprehensive experimental validation. The method is tested on six datasets spanning Alzheimer’s, Parkinson’s, and other population with contact sports. 

The ablation studies are somewhat comprehensive, although there's a few weaknesses in their actual implementation (see below).

### Weaknesses
The main weakness of the paper is that it combines multiple methods instead of making an original novel contribution. 


Several ideas (except the pyramidal projections of connectivity and satisfying SPD constraints) have been proposed before individually. Even the DimMap projection operator that is at the core of the method has been proposed before. Theoretical and computational advancements  for optimization on SPD manifolds both using classical and deep learning approaches have been proposed before. This paper combines several such ideas and applies it to fMRI data. Thus the novelty of the method is low. 

The GeoMix operator is simply an interpolation between SPD matrices on the manifold. The authors introduce it as a new idea but it has been heavily used and studied before. 


Besides SPD manifold optimization, techniques such as dual path stacking have also been proposed before. For e.g. Brain-Mamba Behrouz & Hashemi (2024) performs a temporal and spatial stacking, while FST-Mamba Wei et al. (2025) has proposed a hierarchical stacking formulation. The authors have cited these methods. 


Step (iv) bypassing the Path #1, which disables the initial manifold-aware temporal evolution and
eliminating the sequence fusion under SSM and relying solely on the average time-step matrix for
subsequent computations. Too many things removed. Just disbale the manifold-ware temporal evolution


“While the authors claim a substantial decline when Path #1 is removed, Figure 3 shows only marginal quantitative changes (mostly ≤ 3 % in Accuracy / F1). The benefit of the geometric temporal constraint thus appears more in preserving SPD validity and smooth dynamics than in producing large performance gains.”

In the ablation studies, step iv) is over ablated. To truly test the geometric constraint, the authors should keep Path #1 intact but replace the log–exp manifold operations with Euclidean updates, holding all other factors fixed. The step iv) ablation mixes removal of geometric constraints with also removal of recurrence (SSM), thus weakening the main argument about geometry-aware SSM, which is the main contribution of the paper. Thus it is unclear whether the modest performance drop which is observed (2–4 %) denoted by purple colors, may simply reflect losing temporal recurrence and not necessarily arise from avoiding the geometry constraint itself. A fairer test would retain the same SSM structure but operate in Euclidean space to quantify the direct impact of manifold geometry.

In experimental evaluation, the results of the method should ideally be compared with FST-Mamba, BNMamba, BTMamba to really separate the geometric-constraint contribution and whether the projection steps and the dynamic update steps are truly required for performance. They are certainly required to ensure the appropriate constraints, so that is not under question.



Other manifold temporal models (e.g., Riemannian LSTM, SPD recurrent networks) or covariance-based DPL frameworks could also provide appropriate comparisons (although this is not required).

### Questions
Why aren't the results in Table 1 starred? Were they not significant? 

In the first step following the pearson correlation calculation to construct the connectivity matrices, why not just keep the significant correlations? Depending upon the dataset (ADNI) for e.g., the matrix may be sparse (significance threshold) to begin with. Will it make the signal more noisy?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Geo-Mamba, a deep learning architecture that adapts the selective state-space model (Mamba) to operate directly on the symmetric positive definite (SPD) manifold. This is motivated by the observation that functional connectivity (FC) matrices from fMRI data lie in a non-Euclidean space, making standard Euclidean models geometrically inappropriate. Geo-Mamba leverages Log-Euclidean mappings to move between the manifold and its tangent space, allowing linear operations while preserving intrinsic geometry. The model processes FC matrices through a dual-path design capturing both multi-scale local dependencies and global spatial structure, fused via a geometry-aware mixing operator. The method is evaluated on six fMRI datasets and demonstrates competitive or superior performance to baselines such as Mamba, SPDNet, and STAGIN.

### Strengths
The core idea of adapting the Mamba (SSM) architecture to the non-Euclidean geometry of SPD manifolds is both novel and technically well-motivated.
The use of the Log-Euclidean tangent space for SPD matrices is mathematically justified and enables Euclidean operations while preserving manifold constraints.
The design of multi-scale spatial feature extraction through sliding SPD sub-blocks is a creative way to leverage manifold properties for modeling spatial dependencies in functional connectivity data before passing them into the SSM.
The method is comprehensively evaluated across six public neuroimaging datasets and demonstrates superior or competitive performance compared to a wide range of relevant baselines.

### Weaknesses
While the paper presents a compelling and potentially impactful idea, several ambiguities limit the clarity of the contribution and make it difficult to pinpoint the true source of performance gains and whether certain architectural components are genuinely helpful. Some components are not clearly justified or ablated, and those that are seem to have minimal impact. The experimental setup lacks transparency on key implementation details, such as baseline adaptation and hyperparameter tuning. Below are the concerns in order:

1. While the ablation study in Figure 3 is informative, it lacks standard deviations and shows that the four ablated components do not significantly degrade classification performance (based on the * indicators). This makes it difficult to determine whether modules such as the global descriptor G or the “distillation path” meaningfully contribute to the results. Furthermore, several key components including the self-attention layer, the geodesic mixing used in the output mapping, and the learnable geometric anchor $Q$ are not ablated, and their individual impacts remain unclear.

2. The authors mention that FC matrices are typically ordered by brain regions. The model assumes that ROIs adjacent in the n×n functional connectivity matrix correspond to spatially adjacent regions in the brain. However, cortical organization is inherently three-dimensional, and the 1D sliding-window design in the pyramid-based feature extraction cannot completely capture this 3D spatial topology. This weakens the claim of modeling hierarchical spatial information and limits generalization across modalities.

3. The paper does not provide sufficient detail on how the baseline sequence models (e.g., Transformer, Mamba, STAGIN) were adapted for this task. It is unclear if they were fed the vectorized FC matrix, the same "pseudo-temporal sequence" as Geo-Mamba, or the time-series fMRI data. Since the authors argue that Euclidean operations are unsuitable for FC matrices, comparing Geo-Mamba to baselines trained on Euclidean FC inputs may be unfair. Clarifying these points is crucial to isolate whether improvements stem from manifold geometry or input design.

4. The number of SSM layers used is unspecified, despite claims about capturing “hierarchical temporal evolution” (line 117). It is not demonstrated how stacking helps capture hierarchical spatial information or affects performance.

5. The paper does not clearly describe how $B_t$, $C_t$, and $\lambda_t$ are generated from the GRU or how this differs from the selective parameterization in Mamba. The connection between GRU gating and SPD-preserving updates remains unclear.

6. The “pseudo-temporal sequence” is created by stacking spatial SPD sub-blocks from the averaged n×n connectivity matrix, not by modeling real temporal dynamics. Hence, the terms “pseudo-temporal” throughout the text or “time-dependent” (line 449) are misleading, since the model processes only spatial relationships rather than the actual neural time-series.

7. The term "distillation path" is confusing. In machine learning, "distillation" typically refers to student–teacher knowledge transfer. Using it to mean "dimensionality reduction" is non-standard and should be clarified.

### Questions
1. What is the motivation for including a self-attention layer after the SSM block? Was it ablated, and how did it affect performance or interpretability? 

2. How were key hyperparameters (state dimension, window sizes, number of downsamples) chosen? What is the total parameter count and computational overhead relative to Mamba or SPDNet? Is it scalable to higher-resolution atlases? Clarifying these points is essential to understand the true contribution and applicability of the model.


3. Did the authors observe any overfitting on smaller datasets?


4. In ablation (ii), does removing the “global path” correspond to removing G from $S_{all}$? If so, how critical is this path to multi-scale modeling?

5. How is the attention anchor $Q$ intialized? Was any regularization applied to it during training?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This manuscript introduces Geo-Mamba, a geometric state-space model designed to address the misspecification of conventional Euclidean sequence models when analyzing functional connectivity (FC) data from fMRI, which inherently resides on the SPD manifold. Geo-Mamba employs a dual-path selective state-space design, combining hierarchical modeling with geometry-aware dimensionality reduction, and fusing their outputs via a tailored GeoMix operator. The model was evaluated on six public fMRI datasets against several baseline methods.

### Strengths
- Originality: The method introduces an original combination of hierarchical pyramid multi-spatial scale FC aggregates and stacks them to form a sequence fed to a state space model, which preserves geometry.

- Evaluation: The authors evaluate their approach across several public datasets, encompassing both small and large scales, and compare it to several relevant baselines.

- Neuroscientific Interpretation: The interpretation of the results - although limited to a subpath of the entire architecture - established meaningful connections to existing neuroscientific understanding.

### Weaknesses
- Interpretation analysis was limited to path #2, omitting coverage of the SSM path.
- The title and introduction emphasize SSMs, yet the proposed method incorporates multiple paths and components in addition to the proposed geometric SSM, making it difficult to isolate the source of performance gains.
- The ablation analysis lacks clarity. It would be beneficial to focus on a single key metric (e.g., accuracy) and move others to the appendix. Obvious ablation candidates include removing path #1 and/or path #2 entirely. Another relevant candidate is ablating the self-attention block within path #1.
- Table 4 (appendix) indicates the use of different hyperparameters and multiple learning rates per dataset. The authors should clarify whether they tuned the learning rate of Geo-Mamba for each dataset, how they selected the parameters, and whether the same approach was applied to the baseline methods.

### Questions
- What is the rationale for sharing the query (Q) across all spatial scales?
- Clarify the necessity and role of the self-attention layer positioned between the two SSM blocks, as the original Mamba architecture typically omits self-attention. Include ablation studies demonstrating the impact of removing either the self-attention or the SSM blocks.
- After model fitting, analyze the learned weight alpha in the GeoMix layer to determine the relative importance of path 1 versus path 2.
- The description of the geometry preserving core SSM component (lines 210 to 220), needs significant rephrasing and expansion for clarity.
- There is an incorrect reference at line 330 (Adam et al. 2014).

### Soundness
2

### Presentation
2

### Contribution
2
