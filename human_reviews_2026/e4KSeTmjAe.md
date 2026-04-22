# Reservoir Computing with Spatial Filtering and Manifold Learning for fMRI Classification

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
We introduce a parametric framework that couples discriminative spatial filtering with  reservoir computing to distinguish spatiotemporal structure in resting-state fMRI in two classes. Temporal dependencies are encoded in a reservoir, while supervised spatial filtering on reservoir states isolates condition-specific patterns; parametric Uniform Manifold Approximation and Projection (UMAP) then yields compact nonlinear embeddings fit on training data and evaluated with cross-subject validation. On 163 participants (97 healthy controls, 66 major depressive disorder), the method reaches 87\% accuracy, outperforming network-feature pipelines using LDA, SVM, kNN, and GNN. The framework also generalizes to autism spectrum disorder classification, achieving competitive accuracy on the ABIDE (NYU) benchmark and ranking among top state-of-the-art methods. Interpretability combines spatial-pattern maps with Shapley-value attribution, providing coherent, region-level explanations that consistently implicate cortical and subcortical areas associated with both major depressive disorder and autism spectrum disorder. The framework offers an interpretable route to modeling spatiotemporal organization in clinical and cognitive fMRI.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a pipeline that integrates discriminative spatial filtering with reservoir computing to classify multidimensional spatio-temporal features from high dimensional, multivariate fMRI time series data. The proposed approach applies a supervised spatial filter (CSP) to the reservoir states, thereby enabling the extraction of discriminative spatiotemporal representations. The pipeline also includes a parametric manifold learning via UMAP for nonlinear dimensionality reduction with the final classified carried out by LDA. The authors evaluate the performance of their pipeline on a single HCP dataset for major depressive disorder and claim that their approach outperform LDA, SVM, kNN, and GNN. The paper provides an interpretation of their results on this dataset by consistent identification of cortical and subcortical regions implicated in major depressive disorder.

### Strengths
The proposed approach extends the reservoir computing framework by incorporating discriminative spatial filtering and parametric UMAP to enable efficient classification.

There is a significant amount of ablation studies that illustrate the importance of UMAP

### Weaknesses
While the end to end pipeline seems to be new,  each of the components in the pipelines has been extensively studied in the literature and the paper lack novelty in this sense. 

The empirical evaluation is limited to a single data set. 

The computational complexity of CSP is significant, which is applied to the raw data as well as the reservoir states.

Reservoir computing was introduced to reduce the complexity of recurrent neural networks. However currently there are alternatives that 
work extremely well in practice e.g. LSTM. 

The authors should compare their approach to transformer based approaches that can capture multidimensional spatial and temporal data
The comparison in Table 3 is really unfair and uses different types of inputs for different methods (FC for GNN,  graph metrics for the rest)

### Questions
The authors should try to respond to each of the weaknesses mentioned in the previous section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a classification pipeline for fMRI data by combining spatial filtering (CSP), reservoir computing (RC), UMAP and LDA which it claims gives state of the art accuracy on a MDD vs Controls dataset.

### Strengths
The paper is well motivated in the sense that it aims to utilize both spatial and temporal information present in fMRI data to perform classification. It also does a good job of combining existing methods without trying to reinvent any wheels.

### Weaknesses
This is largely an application paper, which means it really needs to include robust experiments to prove the claims being made. In my opinion it falls significantly short of that.
1. It claims to be state of the art, but the comparisons are only made against baseline methods, rather than making an honest attempt at comparing against a plethora of fMRI classification approaches from the literature.
2. Experiments are provided only on a single dataset, and since it's unclear whether it's publicly available or not, it's hard to know how much work has already been done on that particular dataset. Either way, showing a method works on a single dataset is simply not enough.
3. train/test split. It's unclear what was used for hyperparameter tuning, was there a completely blind test set that was set apart to evaluate performance? or is everything reported here simply on train/test splits with no separation between validation and testing.
4. generalization. It's unclear how well does this approach generalize. Again, it's impossible to establish anything beyond cross subject generalization here. If the method is truly being presented as "SOTA" for MDD classification it needs to be shown that it can generalize out of dataset / different sites etc.
5. ablation. If a pipeline is being proposed, it'd make sense to ablate different components of the pipeline to show how they contribute (just like the paper does for UMAP).

overall writing of the paper can also be improved. please use parentheses for citation and also use consistent notation throughout the paper.

### Questions
see weaknesses.

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
The paper presents a multi-stage pipeline for classifying resting-state fMRI data from individuals with major depressive disorder (MDD) and healthy controls. The approach combines Common Spatial Patterns (CSP) for spatial filtering, Reservoir Computing (RC) for temporal encoding, parametric UMAP for nonlinear dimensionality reduction, and LDA for classification. The authors report an accuracy of 87% (F1 score of 0.83) on a dataset of 163 participants, outperforming basic baselines including SVMs, kNNs, and GNNs trained on functional connectivity matrices. They also provide region-level interpretability analysis via SHAP value back-projection to brain regions.

### Strengths
- Addresses a relevant and interdisciplinary problem (neuroimaging classification with limited data).
- The method is computationally simple and interpretable, combining established signal processing and manifold learning steps.
- Experimental setup and preprocessing are clearly described; evaluation uses subject-wise cross-validation.
- The reported classification accuracy is relatively high for the given dataset.

### Weaknesses
1. Not ICLR-level novelty.
The paper essentially stacks well-known methods (CSP, RC, UMAP, LDA) without demonstrating new theoretical or algorithmic insights. The main idea—applying CSP on reservoir states—is incremental and not well justified conceptually.

2. Pipeline without integration.
The approach feels like a sequence of independent components rather than a coherent model. There is no end-to-end learning (as far as I understand) or extended ablation showing why each component is necessary. 

3. Limited baselines and older references.
The comparison set is narrow and dated (LDA, SVM, kNN, GNN on connectivity). More competitive baselines such as temporal CNNs, GRU-ODE-Bayes, transformers, or contrastive representation learning approaches are missing.

4. Small dataset and potential overfitting.
With only 163 subjects, high reported accuracy (87%) may overstate generalization. The evaluation relies solely on within-dataset cross-validation without a separate held-out or external validation set.

5. Unclear contribution to machine learning.
While the application is interesting, the work reads more like a domain-specific methods report than an ML conference paper. There is no theoretical contribution or clear takeaway for the broader ML community.

6. Writing and framing.
The manuscript is verbose and reads more like a research report than a conference submission. Some methodological descriptions (e.g., CSP, LDA equations) occupy space but add little conceptual clarity.

### Questions
- Can the authors confirm that the parametric UMAP was fit exclusively on the training folds in each cross-validation iteration (to avoid information leakage to the test set)?
- What is the variance across folds—does performance generalize beyond 163 subjects?
- How do your results compare to recent deep representation learning approaches for fMRI (e.g., BrainLM, transformers, or graph contrastive learning)?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a multi-stage pipeline combining Common Spatial Patterns (CSP), Reservoir Computing (RC), and parametric UMAP for classifying resting-state fMRI data from Major Depressive Disorder (MDD) patients vs. healthy controls. The approach applies CSP twice (before and after RC) to extract spatial features, uses RC for temporal encoding, applies UMAP for dimensionality reduction, and performs LDA classification. On 163 participants, the method achieves 87% accuracy, outperforming classical baselines. While the problem is clinically relevant and the interpretability analysis provides biological insights, the work suffers from limited novelty, insufficient theoretical justification, questionable domain transfer from EEG to fMRI, and weak experimental validation.

### Strengths
- **Clinically important problem**: MDD classification from resting-state fMRI addresses a significant challenge in computational psychiatry with potential real-world impact.
- **Interpretability**: The backward propagation of SHAP values to identify relevant brain regions (Fig. 3) provides biologically plausible explanations, highlighting areas like the medial superior frontal gyrus (DMN) that are known to be associated with MDD.
- **End-to-end framework**: The paper presents a complete pipeline from raw fMRI signals to classification with detailed implementation.
- **Comprehensive baseline comparison**: The authors compare against multiple classical methods (LDA, SVM, kNN, GNN) on the same dataset.

### Weaknesses
1. **Limited novelty**:
    - The paper combines existing techniques (CSP from 2000, RC from 2000s, UMAP from 2018) without substantial innovation
    - The main claimed contribution is "applying CSP to reservoir states rather than raw data" - this is an engineering choice, not a methodological advance
    - Even the input matrix G structure is borrowed from Hramov et al. (2024)
    - **For ICLR 2026, this level of novelty is insufficient**
2. **Lack of theoretical justification**:
    - **Critical**: Why should this specific combination of methods work? The paper provides no principled explanation
    - Why apply CSP twice? What is the theoretical motivation for spatial filtering → temporal encoding → spatial filtering again?
    - Why is UMAP necessary after CSP produces discriminative features?
    - The design appears entirely empirical ("we tried this and it worked") without understanding *why*
3. **Questionable domain transfer (EEG → fMRI)**:
    - CSP was designed for EEG/BCI applications with fundamentally different signal characteristics
    - **EEG**: high temporal resolution (~1000 Hz), low spatial resolution, electrical activity
    - **fMRI**: low temporal resolution (~0.5 Hz), better spatial resolution, hemodynamic response
    - **The paper provides no justification for why an EEG-based spatial filtering method should be appropriate for fMRI's slow hemodynamic signals**
    - All Related Works references (Section 2.2) discuss EEG applications, yet the method is applied to fMRI without domain-specific adaptation
4. **Small dataset and overfitting concerns**:
    - 163 subjects is small for neuroimaging in 2024-2025 (typical studies use 500-5000+ subjects)
    - 87% ± 0.05 accuracy on such a small dataset raises serious overfitting concerns
    - No validation on independent datasets or multi-site data
    - Suggest: Test on public datasets (ABIDE, UK Biobank, HCP) to demonstrate generalization
5. **Insufficient ablation studies**:
    - Table 3 only shows: full pipeline, without UMAP, and without RC
    - **Missing critical ablations**:
        - CSP-I only (no RC, no CSP-II)
        - RC only (no CSP at all)
        - CSP-II only (no CSP-I)
        - Single CSP vs. double CSP comparison
    - Without systematic ablation, we cannot assess each component's contribution
6. **Weak baselines**:
    - Classical methods (LDA, SVM, kNN on graph metrics) are outdated for 2025
    - 2-layer GNN achieving only 64% suggests implementation issues
    - **Missing**: Recent deep learning methods (BrainNetCNN 2017+, Transformer-based models 2020+, state-of-the-art fMRI classification methods from 2023-2024)
7. **Misleading terminology**:
    - The paper claims "explicit spatial modeling" but CSP only maximizes variance ratios, not spatial topology
    - Calling this "spatial optimization" implies learnable parameters, but CSP is a closed-form eigendecomposition

### Questions
1. **What is the theoretical justification for applying EEG-based CSP to fMRI?** Given the fundamental differences in signal characteristics (temporal resolution, SNR, physiological basis), why should variance-based spatial filtering designed for EEG work for hemodynamic responses?
2. **Why is the double-CSP architecture necessary?** Please provide ablation studies comparing:
    - Raw → RC → UMAP → LDA
    - CSP-I → RC → UMAP → LDA
    - RC → CSP-II → UMAP → LDA
    - CSP-I → RC → CSP-II → LDA (no UMAP)
3. **What is the actual novelty claim?** If all components (CSP, RC with separated inputs, UMAP, LDA) are from prior work, what is the core contribution beyond empirical combination?
4. **How do you address overfitting concerns?** With 163 subjects and 87% accuracy, what evidence supports generalization? Can you test on independent public datasets?
5. **Why does the baseline GNN only achieve 64%?** This is surprisingly low for graph-based fMRI classification. Is there an implementation issue, or does this suggest the dataset has low signal?
6. Figure 2: What are the axes for the distribution plots? How many reservoir configurations were tested in total?
7. Why use parametric UMAP instead of simpler alternatives like PCA or kernel PCA? What is lost if you use linear dimensionality reduction?
8. How sensitive is the pipeline to hyperparameters? Varying Minp, Mout, reservoir size, etc.?

### Soundness
2

### Presentation
1

### Contribution
1
