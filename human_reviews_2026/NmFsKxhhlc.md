# Neurocircuitry-Inspired Hierarchical Graph Causal Attention Networks for Explainable Depression Identification

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 2

## Abstract
Major Depressive Disorder (MDD), affecting millions worldwide, exhibits complex pathophysiology manifested through disrupted brain network dynamics. Although graph neural networks that leverage neuroimaging data have shown promise in depression diagnosis, existing approaches are predominantly data-driven and operate largely as black-box models, lacking neurobiological interpretability. Here, we present NH-GCAT (Neurocircuitry-Inspired Hierarchical Graph Causal Attention Networks), a novel framework that bridges neuroscience domain knowledge with deep learning by explicitly and hierarchically modeling depression-specific mechanisms at different spatial scales. Our approach introduces three key technical contributions: (1) at the local brain regional level, we design a residual gated fusion module that integrates temporal blood oxygenation level dependent (BOLD) dynamics with functional connectivity patterns, specifically engineered to capture local depression-relevant low-frequency neural oscillations; (2) at the multi-regional circuit level, we propose a hierarchical circuit encoding scheme that aggregates regional node representations following established depression neurocircuitry organization, and (3) at the multi-circuit network level, we develop a variational latent causal attention mechanism that leverages a continuous probabilistic latent space to infer directed information flow among critical circuits, characterizing disease-altered whole-brain inter-circuit interactions. Rigorous leave-one-site-out cross-validation on the REST-meta-MDD dataset demonstrates NH-GCAT's state-of-the-art performance in depression classification, achieving a sample-size weighted-average accuracy of 73.3\% and an AUROC of 76.4\%, while simultaneously providing neurobiologically meaningful explanations. This work represents a significant advancement toward mechanism-aware, explainable artificial intelligence (AI) systems for psychiatric diagnosis.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes NH-GCAT, a neurocircuitry-inspired hierarchical graph causal attention network designed for explainable MDD identification. The model introduces three major modules aiming to incorporate neuroscientific priors into graph-based learning. The authors report advanced performance on the REST-meta-MDD dataset and provide multi-level interpretability analyses demonstrating biologically meaningful findings.

### Strengths
1. The paper tackles an important topic — enhancing both accuracy and interpretability of GNNs for MDD classification — and makes a solid attempt to integrate biological priors (depression-related circuits) with deep learning.

2. The interpretability analyses (frequency-specific validation, hierarchical circuit visualization, causal inter-circuit analysis) are thorough and align well with known MDD mechanisms.

3. The paper is clearly written and provides extensive quantitative results, including LOSO-CV analysis across 16 sites, supporting generalizability.

### Weaknesses
1. Unclear module motivation and mapping between equations and architecture. It is difficult to align the mathematical formulations in Section 3 (Equations 1–21) with the modules illustrated in Figure 2. The description of RG-Fusion, HC-Pooling, and VLCA lacks explicit motivation for each design component — for example, why certain fusion mechanisms, Gumbel-Softmax hierarchical assignments, or causal attention structures were chosen. The rationale for these designs should be better explained or visualized in connection with the biological circuits they represent.
2. Ambiguity in ROI-to-circuit mapping. The paper uses AAL116 for ROI definition, yet defines five circuits based on functional organization. It remains unclear how the authors aligned AAL ROIs to these five circuits. This mapping is problematic because AAL includes cerebellar regions, which are not part of these circuits. The paper should clarify how such ROIs were handled or reassigned — were cerebellar nodes excluded, or mapped to the nearest cortical network based on spatial proximity?
3. Lack of comparison with related literature. The authors cite several interpretable GNNs but omit discussion or comparison with relevant recent studies that also integrate community structure [1-3] or causal learning [4] in brain graphs. 
4. Limited experimental scope. Experiments are only conducted on a single dataset. Given the claim that the model is neurocircuitry-inspired and generalizable, evaluation on at least one other psychiatric or neurological condition (e.g., ASD, AD, schizophrenia, bipolar disorder) would better demonstrate the adaptive ability and robustness of the proposed framework.


[1] Community-Aware Transformer for Autism Prediction in fMRI Connectome. MICCAI 2023
[2] Biologically Plausible Brain Graph Transformer. ICLR 2025
[3] BrainGT: Multifunctional Brain Graph Transformer for Brain Disorder Diagnosis
[4] BrainOOD: Out-of-distribution Generalizable Brain Network Analysis. ICLR 2025

### Questions
See Weaknesses.

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
4

### Summary
The author proposed a hierarchical graph neural network for major depressive disorder (MDD) analysis. A residual gated fusion module was proposed to aggregate BOLD signals at the temporal level. The authors also conducted extensive experiments to show that the model performs better than baselines, that every design is useful, and that the model provides sufficient interpretability.

### Strengths
- Figure 4 includes the ROC and PR curves for better performance evaluation

- Table 2 includes weighted average values, which makes the performance difference clearer. 

- The analysis is comprehensive. While the datasets are somewhat limited, the author discussed them in the future works section.

- A complete ablation is done in table 3 that details the contribution of each component.

### Weaknesses
- This seems to be a resubmission of a previously reviewed work, where the authors promised to discuss how the work differentiates itself from related approaches like https://arxiv.org/pdf/2410.18103, https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10230606 in the related works section. As of the current draft, I don’t see this being done. The current related works section is largely the same as the previous draft. The authors seem to briefly touch upon this in Section. A.3. However, there is not enough comparison with specific works, and no citations were added in the entire section of A.3. Furthermore, no comparison was done against the works that the authors promised to do. 

- While the ROC curve is useful, the implications are limited as the curves for other baselines are not reported. It would be useful to replicate one or two baselines and see how the curves compare.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a novel framework for diagnosing major depressive disorder (MDD) from fMRI data. The method integrates three hierarchical levels of brain information: node (cortical region) level, neural circuit level (for example, default mode and salience networks), and whole-brain network level, within a unified graph neural network (GNN) architecture. The effectiveness of the framework is evaluated on a single fMRI dataset.

### Strengths
The integration of three levels of information makes sense to me, and I also appreciate the general idea of leveraging neural circuits as prior information. However, this prior knowledge does not seem to be fully utilized or to effectively reflect existing neuroscience evidence.

### Weaknesses
The experimental evaluation is too weak. First, only a single dataset is used. Why not evaluate on other MDD datasets such as SRPBS, OpenNeuro, or even the UK Biobank? It would also be more convincing to train on one dataset (for example, REST-meta-MDD) and test on another (for example, SRPBS) to assess the generalization ability of the proposed approach. 

In addition, the comparisons with prior work are neither rigorous nor fair. The results of several state-of-the-art methods appear to be directly copied from the original papers, even though the experimental setups differ substantially. For instance, BrainIB used a 10-fold cross-validation scheme, while the current work adopts 5-fold cross-validation, which makes the comparison unreliable. You should rerun these methods in your own environment for a fair evaluation, especially since many of them have released official implementations. Furthermore, even the baseline results reported here differ from those in other published replications, which raises concerns about reproducibility and evaluation consistency.

The network design appears overly complicated and seems to contradict the stated motivation for interpretability. From a neuroscience perspective, researchers generally prefer architectures that are simple, easy to use, and supported by clear clinical evidence or interpretability. Although you attempt to incorporate circuit-level priors (which might be the only clinically grounded component), the overall network design (especially with several components insufficiently explained) undermines the interpretability of the entire framework.

### Questions
1. Although the integration of three levels of information is conceptually reasonable, the current ablation study, which incrementally adds one module at a time, does not clearly reveal which component contributes most to the final decision. My question is: among the node-level, circuit-level, and network-level information, which source or combination of information plays the dominant role in the diagnostic performance?

2. The network design is not clearly explained. For the RG-Fusion module, there are two inputs, $X^1$ and $X^2$, but it is unclear why they are fused in such a complicated manner. It appears that $X^1$ and $X^2$ are first fused, and then another branch fuses information from $X^1$ again. The motivation for this structure should be clarified. In addition, is the feature dimension $d$ consistent between Equations (1) and (5)?

3. The loss function contains three regularization terms. How are the different $\lambda$ values tuned in practice? It is unclear whether a single set of $\lambda$ values can generalize across different datasets, and I suspect that the optimal configuration might be highly dataset-dependent.

4. I acknowledge that other approaches do not explicitly consider low-frequency oscillatory patterns in BOLD signals. However, it is unclear why you claim that your method captures such information. From my understanding, you simply add a new input (the raw BOLD signal) and apply a basic Transformer architecture. Do you attribute the ability to capture low-frequency oscillations solely to the Transformer design? 

5. It is difficult to clearly understand the source of the reported performance gain. The proposed model takes two inputs, the functional connectivity (FC) matrix $X^1$ and the BOLD signal $X^2$, which effectively makes the framework a multi-view learning system. If I understand correctly, most existing GNN-based approaches use only $X^1$. Since multi-view learning has recently gained attention in neuroscience, it is important to clarify whether the performance improvement primarily stems from introducing an additional input modality rather than from the proposed complex network design itself?

### Soundness
2

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
The paper presents NH-GCAT, a neurocircuitry-inspired model designed for explainable depression identification using fMRI data. The model integrates brain knowledge with graph neural networks through three main modules: RG-Fusion, HC-Pooling, and VLCA, which together capture hierarchical and causal relationships among brain regions. When evaluated on the REST-meta-MDD dataset, NH-GCAT achieves 73.8% accuracy and 78.5% AUC, outperforming previous methods while revealing biologically meaningful patterns in key brain networks associated with depression.

### Strengths
This paper introduces a multi-level modeling framework that includes three hierarchical layers: the region level, the circuit level, and the network level, which together help capture brain functional dynamics from local to global scales. It is also validated on the large-scale REST-meta-MDD dataset, which contains more than 1,600 subjects from 16 research centers.

### Weaknesses
This paper contains many critical methodological and conceptual flaws, as well as unclear details.

1. The overall framework of the paper is outdated. Many existing studies have already proposed similar approaches. Please refer to related works in IEEE TMI, IEEE JBHI, and MICCAI.

2. Several fundamental assumptions in the paper are problematic, particularly regarding the causal inference in the VLCA module. The variational conditional probability assumptions are incorrectly formulated, and the paper completely ignores the prior and posterior distributions.

3. The authors designed Equation 21, but no ablation experiments on the parameter $\lambda$ are presented.

4. According to the description of Counterfactual Reasoning on page~19, the authors set $A^{cf} =\textbf{I}_{C}$ Identity Matrix.

### Questions
No

### Soundness
1

### Presentation
2

### Contribution
2
