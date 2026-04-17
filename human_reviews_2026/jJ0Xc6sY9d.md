# TimeSAE: Sparse Decoding for Faithful Explanations of Black-Box Time Series Models

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
As black box models and pretrained models gain traction in time series applications, understanding and explaining their predictions becomes increasingly vital, especially in high-stakes domains where interpretability and trust are essential. However, most of the existing methods involve only in-distribution explanation, and do not generalize outside the training support, which requires the learning capability of generalization. In this work, we aim to provide a framework to explain black-box models for time series data through the dual lenses of Sparse Autoencoders (SAEs) and causality. We show that many current explanation methods are sensitive to distributional shifts, limiting their effectiveness in real-world scenarios. Building on the concept of Sparse Autoencoder, we introduce TimeSAE, a framework for black-box model explanation. We conduct extensive evaluations of TimeSAE on both synthetic and real-world time series datasets, comparing it to leading baselines. The results, supported by both quantitative metrics and qualitative insights, show that TimeSAE provides more faithful and robust explanations. Our code is available in an easy-to-use library TimeSAE-Lib: https://anonymous.4open.science/w/TimeSAE-571D/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a sparse autoencoder framework using JumpReLU activation to explain time series models. It aims to improve faithfulness and OOD generalization, showing quantitative gains over existing explainers.

### Strengths
1. The presentation of the paper is clear.

2. The experimental evaluation is comprehensive on multiple datasets.

### Weaknesses
1. The motivation focuses on generalization and faithfulness. However, the introduction lacks concrete examples that demonstrate why these issues are significant in real-world time series applications. Providing tangible consequences of poor generalization would make the paper's motivation more compelling.

2. Although the introduction cites many related methods, such as TimeX and TimeX++, the discussion is mostly descriptive and ends with the generic claim that they lack generalization. The authors do not analyze how these methods fail under distribution shift that restrict the OOD performance.

3. For methodology, the core components (SAE and JumpReLU activation) are directly adopted from prior work[1]. The contribution mainly lies in applying these components to time series explainability task. It would be helpful if the authors could clarify the fundamental methodological difference between this work and [1].

4. Adding the training complexity and runtime analysis would strengthen the empirical validity of the method. The proposed framework integrates multiple modules, which likely introduce non-trivial computational overhead.

5. The experimental evaluation could be improved by including comparisons with more recent explainers such as TIMING [2], ORTE [3], and StartGrad [4].

[1] Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders
[2] Timing: Temporality-aware integrated gradients for time series explanation
[3] Optimal information retention for time-series explanations
[4] Start smart: Leveraging gradients for enhancing mask-based xai methods

### Questions
Please see the Weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a technically sound and well-motivated approach, TimeSAE, to time series explainability that explicitly tackles out-of-distribution (OOD) robustness and faithfulness through concept-based sparse decoding and causal intervention. Its integration of JumpReLU, contrastive counterfactual loss, and compositional consistency loss is novel and empirically validated across diverse models and datasets. The introduction of EliteLJ adds practical value for real-world evaluation. However, the theoretical claims rely on assumptions that are not empirically verified, and hyperparameter sensitivity is acknowledged but not systematically mitigated. No direct evidence found in the manuscript for ablation on the necessity of each loss component beyond $L_{cf}$ and $L_{cc}$.

### Strengths
1. This paper provides a novel framework that integrates with Sparse Autoencoders with causal counterfactuals for time series. TimeSAE uniquely combines sparse concept learning with counterfactual generation via interventions on latent concepts.  Couples sparsity with a contrastive counterfactual loss that explicitly encourages CaCE order preservation.

2. This paper conducts comprehensive empirical validation across settings.  Evaluation spans synthetic (FreqShapes, SeqComb-UV) and real-world datasets (ECG, PAM, ETTh, EliteLJ) with known ground-truth explanations.  Black-box models include both trained (Transformer, DLinear) and large pretrained (TimeGPT, Chronos) architectures, demonstrating broad applicability.  Metrics cover both localization accuracy (AUPRC) and faithfulness (Fx), with statistical significance testing (paired t-tests) supporting superiority. 

3. This paper introduces a new benchmark dataset with expert annotations. EliteLJ provides skeletal pose sequences from elite long jump athletes with performance labels and phase annotations, enabling realistic evaluation of temporal explanations in sports analytics. The dataset includes 386 samples with 50-frame sequences and 34-dimensional pose features, filling a gap in human-motion-based XAI benchmarks. Ground-truth performance metrics (jump distance) allow regression-based faithfulness evaluation.

### Weaknesses
1. The theoretical assumptions and equations have some issues. 
- The mathematical symbols used in this article are very confusing and difficult to read. For example, $g$ is the decoder and the generator at the same time. The $\mathcal{E}$ is the encoder, explainer, and counterfactual explanation in Eq. 4.  $\tilde{x}$ is the input in Theorem 1 but embedding in Eq.4. 
- Theorem 1 assumes a small approximation error $\epsilon_{cf}$ (Eq. (5)), but no experiment quantifies this error or validates its boundedness across datasets.  The proof sketch relies on reconstruction and approximation errors being smaller than causal effect gaps, yet no ablation isolates the impact of reconstruction fidelity on faithfulness ordering.  

2. This paper misses some critical experiments in the ablation study.
- Performance heavily depends on dictionary size, Eq. 1, sparsity coefficient $\eta$, and loss weights ($\alpha$, $\lambda$), with optimal values varying across datasets.
- JumpReLU and TopK variants require different hyperparameter schedules, increasing deployment complexity.  

3. Some other unclear parts.
- Some terms are not written consistently, such as CaCE and CACE. The JumpReLU definition introduces a learnable parameter $\phi$ , which should be a matrix, whereas the text then refers to a scalar threshold $\phi_{k}$ for each concept feature $k$.
- Some definitions are not well expressed. In Eq.2 the function $s(\cdot)$ is not well defined. In Sec. 3.4, the $\psi$ is not well defined. The first-order and higher-order contributions should be explained. In Eq. 10, the $\mathcal{L}_{label-fidelity}$ is not well defined.

### Questions
- How is the “true” counterfactual label $y_{cf}$ in Eq. 5 obtained during training, given that the true causal data-generating process is unknown?  
- How does TimeSAE handle multivariate interactions when concepts are learned per-feature versus jointly?  
- Why is label-fidelity (Llabel-fidelity) necessary if $f(\tilde{x}) \approx f(x)$ is already enforced via reconstruction?  
- The proof sketch for Theorem 1 requires the reconstruction error $\epsilon_{rec}$ and the approximation error $\epsilon_{cf}$ to be small enough. Does the paper offer an empirical measure of the reconstruction error $\epsilon_{rec}$ for the different datasets/models?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces TimeSAE, a novel framework for generating faithful explanations for black-box time series models. The authors frame the problem of explainability through the lens of causality, arguing that true faithfulness requires understanding why a model makes a prediction (causal reasoning) rather than just what features it uses (correlational analysis). The proposed TimeSAE method is based on a Sparse Autoencoder (SAE) that decomposes a time series into a sparse set of interpretable "concepts." The key contributions are: (i) a theoretical guarantee that TimeSAE preserves the relative ordering of causal effects under bounded approximation error; (ii) a new sports-focused benchmark dataset with expert annotations; (iii) extensive experiments across 8 datasets.

### Strengths
The paper makes several combined contributions in the field of time series XAI:
* Targeted solution to core time series XAI challenges. Even though previous work has been mentioned, TimeSAE directly addresses three critical pain points: OOD generalization, causal faithfulness, and dead concepts. These design choices are well-motivated by gaps in prior work and align with real-world needs for robust explanations.
* The inclusion of Theorem 1 (proving order-preserving causal faithfulness) enhances the framework’s credibility, as theoretical guarantees are rare in time series XAI. This is practically enforced via a contrastive loss on counterfactuals.
* OOD Generalization: A compositional consistency loss is incorporated to ensure the explainer generalizes to OOD samples by enforcing that the encoder acts as an inverse to the decoder, even for unseen combinations of concepts.
* A New Dataset: The paper introduces EliteLJ, a new open-source dataset for benchmarking time-series explanation methods in sports science, complete with expert annotations.
* Additionally, TimeSAE’s post-hoc, model-agnostic design makes it applicable to real-world black-box models (e.g., proprietary TimeGPT), increasing its practical impact.
*  The paper is very well-written, logically structured, and easy to follow.

### Weaknesses
* The paper successfully demonstrates that its learned concepts are useful for generating faithful explanations at a quantitative level. However, it is less clear how a human user would interpret these learned concepts. Section 3.4 mentions CARs and a functional ANOVA decomposition for interpretation, but these feel somewhat disconnected from the main method? 
* Hyperparameter Sensitivity and Complexity: The final objective function (Eq. 10) is a weighted sum of four different loss terms, involving at least three hyperparameters (η, α, λ). The paper does not discuss the strategy for tuning these hyperparameters or the sensitivity of the results to their values. Given the complexity of the framework, this is an important practical detail. A brief discussion in the appendix would be beneficial.
* Computational Cost: The proposed method involves training a non-trivial autoencoder architecture and includes a contrastive loss that requires generating counterfactuals. A brief comment on the computational overhead would add practical context for potential users.
* Lack of complexity and performance analysis. Is it possible to learn good representations by replacing the integrated TCN and SE modules as the skeleton in TimeSAE.

### Questions
* Can you provide a detailed introduction to the Lcc and Lcf motivation? It is currently unclear whether the true performance comes from them? (other losses are common in baselines)
* Could you elaborate on the hyperparameter tuning strategy for α and λ in Equation (10)? How robust is TimeSAE's performance to variations in these weights?
* How to present and evaluate causal explanations?

### Soundness
3

### Presentation
4

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
The paper proposes TimeSAE, a framework designed to explain black-box models for time series data by combining Sparse Autoencoders (SAEs) and principles of causality. A key feature of the proposed framework is its claimed robustness to distributional shifts. The authors conduct a series of evaluations on both synthetic and real-world time series datasets to demonstrate TimeSAE's performance against leading baseline methods.

### Strengths
The integration of causal concept effects with a Sparse Autoencoder framework is a novel and promising approach for generating explanations.

The proposed compositional consistency loss is a thoughtful addition to address the challenge of out-of-distribution generalization, enhancing the model's robustness.

The paper is supported by extensive experiments on both synthetic and real-world datasets, which effectively demonstrate the performance of the proposed method against relevant baselines.

### Weaknesses
1. **Interpretation of Theoretical Results**: While the inclusion of a theorem on order-faithfulness is appreciated, the paper would benefit from a more in-depth discussion of its practical implications and its significance within the broader literature on explainability.

2. **Rationale for Multiple Sparsity Mechanisms**: The paper employs multiple forms of sparsity (e.g., row-wise normalization, an SAE loss term, and Bernoulli masks). The motivation for including each of these specific mechanisms could be clarified to help the reader understand their individual contributions to the model's interpretability.

### Questions
**Main Concerns**: 

1.	**Definition of Faithfulness**: Faithfulness is defined based on the "predictive capability" of explanations E(x) to reflect the reasoning of the original model f(x). This central concept seems ambiguous. Could the authors provide a more formal or precise definition of "predictive capability" and clarify how it is quantitatively measured?
2.	**Significance of Theorem 1**: Regarding Theorem 1 on order-faithfulness, What is its theoretical and practical insights? For instance, what new understanding does this theorem provide, and how does it advance the literature on explainable AI for time series?
3.	**Concept Interactions**: In the formulation of concept interactions (Line 287), the model appears to consider only successive concepts. What is the rationale for this design choice and whether extending the model to handle more general, non-sequential combinations of concepts was considered?
4.	**Sparsity Function Definition**: Could you please provide an explicit mathematical definition for the sparsity function $s(·)$ used in Equation (2)?
5.	**Interaction Function Definition**: The function $g(c)$ for concept interaction (Line 287) is defined using $\psi_k $. Could the authors please clarify how this basis function $\psi_k $ is defined and implemented?
6.	**Role of Multiple Sparsity Types**: The paper introduces sparsity in several ways: via row-wise normalization (Line 171), within the SAE loss function (Line 188), and through Bernoulli sparsity masks (Line 292). Could the authors provide a more detailed rationale for this multi-faceted approach and explain the distinct role that each sparsity-inducing mechanism plays in the final explanation?

**Minor Concerns**: 

1.	Line 45: There appears to be a typo ("Explaning").

2.	Line 230: There is an inconsistency in the use of epsilon symbols ($\epsilon$ vs. $\varepsilon$). Please ensure consistent notation throughout the manuscript.

### Soundness
3

### Presentation
3

### Contribution
2
