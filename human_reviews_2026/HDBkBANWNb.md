# An Interpretable Contrastive GAN Approach for Identifying Heterogeneous Pathological Imaging Patterns

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4

## Abstract
Despite the rapid development of representation learning applied to neuroimaging, accurately disentangling the heterogeneity of neurological diseases remains a significant challenge. Typically, unsupervised approaches may capture disease heterogeneity that is dominated by confounding factors rather than pathological changes in brain structure or function. Existing semi-supervised methods can reveal disease-specific subtypes or dimensions by contrasting with a background population, but they usually rely on the assumption that non-pathological variations are identically distributed between background and target datasets—a condition often unmet in real-world data. To address this, we introduce InfoSepGAN, a contrastive generative framework designed to separate context (non-pathological) and attribute (pathological) factors between background and target datasets, reducing biases in learned disease-related representations when the assumption is violated. Furthermore, we regularize the learned imaging patterns for continuity, sparsity, and monotonicity, ensuring distinct and interpretable disease-related patterns along each dimension. Finally, InfoSepGAN employs a "synthetic twin" mechanism to perform subject-level counterfactual reconstruction, generating non-pathological counterparts for each patient and providing visualizations of disease-related regions. Experiments on both synthetic and real-world Alzheimer's disease datasets demonstrate that InfoSepGAN effectively extracts pathological imaging patterns while adjusting for potential confounders, outperforming recent baseline methods in both accuracy and interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors present a contrastive generative framework to identify, in a data-driven manner, disease characteristics and subtypes from brain MR images. The core idea is to separate imaging features into what is common to two populations (e.g., variability in brain size), termed the 'context', and what is specific to a patient cohort (e.g., reduced cortical thickness), termed 'attributes'. The principal contribution of this work is that this separation remains robust even when the context is unequally distributed between populations, such as controls and patients. This approach enables the identification of traits specific to a disease population and facilitates the reconstruction of pseudo-healthy features for patients to improve interpretability.

The proposed method builds upon a GAN framework, augmenting it with several key components. These include a mutual information-based regularisation, inspired by InfoGAN, to effectively disentangle context and attributes, alongside a reconstruction loss. To ensure the attributes reflect clinically plausible pathological processes, the authors introduce a suite of specialised regularisation terms. These enforce sparsity (as pathology is often localised), pattern separation (to reflect different patterns of variation), monotonicity (reflecting disease progression), background consistency (to prevent pathological changes in their absence), and decomposition (to mitigate mode collapse and encourage diverse, meaningful representations).

### Strengths
**Methods**: 
The primary methodological strength is the focus on robustness to confounding factors,  i.e. non-pathological effects that differ between the two populations studied, e.g. young controls and old patients. The paper builds a contrastive generative framework upon a GAN base, explicitly designed to separate 'context' from 'attributes' in a way that is stable even when context is unequally distributed between populations. The framework is augmented with a set of well-justified regularisation terms, which are carefully chosen to ensure the derived attributes reflect clinically plausible pathological processes.

**Experiments**:
A key strength is the evaluation strategy, which is designed to progressively validate the model under increasingly challenging conditions. The authors begin with controlled experiments on synthetic data, systematically testing the model's sensitivity to confounding severity, the proportion of affected subjects, and the spatial overlap between confounding and pathological effects. This is followed by an assessment on semi-synthetic data based on the UK Biobank, providing a more realistic testbed. Finally, the framework is applied to real clinical data from ADNI, demonstrating its utility on a genuine problem.

**Results**:
The results validate the core contribution of the method. They show that the proposed approach achieves performance on a par with SOTA techniques when no confounding effects are present. More importantly, the results demonstrate the key strength: in the presence of confounders, the method maintains robust performance, whereas the compared SOTA methods show a marked degradation. This directly addresses a significant challenge in the field.

### Weaknesses
- **Limited baselines**: The comparison could be more comprehensive. While SurrealGAN (2024) is a relevant SOTA baseline, the others are from 2015-2018. The paper would be stronger by including contemporary methods from the "contrastive analysis" family (Abid & Zou, 2019; Aglinskas et al., 2022; Carton et al., 2024; Louiset et al., 2024; Vowels et al., 2020) to better benchmark its contribution to robust pattern discovery under confounding factors.
- **Unclear real-data evaluation**: The results on the real ADNI dataset lack clarity. It is not well-explained how the findings in Figure 4 were obtained.
- **Hyperparameter sensitivity unexplored**: The model uses many loss terms with weights spanning a wide range (0.08 to 500). The paper does not discuss the sensitivity of the results to these choices or the difficulty of tuning them, which is a significant concern for reproducibility and practical application.

### Questions
- **Real-data analysis**: Could you please clarify the exact procedure for obtaining the voxel-wise group differences shown in Figure 4?
- **Method comparison**: To better position the contribution, could the performance be compared against other recent contrastive analysis frameworks?
- **Hyperparameter tuning**: How sensitive are the results to the specific combination of loss weights? Some discussion or analysis of the tuning process and the model's robustness to these choices would be helpful.
- **Monotonicity loss**:  What are the implications of the use of this loss for diseases with less clear progression?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors attempt to address unmodeled confounders in neuroimaging data by creating a generative model that tries to explicitly model confounders and disease-specific attributes as independent. The model is evaluated on a simulated, semi-simulated, and real neuroimaging dataset. The latter is an Alzheimer's disease dataset with many known confounders and disease progression metrics.

### Strengths
I think the authors make some interesting, and good, design choices, such as modeling disease severity as a spectrum. Moreover, it is encouraging that the authors actually find some significant correlations with biomarkers in the CSF in Tables 12 and 13. If the authors can show better generalization and how their design choices affect model performance, I believe this paper can be impactful.

### Weaknesses
Major weaknesses:
1) Lack of ablations. The authors a lot of changes to previous models that they do not ablate, especially the extra regularization terms in their loss function Equation 4 are added, but not ablated.
2) Limited evaluation of generalizability. The authors use two simulation datasets, which is a good start, but they only evaluate their model on a single real neuroimaging dataset. 
3) Unconvincing results. The authors only evaluate against one other GAN, and only on the semi-simulation data. To improve the work, the authors should include more baselines and include ablations for each of their proposed additions and regularization terms as mentioned in 1). Moreover, the authors do not compare their model to a GAN baseline at all on the only real dataset. This makes it hard to verify whether the model actually performs better on a realistic task. Moreover, the results show that age are captured in both the context and attribute embeddings, which means that the authors are not able to separate certain confounders. Similarly, the scanner type is still included in the attribute embedding space. It is potentially arguable that it is impossible to completely decouple age from Alzheimer's disease-specific attributes, but that is definitely not the case for scanner type. Moreover, the authors explicitly assume that the context and attribute embeddings are conditionally independent (L168-169). The model is clearly not learning conditionally independent embeddings, so the authors either need to change their design choices or improve their training objective to enforce the conditional independence.

Minor weaknesses/grammar/spelling errors:
- The authors spend almost 4/9 pages describing their model with mathematical equations, but the presentation, explanation, and intuition behind many of these choices is lacking. In general this section is quite hard to follow, and I am unsure why the authors decided to have the method section take up so much of the paper, especially because the authors leave little room to discuss their results. I would encourage the authors to keep the most important methodological advances, and mostly focus on explaining choices in the loss function. Many of the equations take up a lot of space, but are not novel theoretical results, specifically: equations 1, 2, and 3. Reading these subsections reminded me of the term 'mathiness' as coined by Lipton and Steinhardt [1]. Moreover, the simulation and baseline results are currently in the Appendix, the authors should move these results to the main text. 
- The authors also only use a very specific evaluation metric as their main evaluation metric, the pattern-c-index (PCI). I believe Figure 3 would require multiple evaluation metrics.
- Figure 2 is unclear. What do all the symbols mean? What is the difference between a dashed and solid line etc.?
- The text in Figure 3 is hard to read, also the x-axis of Figure 3a is unclear without reading the text. The authors should add an explanation of each term to the caption. Moreover, what is InfoSepGAN (w.) vs InfoSepGan (w.o.)?
- On L138-139 the authors claim: "While known confounders can be corrected using linear adjustments, it is impractical to account for all potential confounding factors, ...". After reading the paper, it is unclear to me how the authors guarantee that their method actually learns all of the confounding factors, especially given the issue of shortcut learning in medical AI [2].
- The authors should move their discussion on Lines 338-347 to the discussion section.
- L107: "..., these approaches remain limitations ..." -> remain limited
- L210/211: "Maximizing these two terms serves following purposes, ..." -> serves the following
- L255/256: "We adapts and extends the terms ..." -> We adapt and extend


[1] Lipton, Z. C., & Steinhardt, J. (2019). Troubling Trends in Machine Learning Scholarship: Some ML papers suffer from flaws that could mislead the public and stymie future research. Queue, 17(1), 45-77. \
[2] Brown, A., Tomasev, N., Freyberg, J., Liu, Y., Karthikesalingam, A., & Schrouff, J. (2023). Detecting shortcut learning for fair medical AI using shortcut testing. Nature communications, 14(1), 4314.

### Questions
- Some specific choices are unclear, for example the authors use a uniform spectrum for disorders (why a uniform and not, for example a normal distribution?). I understand it is easier in the sense that it is a bounded distribution, and control subjects can be assigned as 0, but is there a neurobiological reason for this design choice? Moreover, for many spectrum disorders it is also believed that controls may lie somewhere on the same spectrum as people diagnosed with a neurological or psychiatric disorder. The authors currently set the attribute factors for controls to 0, have the authors thought about using a different prior on the control subject attributes?

### Soundness
2

### Presentation
1

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
The manuscript proposes a contrastive generative adversarial model that disentangles pathological and non-pathological variability in neuroimaging data. The idea is to separate latent factors into shared and disease-specific components. To achieve that, the authors propose additional information-theoretic regularization and latent-space constraints. The proposed models are shown to improve over SurrealGAN and cPCA in extracting disease patterns.

### Strengths
- The experiments have been performed on synthetic and real datasets
- The proposed model have been compared with multiple baselines

### Weaknesses
- No ablation on hyperparameters for the loss function has been provided, and only mentioned that the parameters have been set to a specific value.
- Authors should support their findings with clinical literature when they say "consistent with characteristic AD pathology"
- Figure 4b is very small and is not readable if the paper is printed.
- No statistical comparison between models in Tables 4, 5, 6, 7, 8

### Questions
- Could the authors clarify how the findings derived from InfoSepGAN differ from conventional voxel-wise group difference analyses, and what unique insights the proposed framework provides beyond such standard methods?
- Since InfoSepGAN is a generative model, one would expect to observe progressive or continuous reconstructions as the attribute variable varies. However, the paper does not present examples of such latent traversals or severity-dependent generations. Could the authors provide visual or quantitative evidence that the learned attribute dimensions indeed capture disease severity in a monotonic and interpretable way?

### Soundness
3

### Presentation
2

### Contribution
3
