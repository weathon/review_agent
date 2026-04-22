# $\text{BrainM}^3$: A Multi-Task Learning Framework Based on A Multi-Level Mixture- of-Experts for Cross-Disease and Cross-Domain Dementia Diagnosis

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Accurate differential diagnosis of dementia subtypes is crucial due to their distinct clinical trajectories and treatment responses. However, rare subtypes such as Lewy Body Dementia (LBD) suffer from data scarcity, and domain shifts across institutions further hinder model generalization. To address these challenges, we propose $\text{BrainM}^3$, a Multi-task learning framework based on a Multi-level Mixture-of-Experts (MoE) architecture for cross-domain and cross-disease Brain modeling. Our model jointly learns Alzheimer’s disease (AD), mild cognitive impairment (MCI), and LBD diagnosis by disentangling disease-shared and specific brain connectivity features. At the domain level, a domain-aware Soft-MoE combined with adversarial training captures domain-invariant foundation brain representations, effectively mitigating scanner and cohort variability. At the task level, task-shared and task-specific Soft-MoEs enable mutual knowledge transfer and facilitate fine-grained pathological feature modeling. Experiments on multi-institutional datasets demonstrate that $\text{BrainM}^3$ consistently outperforms baselines under data heterogeneity. Moreover, our model offers interpretable insights into disease-relevant brain networks, offering potential clinical utility. Our work highlights the promise of an accurate and interpretable model for robust dementia diagnosis in real-world, cross-institution settings. Our code will be published based on acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a model called BrainM3, designed to simultaneously handle the diagnosis of AD, MCI, and LBD, while also accounting for data variability across hospitals and scanners. The model's core design is a hierarchical MoE architecture. Domain-level MoEs first learn universal features that are robust across institutions. Task-shared and task-specific MoEs then use them to enable data-rich tasks like AD to benefit data-limited tasks like LBD. The model also utilizes adversarial training to mitigate domain shift. Experiments show that the multi-task version of BrainM3 performs significantly better than single-task versions, particularly in scenarios with limited data. The authors also analyze the brain regions targeted by the model, which are consistent with established medical literature, demonstrating a degree of interpretability.

### Strengths
The paper's starting point is very realistic. Rather than simply playing with the model in the “clean world” of ADNI, it truly considers the common problem of “model obsolescence if the hospital changes.” The model's structural design is logical: domain --> shared task --> task-specific. This layered arrangement is well-reasoned and not simply a stacking of modules. In experimental comparisons, multi-task performance significantly outperformed single-task performance, especially on the LBD task, demonstrating that the model truly transfers knowledge, not simply sharing parameters. The authors also specifically analyze the brain regions activated by the model. This interpretability is a plus in the medical field and is much more impressive than pure accuracy.

### Weaknesses
1. Although the paper title emphasizes “heterogeneous-feature multi-task learning,” the three tasks actually share the same input format (the structural connection matrix), but the inputs come from different sources. Strictly speaking, these are not considered truly “heterogeneous features.”

2. The baselines used for model comparison are almost all single-task learning, with no comparison to simpler multi-task shared models (e.g., a shared encoder connected directly to three heads). This makes it difficult to determine whether MoE is truly effective, or whether simply training with a shared backbone will suffice. Furthermore, the baseline methods are significantly outdated.

3. The private dataset is relatively small, and the current experiments are more like "domain-aware fine-tuning," so the conclusions about "cross-domain generalization" still have room for improvement.

4. While the model architecture incorporates MoE, the paper does not demonstrate its gating behavior, such as whether different experts truly specialize in different tasks or data sources. Without visualization, one might suspect that MoE simply adds parameters rather than creating a division of labor.

### Questions
1. Have simpler multi-task baselines been developed? For example, using only a shared encoder and task head, without a MoE, how much difference is there in performance? If the performance is similar, then the value of the MoE needs to be redefined.

2. Can cross-domain capabilities be tested more aggressively? For example, if the model's performance is tested directly without training on private data, what will happen? This would further validate the domain-invariant claim.

3. Can you show the activation maps or routing weights of the experts? We could see if some experts clearly favor LBD or ADNI data, which would make the role of the MoE more intuitive.

4. Can the term “heterogeneous features” be more precise? Currently, the input structures of the three tasks are not truly different. In the future, are you considering adding fMRI or PET to achieve truly multimodal heterogeneous input?

### Soundness
2

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
4

### Summary
This paper proposes BrainM³, a multi-task learning framework based on a multi-level Mixture-of-Experts (MoE) architecture for dementia subtype diagnosis under real-world challenges: data scarcity and cross-institutional domain shift. The method jointly models Alzheimer’s disease (AD), mild cognitive impairment (MCI), and Lewy body dementia (LBD) by disentangling domain-invariant foundational representations and task-shared/specific features. The authors validate their approach on a public dataset (ADNI) and an in-house private dataset, demonstrating consistent performance gains over strong baselines and providing interpretable insights into disease-relevant brain networks.

### Strengths
1.The use of Soft-MoE with attention-based pooling enables flexible, region-aware routing, which is particularly suitable for brain connectivity data.

2.The gradient reversal layer (GRL) combined with domain-shared MoE enables model to learn domain-invariant features without sacrificing task discriminability.

### Weaknesses
1. The ANDI dataset used in this study includes only Alzheimer's disease–related tasks, indicating that the multi-task experiments are conducted within the same domain rather than across different domains.

2. The design of task-specific and task-shared MoE architectures is not new; it has been extensively explored in prior studies such as [A, B, C].

  [A]Dynamic modeling of patients, modalities and tasks via multi-modal multi-task mixture of experts. ICLR 2025.

  [B]TaskExpert: Dynamically Assembling Multi-Task Representations with Memorial Mixture-of-Experts. ICCV 2023.

  [C]Exploring Text-enhanced Mixture-of-Experts for Semi-supervised Medical Image Segmentation with Composite Data. MICCAI 2025.

3. The comparison is insufficient and somewhat outdated, as the most recent competing method cited was published in 2022.

4. The discussion is insufficient. There are alternative approaches to achieving domain-invariant representation learning, with one of the most classical being the Fourier-based transformation, which has been widely applied in domain generalization tasks. The authors are encouraged to discuss or compare their method with such alternatives.

5. The interpretability section still requires further refinement, as higher performance metrics naturally lead to more precise sub-network discriminability. However, this alone does not demonstrate the interpretability of the proposed method.

6. Section 2.1 introduces heterogeneous-feature multi-task learning (MTL), yet the remainder of the paper never clarifies how “heterogeneous” MTL differs from “homogeneous” MTL in terms of generalization bounds or optimization difficulty, nor does it explain how the proposed multi-level MoE mitigates the effects induced by these differences.

### Questions
1. The overall contribution appears incremental. Please clarify which specific models are incorporated within the proposed MoE.

2. There are many alternative designs for MoE architectures, such as pMoE (Patch-level Routing in Mixture-of-Experts is Provably Sample-efficient for Convolutional Neural Networks). Please consider discussing or comparing your approach with these alternatives.

3. The current comparisons are not comprehensive enough and could benefit from including more recent or diverse baselines.

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
4

### Summary
The paper introduces BrainM3, a multi-task learning (MTL) framework built upon a multi-level mixture-of-experts (MoE) architecture for cross-domain and cross-disease dementia diagnosis. It aims to address two core challenges: Data scarcity in rare dementia subtypes like Lewy Body Dementia (LBD). Domain heterogeneity arising from different imaging sites and scanners. BrainM3 integrates three levels of feature specialization: Domain-shared Soft-MoE for domain-invariant foundational representations. Task-shared Soft-MoE for common pathological representations. Task-specific Soft-MoE for disease-unique characteristics. The framework also uses adversarial training via a Gradient Reversal Layer to enforce domain invariance.

### Strengths
1. Novel hierarchical architecture: The combination of domain-level, task-level, and expert-level specialization is elegant and technically coherent. Incorporation of soft MoE routing avoids the instability of sparse gating and enhances flexibility.
2. Addresses practical challenges: Cross-domain heterogeneity and data scarcity are real-world problems in neuroimaging, and the proposed multi-level structure effectively models them.
3, Comprehensive evaluation: Comparison against diverse baselines (traditional ML, CNN, GNN, Transformer). Includes ablation studies on expert number and domain-adversarial loss, confirming design choices.

### Weaknesses
1. Limited dataset diversity: Only two datasets are used. The cross-domain setting, though well motivated, remains narrow in scope. Absence of external validation or additional domains limits claims of “generalization”.
2. Potential overfitting to small datasets: The private dataset includes only 147 subjects; despite MoE and regularization, results may reflect overfitting rather than robust generalization.
3. Interpretability remains qualitative: While visualizations are compelling, quantitative interpretability metrics like stability of identified regions and overlap with known biomarkers are not reported.

### Questions
1. Scalability: How would BrainM3 scale to more than two diagnostic tasks or more than two data domains?
2. Generalization and robustness: Did you test model on any unseen external dataset to confirm cross-site generalization beyond ADNI and your private dataset?

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
4

### Summary
The paper proposes BrainM3, a multi-task learning (MTL) framework that leverages a hierarchical Soft Mixture-of-Experts (MoE) architecture to address dementia diagnosis across diseases (Alzheimer's disease [AD], mild cognitive impairment [MCI], and Lewy Body Dementia [LBD]) and domains (different institutions with heterogeneous data). The model utilizes domain-shared MoE with adversarial training to learn domain-invariant representations of brain structural connectivity (SC) from diffusion tensor imaging (DTI). It then employs task-shared and task-specific MoEs to capture common and unique pathological patterns, enabling knowledge transfer from data-rich (AD/MCI) to data-scarce (LBD) tasks. Experiments on ADNI and an in-house dataset claim superior performance over baselines, with interpretability insights into brain networks.

### Strengths
- The work tackles a timely and clinically significant challenge: differential dementia diagnosis under data scarcity (e.g., for LBD) and domain shifts (e.g., scanner variability). This aligns with real-world medical imaging scenarios, where datasets are often heterogeneous and imbalanced. The focus on interpretability (e.g., via expert routing and attention pooling) could provide value beyond accuracy, such as identifying disease-relevant brain sub-networks. 
- The hierarchical MoE design is an acceptable contribution to MTL in heterogeneous settings. Combining domain-shared MoE with adversarial training (via gradient reversal layer) for invariance, followed by task-shared/specific MoEs with residual fusion, is a smart way to disentangle features. This extends Soft-MoE (Puigcerver et al., 2023) to medical imaging, potentially advancing domain generalization in neuroimaging. The brain sub-network tokenization (based on Destrieux Atlas) preserves anatomical structure, making the model interpretable and biologically grounded. 
- The use of multi-institutional datasets (ADNI for data-rich AD/MCI classification, in-house for data-scarce NC/AD/LBD) simulates cross-domain challenges. Comparing against diverse baselines (SVM, XGBoost, CNN/GNNs like BrainNetCNN, and Transformers like BrainNetTF) is appropriate, and the single-task ablation of BrainM3 enables a fair assessment of MTL benefits.

### Weaknesses
- The 80/20 train/test split is mentioned, but no cross-validation (e.g., 5-fold) is described, which is critical for small datasets (ADNI: 418 subjects; in-house: 147), which raises concerns about overfitting.
- Ablation studies (e.g., removing domain adversarial or task-shared MoE) are not done with the required details: does domain invariance truly mitigate shifts, or is it marginal? The hyperparameter sensitivity is not addressed (e.g., λ, number of experts K).
- Using SC from DTI is justified, but why not utilize multimodal input (e.g., fMRI + DTI) for richer features? The tokenization (148 ROIs as sub-networks) assumes symmetry in the matrix, but no handling of noise/artifacts in fiber tracking is discussed—e.g., how robust is it to preprocessing variations?
- Gating is softmax-based, but no load balancing loss (common in MoE to prevent expert collapse) is mentioned. With small datasets, experts might under-specialize; evidence suggests that the embedding dim D=64 and expert counts (8 domains, 4 shared/specific) seem arbitrary (the authors should provide a proper ablation needed).
- λ=1 is fixed in adversarial training; how sensitive is it? The domain classifier is simple (MLP), but no architecture details are provided. Does it overfit on small batches?
- The manuscript provides tests on only two datasets/domains. No external validation (e.g., UK Biobank or other dementia cohorts) is presented. 
- The interpretability claims (e.g., "insights into disease-relevant networks") are vague without examples, for example, which ROIs are routed to which experts for LBD vs. AD?
- The paper acknowledges data scarcity but downplays ethical issues. 
- No mention of computational complexity or accessibility for low-resource labs.

### Questions
- What specific strategies were used to address class imbalance during training?
- If oversampling was applied, was it performed at the subject level or within mini-batches? Was it combined with data augmentation (e.g., random fiber tract perturbations or graph perturbations)? 
- Were these strategies applied uniformly across both tasks, or tailored per task/dataset? 
- Please report per-class performance metrics (precision, recall, F1-score) in addition to overall accuracy/AUC to demonstrate that minority class performance is not sacrificed.
- The paper claims that BrainM3 offers “interpretable insights into disease-relevant brain networks” and that expert routing enables “fine-grained pathological feature modeling.” However, no visualizations or quantitative analyses of gating behavior, attention maps, or expert specialization are provided in the main paper.
Please provide:
- Gating weight heatmaps or t-SNE/UMAP projections of expert assignments per brain sub-network (ROI) across NC, MCI, AD, and LBD subjects. Do certain experts consistently activate for specific ROIs?
- Attention pooling visualizations from Eq. (9): which brain regions receive high attention in task-specific vs. task-shared pathways? Are these anatomically plausible?
- Expert specialization analysis: for each task-specific expert, compute the average activation (gating weight) across disease groups. Is there statistically significant specialization (e.g., via ANOVA or permutation tests)?
- A case study showing how routing differs between a correctly vs. incorrectly classified LBD subject.
- Comparison with Grad-CAM or integrated gradients on baseline models (e.g., BrainNetTF) to quantify whether MoE improves the localization of known pathology.
- Hyperparameter search details, such as grid search for K, λ?
Several critical hyperparameters are fixed without justification:
- Domain-shared experts: 8
- Task-shared experts: 4
- Task-specific experts: 4 per task
- Domain adversarial weight: λ = 1
- Embedding dimension: D = 64
- No load-balancing loss for MoE
Please report the search space and method used for hyperparameter tuning.

### Soundness
2

### Presentation
3

### Contribution
2
