# Test-Time Adaptation without Source Data for Out-of-Domain Bioactivity Prediction

- Decision: Accept (Poster)
- Scores: 2, 2, 6, 6

## Abstract
Accurate prediction of protein-ligand bioactivity is a cornerstone of modern drug discovery, yet current deep learning methods often struggle with out-of-domain (OOD) generalization. The existing methods rely on access to source data, making them impractical in scenarios where data cannot be accessed due to confidentiality, privacy concerns or intellectual property restrictions. In this paper, we provide the first exploration of a more realistic setting for bioactivity prediction, where models are expected to adapt to out-of-domain distributions without access to source data. Motivated by the critical role of binding-relevant interactions in determining ligand-protein bioactivity, we introduce an uncertainty-weighted consistency strategy, in which original samples with high confidence guide their augmented counterparts by minimizing feature distance. This encourages the model to focus on informative interaction regions while suppressing reliance on spurious or non-causal substructures. To further enhance representation discriminability and prevent feature collapse, we integrate a contrastive optimization objective that pulls together augmented views of the same complex and pushes away views from different complexes. Together, these two components enable the learning of invariant, bioactivity-aware representations, allowing robust adaptation under distribution shifts. Extensive experiments across DTIGN, SIU 0.6, and DrugOOD demonstrate that our framework consistently outperforms state-of-the-art baselines under scaffold, protein, and assay based OOD settings. Especially on the eight subsets of DTIGN, it improves Pearson’s $R$ by 8.2\% and Kendall’s Tau $\tau$ by 5.8\% on average over the best baseline, underscoring its effectiveness as a source data-absent solution for OOD bioactivity prediction.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a method for test time adaptation for OOD protein-ligand binding prediction. The method consists in finetuning the encoder of a GNN architecture with two objectives: consistency (a transformation is applied to the input and the method learns to project augmented versions of the same input into the same embedding space, the loss is weighted by the internal certainty of the model, to minimize noise due to transformations that may deform the input) and contrastive learning (the method learns to project close together augmented views of the same input and further apart from augmented views of other inputs).

### Strengths
1. The method is interesting. Its application domain, when no source data exists apart from the features at inference time is highly relevant and captures realistic scenarios.
2. The evaluation datasets are relevant and baselines are comprehensive.

### Weaknesses
1. Title is misleading. I think that protein-ligand binding affinity is a more accurate description of the method rather than bioactivity. Some bioactivities like antibacterial, anticancer, allergenic, do not necessarily involve explicit protein-ligand interactions and this method is limited to modelling those types of interactions.
2. Results should contain a measure of the dispersion of the results across multiple runs, otherwise it is impossible to determine the significance of the differences.
3. Along the lines of 2, the contribution of the contrastive learning, or the consistency learning are difficult to determine. For example, from the baseline model (ERM) to TAB w/o cons there is a difference of 0.008 in RMSE, which does not seem significant, suggesting that the contrastive learning is not providing that much information. Similarly TAB w/o contr has a performance of 1.191 which is only 0.034 from the complete method. Looking at the results in Table 2, it seems like the standard deviation across datasets is higher than those improvements, indicating that there is not enough evidence to make any claims regarding the benefits of TAB.

### Questions
Is there any specific reason that I may be missing as to why the title focuses on bioactivity rather than binding affinity prediction?

### Soundness
1

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
5

### Summary
In the paper, the authors proposed methods for test-time adaptation for OOD ligand-target prediction under the assumption that  the accesses to source (training) data are prohibited. The proposed method called TAB employs data augmentation by randomly masking atoms in the binding graphs and forcing the consistency of the representation of  augmented data and the original data. Besides, contrastive learning was also proposed to enhance the representation discriminabilty and prevent feature collapse.

In the experiments, the authors studies the effectiveness of TAB against baseline approaches across DTIGN, SIU 0.6, and DrugOOD benchmark datasets. The results show that their proposed method outperforms baseline approaches with a significant margin.

### Strengths
The application domain is important, the problems the authors tackle is a challenging problem.

### Weaknesses
The assumption that source data is not available but the pretrained model is available limits the application of the work in practice. Accesses to the model especially the weights of the models usually also come with accesses to the training data as well. Making a strong assumption that is not realistic results in a small variation of  the problem  but limit its scope of application.

The methods proposed in the work such as data augmentation via randomly masking atoms, consistency and contrastive training objectives are all standard techniques.  Therefore the technical contribution of the works are limited.

Even the paper assume that there is no accesses to source data  but it also assume that there is an access to numerous test data used for augmentation and training consistency/contrastive learning.  To perform such adaptation  it requires numerous test data and the OOD setting becomes less practically interesting because in practice OOD test instances are rare.

Under OOD, the assumption that pockets are known is a strong assumption as for new target even pockets are not known in advance. The given assumption restrict the application of the work to known targets.

### Questions
Could you please analyse the effects of having very limitted test data say few-shot settings on the proposed methods?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper tackles the important problem of out-of-domain (OOD) generalization in bioactivity prediction under a source data-absent setting. The authors propose TAB, a test-time adaptation framework combining (1) uncertainty-weighted consistency learning and (2) contrastive optimization. Extensive experiments on DTIGN, SIU 0.6, and DrugOOD show consistent gains over a variety of state-of-the-art OOD generalization baselines.

### Strengths
1.The writing is clear and figures are well-designed, facilitating understanding.

2.The chosen application scenario is highly relevant and meaningful.

3.The experimental results are strong and demonstrate the effectiveness of the proposed method

### Weaknesses
1.Realistic source data-absent scenario. The proposed source-absent setting assumes that the training data are inaccessible. While this may occur in industrial contexts, for example when dealing with large-scale proprietary pre-trained models, the experiments in the paper mainly use a small DTIGN model trained on publicly available ChEMBL data. It is unclear whether TAB’s effectiveness in true source-absent scenarios can be reliably inferred. Evaluating TAB on larger pre-trained models would strengthen the paper.

2.Ablation studies are insufficient. The paper does not provide ablations for the update strategy or the newly proposed masking strategy. Including these analyses would clarify the contributions of each component.

3.Batch update and sequential sensitivity. TAB accumulates model updates using EMA and memory to reduce sensitivity. An ablation study examining the effects of batch size and test sample order would help assess robustness and stability.

4.Clarification on base model usage across datasets. It is unclear whether the same pre-trained DTIGN model is applied to all three datasets or if models are re-trained for each dataset. If the former, the authors should clarify potential overlaps or differences between training and test distributions, particularly for the latter two datasets.

5.Comparison baselines. The reported comparisons are all against source-available methods. It would be useful to include comparisons with source-absent baselines or other approaches designed for realistic deployment constraints.

### Questions
please respond to the weakness

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces TAB, a test-time adaptation framework to improve out-of-domain (OOD) bioactivity prediction in scenarios where the original source data is inaccessible. The proposed method uses an uncertainty-weighted consistency strategy and contrastive learning to adapt a pre-trained model, encouraging it to focus on important binding regions while ignoring spurious substructures. Experiments across multiple benchmarks demonstrate that this source-data-absent approach significantly outperforms existing methods in scaffold, protein, and assay-based OOD settings.

### Strengths
1. This work develops the first Test-Time Adaptation (TTA) method tailored to bioactivity prediction, presenting a novel and practically relevant problem formulation.

2. The proposed framework performs self-supervised optimization during testing without requiring complex architectural changes or extensive additional labeled data, making the overall approach simple and easy to follow.

3. The evaluation on three OOD benchmarks with different focuses covers various distribution shift scenarios, demonstrating strong generalizability and robustness.

### Weaknesses
1. The loss weights for consistency learning ($\alpha$) and contrastive learning ($\beta$) are fixed at 1.0 across all experiments. It would be helpful to include sensitivity analysis for these key hyperparameters to validate model robustness.

2. The existing ablation study (Table 5) validates the necessity of consistency learning and contrastive learning modules. Could authors further isolate uncertainty weighting for separate validation, as this appears to be an important contribution?

3. How is the pre-trained encoder in Fig. 2 or Section 3.2 obtained?

4. Minor:
(a) Typos in Equation 3;
(b) Suggest adding an analysis of TAB's efficiency and computational resource consumption.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
3
