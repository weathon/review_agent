# Sequential Information Bottleneck Fusion: Towards Robust and Generalizable Multi-Modal Brain Tumor Segmentation

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Brain tumor segmentation in multi-modal MRIs poses significant challenges when one or more modalities are missing. Recent approaches commonly employ parallel fusion strategies; however, these methods often risk losing crucial shared information across modalities, which can degrade segmentation performance. In this paper, we advocate leveraging sequential information bottleneck fusion to effectively preserve shared information across modalities. From an information-theoretic perspective, sequential fusion not only produces more robust fused representations in missing-data scenarios but also achieves a tighter generalization upper bound compared to parallel fusion approaches. Building on this principle, we propose the Sequential Multi-modal Segmentation Network (SMSN), which integrates an Information-Bottleneck Fusion Module (IBFM). The IBFM sequentially extracts modality-common features while reconstructing modality-specific features through a dedicated feature extraction module. Extensive experiments on the BRATS18 and BRATS20 glioma datasets demonstrate that SMSN consistently outperforms traditional parallel fusion-based baselines, achieving exceptional robustness in diverse missing-modality settings. Furthermore, SMSN exhibits superior cross-domain generalization, as evidenced by its ability to transfer a trained model from BRATS20 to a brain metastasis dataset without fine-tuning. To ensure reproducibility, the code of the SMSN is provided in the supplementary file.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
To address the problem of missing modality segmentation in multimodal brain tumor MRI, this paper proposes the Sequential Multi-modal Segmentation Network (SMSN). Its core is information bottleneck-based sequential fusion (IBFM), which gradually aggregates modalities to extract modality commonalities while preserving modality specificity. The authors theoretically demonstrate that this sequential fusion approach has a tighter generalization upper bound and is more robust to missing modalities than parallel fusion. The paper also implements SOTA on BRATS18/20.

### Strengths
1. Compared to most existing unified/parallel fusion methods (such as mmFormer, M2FTrans, MMMViT, and IMS2Trans), this paper's approach is highly innovative.
2. The paper's theoretical proofs are rigorous and contribute significantly to the field of modal fusion.
3. The paper's experimental design is comprehensive, with detailed reporting of various scenarios.
4. The paper's visualizations compare existing methods and strongly demonstrate the model's effectiveness.

### Weaknesses
1. The article provides very little explanation of the specific implementation method. Is the modal fusion order designed in the article fixed or random? Figure 3 demonstrates a sequential fusion method for all modalities, but does not include examples of cases where one, two, or three modalities are missing.
2. The article's core innovation builds on the previous work on ITHP, which proposed the core concept of sequential fusion. This paper implements SOTA on a medical dataset and further provides theoretical proof.
3. Compared to most existing unified/parallel fusion methods, does this method consume more computational resources? This article suggests addressing model overhead (number of parameters, FLOPs, throughput, video memory, and training time) to facilitate a fair comparison of lightweight solutions.

### Questions
1. Is the modal fusion order designed in the article fixed or random? Figure 3 shows a sequential fusion method for all modalities, but there are no examples of cases where one, two, or three modalities are missing.
2. Does a different modal order significantly affect model performance? Most existing unified/parallel fusion methods are not affected by modal order and appear more stable.
3. Do the parameter settings for β and γ change depending on the missing modalities, or are they fixed?

### Soundness
3

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
This paper addresses the challenge of brain tumor segmentation in multi-modal MRI when some imaging modalities are missing. Unlike traditional parallel fusion methods that risk losing shared inter-modal information, the authors propose a Sequential Multi-modal Segmentation Network (SMSN) based on an Information-Bottleneck Fusion Module (IBFM). By sequentially fusing modalities, the approach preserves shared information and reconstructs modality-specific features, leading to more robust and generalizable representations. Experiments on the BRATS18 and BRATS20 glioma datasets show that SMSN outperforms parallel fusion baselines, maintaining strong performance even with missing modalities and transferring effectively to a brain metastasis dataset without fine-tuning.

### Strengths
The sequential fusion design preserves shared information, maintaining high segmentation accuracy even when some MRI modalities are absent.
It achieves improved cross-domain performance, successfully transferring to new datasets without fine-tuning.
The method is supported by information-theoretic analysis, ensuring more efficient and reliable feature fusion than parallel approaches.

### Weaknesses
It is confusing for this missing-modality task, as the authors use the BraTS series datasets, which already contain complete modalities. However, in real scenarios, if some modalities are missing, it becomes difficult to obtain accurate segmentation labels. How do the authors explain this situation?

Moreover, the datasets used in this paper are somewhat outdated, which may affect the evaluation of the proposed method.

Besides segmentation, I think other related tasks should also be tested, for example the classification task.

### Questions
N/A

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
This paper proposes the Sequential Multi-modal Segmentation Network (SMSN). The shared information across modalities is gradually extracted via a two-stage Sequential Information Bottleneck Fusion approach, while modality-specific information is separated using a Transformer-based module and orthogonal loss. This design enhances the robustness and cross-domain generalization of segmentation in modality-deficient scenarios. The paper provides theoretical derivations to demonstrate the proposed method’s advantages, and these derivations include a tighter generalization upper bound and Lipschitz boundary comparison. It also conducts extensive modality-deficient experiments on the BRATS18 and BRATS20 datasets. Furthermore, to verify cross-domain generalization, the model trained on BRATS20 is directly transferred to a brain metastasis dataset. Experimental results show that SMSN outperforms multiple parallel fusion baselines in terms of average Dice score and performance under modality absence.

### Strengths
* The advantages of sequential information bottleneck fusion in terms of the Lipschitz upper bound and the mutual information-based generalization bound are proposed and proven, forming a complete theory-proposition-proof chain that is rigorous and reliable.
* Comparisons with multiple parallel fusion and non-fusion baselines were conducted on BRATS18 and BRATS20 datasets, encompassing scenarios with different modal absences, cross-domain tests transferred to brain metastasis datasets, as well as ablation and hyperparameter sensitivity analyses. These experiments are comprehensive and thorough.
* The paper is accompanied by source code and supplementary materials. The method section provides detailed modular implementation details, including two-stage information bottleneck fusion, modal reordering, reconstruction loss and orthogonal loss, to facilitate reproducibility.

### Weaknesses
* While the paper's primary motivation is to address the issue of missing modalities and claims that the sequential information bottleneck framework exhibits greater stability under such conditions, it lacks direct theoretical validation or quantitative analysis to demonstrate that the information bottleneck objective remains valid under the distribution of missing modality scenarios.
* The paper fails to discuss the computational overhead of SMSN, such as the number of model parameters, training or inference time, and complexity in comparison to baseline methods. Sequential fusion, additional reconstruction processes and orthogonality losses may introduce substantial computational costs, which is an important practical consideration for resource-constrained real-world environments.
* In Section 4, this paper introduces a modal reordering strategy, motivated by the need to avoid initiating sequential fusion with a missing modality represented by a zero tensor, thereby preventing degradation of the information bottleneck objective function. While this design is logically sound, the paper lacks sufficient empirical support and provides no comparative experiments like fixed-order fusion versus random reordering.

### Questions
1. In Proposition 2, the authors prove that the sequential information bottleneck model can ensure a tighter upper bound on generalization errors under the condition that the encoder, fusion module, and decoder are all 1-Lipschitz continuous. However, this condition faces challenges in the actual implementation of deep networks. Specifically, when using Transformers, self-attention with softmax often results in a Lipschitz constant greater than 1 during gradient propagation, and other modules also tend to amplify gradients. As a result, the network does not naturally satisfy Lipschitz continuity. How should this issue be addressed?
2. The derivation of sequential information bottleneck relies on the assumption of conditional independence between modalities. Have the authors theoretically proven or empirically verified the approximate validity of this assumption? If there is strong correlation between modalities, are there corresponding mitigation strategies?

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
2

### Summary
This paper proposes a Sequential Multi-modal Segmentation Network (SMSN), a novel method for brain tumor segmentation in the context of missing-modality MRI settings. The key idea is to perform sequential fusion using an Information Bottleneck Fusion Module (IBFM), instead of standard parallel fusion. The authors provide theoretical justification (generalization bound, Lipschitz robustness), and empirically validate SMSN on BRATS18/20 and a metastasis dataset. Results show improved robustness and cross-domain generalization.

### Strengths
1. The paper proposed a novel sequential IB-based fusion with two-stage IB and modality reordering, which is a meaningful contribution.
2. The paper provides generalization and Lipschitz analysis and connects it to empirical behavior.
3. Evaluation on multiple MRI datasets, missing-modality scenarios, and cross-dataset generalization demonstrated the improved performance.

### Weaknesses
See the questions part.

### Questions
1. How sensitive is performance to the chosen bottleneck size and sequential order?
2. Is there a trade-off compared to pure parallel fusion?
3. Can the proposed method scale to more than 4 modalities or other domains, such as CT + MRI?
4. What is the computational overhead of sequential IB vs. attention-based fusion?

### Soundness
3

### Presentation
3

### Contribution
3
