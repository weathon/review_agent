# Revisiting Subsampling and Mixup for WSI Classification: A Slot-Attention-Based Approach

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 3

## Abstract
Whole slide image (WSI) classification requires repetitive zoom-in and out for pathologists, as only small portions of the slide may be relevant to detecting cancer. Due to the lack of patch-level labels, multiple instance learning (MIL) is common practice for training a WSI classifier. One of the challenges in MIL for WSI is the weak supervision coming only from the slide-level labels, often resulting in severe overfitting. In response, researchers have considered adapting patch-level augmentation or applying mixup augmentation, but their applicability remains unverified. Our approach augments the training dataset by sampling a subset of patches in the WSI without significantly altering the underlying semantics of the original slides. Additionally, we introduce an efficient model (Slot-MIL) that organizes patches into a fixed number of slots, the abstract representation of patches, using an attention mechanism. We empirically demonstrate that the subsampling augmentation helps to make more informative slots by restricting the over-concentration of attention and to improve interpretability. Finally, we illustrate that combining our attention-based aggregation model with subsampling and mixup, which has shown limited compatibility in existing MIL methods, can enhance both generalization and calibration. Our proposed methods achieve the state-of-the-art performance across various benchmark datasets including class imbalance and distribution shifts.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes an original architecture for WSI classification which combine MIL (multi-instance learning) with multi-head attention, slot attention and pooling. The model can be seen as summarizing the WSI (split into M patches) into S slots where S is a fixed hyperparameter and is small (e.g. 16). Another model of similar architecture classifies these S slots into K classes. The M patches are first converted to codes using a ResNet-18 pre-trained on Imagenet. The paper also proposes a data augmentation scheme based on patch subsampling and MIXUP. Experiments show that the proposed architecture and data augmentation improves upon SOTA for 3 datasets (CAMELYON-16/17 and TCGA-NSCLC) for cancer/non-cancer or subtype classification. Ablation studies show the effect hyperparameter selection and combination of data augmentation.

### Strengths
* The paper proposes a novel architecture for WSI classification that improves upon SOTA MIL. The architecure choices are well motivated and the results show a clear improvement. The idea of using a fixed number of attention-guided slots to summarize the important patches of a WSI prior to classification is original and powerful.
* A typical drawback of MIL is that it tend to overtrain as the number of bags is generally small compared to the number of instances (a single WSI may generate several thousand patches). The proposed approach of subsampling and MIXUP augmentation appears to be effective at decreasing overtraining issues and improving classification measures. Again those approaches are relatively well motivated in the paper.
* The choice of using ResNet-18 pre-trained on imagenet makes the approach simpler, faster and more reproducible.

### Weaknesses
* The SOTA AUC for all 3 datasets, as reported on the current litterature and Grand-Challenge website is significantly higher than the baselines chosen in the paper. For TCGA-NSCLC, [Zhang22] reports 0.9377 AUC, while the top baseline in the paper is 0.893. For the CAMELYON datasets, the Grand Challenge leaderboard also outperforms the reported baselines. Furthermore, Transmil [2] reports 0.9309 AUC for CAMELYON-16, while the paper reports it at 0.834 ! And for TCGA-NSCLC, the discrepency is: 0.893 vs 0.9603 ! Those are significant differences, that make the proposed approach not SOTA anymore. [update: this issue has been cleared]
[1] Zhang, Jingwei, et al. "Gigapixel whole-slide images classification using locally supervised learning." MICCAI 2022.
[2] Shao, Zhuchen, et al. "Transmil: Transformer based correlated multiple instance learning for whole slide image classification." NIPS 2021.

* Different subsampling rates are applied to the 3 datasets, based on information that is not generally known a-priori (the percentage of positive patches). This unfairly inflates the reported performance.

* It is not clear how the optimal hyperparameters are obtained. At least the subsampling rate seems to be obtained heuristically (see bullet above), so it makes me suspicous about the others too. [update: this issue has been addressed by the authors in the rebuttal]

* Since, one of the claim the authors make repeatedly is that their approach is more computationally efficient, it would be good to include some numbers and compare them to other approaches. [update: this issue has been addressed by the authors in the rebuttal]

Overall, this is a technically solid paper presenting a novel approach to WSI classification using MIL sampling and augmentation. Unfortunately, despite its claims, it doesn't reach SOTA on the reported datasets.

### Questions
see weaknesses

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose an efficient model called SlotMIL that leverages an attention-based mechanism to organize patches into a fixed number of slots. They demonstrate that combining the attention-based aggregation model with subsampling and mixup augmentation techniques enhances both generalization and calibration in WSI classification. 

A key contribution is subsampling/ mixup augmentation, which creates new bags of patches by randomly sampling subsets from the original slides. This helps restrict overfitting to the weak slide-level supervision. They also introduce an efficient model called SlotMIL that summarizes patches into a fixed number of slots using attention. Experiments show subsampling helps make more informative slots and improves generalization.

### Strengths
- The subsampling augmentation is a simple but effective way to create new training bags that reduces overfitting, without altering underlying slide semantics or adding training cost. This is an improvement over complex augmentation techniques.
- The SlotMIL model provides an efficient attention-based aggregation method to summarize patches into discriminative slots. This is more sophisticated than relying only on max pooling approaches commonly used.
- The authors showed that subsampling plus mixup augmentation can work well together, whereas prior work found mixup had limited applicability in MIL frameworks. 
- Thorough experiments on multiple datasets demonstrate state-of-the-art performance, including on class imbalance and distribution shifts.
- Authors considered various relevant baselines (ABMIL, DSMIL) and conducted rigorous ablation experiments to test the various components of the proposed model. 
- The paper is well written and easily to follow.

### Weaknesses
1. MIL attention (https://arxiv.org/pdf/1802.04712.pdf) has been widely used for WSI analysis and several other extensions to the method have been explored in the field. While it is a relevant baseline, the proposed improvements do not significantly improve upon performance (<2-5%) and it is unclear if Mixup augmentation alone is better than other extensions to improve performance in the proposed binary classification task. 
2. The paper focuses solely on binary classification problems. Extending the approach to multi-class classification could be challenging. 
3. In some pathology samples (e.g., in cancer), very small proportion of patches might contain the relevant signal for classification. The subsampling augmentation could potentially discard useful patch information. Strategies that retain all patches may be able to learn more robust features.

### Questions
- What impact does the subsampling rate have on model performance? Is there an optimal sampling fraction or range across datasets?
- How well does the SlotMIL model and subsampling augmentation transfer to multi-class classification tasks? 
The codebase is not public - making it opensource would help the reproducibility efforts.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The papers proposes Slot-MIL, which incorporates the ideas related to inducing points and slot attention for simplifying pooling mechanisms in multiple instance learning for WSIs. Specifically, given two WSIs (bag of instance features), the model should be able to encode them into the identical number of slots for classification. This work assesses performance on CAMELYON and TCGA-NSCLC subtyping, with additional ablation studies performed related to impact of subsampling and number of slots.

### Strengths
- This work performs a comprehensive assessment into different subsampling approaches for WSIs, and how subsampling strategies perform as data augmentation methods for MIL. In addition to performing substantive ablation experiments for the main augmentation method, SubMix (different parameters and combinations of subsampling and Slot-Mixup), this work also thoroughly assesses ablation strategies in combination with other strategies such as RankMix. Additional figures and presentations in the supplement also convey the stability  of how subsampling affects training and validation loss.

### Weaknesses
- Though adapting new techniques such as slot attention, I found this method to have limited novelty as it addresses common concerns such as patch redundancy in MIL. Many works such as DeepAttnMISL (Yao et al. 2020) achieve similar goals as Slot-MIL in filtering the bag to a smaller set of patches. Overall, relative to the performance improvement demonstrated, the contributions presented by the method may still be too limited and lack extensive validation with diverse downstream tasks.
- One of the outlined contributions of this work (#3) is that Slot-MIL reaches state-of-the-art performance on CAMELYON and TCGA-NSCLC. Slot-MIL outperforms baselines relative to the comparisons developed in this work. However, when compared across studies, the reported best performance for C16 on the test set underperforms other reported results by a large margin. For example, on C16, whereas the accuracy / AUC for Slot-MIL+SubMix is 0.890 / 0.921, the reported best performances for ILRA-MIL (FRC) in Xiang et al. 2023 is 0.922 / 0.965. In other works such as MHIM-MIL (DSMIL) by Tang et al. 2023, the reported best performances is 0.925 / 0.965 (evaluated using cross-validation, not on official C16 test), and Bayes-MIL-APCRF by Cui et al. 2023 have best performances of 0.900 / 0.948. Though not using the same splits for TCGA-NSCLC, the AUCs for this task using 10-fold CV is generally 0.930+ (0.977 in Xiang et al. 2023).
- Benchmarks such as C16 and TCGA-NSCLC lack difficulty and can be easily solved without adapting techniques related to WSI augmentation and slot attention. It would be interesting to explore this method on more diverse tasks that would benefit from data augmentation and "sparsity", such as gene mutation prediction (such as MSI prediction in TCGA-COADREAD), survival analysis, and other challenging tasks such as Gleason score grading in PANDA. The tasks evaluated in this work are limited to diagnostically-simple binary classification problems that do not need sparse MIL or virtual augmentation methods to see clinical translation.
- - Additionally, tasks such as C16, TCGA-NSCLC, TCGA-RCC have been over-explored in computational pathology and should no longer be evaluated as the only tasks evaluated for MIL in the reviewer's opinion. C16 already has many state-of-the art performances and is nearly solved from both fully-supervised and weakly-supervised perspectives. Similarly, TCGA-NSCLC and TCGA-RCC can be generally solved without requiring sophisticated MIL approaches. Overall, it would be more interesting to demonstrate how this method would enable more challenging tasks to be solved in computational pathology.

1. Xiang, J. and Zhang, J., 2022, September. Exploring low-rank property in multiple instance learning for whole slide image classification. In The Eleventh International Conference on Learning Representations.
2. Yufei, C., Liu, Z., Liu, X., Liu, X., Wang, C., Kuo, T.W., Xue, C.J. and Chan, A.B., 2022, September. Bayes-MIL: A New Probabilistic Perspective on Attention-based Multiple Instance Learning for Whole Slide Images. In The Eleventh International Conference on Learning Representations.
3. Tang, W., Huang, S., Zhang, X., Zhou, F., Zhang, Y. and Liu, B., 2023. Multiple Instance Learning Framework with Masked Hard Instance Mining for Whole Slide Image Classification. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 4078-4087).
4. Yao, J., Zhu, X., Jonnagaddala, J., Hawkins, N. and Huang, J., 2020. Whole slide images based cancer survival prediction using attention guided deep multiple instance learning networks. Medical Image Analysis, 65, p.101789.

### Questions
See above comments.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
