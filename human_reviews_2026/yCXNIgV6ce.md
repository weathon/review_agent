# Projection-Enhanced Contrastive Learning and Linear Calibration for Exemplar-Free Class-Incremental Learning

- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
Exemplar-Free Class-Incremental Learning (EFCIL) tackles the challenge of learning to discriminate between new and old classes without retaining any past exemplars. Contrastive learning offers a promising direction to mitigate feature-space congestion in EFCIL, yet applying it directly in the classification feature space interferes with the objective of learning class-discriminative features, harming performance. We thus propose a novel EFCIL framework decoupling the contrastive learning and classification via a projection head in order to take advantage of the contrastive learning and preserve rich class-discriminative features in the pre-projection space. To further reduce congestion between old and new classes, we propose an old-class repulsion strategy directly on the pre-projection space. Additionally, we propose to eliminate the computation overhead incurred by current prototype calibration methods through a closed-form similarity-weighted linear regression update, enabling efficient yet effective adaptation of full prototype distributions. By integrating these three strategies, our proposed method outperforms existing state-of-the-art methods across several benchmarks. Code available at [https://anonymous.4open.science/r/iclr2026-D134](https://anonymous.4open.science/r/iclr2026-D134).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper tackles Exemplar-Free Class-Incremental Learning (EFCIL), proposing a framework that decouples contrastive learning from classification via a projection head, introduces an old-class repulsion loss to reduce feature space congestion, and presents a closed-form similarity-weighted linear regression calibration for prototype adaptation. The method outperforms state-of-the-art baselines on CIFAR100, TinyImageNet, and ImageNet100.

### Strengths
1. The use of a projection head to separate contrastive learning from the classification objective is supported by both recent theoretical insights and empirical results. 
2. Table 1 and Tables 3-5 demonstrate consistent improvements over recent SOTA, across three benchmark datasets and various task splits. Figure 2 further visualizes the sustained accuracy advantage through incremental tasks.
3. The analysis in Table 2 and Table 8 provides detailed evidence for each component’s contribution.

### Weaknesses
1. While the method shows promising results, some core ideas (contrastive learning and prototype calibration) are not entirely new and can be viewed as extensions of existing methods (SDC, SSRE, FCS).
2. Despite referring to theoretical works (Ouyang et al.) supporting the decoupling approach, the paper itself does not provide principled guarantees or deeper theoretical insight into why and when the method will provably outperform single-space approaches. The authors mention this as a limitation but punt the analysis to future work (Section 5). This limits the depth and generality of the contribution.
3. While Figure 7 explores sensitivity to hyperparameters ($\alpha$, $\tau$, $\beta$), there’s only a narrow range, especially for $\beta$. Moreover, the method is sensitive to the contrastive loss weight, $\alpha$.
4. The anonymous link only contained the running command without the code.

### Questions
Please refer to Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces a novel exemplar-free class-incremental learning (EFCIL) framework that decouples contrastive learning and classification through a projection head, allowing the model to leverage contrastive objectives without disrupting class-discriminative feature learning. To alleviate feature-space congestion, it further proposes an old-class repulsion strategy operating in the pre-projection space, enhancing separation between old and new classes. Moreover, an efficient prototype calibration method based on closed-form similarity-weighted linear regression is developed to update class prototypes without heavy computation. Together, these innovations improve representation quality and efficiency in exemplar-free continual learning.

### Strengths
1. This paper fully analyzes the limitations of existing methods and has a clear research motivation。
2. The proposed closed-form linear-regression-based prototype calibration method demonstrates effectiveness without requiring additional training.
3. The authors demonstrate the effectiveness of the proposed method through a wealth of experiments, and not just bring about performance improvements.

### Weaknesses
1. Compared to the previous use of projection-based contrastive learning, what is the innovation of this paper?
2. This paper uses AdaGauss as the baseline. How are the proposed multiple loss functions combined with the original AdaGauss loss?
3. Providing detailed information on the accuracy curve in Figure 2 in the appendix will be beneficial for future research.

### Questions
Minor concerns:

- There is no executable code in the provided anonymous repository.

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
This paper proposes a framework for exemplar-free class-incremental learning (EFCIL) that integrates three components: (1) a projection head to decouple contrastive and classification losses, (2) an old-class repulsion loss to reduce feature-space congestion, and (3) a closed-form similarity-weighted linear regression for prototype calibration. Experiments on CIFAR100, TinyImageNet, and ImageNet100 show improved results over existing baselines such as AdaGauss, FCS, and EFC.

### Strengths
1. The paper is clearly written and follows the standard EFCIL formulation with comprehensive experiments on multiple benchmarks.

2. The proposed closed-form linear regression calibrator is computationally efficient and practically valuable.

3. Ablation studies are detailed and help to isolate the contributions of each module.

### Weaknesses
1. Novelty. Using contrastive learning (PASS) together with prototype estimation (SDC) is already common in EFCIL. This paper appears to make only minor modifications, so the technical contribution is limited. Moreover, inserting a projection head after the backbone and then applying a contrastive loss is standard in SimCLR, SupCon, BYOL/SimSiam, etc.; therefore I remain skeptical that “using a projection layer to separate contrastive and classification losses” constitutes a substantive methodological contribution here.

2. Questionable evaluation focus (cold-start only).
In EFCIL, it is typical to start from a pretrained model or to use half-class as the base task. The paper emphasizes a cold-start protocol (training from scratch) as its primary benchmark, which is unusual given practical usage where a pretrained backbone (e.g., ImageNet/CLIP) is readily available. More importantly, the generalization capability provided by pretraining is crucial for subsequent continual tasks, and thus highly relevant to realistic deployments; it is unclear why the authors restrict attention to this single protocol.

3. Potentially unfair comparisons.
Many compared methods (PASS, IL2A, SSRE, FeTrIL, FeCAM) were originally designed under half-class pretraining/warm-start, so poor cold-start results are expected. Though EFC and AdaGauss report cold-start results in their original works; they also provide warm-start results. I strongly recommend adding warm-start comparisons to ensure fairness and practical relevance

### Questions
1. see weakness. My major concerns are novelty and setting.
2. Since the proposed linear-regression calibrator assumes a linear mapping, does it generalize well when feature drift is non-linear?
3. Please provide wall-clock training time and memory comparisons to better demonstrate the claimed “efficiency.”

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper focuses on the cold start class-incremental learning (CIL). It presents a new exemplar-free CIL approach based on AdaGauss. To reduce feature space congestion, it proposes a contrastive learning approach along with the old-class repulsion strategy. It uses a weighted linear regression to calibrate the prototype to address the prototype drift. The paper claims that the proposed method outperforms previous state-of-the-art methods across multiple EFCIL benchmarks.

### Strengths
1. This paper focuses on cold-start CIL, a challenging problem in continual learning.
2. The proposed method achieves competitive performance when compared with previous SOTA methods.
3. Most of the reported data in experiments are averaged over multiple experiments, varying the class order.

### Weaknesses
1. The contribution of this paper is incremental. For example,
    1. Introducing contrastive learning into incremental learning is a plug-and-play and effective trick for improving performance, which has been discussed in previous papers (e.g., [3, 4, 5]). The motivation behind introducing contrastive learning in this article is more like improving performance rather than solving problems in CIL.
    2. The use of the least squares technique to solve the problem of prototype offset has been discussed in DPCR [2]. This paper lacks a comparison with DPCR.
2. The projector-enhanced contrastive learning module may probably improve the performance by enhancing the generalization ability of the backbone network instead of addressing forgetting. Improving the supervised image classification task with contrastive learning has been widely discussed [1]. This is a common technique applicable to all image classification problems, but it is not specifically designed for CIL.
3. The claim that the proposed method outperforms existing state-of-the-art methods is not reliable, as it lacks comparison with recent works (e.g., DPCR [2], which is also a method evolving feature extractor with prototype calibration) published in 2025. 
4. The source code is NOT available at the provided link.

[1] Islam, Ashraful, et al. "A Broad Study on the Transferability of Visual Representations with Contrastive Learning." *2021 IEEE/CVF International Conference on Computer Vision (ICCV)*. IEEE, 2021.

[2] He, Run, et al. "Semantic Shift Estimation via Dual-Projection and Classifier Reconstruction for Exemplar-Free Class-Incremental Learning." *Proceedings of the 42nd International Conference on Machine Learning*. PMLR, 2025.

[3] Zhu, Jitao, et al. "Class incremental learning with deep contrastive learning and attention distillation." *IEEE Signal Processing Letters* 31 (2024): 1224-1228.

[4] Li, Qiwei, Yuxin Peng, and Jiahuan Zhou. "FCS: Feature calibration and separation for non-exemplar class incremental learning." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2024.

[5] He, Run, et al. "REAL: Representation enhanced analytic learning for exemplar-free class-incremental learning." *arXiv preprint arXiv:2403.13522* (2024).

### Questions
My key concerns about this paper are listed in the Weaknesses section. Besides, I list some of my concerns that do not impact my rating:

1. What does Table 4.3 in line 408 refer to?
2. Can you provide an analysis of forgetting in the paper (e.g., a comparison between the proposed method and other methods in avoiding forgetting, an analysis of the effectiveness of different modules in avoiding forgetting)?

### Soundness
2

### Presentation
3

### Contribution
2
