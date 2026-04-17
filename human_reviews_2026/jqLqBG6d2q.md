# CAST: Contrastive Adaptation and Distillation for Semi-Supervised Instance Segmentation

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Instance segmentation demands costly per-pixel annotations and computationally expensive models. We introduce CAST, a semi-supervised knowledge distillation (SSKD) framework that compresses pre-trained vision foundation models (VFM) into compact experts using limited labeled and abundant unlabeled data. CAST unfolds in three stages: (1) domain adaptation of the VFM(s) via self-training with contrastive calibration, (2) knowledge transfer through a unified multi-objective loss, and (3) student refinement to mitigate residual pseudo-label bias. Central to CAST is an \emph{instance-aware pixel-wise contrastive loss} that fuses mask and class scores to extract informative negatives and enforce clear inter-instance margins. By maintaining this contrastive signal across both adaptation and distillation, we align teacher and student embeddings and fully leverage unlabeled images. On Cityscapes and ADE20K, our 11X smaller student improves over its zero-shot VFM teacher(s) by +8.5 and +7.1 AP, surpasses adapted teacher(s) by +3.4 and +1.5 AP, and further outperforms state-of-the-art SSKD methods on both benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
CAST introduces a three-stage Semi-Supervised Knowledge Distillation (SSKD) pipeline that compresses a large Vision Foundation Model (VFM) into a 11× smaller student for instance segmentation. The key novelty is an instance-aware pixel-wise contrastive loss that mines hard negatives by fusing mask + class probabilities; this loss is injected (i) when the teacher is self-trained on unlabeled data and (ii) when the student is distilled.
On Cityscapes and ADE20K with only 10 % labels the student beats its adapted teacher by +3.4 and +1.5 mask AP.

### Strengths
1. First work to unify VFM adaptation, dense pixel-level contrastive learning, and extreme compression for instance segmentation.
2. Novel instance-aware negative sampler that fuses mask & class logits to avoid sampling same-instance pixels; backed by a theoretical guarantee that each gradient step increases expected inter-instance margin (Prop. 3.1).
3. Exhaustive ablations (loss terms, stages, sampling, 6 hyper-parameters, 3 label fractions, 2 datasets).
4. Failure modes (pseudo-label bias) are explicitly discussed and mitigated by Stage 3 supervised fine-tuning.
5. Demonstrates that dense contrastive signals can be cheaply harvested from unlabeled images without extra annotations or human-designed rules.

### Weaknesses
1. Paper fuses Grounding-DINO + SAM-2 but never distills from single teachers. Readers cannot tell whether gains come from the contrastive loss or from ensembling complementary models.

### Questions
I don't have particular questions for this paper.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a semi-supervised knowledge distillation framework for instance segmentation. It proposes an instance-aware pixel-wise contrastive loss to enhance the model's representation capability. Improvements are observed on standard benchmarks like ADE20K and Cityscapes.

### Strengths
1. The paper is clearly written and easy to follow.
2. The motivation of using unlabeled data to improve the labor-intensive instance segmentation scenario is good.

### Weaknesses
1. The core idea is not new. There are many existing works [1, 2, 3] that explore dense contrastive learning for dense perception tasks, especially three or four years ago when contrastive learning was still very popular.

2. The gain of introducing unlabeled images are marginal. For example, on ADE20K, the supervised-only setting already achieves 23.5 mAP. Using 9x more unlabeled data only improves it by 1.0 mAP.

3. The proposed framework does not exhibit a clear empirical advantage over existing frameworks, such as [4].

[1] Semi-supervised semantic segmentation with pixel-level contrastive learning from a class-wise memory bank, ICCV 2021 \
[2] Sepico: Semantic-guided pixel contrast for domain adaptive semantic segmentation, TPAMI 2022 \
[3] Hunting sparsity: Density-guided contrastive learning for semi-supervised semantic segmentation, CVPR 2023 \
[4] Self-training with noisy student improves imagenet classification, CVPR 2020

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents CAST, a semi-supervised knowledge distillation framework to compress large VFMs for instance segmentation. It uses a three-stage pipeline: (1) refining the teacher model, (2) transferring knowledge to a compact student using a new instance-aware contrastive loss, and (3) fine-tuning the student. The resulting student model is 11x smaller in parameters but achieves superior performance compared to both its original teacher and other state-of-the-art methods.

### Strengths
1. Comprehensive Pipeline: CAST offers a well-structured SSKD pipeline that unifies teacher adaptation, knowledge distillation with a pixel-wise contrastive component, and student fine-tuning, systematically addressing the challenges of compressing large VFMs for instance segmentation.
2. Technical Rigor and Theoretical Insight: The paper details a mathematically sound instance-aware pixel-level contrastive loss, complete with negative sampling mechanisms incorporating fused mask-class cues. 
3. Thorough Empirical Evaluation: The main results and ablation studies, along with figures robustly support claims of improved performance, efficiency, and bias reduction. The student notably improves over all relevant baselines and ablations.
4. Ablation and Diagnostic Depth: The paper reports meaningful ablations, including loss term contributions, the necessity of each pipeline stage, pixel-wise negative sampling strategies, hyperparameter sweeps, and architectural effects.

### Weaknesses
1. While the proposal is thoughtfully implemented and the contrastive calibration is well-integrated, the primary technical innovations seem incremental. The instance-aware contrastive loss largely adapts previous contrastive/self-supervised learning techniques with a new but straightforward mask and class score fusion for negative sampling.
2. The claims regarding “robustness” under low-label regimes stem from results on only two standard datasets. Broader generalization or domain robustness (e.g., in-the-wild images, transferred domains, or real edge-based applications) is not empirically validated, contrary to the aspiration mentioned in the introduction and conclusions.

### Questions
1. How robust is the instance-aware negative sampling scheme to highly imbalanced datasets, both between classes and between instances within a single image? Did you observe increased failure or co-located instance confusion under high crowding or many small objects?
2. What is the computational overhead of the contrastive loss (especially the instance-aware negative sampling) in practice relative to prior SSKD/contrastive frameworks?
3. Is the three-stage process of teacher adaptation, student distillation, and final fine-tuning critical? Would performance meaningfully degrade if adaptation and distillation are collapsed into a single objective?

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
This paper introduces CAST, a semi-supervised knowledge distillation framework designed to compress large pre-trained vision foundation models (VFMs) into much smaller, efficient models. Experiments demonstrate the effectiveness of the method.

### Strengths
1. The organization of this paper is clear, which is easy to follow.

2. The experiments are good, and the ablation studies are comprehensive.

### Weaknesses
My main concern is mainly sourced from the insufficient discussion in related works.

(1) As the core design of the method is the instance-aware pixel-wise contrastive loss, there have been many contrastive learning based knowledge distillation methods. However, the paper lacks discussion on these previous works.  

(2) Early works made a contrast between teachers’ and students’ features by employing different views generated from  using various samples’ features [1, 2] as well as gradients [3] stored in the memory buffer. 

(3) In dense prediction tasks, G-DetKD [4] constructed ROI feature pairs and executed soft semantic-guided matching, which promoted the performance in object detection, and CIRKD [5] designed an implicit contrastive method that leveraged both pixel and region representations to learn structured information in the spatial dimension.  Furthermore, for pixel-level mimicking, Af-DCD [6] designs more fine-grained contrastive learning between pixels from the teacher and the student.

(4) Although they have different experimental settings, the paper is also in a related research line. Therefore, adding discussion to the related works, especially for segmentation tasks [4-6], is encouraged. 


[1] Zhiyuan Fang, Jianfeng Wang, Lijuan Wang, Lei Zhang, Yezhou Yang, and Zicheng Liu. Seed:
Self-supervised distillation for visual representation. arXiv preprint arXiv:2101.04731, 2021

[2] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive representation distillation. In
ICLR, 2020.

[3] Jinguo Zhu, Shixiang Tang, Dapeng Chen, Shijie Yu, Yakun Liu, Mingzhe Rong, Aijun Yang,
and Xiaohua Wang. Complementary relation contrastive distillation. In CVPR, 2021.

[4] Lewei Yao, Renjie Pi, Hang Xu, Wei Zhang, Zhenguo Li, and Tong Zhang. G-detkd: towards
general distillation framework for object detectors via contrastive and semantic-guided feature
imitation. In ICCV, 2021

[5] Chuanguang Yang, Helong Zhou, Zhulin An, Xue Jiang, Yongjun Xu, and Qian Zhang. Crossimage relational knowledge distillation for semantic segmentation. In CVPR, 2022.

[6]  Fan, Jiawei, et al. Augmentation-free dense contrastive knowledge distillation for efficient semantic segmentation. In NeurIPS 2023.

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
2
