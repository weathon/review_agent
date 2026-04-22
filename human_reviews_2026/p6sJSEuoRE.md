# iCAS: A In-Context Anomaly Segmentation Framework for Industrial Visual Inspection

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Visual in-context prompting has recently made promising progress, achieving training-free  segmentation with a generalized model derived from large-scale pre-training. However, we observe that these in-context segmentation models fail on the anomaly detection task, e.g., visual inspection. In this study, we propose iCAS, a novel model for In-Context Anomaly Segmentation enabling automatic defect annotation and visual prompting anomaly segmentation. The framework is built upon an in-context mask transformer, further enhanced by a greedy query selection strategy and a mask-level feature matching module to improve both sensitivity and generalization. Further, we propose the General-to-Specific pre-training to solve the weak generalization problem caused by the scarcity of anomalous samples. Finally, we conduct comprehensive experiments under a variety of anomaly detection and segmentation tasks. Evaluations on multiple publicly available datasets show the generalization and effectiveness of our method.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces what the authors claim to be the first in-context segmentation method tailored for industrial visual inspection. By combining a greedy query selection strategy, a mask-level feature matching module, and a general-to-specific pretraining paradigm, the proposed framework not only outperforms existing general in-context segmentation methods, but also delivers strong few-shot anomaly detection performance. Overall, the work is interesting, but there are several issues that should be addressed before it is ready for acceptance.

### Strengths
The paper explores the first in-context anomaly segmentation model designed specifically for industrial inspection.

The general-to-specific pretraining strategy is reasonable and clearly motivated, as training solely on industrial anomaly datasets is often insufficient.

Experimental results demonstrate promising performance.

### Weaknesses
The motivation needs to be strengthened. Why is an in-context segmentation model particularly important for industrial inspection? More discussion is needed.

The related work section is not comprehensive enough. Several recent methods published in 2025 are missing.

The description of the in-context transformer is too brief, making it difficult for readers to understand how it works internally.

The comparisons with PerSAM and Matcher appear unfair. The proposed method leverages both semantic segmentation data and targeted industrial anomaly data for training, while prior in-context segmentation baselines do not use anomaly detection data.

Table 3 reports promising results, but the evaluation metric used is not clearly indicated.

While I understand that space limitations make it difficult to include extensive details, the current description of the proposed method is still hard for readers to follow. Some parts require further elaboration—particularly Section 3.2 on the objective function, which is difficult to understand in its current form.

### Questions
How does the method perform when anomalies have fuzzy or unclear boundaries, which is very common in real industrial settings?

Figure 2 suggests that a mask set is required for training. How is this mask set obtained in practice? Does it require manual annotation?

In line 306, it is mentioned that semantic masks are obtained via SAM. How exactly is this performed?

In Table 1, what do the notations CP, BT, HN, etc. stand for? Please clarify.

### Soundness
3

### Presentation
2

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
This paper proposes the iCAS model together with a General-to-Specific pre-training paradigm. iCAS is based on a mask classification transformer architecture and introduces Greedy Query Selection (GQS) and Mask-level Feature Matching (MFM) to accurately localize anomalous regions. This approach enables robust anomaly segmentation, even under limited anomaly data conditions.

### Strengths
1. This paper evaluates the iCAS model using a wide range of evaluation metrics and a variety of well-structured experiments.
2. This paper defines a new pre-training paradigm, the General-to-Specific approach, that effectively bridges the gap between general semantic segmentation and specialized anomaly segmentation.

### Weaknesses
1. The paper lacks comparison with recent few-shot anomaly detection methods, such as: UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection (CVPR 2025), DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary Lookup (ICCV 2025)

2. Experiments on backbone networks are limited. Ablation studies involving CLIP-ViT and DINOv1 would strengthen the effectiveness of this paper.

3. Greedy Query Sampling (GQS) is likely sensitive to the choice of K, but no ablation studies on this parameter have been presented. Furthermore, if GQS tends to select queries in the normal region, the robustness of the model should be verified on datasets with diverse object locations, such as MPDD or medical datasets, which are not covered in this paper.

4. General-to-Specific pre-training paradigm consists of two stages; an analysis of its computational cost would be valuable.

### Questions
1. We are curious about the performance of iCAS in an unsupervised learning anomaly detection environment and its performance in the presence of anomaly data.
2. For backbone networks, a comparison study comparing DINOv2, CLIP-ViT, and DINOv1 would be interesting.
3. General-to-Specific pre-training paradigm consists of two stages, and an analysis of their computational costs would be important.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces iCAS, a new In-Context Anomaly Segmentation framework designed to generalize visual in-context prompting (e.g., SAM-style models) to industrial anomaly segmentation. The core idea is to enable training-based, prompt-driven anomaly localization using only a few anomaly or normal samples. The method achieves strong performance across diverse datasets, significantly outperforming existing in-context segmentation models and anomaly detection methods.

### Strengths
1.iCAS unifies promptable segmentation (SAM-like) and anomaly detection in a training-based, in-context fashion

2.The proposed GQS and MFM modules are simple yet effective, addressing practical issues of query redundancy and boundary precision.

3.Extensive experiments (five datasets, multiple tasks, and ablations) convincingly show robustness, scalability, and effectiveness of each component.

### Weaknesses
1.While iCAS performs well, its components (GQS, MFM, two-stage training) are mostly adapted from known concepts (active learning, mask matching, transfer learning). The true methodological novelty might be seen as moderate.

2.The paper primarily focuses on industrial surface defects, it is unclear whether iCAS generalizes to other anomaly types (e.g., medical or natural images) or remains domain-specific due to the specialized anomaly-aware pre-training.

### Questions
1.Could the authors clarify why GQS is necessary beyond standard query embeddings? How much does it reduce redundancy compared to random or uniform query sampling?

2.How scalable is the method computationally? Please report training cost and inference time compared to SAM or Matcher.

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
3

### Summary
The paper proposes iCAS, an in-context anomaly segmentation framework designed for industrial visual inspection.  The model extends a mask-classification transformer with two modules: Greedy Query Selection (GQS), which selects representative visual tokens, and Mask-level Feature Matching (MFM), which refines mask alignment across queries.  It further adopts a General-to-Specific pre-training schedule—first training on large-scale semantic segmentation datasets, then fine-tuning on anomaly-focused data (RealIAD, MANTA).  Experiments on MVTec-AD, VisA, and related benchmarks report improved mIoU compared to existing in-context and SAM-based baselines (PerSAM, Matcher, SINE).

### Strengths
•	Ambitious effort to unify semantic, interactive, and few-shot anomaly segmentation within one framework.
•	Broad empirical evaluation across multiple benchmarks with consistent results.
•	Well-executed ablation studies showing measurable effects of GQS, MFM, and the pre-training strategy.
•	Built on open, reproducible components (MaskFormer, DINOv2), which aids transparency.

### Weaknesses
1. The proposed method is primarily a reassembly of existing techniques with limited conceptual advancement.
2. Claims of “training-free” or “in-context reasoning” are overstated, given the reliance on large-scale supervised pre-training and conventional fine-tuning.
3. The contributions of GQS and MFM are empirically validated but not theoretically grounded or well-explained.
4. The paper does not clearly separate iCAS from prior in-context segmentation approaches (SAM, HQ-SAM, SegGPT, SINE, Matcher). GQS and MFM are minor technical variations on existing promptable frameworks and do not introduce a new capability or reasoning mechanism.
5. Existing vision-language anomaly detection models (WinCLIP, AnomalyCLIP, InCTRL, RegAD, MetaUAS) are mentioned but never compared under equal conditions. The reported gains may stem from backbone choice and training scale rather than a new formulation.
6. The General-to-Specific pre-training scheme closely follows the standard “pretrain on generic segmentation → fine-tune on anomaly masks” pipeline already used in RegAD, MetaUAS, and RealNet. Thus, its novelty is marginal.

### Questions
This paper presents a solid empirical study with clear organization and credible results, but the core contribution lies in engineering integration rather than conceptual innovation.

### Soundness
3

### Presentation
3

### Contribution
2
