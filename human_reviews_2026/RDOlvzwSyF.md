# Salient Object Ranking via Cyclical Perception-Viewing Interaction Modeling

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Salient Object Ranking (SOR) aims to predict human attention shifts across different salient objects in a scene. Although a number of methods have been proposed for the task, they typically rely on modeling the bottom-up influences of image features on attention shifts. In this work, we observe that when free-viewing an image, humans instinctively browse the objects in such a way as to maximize contextual understanding of the image. This implies a cyclical interaction between content (or story) understanding of the image and attention shift over it. Based on this observation, we propose a novel SOR approach that models this explicit top-down cognitive pathway with two novel modules: a story prediction (SP) module and a guided ranking (GR) module. By formulating content understanding as the image caption generation task, the SP module learns to generate and complete the image captions conditioned on the salient object queries of the GR module, while the GR module learns to detect salient objects and their viewing orders guided by the SP module. Extensive experiments on SOR benchmarks demonstrate that our approach outperforms state-of-the-art SOR methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposed a method for saliency object ranking. A story prediction module predicts the caption of the image and a guided ranking module predicts the saliency rankings. The cyclical interaction module aligns and refines the caption and the ranking iteratively. The experimental results seemed to show the proposed method outperformed previous SOTA.

### Strengths
- The cyclical interaction uses caption to guide saliency object ranking.
- Ablations shows the effectiveness of the SITA and CMQC in the proposed method.

### Weaknesses
- The segmentation head is unclear. The performance increase could potentially due to using a strong pretrained segmentation model.
- The retained QAGNet has lower scores across metrics compared to the ones reported in the original paper. This is critical since the results of the proposed method does not outperform the reported results of QAGNet.

### Questions
- Is the segmentation head a pretrained segmentation model or else? A strong segmentation model could favor the MAE.
- What is the impact of number of object queries on results? An ablation study will be beneficial to see the impact.
- Would a stronger text decoder leads to better performance?
- What is the reason of decreased performance of retrained QAGNet? Did the authors use different training details or different evaluation settings or else?
- Intuitively, the proposed method could also improve the performance on image captioning task. I am wondering if salient object ranking could help with image captioning. It will be interesting to see results compared with SOTA image captioning methods.

### Soundness
2

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
3

### Summary
This paper proposes a novel framework that models the cyclical interaction between perception and viewing for the Salient Object Ranking (SOR) task. The method introduces two key components: a Story Prediction (SP) module that simulates the human perceptual process through image caption generation, and a Guided Ranking (GR) module that predicts saliency rankings under the guidance of the SP module.

### Strengths
（1）Novel Cognitive-Inspired Framework. 
The paper introduces a cyclical perception–viewing model inspired by human visual cognition, which is strongly supported by established cognitive and psychological theories. And the introduction is easy to follow.

（2）Extensive experiments.
The paper conducts both qualitative and quantitative experiments, and also provides an analysis of inference time. Moreover, the visualized experimental results clearly and intuitively demonstrate the improvements achieved by the proposed method.

### Weaknesses
（1）The paper lacks a clear comparison with the recent top-down method, Language-Guided Salient Object Ranking (CVPR 2025), and its performance remains inferior to the results reported in that study.

（2）In the Method Overview section, the symbols used in the equations do not correspond to those shown in Figure 2, which makes it confusing to understand the inputs and outputs of each module.

（3）The experimental section mainly provides data and setup details but offers limited analysis or discussion to explain the observed results.

### Questions
(1) Explain the differences between the proposed method and Language-Guided Salient Object Ranking (CVPR 2025). Moreover, the performance of this method is still inferior to that of the existing work.

(2) In Eq.(1), when $l=1$, what dose $Q_{l-1}$ refer to?

(3) In Table 2, for Setting II (“independent caption generation”), there is a performance improvement even without interaction between caption and visual features, which is confusing. Could the authors clarify this behavior?

(4) How is the ground-truth (GT) caption obtained?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a Salient Object Ranking (SOR) approach that consists of two modules: the Guided Ranking (GR) module and the Story Prediction (SP) module, whose interaction enhances the overall performance of SOR.

### Strengths
The design of this model aligns well with the human cognitive system, such as predictive coding, and the English writing is good and clear.

### Weaknesses
Some experiments and details are not clearly explained. For example, in the experimental section, how was the choice of 24,000 epochs determined, and why such a large number? Could this lead to overfitting?

In addition, it would be helpful to qualitatively present the interaction between object queries and text features, as well as the results under different values of K.

### Questions
What training data are used for the segmentation head? Was it pre-trained on the COCO segmentation dataset? If it was trained only on the SOR dataset, would its segmentation generalization ability be affected?

Does the random selection of captions influence the results? It is recommended to include a discussion—for example, are the salient objects in the image always located in the main subject position described in the caption?

### Soundness
3

### Presentation
3

### Contribution
3
