# Point2RBox-v3: Self-Bootstrapping from Point Annotations via Integrated Pseudo-Label Refinement and Utilization

- Decision: Accept (Poster)
- Scores: 10, 6, 8, 4

## Abstract
Driven by the growing need for Oriented Object Detection (OOD), learning from point annotations under a weakly-supervised framework has emerged as a promising alternative to costly and laborious manual labeling. In this paper, we discuss two deficiencies in existing point-supervised methods: inefficient utilization and poor quality of pseudo labels. Therefore, we present Point2RBox-v3. At the core are two principles: $\textbf{1) Progressive Label Assignment (PLA)}$. It dynamically estimates instance sizes in a coarse yet intelligent manner at different stages of the training process, enabling the use of label assignment methods. $\textbf{2) Prior-Guided Dynamic Mask Loss (PGDM-Loss)}$. It is an enhancement of the Voronoi Watershed Loss from Point2RBox-v2, which overcomes the shortcomings of Watershed in its poor performance in sparse scenes and SAM's poor performance in dense scenes. To our knowledge, Point2RBox-v3 is the first model to employ dynamic pseudo labels for label assignment, and it creatively complements the advantages of SAM model with the watershed algorithm, which achieves excellent performance in both sparse and dense scenes. Our solution gives competitive performance, especially in scenarios with large variations in object size or sparse object occurrences: 66.09\%/56.86\%/41.28\%/46.40\%/19.60\%/45.96\% on DOTA-v1.0/DOTA-v1.5/DOTA-v2.0/DIOR/STAR/RSAR.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
Due to its potential to replace the time-consuming and laborious manual annotation, the learning method based on point annotation under weak supervision framework has published several related work in high-quality academic conferences or magazines in the past three years, but its indicators are still far from fully supervised oriented object detection. This paper insightfully puts forward that there are two improvements in the existing scheme: more effective label assign and more refined scale loss constraint, and tries to solve it. Two components, PLA and PGDM, are proposed. The PLA component innovatively uses the scale information of the pseudo label generated in the detection pipeline for the label assignment; PGDM component adopts Sam and watershed segmentation algorithms with different characteristics, and observes their advantages in sparse scenes and dense scenes, respectively. The combined use provides a more refined scale loss constraint. This scheme shows a considerable increase in indicators on six test sets (dota-v1.0/1.5/2.0, dior/star/rsar) compared with the previous SOTA baseline Point2RBox-v2. The effectiveness of the scheme is also shown by moving it to partial weakly-supervised tasks.

### Strengths
Strengths:
1. Effectiveness: on the relatively strong baseline Point2RBox-v2, there is still a big improvement: taking DOTA-v1.0 as an example, the end-2-end mode is improved by 8.61 points, and the two stage mode is improved by 3.48 points;
Compared with the fully supervised SOTA index of 75.81, gap has been reduced to less than 10 points for the first time.
2. Generalization: SOTA has been achieved not only on DOTA series datasets (DOTA-v1.0/1.5/2.0), but also on DIOR/STAR/RSAR datasets; In addition to the point supervision task, the effect is also verified on partial weakly-supervised tasks.
3. Easy to follow: it clearly points out two classic applicable scenarios: the target size changes significantly and the sparse target scenario, which is refined and consistent with intuitive cognition and easy for readers to follow.
4. Unified paradigm: based on the verification of the effectiveness of PLA module, it can be predicted that the model paradigms of point supervision, rectangular box supervision and rotating rectangular box supervision tasks will be unified in the future.
5. The logic forms a closed loop: the source of motivation, the design of the method, quantitative indicators, the presentation of ablation experiments, and the comparison of visual images before and after optimization. The logic forms a closed loop self consistently.

### Weaknesses
Weekness:
1. Although point2rbox-v3 has improved significantly compared with baseline, there is still a certain gap in indicators from the fully supervised scheme.
As for the case that the scheme fails to deal with, the paper had better involve this aspect.
2. When displaying the indicators in Tables 1 and 2, bold indicators and underlined indicators should represent Top1 indicators and top2 indicators respectively. However, there is no description in caption.

### Questions
Question:
1. Is the Class-Specific Watered trick first proposed in this paper? Will there be hyper parameters or other prior information? Can its pseudo code or core code be provided? How much more training time will class specific watered take than watered?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Point2RBox-v3, a weakly supervised oriented object detection framework that learns from point annotations through two key modules: Progressive Label Assignment (PLA) for dynamic FPN-level label assignment, and Prior-Guided Dynamic Mask Loss (PGDM-Loss) that combines SAM and watershed masks according to scene density. The approach effectively improves the quality and utilization of pseudo labels, achieving state-of-the-art results across multiple datasets. The writing is clear and easy to follow, and the experimental evidence is solid, though some design motivations and ablations could be further clarified.

### Strengths
1.Writing is clear and logically organized, making the method easy to follow.

2.Proposed PGDM-Loss reasonably integrates SAM and watershed, improving pseudo-label quality.

3.PLA design enhances FPN label assignment and complements PGDM-Loss effectively.

4.The method achieves strong SOTA performance across six benchmark datasets.

5.Ablation studies are complete and show consistent improvements from each module.

### Weaknesses
1.No ablation comparing PGDM with a simple SAM+watershed fusion baseline.

2.The necessity of PLA is not fully clarified when PGDM already refines pseudo labels.

3.The generality of the hyperparameter N_thr across datasets remains uncertain.

4.No analysis or visualization of failure cases where both SAM and watershed fail.

### Questions
1. Can the authors provide a simple fusion baseline of SAM and watershed to isolate the benefit of the prior-guided design?

2. Do PLA and PGDM work independently, or is there performance redundancy between them?

3. Is the same N_thr value used for all datasets, and how sensitive is the model to it?

4. Could the method generalize to non-remote-sensing domains, or are there assumptions limiting its applicability?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This work improves upon the baseline **Point2RBox-v2** by proposing a well-designed framework featuring two innovative components, achieving a remarkable enhancement in detection accuracy and representing a substantial advancement in the domain of point-supervised OOD detection. The proposed **Progressive Label Assignment (PLA)** effectively restores the multi-level feature utilization capability of FPN, thereby compensating for the critical performance gap caused by rigid and inflexible label assignment strategies in point-supervised methods. In addition, the **Prior-Guided Dynamic Mask Loss (PGDM-Loss)** elegantly integrates the robustness of SAM in sparse scenes with the efficiency of the Watershed algorithm in dense scenes, significantly improving pseudo-label quality. Extensive experiments demonstrate the superiority of this approach on six major aerial image datasets, outperforming existing state-of-the-art (SOTA) models.

### Strengths
1. **Clear motivation and precise problem targeting.**

   Point-supervised object detection suffers from the absence of scale information, leading to suboptimal accuracy in mainstream methods. This paper insightfully identifies the inefficiency and poor quality of pseudo-label utilization in existing approaches and proposes a clear and elegant solution. The introduction of PLA enables multi-level label assignment under point supervision through FPN, effectively narrowing the performance gap between point supervision and full or box-level supervision in OOD tasks. Meanwhile, PGDM-Loss achieves an effective balance between accuracy and efficiency by dynamically selecting between the SAM model and the Voronoi-Watershed algorithm.



2. **Superior performance surpassing current SOTA models.**

   The proposed model achieves state-of-the-art results across six popular benchmark datasets (DOTA-v1.0/1.5/2.0, DIOR, STAR, RSAR), demonstrating strong generalization capability and practical impact across diverse datasets and imaging modalities, including SAR imagery.



3. **Methodological transferability across paradigms.**

   The paper successfully extends the method to a *partially weakly supervised learning* setting (PWOOD framework), achieving consistent and significant performance gains under various ratios of weak supervision. This highlights the modular utility and scalability of the PLA and PGDM components beyond pure point-supervised tasks.

### Weaknesses
1. In the performance comparison of categories shown in Table 1, the performance improvement of some categories over the baseline Point2RBox-v2 is not significant, and even decreases in some cases. For example, for BC (basketball court): 79.7 -> 75.7.

2. The authors mentioned using the masks of the SAM Model as pseudo labels to measure the loss. However, training with SAM on the entire image dataset incurs huge computational costs, especially in scenarios such as remote sensing where there may be a dense scene with many instances.

3. Although ( L_{\text{others}} ) is inherited from the baseline model, its specific weight configuration should not be omitted, as this omission may hinder readers' understanding and reproducibility of the model.

### Questions
1. Does the performance improvement of the new method lack stability or have bias, especially in certain challenging categories?

2. The "Class-Specific Watershed" technique introduced by you in the appendix aims to enhance the mask quality of overlapping objects (such as GTF and SBF), and brings a 1.45% performance improvement. Is this technique a necessary patch for the model when dealing with overlapping scenarios? Why didn't you integrate it into the official version of the model?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper builds upon Point2RBox-v2 with incremental improvements. First, it generates more pseudo-label candidates by leveraging features from different levels of the network. Second, it addresses the limitations of the watershed loss in Point2RBox-v2 for simple scenes by incorporating SAM.

### Strengths
The proposed method significantly improves the performance of point-supervised oriented object detection.
The figures in this paper are clear.

### Weaknesses
Unfriendly to non-experts and new readers, as the author likely assumes familiarity with previous versions of Point2RBox.

Recommend formally defining tasks. 

Provide necessary explanations for letters and functions appearing in formulas, e.g., $I$ and $minAreaRect$.

Line 214: gt should be capitalized.

Line 301: Remove equation.

### Questions
1. What is the definition of the $score$ function in Eq. (4)?
2. I don't understand how to dynamically select which mask (SAM or Watershed) to use as the basis for pseudo-labeling. Based on earlier statements in the paper, it seems that the method determines the number of instances in each scene and uses that as the criterion. However, I couldn't find a corresponding step in the actual implementation described in the paper.
3. I'm curious about the proportion of each layer serving as the most likely pseudo-label in Eq. (4). Does the final layer play the most critical role? Because in your two examples, the final layer's pseudo-labels appear to be the most accurate.

### Soundness
3

### Presentation
3

### Contribution
3
