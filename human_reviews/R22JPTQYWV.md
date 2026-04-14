# PointOBB-v2: Towards Simpler, Faster, and Stronger Single Point Supervised Oriented Object Detection

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 6, 8, 6

## Abstract
Single point supervised oriented object detection has gained attention and made initial progress within the community. Diverse from those approaches relying on one-shot samples or powerful pretrained models (e.g. SAM), PointOBB has shown promise due to its prior-free feature. In this paper, we propose PointOBB-v2, a simpler, faster, and stronger method to generate pseudo rotated boxes from points without relying on any other prior. Specifically, we first generate a Class Probability Map (CPM) by training the network with non-uniform positive and negative sampling. We show that the CPM is able to learn the approximate object regions and their contours. Then, Principal Component Analysis (PCA) is applied to accurately estimate the orientation and the boundary of objects.  By further incorporating a separation mechanism, we resolve the confusion caused by the overlapping on the CPM, enabling its operation in high-density scenarios.  Extensive comparisons demonstrate that our method achieves a training speed 15.58$\times$ faster and an accuracy improvement of 11.60\%/25.15\%/21.19\% on the DOTA-v1.0/v1.5/v2.0 datasets compared to the previous state-of-the-art, PointOBB. This significantly advances the cutting edge of single point supervised oriented detection in the modular track. Code and models will be released.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper begins with a comprehensive review of existing point-supervised methods, examining their relative merits with particular emphasis on the limitations of the state-of-the-art approach, PointOBB in Modular WOOD. The key limitations identified include: 1) inefficient pseudo-label generation, 2) excessive GPU memory consumption, and 3) frequent memory overflow issues during RoI proposal generation.
To overcome these limitations, the authors introduce PointOBB-v2, which presents a streamlined pipeline that eliminates the traditional teacher-student architecture. Instead, it employs class probability maps exclusively for precise object contour generation, incorporating efficient Principal Component Analysis (PCA) for object orientation and boundary determination. Additionally, the study proposes a novel vector constraint mechanism for enhanced detection of small objects in high-density scenarios.
Experimental results reveal substantial improvements in pseudo-label generation speed, memory efficiency, and detection performance.

### Strengths
1. The solution proposed by the authors is well-described and well-supported by experimental results. 
2. The paper is easy to follow.
3. The experiments are conducted four datasets of this field.
4. The paper is well-organized, with a clear flow from introduction to conclusion.

### Weaknesses
1. While the paper demonstrates the empirical effectiveness of PointOBB-v2, it could benefit from providing more theoretical justification or intuition behind the design choices, such as the label assignment strategy and the selection of hyperparameters.
2. The authors acknowledge that some hyperparameters, such as the radius in label assignment, are set based on the dataset. While this approach yields strong results on the tested datasets, it may require manual tuning when applied to new scenarios, potentially limiting the method's adaptability.
3.  Although the paper discusses the advantages of modular approaches' flexibility over end-to-end architectures, this method inherently constrains the dynamic updating of pseudo-labels during detector training—a crucial technique that has demonstrated significant efficacy in semi-supervised learning frameworks.

### Questions
1. Please explicitly present the objective function of CPM.
2.  The paper adopts an independent approach for pseudo-label generation and detector training, unlike the conventional semi-supervised teacher-student framework which enables dynamic pseudo-label updates. Please explain it.
3. While identical labels are assigned to positive samples of the same target. Why does the Class Probability Map (CPM) assign higher probabilities to regions surrounding the object centers and along their axes?
4. Please explain the term 'ignored' in Figure 2.
5. Please discuss the hyper parameters ( the radius and α ) in label assignment.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents PointOBB-v2, a novel approach for single point supervised oriented object detection, aimed at providing a simpler, faster, and stronger methodology compared to its predecessor, PointOBB. Specifically, PointOBB-v2 generates Class Probability Maps (CPM) using a positive and negative sample assignment strategy, which helps to capture the approximate contours of objects. To accurately estimate orientation and boundaries, Principal Component Analysis (PCA) is employed, providing a lightweight approach that avoids the need for prior deep network components or cumbersome teacher-student training frameworks. Experiments show that  PointOBB-v2 significantly improves pseudo-label generation efficiency, achieving a 15.58-fold increase in speed and reduced memory consumption while maintaining superior accuracy.

### Strengths
1. The primary contributions of PointOBB-v2 are its significant speed and efficiency improvements. The pseudo-label generation process is 15.58 times faster, and memory consumption is considerably reduced, requiring only around 8GB of GPU memory, compared to the previous version that faced significant out-of-memory issues. It also achieves higher accuracy on DOTA-v1.0/1.5/2.0 compared to PointOBB,  particularly in dense scenarios. 
2. The experimental section of the paper demonstrates a comprehensive evaluation of PointOBB-v2, including comparisons across multiple versions of the DOTA dataset (v1.0, v1.5, and v2.0). The authors present quantitative results that highlight consistent improvements in accuracy and efficiency over the predecessor, PointOBB. The experimental setup effectively uses multiple detectors and datasets to validate the proposed method’s robustness and adaptability, particularly in dense object detection scenarios.
3. The manuscript is well-structured, with a logical flow that clearly explains the motivation, methodology, and outcomes. The authors effectively contextualize their work within the existing literature, highlighting the gap in current oriented object detection approaches and positioning PointOBB-v2 as an advancement in this domain.

### Weaknesses
1. The paper focuses primarily on aerial imagery datasets, and it is not entirely clear how well PointOBB-v2 would generalize to other domains such as autonomous vehicles or medical imaging. Although the aerial domain has its unique characteristics, a brief discussion or experiment on a different type of dataset could help demonstrate the method's broader applicability and flexibility.
2. There are still some grammatical issues in this paper that could be improved. In "Diverse from those approaches relying on one-shot samples," the phrase "Diverse from" is slightly awkward. Consider changing it to "Unlike approaches that rely on one-shot samples." Also, in "achieving significant improvements in the accuracy and generation speed of the pseudo-label, and improving memory efficiency," "improving" is repetitive here. Consider rewriting it as "achieving significant improvements in pseudo-label accuracy, generation speed, and memory efficiency."

### Questions
1. The paper mentions improved performance in high-density scenarios, but what specific attributes of PointOBB-v2 contribute most to these improvements?
2. Why was Principal Component Analysis (PCA) specifically chosen to estimate object orientation and boundary? Could other methods, such as clustering algorithms or deep learning-based orientation estimation, have been a better alternative? What are the trade-offs?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
In this paper, PointOBB-v2 is proposed for point-supervised oriented object detection. The authors introduce the learning of Class Probability Map and use Principal Component Analysis to determine the size and rotation of objects. By employing these paradigms, the capability of the PointOBB-v2 is enhanced and significantly outperforms the PointOBB baselines.

### Strengths
1. The research topic of point-supervised oriented object detection is meaningful and challenging, and the paper is clear and well-organized.
2. As a new version of existing work (PointOBB), the comparison between them is essential. The experiments show the proposed method outperforms the PointOBB baselines in both training speed and accuracy by a considerable amount.

### Weaknesses
1. Since the authors use “the highest resolution feature map of the FPN”, it is not clear whether the resolution while learning the CPM affects the performance. 
2. There are some minor issues: Please check Figure 2, Line 433, and Line 468. Some equations end with a period, while others end with a comma. Please ensure they are consistent.

### Questions
1. What does “the randomness introduced by sampling can cause variance in the result from a single pass” in Line 288 mean? Where does the randomness come from? 
2. Table 2 shows that the improvement on the challenge datasets (DOTA-v1.5/v2.0) is more pronounced compared to DOTA-v1.0. What might account for this difference?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents PointOBB-v2, a faster and more accurate method for single-point supervised oriented object detection that improves upon PointOBB. By generating a Class Probability Map (CPM) with targeted sampling and applying Principal Component Analysis (PCA), PointOBB-v2 effectively estimates object orientation and boundaries without relying on pre-trained models. An added separation mechanism enhances performance in high-density scenarios. PointOBB-v2 achieves 15.58x faster training and significant accuracy gains on DOTA datasets, marking a new state-of-the-art in this domain.

### Strengths
## 1. Significant Contribution to Point-Supervised Oriented Object Detection

This paper presents a notable advancement in point-supervised oriented object detection, achieving improvements in both accuracy and training efficiency.

## 2. Comprehensive Experimental Analysis

The authors provide a thorough evaluation of the proposed method, including detailed ablation studies that effectively demonstrate its efficacy. The method is validated across three aerial image datasets, further underscoring its robustness and applicability.

### Weaknesses
## 1. Incomplete Explanation of the Class Probability Map (CPM)

- While the overall pipeline of PointOBB-v2 is generally understandable from L.198-211, the crucial workings of the CPM remain unclear. 
- Specifically, key details about CPM training are omitted, including the training objective, activation function, and handling of ignored labels.
- The current structure of the Method section adds to the confusion. Section 3.1 describes CPM generation process, followed by Section 3.2, which outlines label assignment. (The first Section 3.1 assumes that we already have the completed CPM generation model, but the readers cannot understand how it can work.)
- Reordering these sections (such as by first describing the model training or merging both subsection) could enhance clarity.


## 2. Limited Justification for the Effectiveness of CPM

- The paper lacks convincing arguments for why the CPM functions effectively. For instance, the use of a fixed radius for positive and negative sampling lacks a clear rationale, particularly in how it helps define object contours.
- Sections 3.1 and 3.2 provide limited insight into how the model generates a CPM with distinct object regions.
- Figure 2, which illustrates the “Training Target” and the “Output CPM”, raises questions about how the model achieves such precise CPM output even when trained on such coarse training labels. Additional explanation on this discrepancy would be valuable.
- Further clarity on Figure 2 would be beneficial: is the red dot an output from the model or a manual illustration? Similarly, the meanings of colors in Figure 3 (blue, pink, and orange) should be clarified.

## 3. Generalization to Other Oriented Object Detection Scenarios
- While the method appears well-suited for densely packed object scenarios (e.g., aerial or retail images), it’s uncertain whether it generalizes to other tasks, such as text detection. Despite aiming to contribute to general object detection, the experiments are limited to remote sensing images, a subset of oriented object detection.


## Additional Comment

- The concept of CPM bears resemblance to the existing method TricubeNet [A], which, although fully supervised, also represents objects within a 2D map. A brief discussion of the similarities and distinctions between these approaches could enhance the paper’s context.

[A] TricubeNet: 2D Kernel-Based Object Representation for Weakly-Occluded Oriented Object Detection (WACV 2022)

### Questions
Please check the weaknesses.

## Justification

This paper presents notable advancements in point-supervised oriented object detection, but key limitations reduce its impact. The explanation of the CPM’s effectiveness is incomplete, particularly regarding how it generates clear object contours from coarse labels. While strong in experimental results, the paper would benefit from clearer methodology. 

My initial rating is weak reject. I will finalize the rating after a discussion with the authors and other reviewers.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, the authors propose a single point supervised oriented object detection method called PointOBB-v2, which can generate pseudo rotated boxes from points without relying on any other prior. Compared with PointOBB, PointOBB-v2 introduces a novel pipeline to improve the accuracy and generation speed of the pseudo-label. Extensive experiments on the DOTA-v1.0/v1.5/v2.0 demonstrate the effectiveness of the proposed method.

### Strengths
1. The paper is innovative, well-organized, and well-written.
2. The proposed method is an improvement of the structure of PointOBB. 
3. The authors conduct comprehensive experiments to demonstrate the effectiveness of the proposed method.

### Weaknesses
1. In related work, it is better to provide a summary of the methods described in section 2.1.
2. The authors conduct experiments on DOTA datasets. How about the effectiveness of the proposed method on other datasets with different object categories and data distributions, such as DIOR?

### Questions
See the Weaknesses section.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper improves PointOBB, an existing weakly supervised oriented object detection method. Oriented object detection is important for analyzing aerial and satellite imagery. The cost of annotation, especially when you want to draw oriented boxes, is high. This paper (and, previously, PointOBB) proposes to annotate an object with just a single point, which is an easier annotation method. The proposed method first generates a class probability map, which creates positive and negative regions. Then, a PCA based analysis is done to estimate principal axes of objects to generate rotated boxes as pseudo annotation. Authors report improved performance over PointBB on the DOTA datasets.

### Strengths
Performance improvement over the baseline is impressive. Also, the proposed method is 15x faster to train.

### Weaknesses
The proposed method (i) is a simple increment to PointOBB and (ii) involves many thresholds, indicating that it might not be an easily generalizable one: 

(i) The method boils down to class probability map (CPM) and PCA.  CPM is a simple positive/negative segmentation. 

(ii) The method has at least 6 thresholds: radius b1 (L244), radius b2 (L262), multiplier to radius (L254), and three others at lines 209, 307 and 320. This suggests that the method is fragile and not easily generalizable to new datasets. I believe it is the authors' responsibility to show that their method is generalizable. Yet, results are reported only on the DOTA dataset. 


The main claim of the paper is that they can do oriented object detection with weak supervision (single point per object). This is good but the performance (around 45)  is drastically behind the fully supervised methods (https://paperswithcode.com/sota/oriented-object-detection-on-dota-1-0), which is around high 70s. So, I think, it  is reasonable to expect a discussion on full annotation cost vs performance degradation. However, none is provided.

### Questions
- What are the annotation costs for full annotation and single point? How do they compare? Is low performance worth the reduced annotation  cost? 
- How does PointOBBv2 perform on DIOR-R and other oriented object detection datasets?


## Post-rebuttal edit: 

The authors provided satisfactory answers to my questions above. They promise to include these answers in the final version of the paper.

### Soundness
3

### Presentation
2

### Contribution
3
