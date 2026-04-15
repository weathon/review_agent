# pDETR: End-to-End Object Detection via Perspective-Aware Transformers

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5

## Abstract
DETR has made notable performance improvements in object detection tasks by leveraging the long-range modeling capabilities of Transformers, but encoding all tokens indiscriminately significantly escalates computational cost and leads to slow convergence. Recent sparsification strategies effectively reduce computational cost through sparse encoders. However, these methods rely heavily on a fixed sparse ratio, which overlooks the coherence of feature representation across levels, leading to performance degradation in complex scenes. To address this issue, we propose a novel object detection approach aimed at constructing consistent representations of multi-level features. The approach composes two steps: First, we introduce a perspective proposal module that leverages the spatial information of high-level foreground features to guide the sparse sampling of low-level features, ensuring both integrity and coherence of multi-scale feature information. Furthermore, we integrated semantic probability to perform hierarchical and dynamic adjustments to the saliency of queries, thereby refining the semantic interaction among foreground queries. Experimental results demonstrate that on the challenging task-specific VisDrone dataset, our pDETR method enhances AP by 1.8% compared to DINO. On the COCO 2017 dataset, the performance improvement of pDETR is even more apparent, achieving a +2.5% increase in AP: under the 1× schedule, pDETR attains an AP of 51.5%, and under the 2× schedule, the AP further increases to 52.0%. Moreover, it exhibits faster convergence, exceeding 40% AP in just 2 training epochs while reducing computational cost by 13% in terms of FLOPs, indicating superior detection capability.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper presents a DETR-based object detection framework, named pDETR, which enhances the existing sparse strategies. The approach offers two key contributions. First, it introduces a Perspective Proposal Module that utilizes the spatial positional information of high-level foreground features to guide sparse sampling in lower-level features, thereby ensuring consistency across different feature levels. Second, it incorporates a module that dynamically adjusts the importance of each token based on target categories within the foreground regions. The experimental results demonstrate that pDETR performs better on the COCO val2017 and VisDrone datasets.

### Strengths
1. pDETR aims to maintain consistency across feature levels through its perspective proposal module and demonstrates enhanced recognition capabilities, particularly for small objects.
2. pDETR is simple and effective, facilitating ease of implementation and comprehension while surpassing the performance of previous methods.

### Weaknesses
1. Unfair Comparison: The baseline presented in Table 3, which does not incorporate the proposed method, already surpasses the Salinece-DETR listed in Table 2. Does this mean that pDETR is trained using different strategies or configurations? Such discrepancies pose an unfair comparison with previous methods and obscure the core contributions of the proposed methods. Besides, FLOPs is missing from Table 2. As an important metric for assessing how many tokens participate in the attention in Encoder, its inclusion is crucial.
2. Insufficient Experiments: The ablation study should include a comparison between the proposed perspective proposal using the positions of current-level foreground tokens and the use of high-level foreground tokens to guide the sampling process. This comparison would more effectively demonstrate the efficacy of the perspective proposal module.
3. Trivial Improvement: As shown in Table 2, pDETR outperforms Salience-DETR by only 0.8 in AP after 24 epochs of training, whereas it achieves a more significant improvement of 2.3 in AP with only 12 epochs of training. This indicates that a shorter training schedule can amplify the observed improvements. So for the ablation results presented in Table 3, the 0.7 gains from combining the perspective proposal module and semantic module may decline with the extended training schedule. These modules may just help the model converge faster but not better. 
4. Fact error: Line 399, $AP_s$ of the model equipped with the semantic-aware module declines to 35.1%, rather than improving. This indicates that the semantic module fails to capture the semantics of small objects.
5. Other minor comments and corrections:  
Line 65: $F_x$ should be explained in the caption of Figure 1.  
Line 123:  Efficient-DETR reference error.  
Line 396: 34.4% is not shown in Table 3, it is 33.9%.  
Line 457: figure 4 reference format error, it should be Figure 4.  
Line 622: A.2 is mostly the same as 3.3, so why not just explain them in 3.3 while there is still room in the main paper?  
Line 671: Querie should be Queries.

### Questions
Please see weakness section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper presents pDETR, an end-to-end object detection model that builds upon DETR by introducing Perspective Proposal and Semantic-Aware Modules. These components aim to address DETR's limitations, such as high computational cost and poor convergence on sparse representations, by selecting and refining critical features at multiple scales. Experimental results show that pDETR achieves improved accuracy and efficiency over state-of-the-art models, particularly on challenging datasets like VisDrone and in detecting small objects.

### Strengths
1. pDETR contributes significantly to research on sparse networks by applying sparse encoding techniques within Transformers, providing a notable improvement in reducing computational redundancy for DETR-based models.

2. The method has been tested on two datasets, COCO and VisDrone, demonstrating a degree of generalization across different object detection scenarios and complex backgrounds.

3.  By using the Perspective Proposal and Semantic-Aware modules, pDETR demonstrates superior performance in small object detection, effectively focusing on important features in challenging scenes.

### Weaknesses
1. The Method section is somewhat rushed, with certain formula inconsistencies (e.g., in Equation 6 where both "m" and "C" are used to represent categories) that could be simplified. Additionally, the difference between Layer-by-Layer and Top-Layer  Perspective is not clearly explained, leaving readers unclear about how each approach functions.

2. Considering that FLOPs and FPS are critical in sparse network research, the paper does not sufficiently highlight this comparison. Although FLOPs for pDETR are listed in Table 7, a direct comparison with other sparse models is missing.

3.  Figure 2 has large visual elements but lacks clarity in conveying the purpose of Background Embedding, and overall, it contributes little to understanding the method. A more detailed or concise illustration could improve the reader's comprehension of the contributions.

### Questions
1. Both approaches are based on position mapping, yet they produce different results. Could you provide a more detailed analysis of their mechanisms, the rationale behind the Layer-by-Layer approach, and how it improves feature consistency through stepwise information transfer?

2. While Salience-DETR applies different sampling rates across layers, pDETR appears to standardize sampling rates between feature maps and encoder layers, yet still achieves better results. Please explain how pDETR balances feature preservation and computational efficiency without varying sampling rates.

3. In Equation 7, are the Q, K, V matrices all token features across layers? If so, would this not cause dimensional mismatches when summing? Additionally, what specific role does $Q^{′}_{T^l}$ play within the encoder, and are similar operations applied to other layers? Further clarification on  $Q^{′}_{T^l}$’s role in encoding would enhance understanding of its functional purpose within the model.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper aims at constructing consistent representations of multi-level features for end-to-end object detection. The authors introduce a perspective proposal module and a semantic aware module to leverage spatial information and semantic probability. Experiments on COCO and VisDrone are shown.

### Strengths
- The ablation study shows the effectiveness of proposed modules.
- The experimental results are given on two different datasets COCO and VisDrone.

### Weaknesses
- The novelty of proposed method is limited. Both two modules are too straightforward without insightful design.
- The proposed method has limited improvement compared to other methods. Moreover, some related methods, such as DDQ DETR and Relation DERT are not compared. These two methods seem having similar performance as the proposed method.
- This paper only reports the performance using R50. It is necessary to give the experiments using R101 and Swin backbones.
- How about the performance on COCO test set.
- It seems that, compared to the baseline method, the proposed method increased computational cost.

### Questions
- The authors can provide more comparisons with the missing methods mentioned in weakness.
- The authors can provide experiments like using different backbones and computational cost.

### Soundness
3

### Presentation
2

### Contribution
2
