# Revisit the Open Nature of Open Vocabulary Semantic Segmentation

- Decision: Accept (Poster)
- Scores: 6, 8, 5

## Abstract
In Open Vocabulary Semantic Segmentation (OVS), we observe a consistent drop
in model performance as the query vocabulary set expands, especially when it
includes semantically similar and ambiguous vocabularies, such as ‘sofa’ and
‘couch’. The previous OVS evaluation protocol, however, does not account for
such ambiguity, as any mismatch between model-predicted and human-annotated
pairs is simply treated as incorrect on a pixel-wise basis. This contradicts the open
nature of OVS, where ambiguous categories may both be correct from an open-
world perspective. To address this, in this work, we study the open nature of OVS
and propose a mask-wise evaluation protocol that is based on matched and mis-
matched mask pairs between prediction and annotation respectively. Extensive
experimental evaluations show that the proposed mask-wise protocol provides a
more effective and reliable evaluation framework for OVS models compared to the
previous pixel-wise approach on the perspective of open-world. Moreover, analy-
sis of mismatched mask pairs reveals that a large amount of ambiguous categories
exist in commonly used OVS datasets. Interestingly, we find that reducing these
ambiguities during both training and inference enhances capabilities of OVS mod-
els. These findings and the new evaluation protocol encourage further exploration
of the open nature of OVS, as well as broader open-world challenges. Project page: https://qiming-huang.github.io/RevisitOVS/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper gives a deep observations on open-vocabulary semantic segmentation. To address the ambiguous category issue, the authors propose mask-wise evaluation protocol and a confusion vocabulary graph for open-vocabulary datasets. The experiments validate method defectiveness.

### Strengths
1. The paper presents an interesting analysis on the openness of open-vocabulary semantic segmentation.

2. The mask-wise evaluation protocol sounds reasonable.

3. The experiments are conducted on multiple existing methods.

### Weaknesses
1. The quality of ambiguous vocabulary graph seems important for performance. Currently, the related experiments are not enough. I think it is better to provide more experiments to verify the quality of ambiguous vocabulary graph.

2. The accuracy for front and back is not very clear. I suggest that the authors give an equation to explain it.

3. The comparison of whether reducing ambiguities during training or not is necessary.

### Questions
Please refer to weakness.  It is important to give more experiments for ambiguous vocabulary graph and more comparsion.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The performance of Open Vocabulary Segmentation (OVS) models will decrease as the query vocabulary size increases, especially when semantically similar category names are present, contradicting the original purpose of OVS. To address this, the authors proposed a mask-wise evaluation protocol based on match/mismatch between prediction and annotation mask pairs, avoiding forced category matching. Key innovations include reducing ambiguity and constructing an ambiguous vocabulary graph. Comprehensive experiments and analysis reveal numerous ambiguous categories in current OVS datasets. Utilizing the proposed protocols during the training and testing stages can help to improve the model’s zero-shot inference capability.

### Strengths
1. Good motivation, authors pointed out the current OVS evaluation sets have many semantic similar categories, which may influence the training&testing stages of model, which further influence the inference ability of current OVS methods. Based on this, authors proposed a new evaluation protocols to alleviate this issue.

2. The whole paper is relatively clear and easy to follow.  

3. Very comprehensive experiment results on multiple datasets and multiple OVS methods.

### Weaknesses
Writing Suggestions: 

1. In the Abstract, authors claim that OVS models perform better under the new mask-wise protocol needs further clarification. To make fair comparisons between the mask-wise and pixel-wise protocols, the authors should add more details about how they determine "better" performance. Providing such details would help readers understand the basis for this improvement claim.

2. In the Abstract, the phrase “enhances zero-shot inference capabilities” likely refers to the capabilities of OVS models. Clarifying this would improve readability. 

3. Given the similarity between open-vocabulary segmentation and open-vocabulary semantic segmentation, the authors should add a brief section comparing these two concepts. Highlighting key differences in their applications or objectives would help avoid potential confusion and clarify the unique focus of their work.

4. For Equation (5), the authors should provide more detailed motivation for choosing this to determine the best threshold, rather than simply listing the source. It would be helpful if they could explain why this method was selected over alternative approaches and how it specifically benefits their evaluation protocol.

5. The equation at lines 324 to 327 is missing a number.

### Questions
1. A significant concern is that the proposed evaluation protocol relies on having sufficient data to identify semantically similar categories. In real-world applications, if the training data lacks adequate masks to differentiate similar categories (e.g., "sofa" and "couch"), the protocol may struggle during testing. To address this, it would be helpful if the authors could analyze the performance of their method with limited training data or provide insights into the minimum data requirements necessary for effective improvement. Additionally, experiments or discussions on the robustness of data scarcity and the impact of potentially misleading information would strengthen the evaluation.


2. While the authors' approach to handling ambiguities through the visual modality is quite interesting, it may be more intuitive to identify similar categories based purely on semantic meaning. For instance, using the text modality to assess semantic similarities could potentially provide greater improvements than relying solely on visual information. To explore this, it would be valuable for the authors to compare their visual-based approach with a text-based semantic similarity approach. Or add more discussions about the potential advantages and disadvantages of incorporating textual semantic information into their method.

### Soundness
3

### Presentation
3

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
This study proposes new evaluation metrics for Open-Vocabulary Segmentation (OVS) tasks. A key limitation of evaluating OVS methods on fixed-category datasets is that traditional image segmentation metrics may misclassify visually similar objects as errors, even when they are semantically related but belong to different categories. This issue intensifies with an increasing number of category labels in the test dataset. This issue becomes more pronounced as the number of category labels in the test data increases. Previous research has addressed this challenge, resulting in improved metrics such as Open-mIoU and SG-IOU. The central premise of this work is to focus evaluation on mask similarity rather than textual similarity.

### Strengths
The primary contention of this manuscript is to shift the focus of evaluation from textual to mask similarity in assessing OVS models. The authors have identified a gap in the current assessment metrics, which are deemed inadequate for evaluating OVS models, and have proposed a novel metric to address this issue.

### Weaknesses
The manuscript exhibits a lack of clarity and organization in its writing.

### Questions
Q1: The analysis in Section 3 appears disconnected from subsequent sections.

Q2: In Figure 2, $\mathbb{A}$ represents a set of predicted binary masks. How are the predicted masks in $\mathbb{B}$ and $\mathbb{C}$ derived from $\mathbb{A}$? If they are matched to GT masks based on IoU using bipartite matching, it seems Figure 2 suggests that the number of predicted masks by the model exceeds that of the ground truth, which is not realistic. Additionally, predicted masks in $\mathbb{B}$ and $\mathbb{C}$ should not overlap according to $\mathbb{C} = \mathbb{A} \backslash \mathbb{B}$.

Q3: The correlation between Algorithm 1 and Section 4 is weak: For example, (1) The CM is not referenced outside the Algorthm 1. (2) The calculations for the core evaluation metrics -- front, back, and errors -- are not represented in Algorithm 1 or any other equations. (3) How is the best threshold $\tau^*$ used in Algorithm 1? 

Q4: What constitutes a good evaluation metric? The last sentence of the introduction (line 83 on page 2) implies that the authors equate higher performance values with better evaluation metrics, which is unreasonable. 
In Figure 3, the authors seem to suggest that more stable evaluation metrics are preferable; however, this should also be compared with other metrics like Open-mIoU and SG-IoU.

### Soundness
3

### Presentation
2

### Contribution
2
