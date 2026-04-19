# DSEG-LIME: Improving Image Explanation by Hierarchical Data-Driven Segmentation

- Decision: Reject
- Scores: 6, 3, 8, 3

## Abstract
Explainable Artificial Intelligence (XAI) is crucial in unraveling decision-making processes in complex machine learning models. LIME (Local Interpretable Model-agnostic Explanations) is a well-known XAI framework for image analysis. It utilizes image segmentation to create features to identify relevant areas for classification. Consequently, poor segmentation can compromise the consistency of the explanation and undermine the importance of the segments, affecting the overall interpretability. Addressing these challenges, we introduce DSEG-LIME (Data-Driven Segmentation LIME), featuring: i) a data-driven segmentation for human-recognized feature generation by foundation model integration, and ii) a user steered granularity in the hierarchical segmentation procedure through composition. We evaluate DSEG-LIME on pre-trained models using ImageNet classes, explicitly targeting scenarios without domain-specific knowledge. Our findings demonstrate that DSEG outperforms most of the XAI metrics and enhances the alignment of explanations with human-recognized concepts, significantly improving interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper describes DSEG-LIME, an improvement over the standard eXplainable AI technique LIME that uses a data-driven, AI-based, algorithm to generate the segmentation needed by LIME to assign feature attribution scores to the (super)-pixels of an image.

The paper is well presented and in the scope of the conference topics, even if it does not fully describe the impact of the presented technique in the context of learnt representations.

The main weakness (see W.2 below) is the fact that the proposed technique operates on an hierarchical segmentation with overlapping regions, which is a very complex and tricky thing to use. In fact, the LIME technique does not seem to be formulated to operate on overlapping features, as it implicitly assumes that segments are independent and form a partitioning of the pixels. 
Such critical change in the LIME assumptions is not well motived, and requires additional explanations.

### Strengths
The paper is well motivated.
The addressed issue (data-driven XAI) is an important component missing in current state-of-the-art XAI methods.

### Weaknesses
**W.1**

Images of the LIME explanations apparently show the first n segments, using the very poor get_image_and_mask() method of LIME. In practice, this approach does not allow to fully understand if the method is correctly assigning large feature attribution to the right regions. A full assessment should entirely and exclusively focus on feature attribution maps, as briefly mentioned in Appendix B.5.

**W.2**

The hierarchical technique is described briefly, but does not come with a clear formulation or pseudo-code. While the intuition is somewhat clear, there are several critical details that are needed and are missing in the presentation for a full assessment.
In fact, LIME is not structurally hierarchical, and such addition would require much more than just a couple of mentions.

**W.3**

A missing analysis of DSEG-LIME is the concept of multi-scale: in principle the XAI method does not know in advance the dimension of the relevant region(s) in the input image. Your proposal account for that using some arbitrary number of iterations d of the algorithm. However, identifying how many and how large the relevant region(s) are is a very relevant problem, which does not seem to be conceptualized and approached very solidly. 
A related part of this issue deals with the fact that you visualize the explanations using the first n segments instead of using feature attribution maps, which hide the actual computed relevance of the segments, and their number and areas. With this approach, the scale of the relevant regions is not well identified.

**W.4**

In Eq. (1) you use a distance function D(), which actually count the number of masked segments. Why is this a fundamental component (181)? I understand that you are using the equations of LIME (which adopt this distance function), but I do not see why this D() function should be actually relevant for your case. Do you have an explanation for why it is so critical to weight samples according to the number of masked segments? This means that the more you mask the image and focus on a specific region, the least the relevance of this sample is in the trained model.
One would guess that D() has actually very little role in the trained model, and it should actually be removed from the equation itself.

**W.5**

A critical component of LIME relates the replacement values that are used to mask the segments. There is very little analysis and reporting of that issue in the paper.

**W.6**

You report that a weakness of SAM is that it may generate overlapping segments, which makes it somewhat troublesome to deal with. The paragraph in 283-301 is not very clear about what your method does on the overlapping segments.
This is particularly critical, as the LIME method assigns feature attribution scores implicitly assuming that segments are non-overlapping partitions. If segments overlap, it is unclear what feature attribution scores means.

**W.7**

(270) What is a grid overlay parameterized by points? 
Please clarify and/or add references.

**W.8**

(1384) Why are you mentioning a "NeurIPS code of Ethics" in a paper submitted to ICLR?

### Questions
**Q.1**
(see W.1) I would strongly suggest to use feature attribution maps to show the generated explanations among all the presented examples in the paper, instead of selecting the n most relevant segments, which is really a poor and arbitrary criteria. It does not really matter that this poor selection criteria of get_image_and_mask() was used in the 2016 LIME paper.

**Q.2**
(see W.2) Please provide a pseudo-code or a clear formulation of your hierarchical extension of LIME dealing with overlapping segments. This can be added in the Appendix.

**Q.3**
(see W.3) Is the identification of the scale of the relevant region totally left to the user which has to tune the d parameter accordingly?
Please, add a clarification to that issue.

**Q.4**
(see W.4) Can you provide intuitions why it is relevant to have the distance function D()?

**Q.6**
(see W.6) Management of overlapping regions is unclear, and require a full formulation and pseudo-code (can be added in the Appendix). 
Dealing with overlapping hierarchical structures is very complex and tricky, and the paper provides very few details about how to tackle such complexity.

- Clarify what you do when you have two feature attribution scores, computed on two segments that overlap. How do you decide which one is more relevant, since the scores are not independent?
- Can you still determine a pixel-level relevance, if segments may overlap? How do you generate the images in Figure 14 when you deal with overlapping segments?
- What happens if the n topmost segments have a non-trivial overlapping region? 

**Minor issues**

- (see W.5) Better clarify which replacements values you are using in the experiments (line 176 is very generic and uninformative).
- Address W.7 and W.8.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes DSEG-LIME, an enhanced version of the LIME framework aimed at improving explainability for image classification tasks. By introducing a segmentation approach, DSEG-LIME leverages SAM to create human-recognizable features, enhancing the consistency and interpretability of LIME's explanations. Additionally, DSEG-LIME incorporates user-controlled segmentation granularity, allowing adjustments to the segmentation hierarchy to suit varying interpretability needs. The authors evaluate DSEG-LIME on pre-trained models with ImageNet classes, particularly focusing on applications without domain-specific knowledge.

### Strengths
The integration of a segmentation process aims to align explanations with human-recognizable features, potentially enhancing LIME's interpretability and making explanations more intuitive for end-users.

By allowing users to adjust segmentation granularity, the method provides flexibility, catering to various interpretability requirements across different use cases.

### Weaknesses
1. The novelty of this manuscript appears minimal, as it mainly combines elements from SAM and LIME without significant advancements.

2. The authors compare their method with traditional segmentation techniques, including Simple Linear Iterative Clustering (SLIC), Quickshift (QS), Felzenszwalb (FS), and Watershed. This comparison seems inadequate for assessing the effectiveness of the proposed DSEG as an XAI method. If the advantage of DSEG is due to SAM outperforming SLIC, QS, or FS, then the unique contribution of DSEG itself remains unclear. A more relevant comparison with other XAI methods is needed.

3. The authors set a threshold 𝜃=500 to filter out small masks, which is problematic, as it risks excluding areas that may contain target of interests. This thresholding strategy should be reconsidered or justified more robustly.

4. Additional visual explanations comparing DSEG with current state-of-the-art XAI methods would strengthen the manuscript by illustrating the practical benefits and effectiveness of the proposed approach.

### Questions
One of LIME’s main weaknesses is its instability. It is unclear if applying DSEG improves LIME’s stability or overall performance, as the results do not appear to outperform other methods. The authors should clarify this aspect and consider adding metrics like Local Fidelity from GLIME for a more comprehensive evaluation.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper discusses LIME's stability problem and hypothesizes that the segmentation step is a cause for it. According to the authors, conventional segmentations are graph or clustering-based and not designed to distinguish between different objects in an image. Keeping that in mind, the authors have proposed using a transformed-based foundational model. This enables LIME's feature generation to be adapted to a novel hierarchical segmentation. Further, the authors support their claims with quantitative and qualitative studies.

### Strengths
1) Paper discusses about LIME stability which is a major concern to users who want to use LIME.
2) The paper performs quantative and qualitative evaluations.
3) The authors have considered a variety of models.
4) The authors integrated their proposed method, DSEG with LIME and it variants.

### Weaknesses
1) The literature review needs to be revised as DLIME (https://arxiv.org/abs/1906.10263) also discusses using hierarchical clustering with LIME. The authors should highlight the key differences between the proposed approach and DLIME.

2) The consistency (specially the Rep. Stability) aspect has not been investigated as there exists prior art (https://openaccess.thecvf.com/content/CVPR2024/html/Bora_SLICE_Stabilized_LIME_for_Consistent_Explanations_for_Image_Classification_CVPR_2024_paper.html) which seems to stabilize LIME explanations without investigating into the segmentation aspect. It would benefit readers if the authors explained how DSEG complements or differs from stabilization methods like SLICE and explained their rationale for focusing on segmentation rather than other stabilization techniques.

3) From Table 1 and Table 2, we can see that DSEG cannot ensure 100% consistency. Hence, I need help understanding how the other metrics (except consistency) were calculated. Does the author use the mean score of 8 runs? To improve clarity, the authors should provide more details on how they calculated metrics across multiple runs, including whether they used mean scores and how they handled any inconsistencies.

4) I need help understanding why BayLIME with non-informative prior was chosen. The authors of BayLIME showed that their method with non-informative prior was almost indistinguishable from LIME. Hence, it would be helpful for readers if the authors explained their rationale for using BayLIME with a non-informative prior and discussed how using a partial or full informative prior might impact their results.

5) The final dataset consists of 50 images, which may not be sufficient. Also, more than the number of runs (i.e., 8) may be needed for the Rep. Stability test. Hence, I would like the authors to justify their choice of dataset size and number of runs by referencing similar studies or discussing statistical power considerations. There is mention of t-statistic and p-value for user-study results, but it would be helpful for readers if the authors explained why they did not use the same for the quantitative evaluation. Further, the authors should also describe their choice of the test and the alternative hypothesis used for ease of readers and complement it with effect size.  

6) For Consistency, SLIC is at par (or very close) with DSEG in many situations (Table 1 and Table 2), (Table 6 and Table 7). It would be helpful to readers if authors explain why SLIC performs similarly to DSEG in terms of consistency in some situations and discuss potential factors that might contribute to this observation.

### Questions
I request the authors to look at the weaknesses and provide their comments. I think the paper addresses an important aspect of LIME and the proposed approach has merits, but some aspects need polishing. I would like to know the authors' comments regarding my questions.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper introduces a new approach for explaining image-classifier models by using hierarchical segmentation, which provides explanations at multiple levels of granularity. This method leverages foundation models to extract concepts that are more aligned with human perception, creating explanations that are easier to understand and interpret. Unlike traditional LIME, which often faces challenges in achieving both human interpretability and faithful representations of model behavior, this approach enhances explainability by structuring explanations in a way that mirrors human visual processing.

### Strengths
- It is an interesting approach with real problems to address.
- It effectively leverages SAM’s capabilities for hierarchical segmentation, enhancing explainability by capturing meaningful image segments.
- The user study demonstrates that the extracted concepts are recognizable and understandable for humans.
- Empirical results support the effectiveness of the proposed method in generating more interpretable and faithful explanations compared to traditional techniques.

### Weaknesses
- Does SAM consider an image as an input or extracted features from a classifier as an input, figure 2 points at extracted features, while equation 3 points at input image?
- In the spirit of LIME the surrogate model is trained to mimic the behaviour of the given classifier, but in this scenario, the surrogate model not only considers true model, but also SAM masks with input.
    - Given the above statement, the conducted human evaluation states SAM’s capabilities to capture human-recognisable features and not the classifiers perception of these concepts, I would like to know authors thoughts on this?
    - Additionally, DSEG offsets computational cost of extracting explanations as compared to LIME
- I would like to know authors view-point on explaining one deep-learning model using another much deeper more complex model?
- The output from the foundation model isn’t always human-interpretable, highlighting the need for more quantitative studies to evaluate interpretability in greater depth.

### Questions
see weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
