# Shapley Image Explanations With Data-Aware Binary Partition Trees

- Decision: Reject
- Scores: 3, 6, 6, 6

## Abstract
Extracting a visual interpretation of a learned representation of a machine learning model applied to image data is a relevant task in eXplainable AI (XAI). 
Pixel-level feature attributions serve as a valuable tool in this context, as they identify the regions within an image responsible for the classification outcome.
The hierarchical Owen approximation of the Shapley values has proved to be an effective strategy for this task. 
However, existing approaches lack data-awareness, leading to poor alignment between the pixel-level attributions and the actual morphological features of the classified image.

This paper introduces ShapBPT, a novel XAI method that computes the Owen approximation of the Shapley coefficients following a data-aware binary hierarchical coalition structure derived from the Binary Partition Tree computer vision algorithm. 
By aligning with the morphological features of the image, the proposed method significantly enhances the identification of relevant image regions.
Experimental results confirm the effectiveness of the proposed method.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes an XAI method which combines the SHAP method with a Binary Partition Tree (BPT) representation of an image. The BPT is used to segment the image into regions which are masked to determine their contribution to a model’s classification decision. The proposed method is evaluated by comparing the selected “important” regions to the ground-truth object segment in an object classification task. The proposed method outperforms methods which segment the image into superpixels or grids.

### Strengths
- The paper is clearly written and the results are well-presented. In particular, the figures are well designed to illustrate how the method works.
- The paper does include an experiment where the model's ground truth is known (E2), which is something that's lacking in a lot of XAI research. I have some concerns with this experiment (the chosen ground truth seems to strongly favor this model over the comparison models), but it's still nice to see this kind of evaluation in an XAI paper.

### Weaknesses
- The main novelty seems to be using BPT instead of a simpler image segmentation like a grid or superpixels.
- The method is only compared to methods that use these simpler segmentation approaches. It's not surprising that this approach performs better.
- The “ground truth” explanation region used in the evaluations is a human-annotated semantic segmentation of the object (Gao, et al., 2022). This probably isn’t the actual ground truth region for a classification model -- it's likely a model focuses on just a few parts of the segment that are most critical for recognition -- but this choice of ground truth seems to clearly favor a model that can produce object-like segments over models that only use very simple segments like grids or superpixels. E2 seems to be intended to address this issue, but E2 also uses semantic segments as the ground truth "important" region for the model. It would be more convincing to show that this model still work better than other approaches when the ground truth is not a semantic segment.

### Questions
- How well does this method work when the "ground truth" important region is not a whole object or whole semantic segment (what if the critical feature is an edge or corner or just part of an object)? This approach seems to assume that connected same-color regions are the same thing and contribute equally to importance, and it's only tested with data that matches this assumption.
- Why is it not possible to swap out BPT with other segmentation methods? The paper says that "fixed" segmentations cannot be used but most segmentation methods allow hierarchical merging; it wasn't particularly clear to me why none of them are applicable in approach. It does seem like a potential limitation of this approach that the user cannot use alternate segmentation algorithms.

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
4

### Summary
This paper presents an explainable AI (XAI) method that combines the SHAP partition explainer with the partition hierarchy of the BPT algorithm. While the idea of integrating these two algorithms is straightforward, it results in greater focus on the object region compared to previous methods, as evidenced by numerical experimental results.

### Strengths
+ The technical details are carefully and thoroughly described. 
+ Although the proposed method simply combines SHAP and BPT algorithms, it shows great promise based on the qualitative comparison in Figure 4 and the quantitative comparisons in Figures 4 and 5. This combination is novel in the context of XAI and demonstrates effective results.
+ All code and notebooks required to reproduce the paper's results are provided and will be made publicly available upon acceptance, facilitating readers’ understanding of the method. Additionally, the proposed XAI method can be immediately implemented by users.
+ The discussion of alternative candidate modules for each stage in Section 5 further enhances understanding of the approach presented in this paper.

### Weaknesses
- I have a concern about the ground truth in Figure 4. It seems that the authors are claiming that high scores should be assigned to the target object only and no score should be assigned to any other region, with which I do not fully agree. In some cases, the background information or high-order interactions between the object and the background should also be important and considered. However, in the definition in this paper, such cases are eliminated. Here are my questions or requests:
  - The authors might want to justify their “Truth.”
  - The authors might want to discuss the limitations of their ground truth definition and consider how their method might be evaluated in cases where background or contextual information is important for classification.
  - The authors might want to discuss using alternative ground truth definitions that account for these factors.

- It seems there is a strong assumption: there is only one object in the target image. Here, I have some questions to the authors
  - What kind of responses does the proposed model would yield if there are multiple objects such as two or three butterflies in the image in the case of Figure 4(a)?
  - What would happen if this model is applied to subjective classification tasks such as good/bad quality photos or to regression tasks such as image aesthetics score predictions?

- The choice of the BPT algorithm should be justified. For instance, applying the segment anything model (SAM) or bilateral filter-based region segmentation in the pre-processing stage would also be possible rather than using the BPT algorithm. It would be helpful if the authors could add a more detailed explanation why the BPT algorithm was employed or additional experiments.

Here are some comments to improve the paper (not considered for the paper score):
- There are many mistakes such as undefined abbreviations, grammatical errors, inconsistent reference styles, etc.
- x in Section 2 is preferred to be a bold face because it is a vector.
- Figure 2 is referred to only from the appendix. Therefore, the authors might want to reconsider if this figure is really needed here. Or, if this figure is important to convey the concept of BPT, the authors might want to add more detailed description with proper reference in the main body.
- The authors might want to better illustrate Figure 5 because the current figure is too busy and the values are not readable.

### Questions
Please answer my comments and questions in the Weakness part.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper discusses about aligning morphological features of an image to enhance the identification of relevant regions for extracting a visual interpretation of a machine learning model for XAI. The authors try to mitigate the poor alignment of pixel level attribution and the morphological features of an image and propose a method to compute the Owen's approximation of Shapley coefficients following a hierarchical coalation structure from BPT.

### Strengths
1) Discusses an important aspect of XAI i.e. poor alignment of pixel level attribution and morphological features.
2) Shows experiments with several settings.
3) Considers a wide range of metrics.
4) Discusses about 5 different replacement values (although in Supplementary) and evaluates the same. 
5) Provided code in the form of library for reproducibility.

### Weaknesses
1) The paper's organization could be clearer, and it isn't easy to understand, especially the section on experimental assessment.

2) For the reader's benefit, it would be nice to put some markers expressing whether the metrics' higher or lower value is better or worse. For example, in the captions of Figures 5, 7, 9, etc. 

3) The authors talk about the ANOVA test (in lines 403-409) but don't refer to any table (in the paper or supplementary) related to the test. Please include a table in supplementary or write about the range of p-values so that the readers can get some idea.  

4) The evaluation was done on 574 images, which may have needed more. It would be great if the authors could cite previous work using similar dataset sizes for similar evaluations.

5) In Figure 5, specific methods (e.g., IDG in AUC-, GradCAM in AU-IoU) appear to outperform the proposed approach, but the distributions seem to have high overlap. For the benefit of readers, it would be helpful to include statistical test (like Wilcoxon signed rank test or paired t-test) to provide a clearer understanding of whether the observed differences are statistically significant or not.

### Questions
The paper discusses an important aspect, namely, aligning pixel-level attribution with morphological features of an image, and that's why I am leaning towards mild acceptance. However, I would like to know the authors' comments regarding the points I mentioned in the weakness section.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces ShapBPT, a novel method for explainable AI designed to calculate an Owen approximation of Shapley coefficients. The approach employs a data-aware binary hierarchical structure, derived from a Binary Partition Tree (BPT), to improve the accuracy and efficiency of Shapley value computation. The authors validate the effectiveness of ShapBPT through extensive theoretical analysis and experimental results.

### Strengths
- The paper is clear and easy to follow
- The proposed method is novel and makes a meaningful contribution to the field of explainable AI by advancing Shapley-based explanation techniques. 
- Theoretical insights are paired with empirical validation, demonstrating the efficacy of the approach across various settings.

### Weaknesses
- The method is only evaluated on pretrained classification models with different architectures, without comparisons to other training methods, such as supervised and self-supervised approaches. 
- The experiments focus on image-based explanations, with no exploration of potential applications to video classification tasks.

### Questions
- How might the findings from 1 on the limitations of Shapley values affect the explainability claims made in this work? 
- Did the authors consider a comparison with attention-based mechanisms that identify image regions contributing to classification decisions? 

1. On the failings of Shapley values for explainability
https://www.sciencedirect.com/science/article/abs/pii/S0888613X23002438 
2. Attention-based Interpretability with Concept Transformers https://research.ibm.com/publications/attention-based-interpretability-with-concept-transformers
3. Attention Flows are Shapley Value Explanations https://arxiv.org/pdf/2105.14652

### Soundness
3

### Presentation
3

### Contribution
3
