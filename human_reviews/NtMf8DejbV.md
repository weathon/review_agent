# Segment as You Wish: Free-Form Language-Based Segmentation for Medical Images

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 1, 3, 3

## Abstract
Medical imaging is crucial for diagnosing a patient’s health condition, and accurate segmentation of these images is essential for isolating regions of interest to ensure precise diagnosis and treatment planning. Existing methods primarily rely on bounding boxes or point-based prompts, while few have explored text-related prompts, despite clinicians often describing their observations and instructions in natural language. To address this gap, we first propose a RAG-based free-form text prompt generator, that leverages the domain corpus to generate diverse and realistic descriptions. Then, we introduce FLanS, a novel medical image segmentation model that handles various free-form text prompts, including professional anatomy-informed queries, anatomy-agnostic position-driven queries, or anatomy-agnostic size-driven queries. Additionally, our model also incorporates a symmetry-aware canonicalization module to ensure consistent, accurate segmentations across varying scan orientations and reduce confusion between the anatomical position of an organ and its appearance in the scan. FLanS is trained on a large-scale dataset of over 100k medical images from 7 public datasets. Comprehensive experiments demonstrate the model’s superior language understanding and segmentation precision, along with a deep comprehension of the relationship between them, outperforming SOTA baselines on both in-domain and out-of-domain datasets.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper presents FLanS, a novel medical image segmentation model that can understand and respond to free-form text prompts. It features a RAG-based text prompt generator and a symmetry-aware canonicalization module.

### Strengths
FLanS is trained on over 100k images and demonstrates strong language understanding and segmentation accuracy across various datasets. The key contributions are:
A RAG-based generator for diverse text prompts.
The FLanS model for text-driven medical image segmentation.
A symmetry-aware module for consistent segmentation across different scan orientations.

### Weaknesses
1. Comparison to the state-of-the-art baselines from natural images in the tasks like RES(Referring Expression Segmentation)[1]. The paper claims superior performance over SOTA baselines. However, it would be beneficial to see more direct comparisons, including visual examples and error analysis, to better understand where and how FLanS outperforms existing methods.
2. Lack of theoretical analysis, as a paper submitted to ICLR.
3.More details of making such a dataset.


[1] Liu, Chang, Henghui Ding, and Xudong Jiang. "GRES: Generalized referring expression segmentation."Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

### Questions
1. Comparison to the state-of-the-art baselines from natural images in the tasks like RES(Referring Expression Segmentation)[1]. The paper claims superior performance over SOTA baselines. However, it would be beneficial to see more direct comparisons, including visual examples and error analysis, to better understand where and how FLanS outperforms existing methods.
2. Lack of theoretical analysis, as a paper submitted to ICLR.
3.More details of making such a dataset.


[1] Liu, Chang, Henghui Ding, and Xudong Jiang. "GRES: Generalized referring expression segmentation."Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
1

### Rating Number
1

### Confidence
5

### Summary
The authors introduce FLanS, a medical image segmentation model capable of handling diverse free-form text prompts, including both anatomy-informed and anatomy-agnostic descriptions. By integrating equivariance, it ensures accurate and consistent segmentation across varying scan orientations. Trained on over 100,000 medical images from seven public datasets covering 24 organ categories, FLanS outperforms baselines in both in-domain and out-of-domain tests.

### Strengths
1.	The paper is well-written, and the core idea is easy to follow.
2.	The proposed method is evaluated on diverse organ segmentation datasets.

### Weaknesses
## 1. Insufficient Motivation for FLanS:
The authors propose FLanS, a model designed to perform organ segmentation in CT scans based on free-form text prompts. However, the motivation behind this approach appears weak for the following reasons:

(1) Existing Tools: The authors claim that in real-world scenarios, clinicians often rely on natural language commands (e.g., "Highlight the right kidney" or "Segment the largest organ"). However, tools like TotalSegmentator[1,2,3] already segment most organs in CT and MRI scans effectively. These tools are widely integrated into platforms such as 3D Slicer[4], allowing clinicians to obtain segmentation results without needing text-based interactions.

(2) Questioning the Need: Given that TotalSegmentator covers a vast range of anatomical structures, the need for a text-based interaction model seems redundant. The authors do not provide a clear explanation of what organs or structures remain uncovered by existing solutions.

## 2. Addressable Challenges Without FLanS
The authors identify another challenge related to orientation differences between preprocessed datasets:

Orientation Standardization: The authors argue that varying scan orientations pose a problem that requires a "Canonicalized Network." However, this issue can be readily addressed with widely available Python libraries such as SimpleITK and nibabel, which can standardize 3D medical image orientations. Training an additional network specifically for orientation seems unnecessary and overcomplicated.

### Questions
In conclusion, the paper’s primary motivation for proposing FLanS is questionable. The authors' claim of a real-world need for text-prompt-based segmentation lacks sufficient justification, given that existing tools already handle the tasks effectively. Additionally, the technical challenges outlined, such as orientation standardization, can be addressed through simpler, more established methods, making the proposed solution less compelling.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
### Summary:
This paper introduces a free-form language-based segmentation algorithm termed FLanS. This model allows language-prompt-based segmentation of 24 different organs by creating text prompts in a RAG fashion. They differentiate into two type of prompts, anatomy-informed and anatomy-agnostic text prompts that are supposedly commonly used in practice and should allow for easier segmentation given a new clinical image. The authors evaluate their method on a variety of abdominal organ CT downstream datasets, namely FLARE22, WORD and RAOS showing improved performance over the chosen baselines.

### Strengths
I agree with the authors that language inclusions are currently underutilized in the domain of medical image segmentation and I like their creation of additional text reports through the RAG approach. Moreover, the description of position to steer predictions is novel.

### Weaknesses
I am highly skeptical that the current capabilities of the method are introducing new capabilities over existing methods in any meaningful way, which I will elaborate on in the following.

### Main concerns:
While I believe the integration of language to steer segmentation is very useful, I believe the author’s current use-case is not convincing at all. The following are the main points of critique:
1. Every clinician or experienced user is able to easily state which organ they are interested in. There is no added benefit of trying to predict which organ one refers to from a report description, this raises questions about the anatomy-informed prompt segmentation and also compromises the contribution of the RAG component that emulates clinical prompts.
2. The anatomy-agnostic setting is also not useable in the current format: In every clinical setting patients are present in 3D format. Given that this method is 2D, the “largest” organ will not be consistent across all 2D slices when one does whole-volume inference. So if one wants to use this method the clinician/user would have to adapt the slices to infer or create unique prompts for unique slices, which is both very unpractical.
3. Due to the proposed method being closed-set and being constrained to basic organ segmentation, any supervised model that does organ segmentation (and predicts all organs) would currently have a very similar capability, without the prior issues. The user would just have to discard all segmentations he is not interested in.
4. Regarding canonicalization, I would like to see the prevalence of these cases. Generally, all 3D images come with meta-information in the image header that should allow to re-orient it in a canonical way already. Hence I would like to have a quantification of how often this overall occurs to convince me that this is actually a problem worth solving.  
5. Evaluation: It currently seems like the authors are only testing their capabilities in a closed-set setting. The final anatomy-informed prompts are the same as they used during training. To actually show that their method provides novel capabilities experiments where they try to predict novel classes given the text guiding could be conducted, which would greatly improve the utility of their proposed method. 
6. Baselines: Since this is very close to supervised organ segmentation I would like to see supervised performance as reference (At least as a baseline to know where the performance is relative to a supervised 3D Model.

### Recommended improvements:
In my opinion, the utility of the method hinges entirely on being able to apply it in an open-set setting (a setting where they predict novel structures that they did not train on.  The way to show this would be to:
A) Predict a withheld/additional abdominal organ that was not trained on or
B) Predict a novel anatomy-agnostic prompt that they did not train on. An example for B could be: “Predict the organ above the liver.” Something that was not seen before and that could show that the model actually has learned more than the artificially created prompts.
C) Change predictive behavior for a certain organ. E.g.: “Predict the liver, but without the hepatic vessels.”
Any of these use-cases would actually bring novel capabilities, that would not be achievable or solvable by a simple supervised model. 
This would likely be much easier if the authors used pathologies and described the pathology "In the lower left lobe of the liver is a hyperintensity, please segment it for me" -- This would likely make it much easier to become open-set and do something that basic supervised methods cannot do.

Summarizing, I find the currently displayed capabilities and results not convincing, the author’s method seems to be capable of inferring which organ is desired given a prompt, which lacks overall motivation.

### Questions
Q1: What are the use cases where FLanS is preferable over a TotalSegmentator? / Why shouldn't I ask ChatGPT to predict the class label and then just have a script retrieve the TotalSegmentator mask instead? (For anatomy-informed prompts)
Q2: How did you account for the "Segment the leftmost organ" not changing when iterating along 2D slices? -- How would you imagine this prompt should be applied in practice?
Q3: Did you evaluate fully in 3D or did you only consider a 2D slice by itself?

If the authors are able to provide experiments regarding A, B or C in the rebuttal (should be do-able since it should be just about providing new prompts) or convince me that their free-form text organ segmentation shows benefits over a TotalSegmentator model I am certainly willing to raise my score.

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
This paper aims to explore the capability of text for segmentation in real clinical setting, the main contributions of this paper can be summarized as following:
1) Proposed a retrieval-augmented generation (RAG) based free form text prompt generator to enhance the segmentation performance
2) Proposed a symmetry-aware canonicalization module to ensure same scan orientations and reduce anatomical position
3) Demonstrated effectiveness on both in-domain and out-of-domain datasets

### Strengths
The strength of this paper can be summarized as follows:
1) It is interesting to see how the variable of text description affects the segmentation performance
2) Adapting RAG to generate description is an interesting way to tackle the generalization capability using text

### Weaknesses
The weakness of paper can be summarized as follows:
1) Lack of experiments of comparing with 2D current state-of-the-art supervised model (i.e. nn-UNet, Swin-UNet) 
2) The image orientation to train a model cannot be counted as a novel problem to handle, as it is not a problem and it is just a correction to the experiment setting.

### Questions
1) When we perform training with medical images, changing the orientation to a consistent orientation (i.e. RAS) is a usual step to preprcoess the data. It will be great if you can clarify why you think it is a clinical problem and needs a machine learning model to correct the orientation?

2) From Figure 3, you are using point / id to identify the organ semantic in the image and want to use the generator to correspond each organ with text, which is an interesting idea. However, how about some small organs in the abdominal region? It is challenging to describe all the organ location is, as they are nearby and is there any experimental setting that you have tried to adapt all organs?

3) Similar work have been done by adapting text into segmentation, please cite this paper in the related work and it will be great to see if there is an experiment comparison:
- Zhao, Theodore, et al. "BiomedParse: a biomedical foundation model for image parsing of everything everywhere all at once." arXiv preprint arXiv:2405.12971 (2024).

4) For Figure 7, the t-SNE plot for text-prompt embedding should be completely separable, because the text is pointing towards different organs. A meaningful visualization towards your model should be an attention map of computing the correlation between text and the image. It will be great to have the attention map that demonstrates the shape of the corresponding organs and show the correlated semantics between text and our segmentation target.

### Soundness
3

### Presentation
3

### Contribution
2
