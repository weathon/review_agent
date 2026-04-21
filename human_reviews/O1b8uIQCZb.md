# VEGA: Visual Expression Guidance for Referring Expression Segmentation

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3

## Abstract
Referring expression segmentation aims to segment a target object described by a given linguistic expression in an image. Unlike the unimodal segmentation taking predefined categories, this task takes the free-form linguistic expression that contains a single attribute or more than one attribute (e.g., location, color and action) related to the target object. However, the given linguistic information is only some part of information on the target object. In contrast, the image contains more additional information for the target object, including the unique information that is hard to describe in linguistic expression. Motivated by this, we propose a novel Visual Expression GuidAnce framework for referring expression segmentation, VEGA, which enables the network to refer to the visual expression that complements the linguistic expression information to improve the guidance capability. Since the image includes information related to both target and non-target regions, it needs to meticulously identify and selectively extract the useful information relevant to the target object. Therefore, we introduce a novel visual information selection module that flexibly selects the semantic visual information related to the target object to produce the visual expression, enhancing the adaptability to diverse linguistic and image contexts for robust segmentation. Furthermore, the proposed module allows each token of the visual expression to consider the visual contextual information by exploiting the global-local linguistic cues, thereby enhancing the capacity to understand the context of the target region. Our method consistently shows strong performance on three public benchmarks for referring expression segmentation, where it surpasses the existing state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors propose a visual expression guidance framework for referring expression segmentation, called VEGA. VEGA enables the network to refer to the visual expression that complements the linguistic expression information by providing relevant visual information to the target regions. A visual information selection module is introduced to select the semantic visual information related to the target regions, enhancing adaptability to various language expressions and image contexts. Experiments show the proposed method obtain good performance.

### Strengths
* The proposed method achieves good performance on three referring expression segmentation datasets.

* The paper writing is good and easy to follow.

### Weaknesses
* The innovation of the Visual Information Selection module is limited. K-Net [1] has already proposed a method for selecting top-k enhanced visual features in the universal segmentation field. SADLR applies the idea of K-Net to referring expression segmentation tasks, and PPMN [2] also applies top-k selection to enhance phrase features in a similar panoptic narrative grounding task. Therefore, the reviewers believe that this module has minimal differentiation from previous methods.

* The reviewers have some doubts about the implementation of the Visual Information Selection module. In Equation (4), S_norm already sets the similarity of pixel tokens not belonging to the top-k to 0, so why is it necessary to multiply it with M? Additionally, there are two multiplication operations in the middle part of Equation (4) that use the same symbol. If they are both element-wise multiplication or matrix multiplication, it seems incorrect in terms of dimensions. Furthermore, E is already an image feature obtained by weighted summation of K pixels, so what is the purpose of performing cross-attention again?

* From the ablation experiments, the performance improvement from selecting top-k and visual expression seems marginal. The state-of-the-art performance of this paper may be achieved based on a strong baseline, which does not provide strong support for the effectiveness of the proposed innovations.

[1] Zhang et al. K-net: Towards unified image segmentation. NeurIPS 2021

[2] Ding et al. PPMN: Pixel-Phrase Matching Network for One-Stage Panoptic Narrative Grounding. ACM MM 2022.

### Questions
See Weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a Visual Expression GuidAnce framework for referring expression segmentation, which enables the network to refer to the visual expression that complements the linguistic expression information to improve the guidance capability. The proposed semantic visual information selection leverages the similarity between word tokens and pixel tokens to select top-$k$ pixel tokens for each word token, which are used to collect the semantic information relevant to the target regions by cross-attention mechanism. Extensive experimental results on three benchmark datasets show the effectiveness of the proposed method.

### Strengths
The proposed method is well-motivated and technically sound. The paper is well-organized and shows state-of-the-art performance. The qualitative results are also adequate the visually show its effectiveness.

### Weaknesses
1. The proposed method is with limited novelty. The visual information selection module, which includes the top-k selection and visual expression extraction, is simple and straightforward.
2. The performance gain over existing methods is marginal. Besides, according to Table 2, the proposed visual expression is of limited effect.

### Questions
What is the limitation of this method? Please show some failure cases of this method.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes a new framework to tackle referring image segmentation. In contrast to other works that only use text tokens to segment the target object, the proposed framework makes use of both visual tokens and text tokens to guide the segmentation. To do so, they develop a selection module that first gets top-k image features based on their similarity with text tokens, then goes through a set of transformer layers to obtain the visual tokens. Experiments on several datasets show that the proposed method is robust and effective.

### Strengths
1. This paper is well-written and easy to follow.
2. Experimental results show improvements on 3 datasets.

### Weaknesses
1. The motivation is confusing. Why do we need to use visual knowledge to complete text? If the text query is enough to localize the target object, it is not necessary to complete it. If not, how can we know the target object and complete the text?
2. What will happen if noisy complements are generated?
3. Many previous works also incorporate vision knowledge into text, such as [A-C]. What is the difference between the proposed method with them?

[A] Key-word-aware network for referring expression image segmentation. In ECCV, 2018.
[B] Cross-modal self-attention network for referring image segmentation, In CVPR 2019.
[C] See-through-text grouping for referring image segmentation. In ICCV, 2019.

### Questions
Please see weakness.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
