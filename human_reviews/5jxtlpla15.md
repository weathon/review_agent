# Diffusion Models for Open-Vocabulary Segmentation

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 5, 6, 3

## Abstract
The variety of objects in the real world is unlimited and is thus impossible to capture using models trained on a closed, pre-defined set of categories. Recently, open-vocabulary recognition has garnered significant attention, largely facilitated by advances in large-scale vision-language modelling. In this paper, we present OVDiff, a novel method that leverages the generative properties of text-to-image diffusion models for open-vocabulary segmentation. Specifically, we propose to synthesise support image sets from arbitrary textual categories, creating for each category a set of prototypes representative of both the category itself and its surrounding context (background). Our method relies solely on pre-trained components: segmentation is obtained by simply comparing a target image to the prototypes without further fine-tuning.  We show that our method can be used to ground any pre-trained self-supervised feature extractor in natural language and provide explainable predictions by mapping back to regions in the support set. Our approach shows strong performance on a range of open-vocabulary segmentation benchmarks, obtaining a lead of more than 10% over prior work on PASCAL VOC.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper introduces a novel method for open vocabulary segmentation. Without the need for training,  it leverages diffusion models to generate examples for the categories uses the clip/dino to extract prototypes, and uses the prototypes to make segmetation.

### Strengths
1. This work is novel and interesting.  It provides a new idea to tackle open vocabulary segmentation. 
2. The motivation is clear writing is easy to understand. 
3. Thanks to the generalization ability of SD, CLIP, and DINO models, the proposed methods show a strong generalization ability for "zero-shot" tasks.

### Weaknesses
1. The definition of "zero-shot". As the authors use diffusion models to generate images for the potential categories, I suggest the authors not claim "zero-shot". Because using SD to generate images is somehow equivalent to collecting the target images from the internet. The categories are no longer "unseen".  From my perspective, "open-vocabulary" is acceptable but "zero-shot" is not. 

2. Burdens of generating a large support set.  Although this work does not require training, generating and storing the support set might be heavy when the category list becomes large.

### Questions
See the weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper describes OVDiff, a model that uses text-to-image diffusion models for open-vocabulary segmentation.

The basic approach is to: 
a) use text queries with a text-to-image model to produce sample images that constitute a support set.
b) unsupervised instance segmentation is used (e.g., CutLER) along with cross-attention maps of the diffusion model to distinguish between foreground and background in the support set images.
c) from the support set, prototypes are learned for the class, instance and parts. Both the object and the background are used for positive and negative prototypes.
d) finally, a segmentation map is obtained by comparing dense image feature to prototypes using cosine similarity.

Experiments are performed with several image encoders; DINO, MAE, SD (stable diffusion) and CLIP, and on several datasets; PASCAL VOC, Pascal Context, and COCO-object. Several ablations are performed, e.g., combining image features outperforms any individual feature.

### Strengths
Overall the paper is well written and clearly lays out the approach and the experiments. 

The proposed method relies on off-the-shelf pretrained components, and it is relatively straightforward. 

Experiments explore a few interesting ablations, e.g., the contribution from the background components, the distinction between stuff and things, etc.

### Weaknesses
Several related missing references: 
Learning to Detect and Segment for Open Vocabulary Object Detection, T. Wang, N. Li, CVPR 2023
Semantic-SAM: Segment and Recognize Anything at Any Granularity, Feng Li et al., https://arxiv.org/pdf/2307.04767.pdf

The paper does not describe results on LVIS, commonly used for open set segmentation. 

In distinguishing between stuff and things, the authors describe asking ChatGPT. It was unclear to me whether this was necessary for the paper as it was a relatively small number of classes and the results contained some errors. It's possible that better prompting could have produced more accurate results. 

The results shown in Fig. 5 show a few issues w/ OVDiff; e.g., small false positive "corgi" patches, issues w/ the donut image, a small false positive patch of "bus". 

Editing Comments
p. 6: As the approach does note require --> As the approach does not require
p. 8: Though sometimes region --> Though sometimes the region?
p. 8: not fully align with whole --> not fully align with the whole?

### Questions
a) Please clarify the overall novelty and contribution of the paper.
b) In the text-to-image methodology, it seems that only single-class prompts are used (e.g., "a good picture of a cat" or "a good picture of a dog") rather than more complex queries that could provide more shape information when segmenting. Does this limitation impact performance?
c) Comment on whether it would be useful to benchmark on LVIS?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes to leverage the generative text-to-image diffusion models to enhance open-vocabulary segmentation. The proposed method OVDiff synthesizes support image sets from category names and collect the representative prototypes for each category. The segmentation is performed by comparing a target image with the prototypes.

### Strengths
* The proposed idea of using text-to-image generated samples as support set  to perform the image-image feature comparison seems novel.
* OVDiff achieves the state-of-the-art on VOC, Context and Object benchmarks.
* OVDiff also exhibits reasonable segmentation on the in-the-wild examples.

### Weaknesses
* The background segmentation requires a pre-computation and the use of external module CutLER (Wang et al. 2023). 

* It seems requiring a careful curation and parameter control to achieve the accurate foreground/background segmentation and to collect the good representative prototypes.

* As the synthesized images are mostly object-centric, it is not clear whether the method can still work on large images with multiple fine-grained objects. 

* When evaluated on the context-59 and ADE-150 datasets with more fine-grained objects, OVDiff performs worse than some of the recent SOTA methods.

* While running speed is not a main benchmark in open-vocabulary segmentation, the proposed pipeline of image synthesis, prototype collection, background computation clustering seems involving quite a bit of computation.

### Questions
* Would OVDiff run faster or slower than the SOTA methods?
* Please see Weaknesses section for other questions.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper present OVDiff, a novel method that leverages the generative properties of text-to-image diffusion models for open-vocabulary segmentation. The proposed method shows good results on PASCAL VOC.

### Strengths
1. The proposed method achieve SOTA performance on challenging benchmarks
2. Figures in the paper are clear and easy to follow

### Weaknesses
1. The paper mentions the use of diffusion to generate images and extract the corresponding feature prototypes. However, this approach may introduce bias due to the potentially limited diversity of the generated images, leading to biased results. Generating a larger number of images to address this issue would result in a significant increase in time, as creating a single image with diffusion methods is time-consuming, requiring at least 2-3 seconds even when accelerated by methods like Denoising Diffusion Implicit Models (DDIM).

2. The core insight of your study is not immediately clear. Could you succinctly summarize the key findings and the experimental evidence that supports them? The method section could be simplified for better readability; currently, the intertwining of motivation within the methodological steps detracts from a clear understanding.
Furthermore, there is a discrepancy between the subtitles in Figure 1 and the corresponding method section headings, which disrupts the flow for the reader. The methods section itself seems overly intricate, resembling a layered application of preexisting large models and techniques from other studies, which dilutes the novelty of your work.

3. I am concerned about the complexity of the process and would like to know detailed information on the time required to process a single image, including the computational costs needed for the entire procedure from image generation to final segmentation results. Additionally, how does this compare to other methods? A report on the processing times for alternative methods is also necessary for a thorough comparison.

### Questions
Please refer to the weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
