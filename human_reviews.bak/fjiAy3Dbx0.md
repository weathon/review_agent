# Desigen: A Pipeline for Controllable Design Template Generation

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 5

## Abstract
Templates serve as a good starting point to implement a design (e.g., banner, slide) but it takes great effort from designers to manually create. In this paper, we present Desigen, an automatic template creation pipeline which generates background images as well as harmonious layout elements over the background. Different from natural images, a background image should preserve enough non-salient space for the overlaying layout elements. To equip existing advanced diffusion-based models with stronger spatial control, we propose two simple but effective techniques to constrain the saliency distribution and reduce the attention weight in desired regions during the background generation process. Then conditioned on the background, we synthesize the layout with a Transformer-based autoregressive generator. To achieve a more harmonious composition, we propose an iterative inference strategy to adjust the synthesized background and layout in multiple rounds. We construct a design dataset with more than 40k advertisement banners to verify our approach. Extensive experiments demonstrate that the proposed pipeline generates high-quality templates comparable to human designers. More than a single-page design, we further show an application of presentation generation that outputs a set of theme-consistent slides. The data and code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The focus of the paper is on automating the process of design template generation.

The main contribution is a design template generation pipeline, consisting of two stages: background generation and layout generation. First, a background image is generated with an extended T2I diffusion model that imposes saliency constraints on the cross-attention activations to preserve space for subsequent layout elements. Then, a layout on the generated background is generated with an autoregressive Transformer, which is then refined together with the background in an alternating fashion for a harmonious composition.

A large-scale banner dataset with rich annotations is constructed for training and testing the method, and an application of the method to multi-page design template generation is demonstrated.

### Strengths
1. Automatic generation of design templates is an important problem to study.

2. The constructed dataset, if can be publicly released, will be of value to the graphic design synthesis community.

### Weaknesses
1. The scale of technical novelty is limited. First, the relationship between subjects in the generated image and their attention activations has already been explored in a recent work, Attend-and-Excite. In view of this work, the finding at the beginning of Sec. 4.2 (and in  Fig.3) is unsurprising and the high-level idea of modifying attention values to control the generated images is not new. Second, the proposed spatial control strategy in Sec. 4.2 (including salient attention constraint and attention reduction) is simple and straightforward, which does not bring much novel technical insight. Third, the layout generator is just a previously proposed technique, i.e., LayoutTransformer. More importantly, the proposed iterative inference strategy for background and layout refinement seems to be ad-hoc. It would be desirable to propose a more unified, perhaps learning-based, approach to capture the dependency between the background and layout, e.g., by modeling their joint distribution, which will be of more interest to the ICLR community.

2. The evaluation is insufficient. For background generation, the diversity of generated images is not evaluated. For layout generation, more evaluation metrics such as FID and max IoU as in [Kikuchi et al., 2021] should be used but are missing. Furthermore, background and layout are now evaluated separately, but the overall design template, which is formed by composing the two, is not evaluated. This is unreasonable since the paper is claimed to be aimed for design template generation instead of background or layout generation. Thus, an evaluation on the quality of generated design templates, e.g., at least through some human studies, is needed to support the paper’s main claim, which is missing in the current paper.

### Questions
How is the attention map A obtained in Eq. (2)?

### Soundness
1 poor

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper generates single-image design templates, e.g., for slides and advertisements, based on a generator trained to do so, given an image prompt layout for which parts of the image should be empty.  The paper notes that the cross-attention values often correspond to saliency, providing a direct way to control saliency.

### Strengths
The paper solves a novel and useful task. The results seem promising. The cross-attention/saliency observation is interesting.

### Weaknesses
I'm not sure that this task is necessarily of great interest to the ICLR community; it may be better-suited for another venue.

The layout-generation component of the work does not seem too novel and is not compared to the state-of-the-art:

PosterLayout: A New Benchmark and Approach for Content-Aware Visual-Textual Presentation Layout
Hsiao Yuan Hsu, Xiangteng He, Yuxin Peng, Hao Kong, Qing Zhang; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 6018-6026


For the image-generation component, much of the same effect can be achieved with existing diffusion models. For example, I tried prompts like `“a background image with empty space for a shoe advertisement”` or `"a background image of a squirrel with empty space for advertising text"`, I got reasonable results with empty spaces. This doesn't offer the same level of control as the proposed system, but seems much simpler, and one could run many generations to find reasonable layouts.

### Questions
None

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper aims to devise a model that could automatically generate a background image with the layout of multiple elements on the image, which could be useful for designing templates for slides, ads, webpages, etc. The desired properties of such templates are: i) background image should leave enough space for overlaying elements (like texts); ii) texts and titles should fit proportionally in the blank space on the image, not occluding each other. To achieve the above goal, the proposed method first generates an initial background with with spatial control using cross-attention on saliency maps, and then a transformer based layout generation model that iteratively refines the background image and the positions and proportions of the elements on it. Qualitative and quantitative results were presented to show that the proposed method generates images that are more suitable as background and also generates layouts with more harmonious elements than baseline methods.

### Strengths
- This work proposes clear and extensive automatic metrics to evaluate the success of layout generation (salient ration, alignment, overlap, occlusion).
- After defining the goal as generating a background image with enough blank space for overlaying texts, the proposed method is effective in achieving the goal.
- Ablation study for each proposed component is reported.

### Weaknesses
- If more discussions can be included on exactly what prompts were experimented for baselines like DALL-E2 and Stable Diffusion, it would really help me understand what existing work is missing that this work provides. Currently the paper reads like baselines and proposed method use the same prompts to generate background images. Is that a fair setting? Baseline models were not trained to generate background only images, so without giving them more explicit prompts specifying that the output should should be a background, it seems natural to me that the output images from baselines are not suitable for background. 
- I wonder why the number of elements in a layout (and their minimal and maximal allowed size) is not an input to the background image generation model? It seems to me that the space to leave blank and its position depends on the potential space the elements would take, so this can be a crucial context for the generation model. While an iterative refinement mechanism is used to adjust the background image and the layout, the iterative nature can be confining the model to only make improvements in a local range of the initial image.

### Questions
- What was the prompts accompanying each image in the newly collected Web-design dataset, and how were they obtained?
- In Section 5.1 Implementation Details, it mentions that all training images are resized to 512x512. Does this mean that during inference time the model also generates images of size 512x512? It seems to me that advertisement images can come in a wide range of aspect ratios, would the resizing and squared output size limit the use case of this model?
- From the qualitative examples, it seems like each layout can have different numbers of elements on it. How are the elements in each layout determined?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
