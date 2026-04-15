# Text-Driven Image Editing using Cycle-Consistency-Driven Metric Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 6, 3, 3

## Abstract
We present a simple but effective training-free method for text-driven image-to-image translation based on pretrained text-to-image diffusion models. Since a naive application of the pre-trained diffusion models for the manipulation tasks often significantly destroys the structure or background of the source image, we revise the original backward process for the target image by meaningfully aligning better with a given target task while preserving the background or structure of a source image. We derive a new guidance objective term that is a combination of maximizing the similarity with target prompts rather than the source prompt based on the pre-trained CLIP and minimizing the distance with the source latents. Moreover, contrary to existing methods based on the diffusion models, we exploit the cycle-consistency objective in order to further maintain the background of the source image, where we perform an iterative optimization process by alternately optimizing the source and target latents. Experimental results demonstrate that the proposed method achieves outstanding editing performance on various tasks when combined with the pre-trained Stable Diffusion.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work presents a training-free approach for text-driven image-to-image translation, building on a pre-trained text-to-image diffusion model. The authors revise the process to align better with the target task. They introduce a new guidance objective, which combines maximizing similarity to the target prompt (measured by CLIP score) and minimizing the distance to the source latent variables. Moreover, they employ a cycle-consistency objective to maintain the source image background by iteratively optimizing source and target latent variables. Experimental results demonstrate the exceptional performance of this method.

### Strengths
The article introduces a simple yet effective approach for text-driven image-to-image translation. 
1. In contrast to other methods, this approach places a strong emphasis on preserving the structure and background of the source image during image editing. It accomplishes this by revising the process for generating target images to better align with the target task.
2.  The article introduces a new guidance objective that combines maximizing similarity to the target prompt (measured by CLIP scores) and minimizing the distance to the source latent variables, resulting in improved quality of generated outputs.
3. To maintain the background of the source image, the article utilizes a cycle-consistency objective. This involves iteratively optimizing source and target latent variables, enhancing the feasibility of the method.

### Weaknesses
I find this method to be intuitive, but it appears to lack enough technical innovation, as similar concepts have been previously mentioned in prior works. My primary concerns are related to the experimental aspects:

1. The authors should also conduct experiments on some of the datasets or images provided in their previous work.

2. The quantitative experiments in the study appear to be insufficient. Since the authors have collected a dataset, it would be better for them to report average metrics on this dataset.

3. There is a shortage of comparison with other methods. Given the wide attention in this field, it would be beneficial to compare this approach with more recent works. It is also better to include some fine-tuning-based methods (like SINE, Text-Inversion)  to provide a more comprehensive evaluation.

4. The running costs, such as time and GPU resource consumption, should be reported and compared to help readers understand the resource requirements when using this method.

5. The authors have not listed the limitations of their method. As this approach is positioned as a general method, it is essential to clarify whether it supports general scenarios, like the removal or addition of specific elements in the target image, to better inform users about its applicability.

6. When comparing "Prompt-to-prompt," it seems that the authors have not adopted a strategy that specifically considers the background region. This might impact the accuracy of the experimental results.

### Questions
See Weaknesses.

### Soundness
2 fair

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
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper  presents a training-free approach for text-driven image-to-image translation using a pretrained text-to-image diffusion model.

### Strengths
1.The paper introduces a new guidance objective term, which combines maximizing similarity to the target prompt (based on the CLIP score) and minimizing the distance to the source latent variables.

2.Unlike many existing methods based on diffusion models, the paper leverages a cycle-consistency objective to preserve the background of the source image.

### Weaknesses
1. The time consumption of the proposed method compared to other methods should be given.


2. Comparable works such as "Negative-prompt Inversion" and "Null-text Inversion for Editing Real Images using Guided Diffusion Models" demonstrate robust image reconstruction and content editing capabilities while preserving the original background. These works also support flexible target category transformations. A comprehensive comparison with these similar works could further support the paper's novelty and performance in relation to existing solutions.

3. A user study is encouraged to be carried out.

### Questions
see above

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces a unique method for free text-driven image editing by utilizing pre-trained text-to-image diffusion models. Central to this approach is a new guidance objective term, which maximizes similarity to the target prompt (as opposed to the source prompt) based on the CLIP score. In tandem, it minimizes the distance to the source latent variables. Additionally, the authors incorporate a cycle consistency objective to retain the background details.

### Strengths
- **Simplicity & Effectiveness**: The proposed method is both straightforward and seemingly efficacious, as evidenced by the results presented in the paper.

### Weaknesses
- **Evaluation Methods**: The evaluation could be more robust. The prompts used for evaluation are closely related to the original noun, reducing diversity and potentially biasing results.
- **Aspect Ratio Concerns**: The samples used for evaluation have been altered from their original aspect ratios. This could inadvertently disadvantage competing methods.
- **Comparison Choices**: The results from prompt-to-prompt evaluations seem to perform well on generated images rather than inverted real ones. The absence of a comparison with Null-text-inversion, which might be a more apt benchmark, raises questions.
- **Efficiency Metrics**: The paper would be more informative with a runtime efficiency comparison against other methods.

### Questions
None

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a method for editing without training using additional cycle-consistency and triplet-based distance guidance. The triplet-based distance ensures that source and target images at the same time step are mapped closer together than those at different time steps, in addition to using a general feature similarity-based distance. The cycle-consistency objective is employed to ensure that two images, one with guide in the forward process and the other with guide in the backward process, produce identical results.

### Strengths
The method produces better structure-preserved editing results. Also, compared to other training-free algorithms the proposed method achieves better quantitative results.

### Weaknesses
Cycle-constistency may overly fix the structure and may make the result unnatural with object with different structure. Also, the argument that different time-step target images should be farther apart than same time-step source and target images seems to lack sufficient justification. And there is no comparison with papers such as null-text inversion.

### Questions
It seems there are only subtle differences with naive distance and the triplet distance guidance results. Is there more basis for the triplet loss that makes different time-step images of the same image distant from each other?
Also, how does the performance compare to recent papers such as null-text inversion and similar approaches?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
