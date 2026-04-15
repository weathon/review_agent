# Exploring Local Memorization in Diffusion Models via Bright Ending Attention

- Decision: Accept (Spotlight)
- Scores: 6, 8, 8

## Abstract
Text-to-image diffusion models have achieved unprecedented proficiency in generating realistic images. However, their inherent tendency to memorize and replicate training data during inference raises significant concerns, including potential copyright infringement. In response, various methods have been proposed to evaluate, detect, and mitigate memorization. Our analysis reveals that existing approaches significantly underperform in handling local memorization, where only specific image regions are memorized, compared to global memorization, where the entire image is replicated. Also, they cannot locate the local memorization regions, making it hard to investigate locally. To address these, we identify a novel "bright ending" (BE) anomaly in diffusion models prone to memorizing training images. BE refers to a distinct cross-attention pattern observed in text-to-image diffusion models, where memorized image patches exhibit significantly greater attention to the final text token during the last inference step than non-memorized patches. This pattern highlights regions where the generated image replicates training data and enables efficient localization of memorized regions. Equipped with this, we propose a simple yet effective method to integrate BE into existing frameworks, significantly improving their performance by narrowing the performance gap caused by local memorization. Our results not only validate the successful execution of the new localization task but also establish new state-of-the-art performance across all existing tasks, underscoring the significance of the BE phenomenon.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a novel perspective on memorization in text-to-image diffusion models by focusing on local memorization, where specific regions of a training image are memorized rather than the entire image. It identifies a unique "bright ending" (BE) anomaly in cross-attention patterns, where memorized regions show higher attention to the end token during the final denoising step. This insight enables the extraction of localized memorization regions without access to training data. The paper proposes a new task—locating these localized memorization regions—and integrates this approach with existing evaluation, detection, and mitigation strategies. The paper demonstrate that incorporating BE significantly improves performance in existing tasks, particularly in narrowing the gap caused by local memorization, achieving state-of-the-art results.

### Strengths
The paper introduces the concept of "bright ending" as a novel phenomenon in diffusion models, which has not been explored in prior research. This unique observation enables a new task of localized memorization detection, which broadens the scope of understanding memorization in generative models.

The paper is good written. The paper clearly differentiates between global and local memorization, offering detailed explanations of how BE is used for local region detection. Figures and visualizations of attention maps provide clarity on how the BE mechanism operates in practice, enhancing understanding of the proposed method.

The motivation of the paper is sound. It has implications for legal and ethical considerations, especially in cases where models might unintentionally replicate copyrighted content. The findings are particularly relevant for improving privacy-preserving techniques in model training and inference.

### Weaknesses
The efficiency of the proposed approach needs to be further demonstrated. The BE-based approach requires analysis at the end of the inference process, making it slower compared to some existing global methods that might detect memorization earlier in the process. 

While the BE phenomenon is well-documented for text-to-image diffusion models, it is not clear how well this method would generalize to other types of diffusion models, like recently the flow-matching based models.

More evaluation metrics for local memorization could be explored for a comprehensive study.

### Questions
How does the proposed BE method perform in scenarios where memorization is more subtle or occurs in highly abstract image regions? Is there a threshold of memorization size or intensity where BE becomes less effective?

Is it possible for the method to be generalized to other types of generative models? Can the BE-based method be generalized to generative models that do not rely on text conditioning, such as unconditional image generation models? Can it be applied to video generation?

Given the increased inference time due to analysis at the final denoising step, do the authors have suggestions for optimizing the process to make it more efficient for real-time applications?

### Soundness
3

### Presentation
3

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
This paper presents a problem with existing techniques to detect memorization in diffusion models in their lack to detect instances of memorization localized to specific subsections of generated images. To resolve this problem, they propose a new technique that creates a memorization mask for a generation.  The technique is based on a correlation between high attention values and memorization that the authors observe at the end of the diffusion process.

### Strengths
1. The problem presented by the authors is compelling, and they demonstrate failure cases of existing techniques in the literature
2. The authors propose a novel and effective method to address the problem
3. The proposed method seems to solve this problem while also maintaining the success cases of earlier techniques

### Weaknesses
1. Some of the figures and text are somewhat difficult to understand, especially relating to the method itself. For example, Table 1 and Figs 6/7 are not explained thoroughly enough, even though they seem to be central results for the paper. Section 5.1 could also benefit from further elaboration to ensure that the method and its formulation are clear.
2. The evaluation is not well explained in my opinion. For example, more details about the manual labelling process will make the results more reproducible.
3. Sharing the code and the dataset will help making the method more reproducible

### Questions
1. What is the motivation for incorporating SSCD similarity into the metric? Why not just use some function of the mask itself to generate a score?
2. Could you please share more details about the manual labeling process and the dataset itself?
3. Will you share the code implementing your method?

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
4

### Summary
This paper is a followup work of Wen et al. (2024), which discovered interesting patterns that "abnormally high predicted noise magnitude" indicate "global memorization". This work finds similar patterns between "abnormally high cross-attention values" of the EOS token in the prompt with the image tokens indicate "local memorization". The experiments validate such patterns can indeed be used to find local memorization.

### Strengths
1. The proposed idea is simple and is a natural extension of Wen et al. (2024).
2. The occurrence of "local memorization" has important practical values but is under-explored.

### Weaknesses
1. In multiple places, the authors state that the "bright ending" is obvious at the final denoising step. However, in the experimental results presented in Table 1, the 3 designs were using "First Step", "First 10 Steps", and "All Steps", respectively. Only "All steps" include "the final denoising step", but apparently "the final denoising step" plays only a minor role in "All steps". Why this discrepancy?
2. In section 5.1, "we propose element-wise multiplication of the magnitude by our memorization mask extracted via BE". However, the authors didn't give details about how to extract a memorization mask from BE in Section 4. Is this done using a fixed threshold? a dynamic threshold? Or a threshold predicted by a dedicated NN?

### Questions
1. AFAIK there's no dedicated EOS token used by stable diffusion. Instead, there are only 77-N padding tokens which are almost identical to each other (except for being added with different positional encodings). Suppose the prompt contains 15 tokens, then there will be 77-15=62 padding tokens which have identical token embeddings. I believe the cross-attention maps of these tokens are highly similar to each other. Do you compute the cross-attention map by averaging these 77-N attention maps? Or do you choose the last padding token to extract the cross-attention map?

EDIT after author response:
Apparently I got it wrong. There's indeed an EOS token inserted before padding tokens.

### Soundness
2

### Presentation
2

### Contribution
2
