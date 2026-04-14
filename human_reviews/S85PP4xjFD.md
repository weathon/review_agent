# Progressive Compositionality in Text-to-Image Generative Models

- Decision: Accept (Spotlight)
- Scores: 6, 8, 8, 8

## Abstract
Despite the impressive text-to-image (T2I) synthesis capabilities of diffusion models, they often struggle to understand compositional relationships between objects and attributes, especially in complex settings. Existing approaches through building compositional architectures or generating difficult negative captions often assume a fixed prespecified compositional structure, which limits generalization to new distributions. In this paper, we argue that curriculum training is crucial to equipping generative models with a fundamental understanding of compositionality. To achieve this, we leverage large-language models (LLMs) to automatically compose complex scenarios and harness Visual-Question Answering (VQA) checkers to automatically curate a contrastive dataset, ConPair, consisting of 15k pairs of high-quality contrastive images. These pairs feature minimal visual discrepancies and cover a wide range of attribute categories, especially complex and natural scenarios. To learn effectively from these error cases (i.e., hard negative images), we propose EvoGen, a new multi-stage curriculum for contrastive learning of diffusion models. Through extensive experiments across a wide range of compositional scenarios, we showcase the effectiveness of our proposed framework on compositional T2I benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces CONTRAFUSION, a novel framework designed to improve the compositional understanding of text-to-image (T2I) diffusion models. The authors address these models' common issues, such as incorrect attribute binding and flawed object relationships, by introducing COM-DIFF, a high-quality contrastive dataset consisting of minimally differing image pairs across diverse attribute categories. Using a curriculum-based contrastive learning approach, CONTRAFUSION progressively fine-tunes models from simple to complex compositional scenarios. By leveraging large-language models (LLMs) and Visual-Question Answering (VQA) systems for enhanced image-text alignment, the framework significantly boosts model performance on compositional benchmarks, setting a new standard for T2I generation.

### Strengths
The paper introduces a novel methodology targeting the major shortcomings of image generation models, such as the lack of expressive capability for multiple objects or attributes. It explains the creation of a benchmark to address these issues and the application of contrastive learning for model fine-tuning to achieve improvement, demonstrating superior performance across various models.

### Weaknesses
1. The argument that three different sub-tasks were resolved solely using contrastive learning is not very convincing, especially since the paper does not propose a novel loss function but rather uses an existing one. Given that the main contribution of this work is the creation of a benchmark, more justification is needed to support the claim of achieving multiple improvements solely from this contribution.

2. With the text prompts structured into eight categories, there seems to be potential for generating a higher number of combinations than the 15,000 currently provided, which feels somewhat limited.

3. The method used to measure text-image alignment has already been employed in several papers, such as TIFA and other works evaluating text-to-image generation using VQA and LLMs. These approaches have limitations, such as not being able to account for all elements in the generated images using a limited QA set, and issues like hallucination from LLMs and VLMs. It appears that these concerns have not been adequately addressed.

4. The idea of modifying text prompts to match generated images, as mentioned around line 250, is unclear. It’s difficult to imagine how such prompts would be presented, especially given that generated images often have subtle and complex characteristics that are hard to describe accurately, as exemplified by the top-right part of Figure 2.

5. The description of the three stages appears for the first time on page 5, but sufficient explanation should be provided earlier, such as in Table 2 or preceding sections.

6. The claim that three annotators randomly assessed image-image similarity is questionable, as it would have been better to review all images comprehensively. There are likely cases where the differences were not properly captured. Alternatively, some automated method should have been used to ensure accuracy.

### Questions
1. How do you ensure that when generating negative images, only the specified attribute or relationship described in the prompt was changed? It’s unclear how this was rigorously verified.

2. In Figure 3, isn't the dog's head in the second image of the first row incorrectly generated? It looks like there might be an issue with the image.

3. What is the training cost of the proposed approach? Considering the computational expense, it’s not entirely clear if the performance improvement is significant enough to justify the cost.

4. Are there any adverse effects, such as a decrease in the model's ability to generate diverse images, after fine-tuning?

5. Given that CLIPScore has several limitations as an evaluation metric, such as being inaccurate in counting objects, isn't it contradictory to use CLIPScore for measuring alignment in a paper that aims to overcome these limitations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces ContraFusion, a framework designed to enhance the compositional understanding of diffusion models through a combination of contrastive learning and multi-stage fine-tuning. This approach addresses the prevalent challenges faced by text-to-image (T2I) synthesis models, particularly in managing complex relationships between objects and attributes. It introduces the Com-Diff dataset, which comprises 15,000 carefully curated contrastive image pairs, tailored to refine the model's performance.

### Strengths
- The paper addresses a critical issue in text-to-image generative models, focusing on the challenge of compositionality. Despite significant advancements in this field, even the most sophisticated models often struggle with accurately rendering complex compositional details. This work provides an approach to mitigate these limitations.
- While datasets for compositionality already exist, this paper makes a valuable contribution by introducing a new dataset that features contrastive pairs of images generated from positive and negative prompts. This dataset is specifically designed to enhance model performance on compositional tasks, making it a valuable resource for further research.
- The paper applies two techniques—contrastive loss and multi-stage training—for fine-tuning Stable Diffusion on compositional data. It demonstrates the effectiveness of each technique through quantitative analysis, providing evidence of their impact on improving model accuracy and handling of compositional elements.

### Weaknesses
- The ContraFusion model is compared against other methods using the T2I-CompBench dataset. However, it is trained on the Com-Diff dataset, which likely overlaps noticeably with the T2I-CompBench test set. To address this, I recommend that the authors provide a detailed analysis of any overlap between Com-Diff and T2I-CompBench and discuss its potential impact on the reported results. Furthermore, I suggest evaluating the model on a completely separate test set to ensure the robustness and generalizability of the findings.
- For improved clarity, it would be beneficial if the paper explicitly stated the metrics used in Table 3 and other relevant sections. For instance, the BLIP VQA score is employed for evaluation in Table 3, but this is not directly mentioned. Clearly indicating this metric would help readers better understand the basis of the model’s evaluation and the reported results.
- The paper would benefit from more comprehensive details about the experimental setup. For example, it should specify the number of samples generated for calculating the BLIP VQA score, discuss the potential variability in results by indicating the standard deviation across different seeds, and provide a more detailed description of the user study, including the number of participants involved.
- Incorporating a dedicated subsection on relevant background research would provide a richer context for understanding the current contributions. Specifically, discussing recent methodologies like those in [1] and [2], which offer new perspectives on addressing compositionality issues, would demonstrate awareness of and engagement with the latest developments in the field.

[1] LayoutGPT: Compositional Visual Planning and Generation with Large Language Models, Feng et al, 2023.

[2] Understanding and Mitigating Compositional Issues in Text-to-Image Generative Models. Zarei et al, 2024.

### Questions
- Do you have evaluation results for training on the Com-Diff dataset using multi-stage fine-tuning without incorporating contrastive loss? 
- Regarding the multi-stage fine-tuning process, have you considered reintroducing a small number of samples from earlier stages to maintain performance on simpler compositional prompts? Observations from Table 4 suggest a larger performance gap in complex scenarios (excluding texture); thus, recycling simpler samples might help bridge this gap and enhance overall model robustness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors make a dataset of images with hard negatives for a text prompt. They use LLMs to generate a hard negative prompt, stable diffusion3, and an image editing method (MagicBrush) to make pairs of positive and hard negative samples. Finally, they train a Stable diffusion model with an additional contrastive loss using the negative images on the denoising encoder representation. This leads to better compositional generation.

### Strengths
- Introducing a contrastive loss in the denoising encoder representation is an interesting idea. 
- The experimental setups are clear, and there are good comparisons to baselines and ablations on their design choices.

### Weaknesses
- A bit more detail on how the contrastive loss is incorporated would be nice. Is the contrastive loss of the denoising encoder representation at any arbitrary time step t? I am assuming while training, the denosing encoder just predicts the noise to be subtracted for an arbitrary time step t. Hence, the losses would be the MSE on the noise prediction and the InfoNCE loss? What proportions are the added in?  

- A few other datasets with hard-negative compositional images exist, albeit small. Two examples are COLA (https://arxiv.org/abs/2305.03689, they have a hard negative attribute-object composition image for a caption) and Winoground (which the authors already note).  A nice baseline to show would have been comparing using the author's synthetic dataset for tuning over using a combination of the real hard negative image datasets.

### Questions
See weaknesses.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper addresses a complex text-to-image generation approach that aims to achieve accurate and high quality generations of complex samples.  The authors develop a data generation pipeline where they use a LLM and a VQA model to create fine-grained differences in images to help train their mode.  The authors then use curriculum learning to train a diffusion model using their data.  The authors evaluate their approach using compositional T2I benchmarks and provide a user study of their approach.

### Strengths
1. The approach is well motivated and addresses a critical shortcoming of T2I models
2. The paper is relatively well written
3. The experiments cover a number of key comparisons, including a users study, to help validate the effectiveness of their approach

### Weaknesses
1. The data pipeline relies on VQA method's ability to measure alignment.  However, just as T2I models struggle with this, VLMs also struggle, e.g., [A,B].  As such, it seems quite questionable that this would provide a high quality dataset. The statement on L259 on quality control is also ambiguous. How many total annotators were used? How many samples?  How were the samples selected (i.e., how did you ensure coverage of all the elements of the dataset?). The sentence there is vague and gives no details.

2. The authors motivate the need to use generated data due to the cost of collecting pairs.  While I tend to agree with this generally, in practice it isn't clear if this is true.  Some benchmarks like [B] show that fine-grained negatives can be extracted from existing datasets.  There are also datasets with fine-grained pairs of images [C].  What would make this stronger is if the authors compared directly against trying to use some of the competing datasets in their model (including balancing the number of samples).  This would help demonstrate that the differences are important, or the effect of using generated vs. real pairs

3. The user study in figure 8 is not convincing that the proposed approach is better.  The alignment has improved, but at the cost of the generation quality.  I don't have a specific recommendation here- it seems to be simply a weakness of the approach.  That said- the discussion could be more forthcoming and perhaps discuss why despite the drop in aesthetics this is still important.

[A] See, Say, and Segment: Teaching LMMs to Overcome False Premises. CVPR 2024

[B] Cola: A Benchmark for Compositional Text-to-image Retrieval. NeurIPS 2023

[C] Evaluating Text-to-Image Matching using Binary Image Selection (BISON). 2019

### Questions
I proposed some specific questions in the weaknesses.  Generally I would mostly like to hear about W1 and W2

### Soundness
3

### Presentation
3

### Contribution
3
