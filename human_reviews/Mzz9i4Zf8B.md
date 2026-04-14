# CLoRA: A Contrastive Approach to Compose Multiple LoRA Models

- Decision: Reject
- Scores: 6, 5, 5, 5, 6

## Abstract
Low-Rank Adaptation (LoRA) has emerged as a powerful and popular technique for personalization, enabling efficient adaptation of pre-trained image generation models for specific tasks without comprehensive retraining. While employing individual pre-trained LoRA models excels at representing single concepts, such as those representing a specific dog or a cat, utilizing multiple LoRA models to capture a variety of concepts in a single image still poses a significant challenge. Existing methods often fall short, primarily because the attention mechanisms within different LoRA models overlap, leading to scenarios where one concept may be completely ignored (e.g., omitting the dog) or where concepts are incorrectly combined (e.g., producing an image of two cats instead of one cat and one dog). We introduce CloRA, a training-free approach that addresses these limitations by updating the attention maps of multiple LoRA models at test-time, and leveraging the attention maps to create semantic masks for fusing latent representations. This enables the generation of composite images that accurately reflect the characteristics of each LoRA.  Our comprehensive qualitative and quantitative evaluations demonstrate that CloRA significantly outperforms existing methods in multi-concept image generation using LoRAs. Furthermore, we share our source code and benchmark dataset to promote further research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces CLoRA, an approach to enhance the use of Low-Rank Adaptation (LoRA) models for multi-concept image generation. CloRA aims to address the limitations of existing methods, which struggle with overlapping attention mechanisms in multiple LoRA models. CloRA updates attention maps at test-time, using these maps to create semantic masks for better fusion of latent representations. The authors claim that CloRA significantly outperforms existing methods in generating composite images that accurately reflect the characteristics of each LoRA. They also provide a benchmark dataset to facilitate further research.

### Strengths
1. The paper addresses a relevant and challenging problem in the field of personalized image generation, making it a meaningful contribution to the community.
2. The authors provide a benchmark dataset, which is valuable for systematic study and future research.
3. The paper is well-structured and clearly written, making it easy to follow and understand the proposed approach and its benefits.

### Weaknesses
1. The author should report the time and computational consumption of their test-time adaptation strategy.  The training-free attribute is less attractive if it takes a similar amount of time or memory usage to methods that require fine-tuning. 
2. The technical novelty of this work requires further justification. Adjusting attention weights is a common strategy to alleviate the problems of concept ignorance or incorrect combination, and has been studied in many works [1,2,3,4].
3. Missing Citation and Comparison. The authors are suggested to make comparisons with Lora-Composer [4] which also targets multi-LoRA composition and adjusts attention weights through test-time optimization.
4. The authors are suggested to make a comprehensive evaluation of the scalability of the method. For example, what is the maximum number of LoRA models the proposed method can handle? Are there any trends in performance as the number of LoRA models increases?

[1] Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models

[2] Grounded Text-to-Image Synthesis with Attention Refocusing

[3] BoxDiff: Text-to-Image Synthesis with Training-Free Box-Constrained Diffusion

[4] LoRA-Composer: Leveraging Low-Rank Adaptation for Multi-Concept Customization in Training-Free Diffusion Models

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper addresses the challenges of using multiple Low-Rank Adaptation models for personalized image generation, where combining multiple LoRAs often leads to issues like attention overlap and attribute binding. To resolve this, the paper proposes CLoRA, a training-free method that updates attention maps at test-time, ensuring each LoRA model focuses on its respective concept. CLoRA improves image generation by refining attention maps and using semantic masks to blend representations, outperforming existing methods. It works without specialized LoRA variants or training and also includes a benchmark dataset for evaluation.

### Strengths
1.	The paper proposes a training-free method to merge LoRAs.
2.	It introduces a diverse benchmark dataset comprising various LoRA models
3.	The work is complete, providing a new contrastive approach CLoRA and a benchmark dataset. CLoRA achieves better performance compared to previous methods.
4.	This is the first comprehensive work to observe and address attention overlap and attribute binding within LoRA-enhanced image generation models, which offers a new perspective of attention map for this task.

### Weaknesses
1. Writing weakness:
   1. The style is not uniform(line 263-264 $L_{1}$-applied, $L_{2}$-applied and line 323 $L_1$). It is same as $L_1$ in red color?
   2. The meaning of "N" in equation 2 is unclear.
2. Poor visualization quality:
   1. In Figure 3, the visual similarity between the animals and the reference images is lacking, and it seems to exhibit an animated style without applying the style tokens indicated in the prompts.
   2. In Figure 4, the depiction of the book loses essential characteristics, and the cup does not closely resemble the reference images.
3. Insufficient evaluation metrics: The paper only considers image similarity, neglecting text similarity, which is commonly evaluated in baseline papers.
4.	Lack of analysis of the model efficiency. Efficiency is naturally crucial for test-time approach, but the analysis is absent.

### Questions
1. Does the approach work in cases where two similar concept categories are generated (e.g., “catA” and “catB”)? I am concerned that the contrastive loss might fail in such scenarios.
2. In the results depicted in Figure 4 for the Mix-of-Show, are conditions such as human pose or canny edge used as inputs? If so, could you please describe how these input conditions are generated?
3.	Are there more detailed analysis of the model efficiency to prove its test-time usability?
4.	Are there some small errors of the subscripts of $Z$s in Figure 2?

### Soundness
2

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
5

### Summary
The paper introduces CLoRA, a novel approach for composing multiple Low-Rank Adaptation (LoRA) models to generate images that represent a variety of concepts in a single image. CLoRA addresses the challenges of attention overlap and attribute binding that arise when multiple LoRA models are used together. By revising the attention maps at test-time and using contrastive learning, CLoRA successfully creates composite images that accurately reflect the characteristics of each LoRA model. Also, the paper provides a comprehensive evaluation, demonstrating CLoRA's superiority over existing methods in multi-concept image generation using LoRAs.

### Strengths
1 CLoRA's ability to adjust attention maps at test-time allows for dynamic composition of multiple LoRA models without the need for retraining.
2 The use of contrastive learning to refine attention maps is a clever strategy that enhances the model's ability to generate images that respect the boundaries and characteristics of each LoRA model.
3 CLoRA directly addresses the issues of attention overlap and attribute binding, which are critical in the context of multi-LoRA composition.

### Weaknesses
1 The prompts used in the evaluation phase of the paper are relatively simple and do not fully explore the model's performance in handling complex scenes involving interactions between multiple objects. Therefore, the capability of CLoRA in generating composite images with complex interacting objects has not been sufficiently validated.
2 The use of contrastive learning in the paper aims to align the attention of objects introduced by LoRA with specific concepts. A possible question is why not directly generate masks based on the attention of these specific concepts, rather than adopting the method of contrastive learning?
3 CLoRA has a limitation in the number of LoRA objects it can support, with the current method only effectively handling combinations of up to four objects. This poses a limitation for creating richer or more complex scenes, especially in creative tasks that require combinations of more elements.

### Questions
None

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper presents a contrastive approach to seamlessly integrate multiple content and style LoRAs simultaneously, which works in test-time and does not require training. It effectively solves the issues of attention overlap and attribute binding.

### Strengths
1. The proposed method is simple and easy-to-follow.
2. The experimental results verify the effectiveness of the method, which surpasses the baseline methods.

### Weaknesses
1. The issue of attribute binding has been proposed in many works in the field of multi-concept image customization, such as [1] and [2]. The observation mentioned in this paper is not novel.
2. There has been some works on designing attention mechanisms to address attention overlap, such as [2]. It needs additional experiments to verify the advantages of the proposed contrastive way, 
3. Test-time optimization introduces additional inference time, which requires experimental comparison.
4. The dataset in the proposed benchmark only has 20 more characters than CustomConcept101, which lacks a sufficient contribution.
5. The evaluation metric used in quantitative experiments are not enough. Clip-T and Clip-I should also be compared.

[1] Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG), 42(4):1–10, 2023.

[2] Attention Calibration for Disentangled Text-to-Image Personalization. CVPR 2024.

### Questions
As seen in weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a new training-free approach to utilizing multiple LoRA models to generate images with high quality and distinctive features, which solving the attention overlap problem and attribute binding problem by updating the attention maps of multiple LoRA models at test-time. The result shows improved performance in both qualitative assessment and quantitative assessment.

### Strengths
1. This paper is detailed in the background introduction. Specific and vivid examples are used to illustrate the pitfall of previous work and the advantage of cLoRA. It explains the root causes of existing work defects and get nice result using new model.
2. A contrastive objective-based approach to integrate multiple content and style LoRAs simultaneously without training.
3. A method that can use community LoRAs in a plug-and-play manner without needing specialized variants.

### Weaknesses
1. In user study of Section 4, it says the user study involves 50 participants, but does not clarify how many samples each participant was shown. Also, this paper seems to be too brief in the limitation and conclusion sections. For example, in Section 5 when mentioning processing times and the number of LoRA models, the data is not showed to better illustrate. 
2. The core innovation of the method is to differentiate between different concepts using distinct attention maps. However, this design lacks novelty, as similar approaches have been employed in previous research. Additionally, while attention maps are primarily used to control the separation of different concepts, they also restrict information exchange between these concepts. This phenomenon is evident in the results presented.
3. The inference phase consumes more computational resources and the computational complexity increases linearly with the number of merged LoRA models

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
2
