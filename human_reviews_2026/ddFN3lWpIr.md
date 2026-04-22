# Everything in Its Place: Benchmarking Spatial Intelligence of Text-to-Image Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 2, 4, 8

## Abstract
Text-to-image (T2I) models have achieved remarkable success in generating high-fidelity images, but they often fail in handling complex spatial relationships, e.g., spatial perception, reasoning, or interaction. These critical aspects are largely overlooked by current benchmarks due to their short or information-sparse prompt design. In this paper, we introduce SpatialGenEval, a new benchmark designed to systematically evaluate the spatial intelligence of T2I models, covering two key aspects: (1) SpatialGenEval involves 1,230 long, information-dense prompts across 25 real-world scenes. Each prompt integrates 10 spatial sub-domains and corresponding 10 multi-choice question-answer pairs, ranging from object position and layout to occlusion and causality. Our extensive evaluation of 23 state-of-the-art models reveals that higher-order spatial reasoning remains a primary bottleneck. (2) To demonstrate that the utility of our information-dense design goes beyond evaluation, we also construct another SpatialT2I dataset. It contains 15,400 text-image pairs with rewritten prompts to ensure image consistency while preserving information density. Fine-tuned results on current foundation models (i.e., Stable Diffusion-XL, Uniworld-V1, OmniGen2) yield consistent performance gains (+4.2%, +5.7%, +4.4%) and more realistic effects in spatial relations, highlighting a data-centric paradigm to achieve spatial intelligence in T2I models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces SpatialGenEval, a new benchmark designed to systematically evaluate the spatial intelligence of T2I models. SpatialGenEval consists of 1,230 long, information-dense prompts across 25 real-world scenes  and 10 spatial sub-domains. SpatialGenEval moves beyond existing benchmarks by creating information dense prompts and evaluate 21 state-ofthe-art models. Evaluation reveals that higher-order spatial reasoning remains a primary bottleneck in existing models. Further, SpatialT2I dataset is generated, which consists of 15,400 text-image pairs, fine-tuning on which leads to performance improvement in models such as Uniworld-V1 and OmniGen2.

### Strengths
Having worked on spatial relationships in T2I models, I thank the authors for working on a very timely problem and doing a good job of it!

1. The motivation is outlined very well and the paper is easy to follow with supporting examples for easier readability.
2. Existing benchmarks, as rightly pointed by the paper, only present simpler prompts. SpatialGenEval does a comprehensive job of covering 10 spatial constraints, enabling deeper probing of models. 
3. Holistic evaluation of multiple models, both open and closed-source along with difference architectures such as diffusion / AR models.
4. While the prompts and corresponding questions are generated from VLM's, a subsequent round of human validation is performed, which ensures a higher quality and avoids potential bias emanating from a proprietary LLM.

### Weaknesses
1. Reliance on VLM for evaluations - This is the major weakness of the current work. Generated images are evaluated with VLM's which themselves struggle with spatial reasoning. For example, a question such as :
 "Based on the toddler's movement, what is the toddler facing towards?" -- A: The ficus tree B: Away from the rubber duck C: The colorful blocks D: The window E: None ; will be difficult for most VLMs to answer as well. Despite Table 3 and 4, where meta comparisons to other benchmarks are shown, performing an alignment with human judgement will solidify the findings. 

2. What is the design choice behind having all prompts having all the 10 spatial constraints? Incremental complexity would paint a better picture of where models fail; additionally since all these constraints are essentially "prompts"; some of these shortcomings might just be text encoder shortcomings instead of the core image generator blocks. This also increases complexity on the VLM's to "generate all 10 QA pairs at once".

3. While the intention is correct, some prompts/questions are ambiguous; for example in Spatial Causal Interaction - "Under a spotlight, a large opera singer in a tuxedo stands center stage. <···>. The sound wave hits the glass, causing it to shatter" - Since its an image, the glass can be shattered by a multiple number of possibilities; such as a rock thrown by someone in the crowd -- How are these cases handled?

4. The SFT contribution seems like an after thought, some more details will be helpful; for example: was the text encoder fine-tuned as part of the experiments? Please refer to SPRIGHT (https://arxiv.org/abs/2404.01197) where authors discuss useful training details for improving spatial relationships in T2I models and covers multiple related data-centric nuances mentioned about in the current work such as relative sizes, spatial proximity etc.

I will increase my score if the above points (especially 1) are addressed.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces a new benchmark for critically evaluating the spatial understanding of current text-to-image models.

### Strengths
* Well-motivated problem statement
* Diverse evaluation dataset
* Well-guiding principles behind dataset construction
* Diverse results from many different models

### Weaknesses
* Unclarity in writing
* Lack of QA in MLLM-synthesized prompts
* Emphasis on using 77 tokens (L259) despite the fact that modern models go well beyond this context length

### Questions
* L85 - L86: It's not clear what the sentence means. Does it mean high-scoring samples from T2I models?  Also, which prompts are being referred to here?
* Figure 1: Are the bottom formats from different eval frameworks? If so, which ones?
* Could the authors provide comparisons to Nano Banana [1] in Figure 2 and in general?
* Could the authors provide some broader trends from benchmark results? For example, do models with LLMs as text encoders (QwenImage [2], for example) show better performance than the ones without?
* "Refuse to answer (not guess)" -- how is this taken into consideration during the evals?
* Could the authors consider using the SPRIGHT [3] fine-tuning strategy and see if that helps? The strategy is to only train on images where the number of objects exceeds a certain threshold.
* The SFT experiments include no diffusion-based models. Could the authors include an experiment with a diffusion-based model, say QwenImage?

Misc:

1. Some missing references: SPRIGHT [1], HEIM [4]

References:

[1] Nano Banana; https://gemini.google/us/overview/image-generation/?hl=en

[2] Qwen-Image Technical Report; Wu et al.; 2025

[3] Getting it Right: Improving Spatial Consistency in Text-to-Image Models; Chatterjee et al.; 2024

[4] Holistic Evaluation of Text-To-Image Models; Lee et al.; 2023

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a new T2I benchmark, SpatialGenEval, designed to evaluate the spatial understanding of T2I models. This benchmark features long, information-dense prompts that incorporate complex spatial descriptions, offering a more challenging evaluation compared to previous benchmarks. It categorizes the prompts into four main domains and ten subdomains to enable a more detailed analysis of model performance.
The paper also introduces a new evaluation metric that employs a Vision-Language Model (VLM) to analyze the generated images and answer multiple-choice questions based on them. Using this benchmark, the authors conduct a comprehensive evaluation of four different architecture of T2I models, covering both pioneering and state-of-the-art approaches. The results highlight the challenges these models face when generating images with more advanced spatial relations—particularly those involving 3D spatial reasoning.
Finally, the paper demonstrates the utility of the proposed benchmark by fine-tuning a T2I model using the benchmark prompts. The fine-tuned model shows some improvements in spatial understanding in image generation.

### Strengths
- The paper introduces a new benchmark with complex and information-dense prompts, which significantly improve upon previous benchmarks that rely on shorter prompts and limited number of spatial relations.
- The paper also introduces a new evaluation protocol that incorporates a vision-language model (VLM) to assess the completeness of generated images through visual question answering, specifically designed to evaluate spatial understanding.
- The benchmark includes human experts in refining prompts and evaluating questions to ensure quality, soundness, and appropriate complexity.
- The benchmark reveals performance gaps across different categories of spatial relations, enabled by the detailed annotations proposed in the benchmark.
- The paper illustrates the usefulness of the evaluation framework for improving model performance in text-to-image generation.
- Well-document and easy to follow paper. Especially figures make the paper easy to follow and understand.

### Weaknesses
• Although the paper proposes a benchmark with complex spatial relations, the evaluation protocol raises some concerns. As demonstrated by several prior works VLMs still struggle with spatial understanding—particularly in 3D spatial relations[1, 2, 3]. Therefore, the interpretation of the results is limited, as it remains unclear whether the observed issues stem from the T2I model itself or from the evaluation protocol.

• Although the authors demonstrate that fine-tuning improves performance, the paper does not explore other potential methods or strategies to address the lack of spatial understanding in T2I models.

• While the authors incorporate a comprehensive human-in-the-loop process for dataset creation, the study does not include a comparison against a human baseline. This is significant, especially when introducing a new evaluation protocol, as it is unclear whether the proposed metric aligns with human judgment. The significant of VQA settings for evaluation is drop without showing this can compared with previous methods.

• The error analysis is based solely on four domain types and relies on numerical accuracy. A qualitative, human-centered analysis of failure cases is needed to provide deeper insights into model weaknesses.

[1] Liu et al. Visual Spatial Reasoning, TACL 2023.

[2] Chen et al. SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities, CVPR 2024. 

[3] Zhang et al. Do Vision-Language Models Represent Space and How? Evaluating Spatial Frame of Reference under Ambiguities, ICLR 2025.

### Questions
There are no additional questions beyond those asked in weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a new text-to-image generation benchmark focused on testing spatial capabilities. The benchmark consists of 1.2k prompts, generated by an LLM prompted with different combinations of domains and themes, and covering 10 axes of spatial capability. After refinement, for evaluation, an MLLM is given the prompts and asked to produce QA pairs aligning with the 10 spatial aspects. Newer T2I models are tested on this benchmark and findings show that spatial comparison and occlusion are two main bottlenecks in improving spatial capabilities.

### Strengths
The work introduces previously untested axes of T2I capability, and provides a large benchmark designed to test models along these new dimensions. The benchmark consists of LLM-generated prompts with automated model-based evaluation, where the generation prompts and evaluation QA pairs are refined manually by humans. While the integrity of the model-based evals could be contested, the diversity of prompts and structured scoring (evenly across the clearly defined 10 sub-domains) is clear.

The benchmark also remains inclusive through information-dense prompts that are short enough to fairly evaluate older T2I models while being difficult enough to distinguish between frontier models.

### Weaknesses
In the analysis of error rates and failure cases, the authors mention "This clear hierarchy demonstrates that models learn skills in a specific order". However, the hierarchy is not clear from Figure 5; it seems for example that motion, which appears after relational reasoning, is already easier. Some clarification would be helpful here.

It is unclear how impactful the contribution of the training data is. While the results show that SFT on the new data improves performance on their benchmark, this is to be expected since the method of training data generation/collection is closely tied to the benchmark. It would be helpful to see numbers from other benchmarks showing whether this improvement also carries into other metrics and maintains the same rank correlation.

### Questions
1. What informs the choice of Gemini 2.5 Pro as the model to create T2I prompts?
2. How strongly do individual per-question scores in the QA portion align with human judgment? While the strong correlation with other benchmark numbers is convincing, if the goal is a breakdown of model capabilities by sub-domain, it is important that these scores also correlate strongly with human or other benchmark judgment.
3. In human refinement of the QA pairs, you mention revising questions that can be answered from the prompt alone. Is this determined by whether a human can answer correctly without an image, or a model? Often, models can still end up being biased towards the correct answer, even if it is not apparent from human inspection.
4. Is there any analysis of bias towards/against image quality or image style, independent of spatial correctness? E.g., do models that output photorealistic vs digital art-style images exhibit higher or lower scores on the benchmark despite having similar true spatial capabilites?

### Soundness
2

### Presentation
4

### Contribution
3
