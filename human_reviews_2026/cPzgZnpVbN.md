# FLUX-Reason-6M & PRISM-Bench: A Million-Scale Text-to-Image Reasoning Dataset and Comprehensive Benchmark

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
The advancement of open-source text-to-image (T2I) models has been hindered by the absence of large-scale, reasoning-focused datasets and comprehensive evaluation benchmarks, resulting in a performance gap compared to leading closed-source systems. To address this challenge, We introduce FLUX-Reason-6M and PRISM-Bench (Precise and Robust Image Synthesis Measurement Benchmark). FLUX-Reason-6M is a massive dataset consisting of 6 million high-quality FLUX-generated images and 20 million bilingual (English and Chinese) descriptions specifically designed to teach complex reasoning. The image are organized according to six key characteristics: Imagination, Entity, Text rendering, Style, Affection, and Composition, and design explicit Generation Chain-of-Thought (GCoT) to provide detailed breakdowns of image generation steps. PRISM-Bench offers a novel evaluation standard with seven distinct tracks, including a formidable Long Text challenge using GCoT. Through carefully designed prompts, it utilizes advanced vision-language models for nuanced human-aligned assessment of prompt-image alignment and image aesthetics. Our extensive evaluation of 19 leading models on PRISM-Bench reveals critical performance gaps and highlights specific areas requiring improvement. Our dataset, benchmark, and evaluation code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces FLUX-Reason-6M and PRISM-Bench. FLUX-Reason-6M is a 6-million-scale synthesized dataset curated by Flux, with the first multi-dimensional, million-scale generation chain-of-thought annotations.  PRISM-Bench serves as a comprehensive and discriminative benchmark with 7 independent tracks that evaluate Imagination, Entity, Text rendering, Style, Affection, and Composition. By evaluating 19 leading models, the work identifies critical performance gaps that require improvement.

### Strengths
Unlike many datasets that are simple collections of image-caption pairs, FLUX-Reason-6M is specifically designed to teach complex reasoning capabilities to T2I models. It moves beyond merely describing what is in an image to also explaining why it is composed in a particular way.

A major innovation is the introduction of GCoT, which provides detailed breakdowns of image generation steps. These explicit reasoning paths deconstruct the semantic and compositional logic of the final image, offering powerful intermediate supervisory signals for training models to understand underlying structures and artistic choices.

PRISM-Bench provides a multi-dimensional, fine-grained, and human-aligned evaluation, especially a challenging "Long Text" track that leverages GCoT captions.

### Weaknesses
1. The paper would be far more convincing if the authors had fine-tuned an existing open-source model on this new dataset and shown corresponding performance improvements on PRISM-Bench or other benchmarks. This is the most direct way to validate the curated dataset's value.

2. The effectiveness of GCot is not evaluated. A comparison between models with and without GCoT on PRISM-Bench is encouraged.

3. An analysis comparing the proposed VLM-based evaluators (GPT-4.1, Qwen2.5-VL-72B) against traditional metrics (e.g., CLIPScore, FID) would be valuable to validate PRISM-Bench as a superior evaluation standard.

4. The abstract claims "human-aligned assessment," but the paper provides insufficient details about the human-in-the-loop refinement process. Specifically, the background, expertise and exact guidelines for the human reviewers are not described. These details are necessary to verify the quality and reliability of the claimed human alignment.

5. The paper would benefit from more qualitative visualizations of model outputs.

### Questions
The paper relies heavily on VLM-based automation: Qwen-VL for dataset filtering and GPT-4.1/Qwen2.5-VL-72B as judges for benchmark evaluation. However, the alignment of these automated judgments with human evaluation is not sufficiently detailed. 

In dataset construction, are the filtering results of Qwen-VL aligned with human checking? 

Similarly, how well do the VLM judges (GPT-4.1, Qwen2.5-VL-72B) align with human preferences and scoring on the PRISM-Bench tasks? While an automatic pipeline is valuable, its quality at each step needs to be validated against human assessment.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents two contributions to the text-to-image generation field, primarily aimed at improving open-source models by focusing on complex reasoning. The contributions are: 1) FLUX-Reason-6M: A large-scale dataset of 6 million high-quality synthetic images paired with 20 million bilingual captions. The dataset is structured around six key characteristics and notably includes "Generation Chain-of-Thought" annotations to provide richer, structured supervision for reasoning. 2) PRISM-Bench: A new, 7-track benchmark designed to evaluate T2I models on these characteristics, including a "Long Text" track based on GCoT. A key feature of this benchmark is its use of advanced Vision-Language Models as judges to score both prompt-image alignment and image aesthetics, moving away from simpler, saturating metrics. The authors use this benchmark to evaluate 19 different models, identifying "Text rendering" and "Long text" comprehension as major, persistent challenges across the board. The authors intend to release both the dataset and the benchmark.

### Strengths
1.	The FLUX-Reason-6M dataset is impressive in its scale (6M images, 20M captions) and its use of a high-quality generator (FLUX.1-dev) ensures a strong baseline of visual quality, avoiding the noise of typical web-crawled data.

2.	The Generation Chain-of-Thought is a novel and important contribution. Providing explicit, multi-step reasoning breakdowns as a supervisory signal is a promising direction for teaching models complex compositional logic, which is a key limitation of current T2I systems.

3.	PRISM-Bench represents a significant improvement over existing T2I evaluation methods. The 7-track design allows for a much more granular analysis of model strengths and weaknesses than a single, aggregate score.

4.	The comprehensive evaluation of 19 models yields useful insights, particularly the clear confirmation that even SOTA closed-source models struggle immensely with reliable text rendering and long, complex instruction following. This helps focus the community on the next set of hard problems.

### Weaknesses
1.	The dataset is entirely synthetic, generated by a single model (FLUX.1-dev). This creates a significant risk of "dataset bias" or "domain gap." Models trained on FLUX-Reason-6M may learn the specific style, quirks, or common failure modes of the FLUX generator, and this improved performance might not generalize to real-world images or other domains.

2.	The PRISM-Bench relies entirely on VLM judges. While the use of VLMs is well-motivated, the central claim that these judges are "human-aligned" is asserted rather than proven. Without a correlation study comparing the VLM scores to scores from human evaluators, the validity of the benchmark is not fully established. This is a omission for a paper introducing a new evaluation standard.

3.	There is a potential self-referential loop in the methodology. The GCoT reasoning annotations are generated by a VLM, and the benchmark is then evaluated by other VLMs. This creates a system where models are trained on VLM-generated text and then evaluated by VLM judges. This VLM-in-a-loop system risks optimizing for what VLMs think is good reasoning or good alignment, which may not perfectly overlap with true human intent or logic.

### Questions
1.	Could the authors elaborate on the quality control process for the GCoT annotations? Specifically, how did the authors ensure that the VLM-generated reasoning steps were logical, accurate, and of high quality?

2.	The authors state that the Qwen-VL judge yields “consistent rankings” with the GPT-4.1 judge. Could the authors provide a more quantitative measure of inter-judge agreement between the two VLM judges across the seven tracks to substantiate the claim of benchmark robustness?

3.	Have the authors conducted any preliminary experiments to examine whether training on FLUX-Reason-6M, particularly with GCoT supervision, improves performance on other existing benchmarks compared with training on other large-scale datasets? If so, what do these results suggest about the practical benefits of the proposed dataset and GCoT supervision?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work present a training dataset, FLUX-Reason-6M, and a benchmark, PRISM-Bench. FLUX-Reason-6M contains 6M FLUX-generated images paired with carefully curated prompts that requires reasoning during image generation. Each text-image pair comes with multiple captions and a Generation Chain-of-Thought (GCoT) that provide details breakdowns of the image generation process.

The PRISM-Bench is a challenging benchmark that covers the complex reasoning abilities introduced in FLUX-Reason-5M. The benchmark uses strong MLLMs like GPT-4.1 and Qwen2.5-VL-72B to do evaluations. Experiments show big performance gaps across models, and hights the areas that need improvements.

### Strengths
1. The field needs resources toward more complex and reasoning-heavy text-to-image generation tasks. This paper provides both the training data and the evaluation benchmark for it.

2. The paper contains detailed description of the carefully designed data curation pipeline. The pipeline make sense and can generate data of high quality. The benchmark is further examined by humans to ensure quality. These make the datasets included in this paper valuable to the community.

3. The paper is well-written and easy to read.

### Weaknesses
1. The dataset FLUX-Reason-6M only used a single text-to-image model, FLUX.1-dev to create images. It potentially has the model's biases and inherits its failure modes.

2. There is no experiments that trains any model with FLUX-Reason-6M. There lacks evidence that this dataset is helpful for model training.

3. PRISM-Bench relies on MLLM for evaluation. However, MLLMs can be unreliable, especially with these challenging tasks. There is no human experiments that show how well this benchmark aligns with humans.

### Questions
1. Have you tried train any model with FLUX-Reason-6M? Can it yield improvements on tasks?

2. Is there any evidence that GCOT can improve model training? Any ablation experiments showing that?

3. Have you conduct any human evaluation on PRISM-Bench? How well does the evaluation metrics align with human judgements?

### Soundness
3

### Presentation
3

### Contribution
4
