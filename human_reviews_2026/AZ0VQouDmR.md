# DetailMaster: Can Your Text-to-Image Model Handle Long Prompts?

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
While recent text-to-image (T2I) models show impressive capabilities in synthesizing images from brief descriptions, their performance significantly degrades when confronted with long, detail-intensive prompts required in professional applications. We present DetailMaster, the first comprehensive benchmark specifically designed to evaluate T2I models' systematic abilities to handle extended textual inputs that contain complex compositional requirements. Our benchmark introduces four critical evaluation dimensions: Character Attributes, Structured Character Locations, Multi-Dimensional Scene Attributes, and Spatial/Interactive Relationships. The benchmark comprises long and detail-rich prompts averaging 284.89 tokens, with high quality validated by expert annotators. Evaluation on 7 general-purpose and 5 long-prompt-optimized T2I models reveals critical performance limitations: state-of-the-art models achieve merely ~50% accuracy in key dimensions like attribute binding and spatial reasoning, while all models showing progressive performance degradation as prompt length increases. Our analysis reveals fundamental limitations in compositional reasoning, demonstrating that current encoders flatten complex grammatical structures and that diffusion models suffer from attribute leakage under detail-intensive conditions. We open-source our dataset, data curation code, and evaluation tools to advance detail-rich T2I generation and  enable applications previously hindered by the lack of a dedicated benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces DETAILMASTER, the first benchmark for evaluating T2I models on long detail-rich prompts, focusing on four dimensions: Character Attributes, Structured Character Locations, Multi-Dimensional Scene Attributes, and Spatial/Interactive Relationships.

It identifies T2I models’ flaws (compositional reasoning issues, attribute leakage) in long prompts, sets a standard for evaluation, and enables progress in professional applications like industrial prototyping hindered by poor detail fidelity.

### Strengths
1. It fills the gap with long-prompt T2I evaluation by proposing DETAILMASTER— they claim as the first benchmark (actually concurrent with [1] TIT-Score) with avg 284.89-token prompts (vs. existing <100-token ones) and 4 targeted dimensions (e.g., Structured Character Locations).

2. It reveals critical T2I flaws (prompt-length accuracy degradation, backbone limits) and guides research (prioritize long-prompt training over context windows). 

[1] TIT-Score: Evaluating Long-Prompt Based Text-to-Image Alignment via Text-to-Image-to-Text Consistency (Arxiv Oct 3rd.)

### Weaknesses
**1. Incomplete Failure Mechanism Validation:** It attributes poor performance to "encoder grammar flattening" and "diffusion attribute leakage" but lacks causal tests. For instance, no ablation comparing T5 (grammar-aware) vs. CLIP (flat) encoders on prompt structure preservation, or attribute leakage tracking via attention visualization. Adding such experiments would confirm root causes and guide model improvements. Instead of proposing a benchmark, which is more like an effort job, readers may want to know more insights from the huge benchmark experiments. 

For example: 

(1) what is the reason to make the T2I models behave diversely? The training data, training scheme or text encoder differences? 

(2) Any potential to solve the problem with your proposed benchmark? 

(3) How are the unified models (BAGEL, Blip3-o, Janus, Janus-pro, Janus-flow, etc.) perform on your benchmark? 

(4) Would better LLM/VLM as the encoder benefit the generation?

**2. High Evaluation Computational Barrier:** The MLLM-based evaluation requires 20GB–39GB GPU VRAM and 10+ hours per run, excluding small teams. It could adopt lightweight MLLMs (e.g., Qwen2.5-VL-2B) fine-tuned on its annotation data, or distill the evaluator into a smaller model. This would reduce resource needs while preserving accuracy.

### Questions
refer to the weaknesses.

### Soundness
2

### Presentation
3

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
This paper proposes the comprehensive benchmark to address extended textual inputs that contain complex compositional requirements. The benchmark introduces four critical evaluation dimensions: Character Attributes, Structured Character Locations, Multi-Dimensional Scene Attributes, and Spatial/Interactive Relationships.

### Strengths
1. The paper proposes the comprehensive compositional dataset on long, complex prompts.
2. The paper is well-written and easy-to-follow.
3. The experiments are extensive.

### Weaknesses
1. The paper lacks discussion of ConceptMix, which targets at compositional T2I generation.
2. The attribute pipeline relies on MLLM (e.g.,  use MLLM to identify its background composition, lighting conditions, and stylistic elements), which may introduce hallucinations or mistakes. And use MLLM as evaluators may still introduce problems though authors tried to mitigate. For example, the evaluation results are not easy to reproduce.

[A] Wu X, Yu D, Huang Y, et al. Conceptmix: A compositional image generation benchmark with controllable difficulty[J]. Advances in Neural Information Processing Systems, 2024, 37: 86004-86047.

### Questions
1. What are the ways to prevent hallucinations of MLLM in pipeline construction?

### Soundness
3

### Presentation
2

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
This paper proposes DETAILMASTER for evaluating the consistency between images and text generated by T2I models under complex long-text conditions. DETAILMASTER starts from open-source image-caption datasets with dense annotations, further expanding on Character Attributes, Structured Character Locations, Spatial/Interactive Relationships, and Multi-Dimensional Scene Attributes in the prompt through MLLM & LLM, synthesizing the final version of DETAILMASTER. The paper evaluates several general T2I models as well as T2I models specifically optimized for long text, demonstrating that even the most advanced T2I models still need further improvement in aligning images and text under complex long-text prompts.

### Strengths
1. The consistency of text and images in complex long prompts is crucial for evaluating the capabilities of T2I models, and existing benchmarks are indeed lacking in this aspect;
2. The benchmark synthesis process in the paper comprehensively considers various aspects under long text prompts, such as Character Attributes, Structured Character Locations, and Multi-Dimensional Scene Attributes;
3. The paper conducts extensive experiments on existing open-source and closed-source models, indicating that the current state-of-the-art models still need further improvement in handling complex long texts.

### Weaknesses
1. Recent diffusion models that use MLLM as a text encoder, such as Hunayuan Image 3.0 and Qwen-Image, possess stronger text understanding capabilities. How do these models perform on DetailMaster?
2. During evaluation, DetailMaster needs to detect the bounding box for each character based on the Character List. How does it handle cases when the prompt contains multiple repeated characters and there are interactions between these repeated characters?
3. Due to the inherent hallucinations of LLMs/MLLMs, there might be inconsistencies between the final prompt and the image content in DetailMaster's benchmark creation process. The paper does not develop a secondary verification process to ensure higher accuracy.
4. In section 3.2.3, the paper admits that using a single MLLM family (i.e., QwenVL) for both data curation and evaluation could raise concerns about potential self-enhancement bias. Although evaluation results on previous t2i models using InternVL and QwenVL indicate consistent relative rankings, I am still curious whether the evaluation results of diffusion models like Qwen-Image, which use QwenVL as a text encoder, are consistent between QwenVL and InternVL evaluations in DetailMaster.

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces a benchmark for assessing T2I models on their ability to faithfully handle long prompts. To craft the benchmark, the authors follow a data curation pipeline for the evaluation dataset. The paper also includes analyses and insights into the limitations of current diffusion models w.r.t handling long prompts.

### Strengths
* Data curation pipeline 
* Analysis of limitations of current benchmarks pertaining to long prompts
* Robustness and validity of the benchmark through human evaluation

### Weaknesses
* Lack of controlled experiments to drive the insights and analyses in Section 4. It would have been much better to take a single model architecture and ablate it under different setups to drive the insights. More on this in "Questions".
* Lack of results with several models that are known for handling long and complex prompts (such as SANA [1], Lumina-Next [2], and QwenImage [3]).
* It's said in the paper multiple times (L39, for example) that T2I models are trained on short-length prompts. However, many recent models actually leverage denser prompts during their training phase (SANA, for example).

**References**

[1] SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers; Xie et al.; 2024.

[2] Lumina-Image 2.0: A Unified and Efficient Image Generative Framework; Qin et al.; 2025.

[3] Qwen-Image Technical Report; Wu et al.; 2025.

### Questions
> Our analysis reveals fundamental limitations in compositional reasoning, demonstrating that current encoders flatten complex grammatical structures and that diffusion models suffer from attribute leakage under detail-intensive conditions.

Could the authors reference the sections where this was analyzed? Section 4 didn't seem to address these points.

> To achieve higher precision and greater detail, we subsequently reprocess the corresponding 4,565 samples using Qwen2.5-VL-72B-Instruct as both the LLM and the MLLM. This new process generates an improved dataset containing 4,116 prompts with an average token length of 284.89.

How is the number of prompts changing from 4,565 to 4,116? How were they validated? It would also be beneficial to the community if the authors could include all the system prompts that were used throughout this work.

> Section 4 analyses

It would have been better and more helpful to fine-tune a specific architecture on image and long prompt pairs, and then study the effects. Models like QwenImage already leverage an LLM backbone for encoding the text prompts and were also trained with a longer sequence length. So, I believe this is feasible. The models mentioned in "Long prompt training matters more than increasing token capacity." section all have varying amounts of confounding factors and hence it makes it difficult to drive educated observations.

> Datasets

It's unclear what proportions of the samples from each dataset were used to construct the benchmark. It could also be beneficial to include some commentary about how the original captions aren't long enough and miss the important, desirable details (possible through some quantifiable numbers). For example, what character attributes are missing in the original captions?

### Soundness
3

### Presentation
3

### Contribution
2
