# ImageRAG: Dynamic Image Retrieval for Reference-Guided Image Generation

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 4

## Abstract
While recent generative models synthesize high-quality visual content, they still struggle with generating rare or fine-grained concepts.
To address this challenge, we explore the usage of Retrieval-Augmented Generation (RAG) for image generation, and introduce ImageRAG, a training-free method for rare concept generation. 
Using a Vision Language Model (VLM), ImageRAG identifies generation gaps between an input prompt and a generated image dynamically, retrieves relevant images, and uses them as context to guide the generation process. 
Prior approaches that use retrieved images require training models specifically for retrieval-based generation. In contrast, ImageRAG leverages existing image conditioning models, and does not require RAG-specific training.
We demonstrate our approach is highly adaptable through evaluation over different backbones, including models trained to receive image inputs and models augmented with a post-training image-prompt adapter. 
Through extensive quantitative, qualitative, and subjective evaluation, we show that incorporating retrieved references consistently improves the generation abilities of rare and fine-grained concepts across three datasets and three generative models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper aims to address the recent generative models problems in generating rare or fine-grained concepts. ImageRAG framework is introduced as a training-free method for rare concept generation. ImageRAG identifies generation gaps with the guidance from a Vision Language Model. The VLM retrieves relevant images while ImageRAG uses them as context to guide the generation process.

### Strengths
1. The idea of applying RAG in image generation is up-to-date.

2. The method illustration is clear.

3. Writing is easy to follow.

### Weaknesses
1. As the most relevant work, RealRAG[A] is only mentioned in the related work section, while ignoring in the other sections, especially for the introduction and experiments.  

a. Line-090: "previous works employing image retrieval for better image generation, train models specifically for the task, hindering wide applicability." The reviewer believe RealRAG does not train models specifically for task. 

b. In the experiments, only several base models are included for comparison, more methods in this area should be compared, especially for RealRAG[A].

[A] Lyu Y, Zheng X, Jiang L, et al. RealRAG: Retrieval-augmented Realistic Image Generation via Self-reflective Contrastive Learning[C]//Forty-second International Conference on Machine Learning.

2. In the method, CoT is used for identifying challenging concepts. Could the author provide more discussion regarding this part? Is there a  trade-off between the CoT length and the generation performance? Is there any over thinking problem in this scenarios?

### Questions
### **1. On Related Work and Comparison with RealRAG**

**a. Conceptual Differences**

* In this paper, *RealRAG* (Lyu et al., ICML 2024) is mentioned only briefly in the related work section but not discussed elsewhere.
  Could the author elaborate on how the method differs conceptually from RealRAG, especially given that RealRAG also employs retrieval-augmented generation but does **not** require task-specific model retraining?
* What are the main methodological distinctions between ImageRAG and RealRAG in terms of:

  * Retrieval mechanisms and
  * Integration with the generation process?

**b. Experimental Comparison**

* In the experiments, only a few baseline models are included. Could the authors clarify why RealRAG and other recent retrieval-based image generation methods were excluded from the comparison?
* Have the authors evaluated the framework under the same experimental conditions as RealRAG (e.g., dataset, retrieval backbone, evaluation metrics) to ensure a fair comparison?

---

### **2. On Chain-of-Thought (CoT) Reasoning for Concept Identification**

* The authors mention that CoT reasoning is used to identify challenging or ambiguous concepts. Could the authors expand on how the CoT process interacts with the retrieval or generation stages?
* Is there a trade-off between **CoT length** and **generation quality** (e.g., longer CoTs improving reasoning but reducing visual fidelity or inference speed)?
* Have the authors observed any “overthinking” effects — cases where extended reasoning sequences introduce redundancy or semantic drift — and if so, how do the authors mitigate them?

### Soundness
2

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
5

### Summary
The paper propose a image Retrieval-Augmented Generation (RAG) framework for image generation named ImageRAG. ImageRAG using VLM to retrieves relevant images to enhance custom image generation. Through extensive quantitative, qualitative, and subjective evaluation, paper show that incorporating retrieved references consistently improves the generation abilities of rare and fine-grained concepts.

### Strengths
1. The paper proposes a novel pipeline designed for image retrieval-augmented generation (RAG), which can retrieve relevant images to enhance the generative performance of diffusion models.
2. The proposed framework automates the common engineering practice of improving generation quality through reference images, demonstrating strong practical value.
3. The paper introduces an efficient approach for leveraging image knowledge database, which offers valuable insights for the development of multimodal large models.
4. The paper is clear and easy to follow.

### Weaknesses
1. If the target concept to be generated does not exist in the retrieval dataset, is there a possibility that irrelevant images may be selected, thereby potentially affecting the generation quality adversely? The paper should include corresponding quantitative analyses.  
2. [Retrieval-Augmented Diffusion Models] has proposed a similar concept. The paper should provide a detailed comparison with this work with same base model and dataset, and analyze advantages of ImageRAG.  
3. Will the diversity of generated images be constrained by the retrieval dataset, leading to visually similar outputs for same prompts? The paper should include relevant analyses to clarify this.

### Questions
SDXL+IPA have different image and text input interface, and cannot use the prompt template in Figure 3. How dose the ImageRAG handle models like this? Are the images correctly associated with the corresponding text snippets?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces RAG into text-to-image generation process for some rare or fine-grained concepts. This approach is training-free and model agnostic, and can operate with off-the-shelf image-conditioning interfaces. Image RAG contains four steps: 1/ rough image generation from a text prompt; 2/ use VLM to identify the missing concepts in a CoT process,  3/ retrieve reference images corresponding to the missing concepts from an external dataset; 4/ regenerate the image using these reference images by a generative model. The authors evaluated the method using three different base models, including SDXL, FLUX, and OmniGen, on three benchmarks.

### Strengths
- The adaptation of RAG to text-to-image generation is inspiring. It focuses on "polish" the generated image at inference time rather than retraining.

- The formulation of CoT-based retrieval is promising, which is effective for identifying the missing visual elements.

- The evaluation is comprehensive, including qualitative comparisons, user study, and ablation studies. It is good to see that failure cases are included in this paper, which help the reader to better understand the limitation of this paper.

### Weaknesses
- The applicable scenario is limited. ImageRAG suits most for the scenario where rare or even weird concepts exist, such as "a boston bull", but the successfulness greatly depends on the coverage of the retrieval dataset (e.g. LAION-350K subset). It would be better to see that this method can handle well with the complex and lengthy prompts that are usually more customer-driven. 

- No report or measurement of the latency and computational cost of this multi-step pipeline, which is even more important than the metrics measuring image quality. We should always consider the latency for real-time or interactive use cases.

- The technical novelty is limited. Even though the concept of ImageRAG is good, each component in this framework is individually normal. And the success of ImageRAG heavily rely on the performance of the submodule such as CLIP, GPT-4o and T2I, etc. 

- The performance improvements in CLIP or DINO metrics are in general very limited (<0.01 absolute).

### Questions
- Since the limitations are listed in this paper, is there any solution or strategy to improve them?

- Is there any failure cases that VLM misidentify the gaps between text prompts and rough images?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ImageRAG, a training-free retrieval-augmented generation method that enhances text-to-image models by dynamically identifying generation gaps using a vision-language model and retrieving relevant reference images to guide the synthesis of rare or fine-grained concepts. The approach is model-agnostic, demonstrated through evaluations on diverse backbones like OmniGen, SDXL, and FLUX, showing improved alignment with prompts without requiring additional training.

### Strengths
The method's adaptability is a key strength, as it seamlessly integrates with existing image-conditioning models like IP-Adapter and OminiControl, enabling broad applicability across different architectures. Extensive experiments, including quantitative metrics, qualitative examples, and human studies, robustly validate that ImageRAG consistently improves rare concept generation, with users preferring it over baselines in text alignment and visual quality.

### Weaknesses
1. The method's efficacy is highly dependent on the retrieval dataset; if it lacks relevant images (e.g., specializing in birds when generating dogs), performance may not improve, as illustrated in the retrieval data limitations.

2. It relies on the VLM's accuracy for gap identification; errors in concept detection (e.g., false positives in alignment checks) could lead to missed enhancements, though the paper notes robustness issues with some VLMs.

3. While the paper notes VLM API calls add 10–30 seconds, it lacks comparisons to baseline T2I inference latency and does not explore optimizations (e.g., cached embeddings for small datasets) for real-time use cases.

4. Although diversity is mentioned, the paper does not quantify how ImageRAG impacts generation diversity across models (e.g., SDXL’s low diversity vs. OmniGen’s high diversity) or whether retrieval introduces bias toward reference image styles.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3
