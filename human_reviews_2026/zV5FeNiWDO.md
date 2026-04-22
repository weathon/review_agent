# Lavida-O: Elastic Large Masked Diffusion Models for Unified Multimodal Understanding and Generation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
We propose Lavida-O, a unified Masked Diffusion Model (MDM) for multimodal understanding and generation.  Unlike existing multimodal MDMs such as MMaDa and Muddit which only support simple image-level understanding tasks and low-resolution image generation, Lavida-O presents a single framework that enables image-level understanding, object grounding, image editing, and high-resolution (1024px) text-to-image synthesis. Lavida-O incorporates a novel Elastic Mixture-of-Transformers (Elastic-MoT) architecture that couples a lightweight generation branch with a larger understanding branch, supported by token compression,  universal text conditioning and stratified sampling for efficient and high-quality generation. Lavida-O further incorporates planning and iterative self-reflection in image generation and editing tasks, seamlessly boosting generation quality with its understanding capabilities. Lavida-O achieves state-of-the-art performance on a wide range of benchmarks including RefCOCO object grounding, GenEval text-to-image generation, and ImgEdit image editing, outperforming existing autoregressive models and continuous diffusion models such as Qwen2.5-VL and FluxKontext-dev, while offering considerable speedup at inference. These advances establish Lavida-O as a new paradigm for scalable multimodal reasoning and generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Lavida-O, a unified Masked Diffusion Model for multimodal understanding and generation. Building upon LaViDa, the model integrates an Elastic Mixture-of-Transformers (Elastic-MoT) architecture with asymmetric branches for efficient parameter sharing. It supports diverse tasks such as image understanding, object grounding, image editing, and high-resolution text-to-image generation, achieving state-of-the-art performance on multiple benchmarks.

### Strengths
1.Proposes a unified framework that effectively bridges image understanding and generation.

2.The Elastic-MoT architecture is efficient, reducing training cost while maintaining strong performance.e

3. Demonstrates impressive empirical results across several benchmarks, indicating strong generalization.

### Weaknesses
1.The novelty of the work appears somewhat incremental — the Mixture-of-Transformers (MoT) paradigm has been extensively studied, and the adoption of discrete diffusion is also not particularly new. The paper’s main contribution seems to lie in engineering integration and architectural refinement, rather than proposing a fundamentally new concept.

2.The training cost appears rather high. According to the appendix, the model requires around 34 days of training, which may be excessive for a unified framework. It would be helpful if the authors could discuss potential efficiency improvements or training optimizations that make the approach more practical.

### Questions
see the weakness.

### Soundness
3

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
This paper introduces Lavida-O, a unified Masked Diffusion Model (MDM) designed to handle a wide range of multimodal tasks, including image-level understanding, object grounding, image editing, and high-resolution (1024px) text-to-image synthesis. The key idea is to build a single, efficient framework that surpasses existing specialized models and other unified approaches (like AR or AR+diffusion models).

### Strengths
- The proposed architecture is efficient and effective. It intelligently combines a large 8B understanding model with a smaller 2.4B generation model, improving training/inference efficiency (Fig 5) while maximizing performance by leveraging the pre-trained understanding branch.

- The integration of planning (layout generation) and self-reflection (iterative refinement) is a key innovation. It provides a concrete mechanism for the model's understanding and generation capabilities to mutually benefit each other, leading to large, quantifiable gains in prompt-following and editing (Tables 3, 13, 14).

- The paper is packed with valuable technical innovations for MDMs, including stratified random sampling (improves FID, Table 10), universal text conditioning (improves control), and coordinate quantization (enables parallel decoding for grounding).

### Weaknesses
- The paper acknowledges that the model's text rendering capability is "very limited." This is a significant weakness for a model aiming for general-purpose multimodal capabilities, as text is a common element in user requests. The authors attribute this to the VQ tokenizer and data, but it remains a key area for improvement.
- The model suffers from a "pixel shift" problem, where non-edited regions are altered during editing. While attributed to the training data, this lack of precision detracts from the quality of the editing feature.
- While performance on MathVista improved over the base model, it still "lags behind state-of-the-art AR models" (Sec D, Fig 13). This indicates that the MDM architecture, or at least this implementation, has not yet closed the gap on high-level, abstract reasoning.

### Questions
- The planning and reflection mechanisms appear to be invoked via specialized prompts (e.g., "please generate a layout...", Sec A.7). Does the model have any capability to *autonomously* decide when to use planning or reflection based on its own assessment of the prompt's complexity? Or must the user always explicitly invoke these modes?
- Regarding the "pixel shift" limitation: Beyond identifying the training data as the source, have the authors explored any mitigation strategies? For example, could a mask-preservation loss (e.g., L1 loss on non-edited regions) be applied during the image editing finetuning stage?
- The modality-aware masking (Sec 3.1.2) is shown for single-turn planning/reflection. How robust is this mechanism for more complex, multi-turn interleaved tasks (e.g., generating a story with multiple images and text blocks)? What are the practical context length limits when history includes multiple VQ-tokenized images?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes LaViDa-O, a unified masked diffusion model that brings image understanding, object grounding, text-to-image generation, image editing, planning and self-reflection into one architecture. It includes several techniques, like Elastic-MoT, modality-aware masking, universal text conditioning, and stratified sampling. Empirically, the planning and reflection produces noticeable gains on text-to-image and editing benchmarks, and the model also improves understanding relative to the LaViDa base.

### Strengths
1. The paper is clearly written and well organized.

2. The method combines multiple ideas designed for different challenges (e.g., stratified sampling). There is a clear motivation before proposing each.

3. Elastic-MoT is a sensible design for improving efficiency, and the appendix provides informative ablations exploring its effectiveness.

4. The single model shows clear performance gains across multiple dimensions (e.g., generation and understanding), particularly when incorporating planning and reflection.

Overall, the paper introduces several clever design choices, such as modality-aware masking and stratified sampling, each of which is well-motivated and justified. So, I believe this work will positively influence the development of masked diffusion models.

### Weaknesses
1. Reporting results on some components of T2I-CompBench++ would strengthen the evaluation, especially since some of its focused prompts (e.g., for 3D and texture) are not currently covered by GenEval.

2. The contribution of universal text conditioning is not quantitatively assessed. A user study or automated evaluation demonstrating its benefit would be helpful.

### Questions
1. For editing images, it may be the case that other conditioning modalities (e.g., keypoints, masks) might be more beneficial than bounding boxes. Is there a way to extend the grounding mechanism to such modalities?

2. The paper states that bounding boxes are denoised in a single step. Could the authors clarify how this is implemented (when to unmask)? Also, is the timestep necessarily before the denoising step of an [exp] token?

3. During training, do image tokens follow a separate masking schedule until $t_{\text{exp}}$? If so, would unmasking an [exp] token before or after $t_{\text{exp}}$ introduce a train-inference mismatch?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a unified masked model that is able to tackle multimodal understanding and generation. Specifically, several components, including Elastic Mixture-of-Transformers, planning, and iterative self-reflection, were proposed. Extensive experimental results on understanding and generation benchmarks demonstrated the effectiveness of the proposed approach.

### Strengths
1. This paper is well-organized and easy to read.  
2. The proposed approach demonstrates promising results on several understanding and generation tasks, such as grounding and editing.  
3. Comprehensive evaluations are conducted.

### Weaknesses
1. The proposed masked diffusion language model is not relatively interesting, which was explored in UMDD[1].  
2. The overall methodologies, including mixture-of-transformer, discrete language model, and discrete diffusion, are well-studied by the existing works [1,2,3,4,5]. I cannot find a significant contribution of this work.  
3. There are several recent works with models of around 2B parameters, which should be fairly compared in the benchmarks.


[1] Unified Multimodal Discrete Diffusion.
[2] LMFusion: Adapting Pretrained Language Models for Multimodal Generation.
[3] Emerging Properties in Unified Multimodal Pretraining.
[4] MMaDA: Multimodal Large Diffusion Language Models.
[5] Show-o: One Single Transformer to Unify Multimodal Understanding and Generation.

### Questions
See weaknesses. I would adjust my initial rating according to the authors' response.

### Soundness
2

### Presentation
3

### Contribution
2
