# Reconstruction Alignment Improves Unified Multimodal Models

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 8

## Abstract
Unified multimodal models (UMMs) unify visual understanding and generation within a single architecture.
However, conventional training relies on image–text pairs (or sequences) whose captions are typically sparse and miss fine-grained visual details, even when they use hundreds of words to describe a simple image. We introduce **Reconstruction Alignment (RecA)**, a resource-efficient post-training method that leverages visual understanding encoder embeddings as dense “text prompts,” providing rich supervision without captions. Concretely, RecA conditions a UMM on its own visual understanding embeddings and optimizes it to reconstruct the input image with a self-supervised reconstruction loss, thereby realigning understanding and generation. Despite its simplicity, RecA is broadly applicable: across autoregressive, masked-autoregressive, and diffusion-based UMMs, it consistently improves generation and editing fidelity. With only 27 GPU-hours, post-training with RecA substantially improves image generation performance on GenEval (0.73 → 0.90) and DPGBench (80.93 → 88.15), while also boosting editing benchmarks (ImgEdit 3.38 → 3.75, GEdit 6.94 → 7.27). Notably, RecA surpasses much larger open-source models and applies broadly across diverse UMM architectures, establishing it as an efficient and general post-training alignment strategy for UMMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Reconstruction Alignment (RecA), a lightweight post-training framework designed to improve semantic consistency between the visual understanding and visual generation modules of unified multimodal models (UMMs). The key idea is to use the embeddings from the vision encoder as dense semantic prompts for self-supervised image reconstruction. By enforcing a reconstruction loss, the model explicitly learns to map semantic embeddings to visual outputs, thereby refining the alignment between understanding and generation. RecA is model-agnostic and can be applied to multiple architectures. Extensive experiments on GenEval, DPGBench, and multiple image editing benchmarks demonstrate consistent improvements with minimal computational cost.

### Strengths
The proposed reconstruction alignment is conceptually simple yet effective. By reusing visual embeddings as dense prompts, it provides a self-supervised approach to enhance multimodal coherence without relying on additional labeled or distilled data. Across multiple UMM architectures (e.g., Harmon and Show-o), RecA consistently improves the fidelity of text-to-image generation and editing. Furthermore, RecA demonstrates practical efficiency, requiring only 27 A100 GPU hours for deployment.

### Weaknesses
- Unified multimodal models (UMMs) aim to integrate both visual understanding and generation within a single architecture. However, RecA is based on a reconstruction task, where the training (I2I) and testing (T2I or I2T) processes differ. It is expected to further clarify how RecA benefits T2I generation tasks as well as visual understanding tasks, especially considering that visual understanding requires both detailed and high-level comprehension.

- Although the work claims to enhance unified multimodal alignment, the quantitative results primarily focus on generation tasks (T2I and editing). Direct evaluation on understanding tasks (e.g., VQA) is needed to substantiate the claim of improving the "unified" aspect of UMMs, as the current results do not fully address this dimension.

- The mechanism by which semantic embeddings improve alignment is primarily intuitive. A deeper representation analysis (e.g., embedding similarity before/after RecA) would strengthen the causal understanding of why RecA works.

- RecA primarily operates as a post-training technique, which inherently limits its scalability.

- Some important technical details are missing. For instance, the paper does not clearly specify where and how the visual embedding is integrated into the model. Additionally, it remains unclear whether hyperparameter tuning is required for each model, which limits the reproducibility of the results.

### Questions
Please see weaknesses.

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
3

### Summary
This paper introduces Reconstruction Alignment (RecA), a simple yet effective post-training method for Unified Multimodal Models (UMMs) that unifies understanding and generation. The key idea is to condition a UMM on its own visual understanding encoder embeddings and optimize it to reconstruct the input image using a self-supervised reconstruction loss.

### Strengths
1. Simple yet effective. The key idea is very simple, but surprisingly works well. It yields consistent improvements in image generation and editing performance while being computationally efficient (27 GPU hours). 

2. Broad generality: RecA is architecture-agnostic and demonstrated across various UMM families (autoregressive, masked-autoregressive, and diffusion-based). Works across AR, MAR, and AR+Diff UMMs with consistent gains.

3. Good experimental results. Experiments show that a 1.5B-parameter UMM post-trained with RecA outperforms much larger models, achieving GenEval 0.90 and DPGBench 88.15, and improves editing benchmarks such as ImgEdit (3.38→3.75) and GEdit (6.94→7.27).

4. Clarity and reproducibility: Strong visualizations, clear methodology, and detailed appendices.

### Weaknesses
1. Limited theoretical grounding: The paper would benefit from more analysis on why reconstruction alignment so effectively bridges understanding–generation gaps (e.g., mutual information or feature alignment metrics).

2. What are the failure cases—does RecA ever degrade diversity or overfit to reconstruction-style artifacts?

### Questions
1. Could RecA be extended to cross-modal reconstruction (e.g., text→image→text) for deeper alignment?

2. Could combining RecA with RL-based alignment (e.g., DPO/GRPO) further enhance reasoning alignment?

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
This paper presents Reconstruction Alignment (RecA), a simple yet remarkably effective post-training technique for unified multimodal models (UMMs). RecA addresses the limitation of sparse text-caption supervision by leveraging dense supervision from visual encoder embeddings, training the model to reconstruct the original image. This process realigns the model's understanding and generation capabilities without requiring any additional labeled data. Extensive experiments across diverse UMM architectures (autoregressive, hybrid, diffusion-based) demonstrate consistent performance gains on both generation (GenEval, DPGBench) and editing (ImgEdit, GEdit) benchmarks. Notably, the method is computationally lightweight.

### Strengths
The proposed method is characterized by its simplicity and broad applicability, ensuring easy implementation with low computational overhead. It demonstrates strong empirical performance, evinced by consistent improvements across four distinct model families. The study maintains rigorous experimental standards, including fair comparisons and thorough ablation studies (e.g., SFT vs. RecA, encoder choice, resolution). The approach has high practical relevance as a plug-and-play component for future UMM pipelines, and its benefits are clearly illustrated through intuitive figures and examples

### Weaknesses
1. The heavy reliance on simple benchmarks (GenEval and DPGBench) raises questions about true generalization. The absence of tests for real-world prompt fidelity and compositional reasoning

### Questions
1. Could the authors provide an analysis, perhaps using feature attribution methods, to identify where RecA contributes most significantly, such as spatial layout, geometry, or fine-grained attributes?
2. Could applying RecA iteratively—through multiple cycles of reconstruction—lead to further improvements in the output?
3. How does RecA perform when given open-ended or abstract prompts for which no concrete visual reference exists?
4. Is there a quantifiable trade-off between the fidelity of the reconstructions and the diversity of the generated samples?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper identified an important misalignment and information gap during the training of image generation (text to image): text input provides only sparse visual information while the model learns to fine-grained visual details. The sparsity holds even when the prompt is completed with hundreds of words. In the context of a unified multimodal model (UMM), the mapping from visual features to language space gives rise to a new source of dense supervision for image generation, i.e., the semantic embeddings of the target image itself. Built on this insight, the authors proposed a post-training technique named Reconstruction Alignment (RecA), where images are abstracted by a visual encoder and mapped to the input space of LLM, then the generation part of the UMM reconstructs the input image. This enables the learning of structural and low-level details of images that may be rare in image captions. RecA was experimented on various types of UMM and brought consistent gains for both image generation and editing.

### Strengths
1. The core idea of the paper, i.e, dense conditioning for image generation, is intuitive and well-motivated. It has been a long-standing issue in multimodal learning that generating visual details is much easier than perceiving them. The authors' perspective on the information density of condition signals is insightful.

2. And the solution, a simple reconstructive supervision, is widely applicable to various UMM paradigms and is effective across different benchmarks. Particularly, the results on GenEval are exceptional, without relying on gpt-distilled data samples.

3. The paper is well written and easy to follow.

### Weaknesses
1. The proposed method seems limited to the post-training stage. It is unclear whether ReCA will also benefit the pretraining of image generation models or unified models.

2. More reconstruction results are expected for all the models studied. I believe the reviewers and readers would be interested in how well the images can be reconstructed.

3. It is unclear how the performance would be affected by the training images, e.g., the scale of the data, visual quality, real data v.s., synthetic data.

4. A factual error: Show-o is not an AR model for image generation. Instead, it is a mix of AR for understanding and discrete diffusion for generation.

### Questions
1. The improvement on Harmon is exceptionally larger than on other models. Can the authors provide more insight? Would it be due to the shared visual encoder or because Harmon mainly uses short captions for training, thus leaving a large space for improvement with additional dense supervision?

2. The performance gains on GenEval are crystal clear. However, GenEval prompts are all short, and the evaluation mainly focuses on high-level image understanding like object detection and counting. Why would the dense supervision be so effective?

### Soundness
4

### Presentation
4

### Contribution
4
