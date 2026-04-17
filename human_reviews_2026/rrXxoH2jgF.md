# VMDiff: Visual Mixing Diffusion for Limitless Cross-Object Synthesis

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Creating novel images by fusing visual cues from multiple sources is a fundamental yet underexplored problem in image-to-image generation, with broad applications in artistic creation, virtual reality and visual media. Existing methods often face two key challenges: coexistent generation, where multiple objects are simply juxtaposed without true integration, and bias generation, where one object dominates the output due to semantic imbalance. To address these issues, we propose **Visual Mixing Diffusion (VMDiff)**, a simple yet effective diffusion-based framework that synthesizes a single, coherent object by integrating two input images at both noise and latent levels. Our approach comprises: (1) a **hybrid sampling process** that combines guided denoising, inversion, and spherical interpolation with adjustable parameters to achieve structure-aware fusion, mitigating coexistent generation; and (2) an **efficient adaptive adjustment module**, which introduces a novel similarity-based score to automatically and adaptively search for optimal parameters, countering semantic bias. Experiments on a curated benchmark of 780 concept pairs demonstrate that our method outperforms strong baselines in visual quality, semantic consistency, and human-rated creativity.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces an inference-time method for generating novel, visually coherent hybrid objects from customized image inputs. It requires no retraining or fine-tuning of the underlying diffusion model; instead, it optimizes a small set of fusion parameters during inference to effectively combine visual and semantic information from two distinct sources.

The authors identify two key failure modes in existing pretrained models when performing cross-object synthesis: coexistent generation, where objects are merely overlaid or placed side-by-side without true integration, and bias generation, where one object dominates the output while the other is largely ignored due to semantic imbalance. To address these issues, VMDiff proposes a Hybrid Sampling Process (HSP) and an Efficient Adaptive Adjustment (EAA) module. HSP integrates inputs at both the noise and latent levels, enabling seamless fusion through curvature-aware spherical interpolation. Meanwhile, EAA automatically searches for optimal fusion parameters by maximizing a similarity-based score that balances visual fidelity, semantic consistency, and inter-object symmetry.

Experiments on the newly curated IIOF benchmark (780 images) demonstrate that VMDiff consistently outperforms state-of-the-art baselines. User studies also show a strong preference for VMDiff’s outputs.

### Strengths
1. This work clearly defines the visual mixing task and identifies two key failure modes—coexistent and bias generation—in existing methods.
2. The proposed Hybrid Sampling Process effectively combines noise-level and latent-level fusion, enabling both detail preservation and semantic integration.
3. The method is evaluated on a new benchmark with diverse object pairs and shows consistent improvements over strong baselines in both automatic metrics and human studies.
4. This work shows a clear generation process of the proposed pipeline

### Weaknesses
1. **Lack of multi-image generalization**: All experiments are limited to pairwise fusion; the method’s extensibility to three or more input images is neither explored nor validated.  

2. **No cross-model evaluation**: As an inference-time approach, its compatibility with other diffusion backbones (e.g., SD3, SDXL) is not tested, leaving the potential of this method unclear

3. **Missing time-cost ablation**: Owing to high inference latency (166 sec/pair), the paper should break down the computational overhead of individual components (e.g., BNoise, α/β search, SS scoring), to justify the necessity of each design

### Questions
1. Is manual intervention feasible—for instance, to selectively enhance the identity features of one image or the stylistic attributes of another?
2. In certain scenarios, fine-grained control over specific regions may be required—such as replacing a human hand with a robotic arm, or even transforming a foot into a robotic arm. Can this method reliably discern subtle distinctions in the prompt and generate outputs that accurately reflect such targeted modifications?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a framework (VMDiff) for visual cues fusion and mitigates the coexistent generation and bias generation problems that exist in previous works. The pipeline incorporates two important modules: the hybrid sampling process and the adaptive parameter adjustment for efficient latent space manipulation for semantic information integration and conditioning. VMDiff generally outperforms previous approaches under different evaluation settings.

### Strengths
1. The method proposed in the paper introduces free lunch, which can be seamlessly integrated into different diffusion architectures, which could be an interesting direction to explore the general usage cases.

2. Detailed information on design choices over qualitative figures and quantitative numbers is given to showcase the effectiveness of the proposed approaches.

3. A benchmark is proposed to demonstrate the effectiveness of the proposed method.

### Weaknesses
1. The quantitative evaluation details are ambiguous. For example, in Table 1, apart from the proposed approach, MIP-Adapter outperforms OmniGen and DreamO; however, this seems opposite in Figure 6, where MIP-Adapter is less photorealistic compared to all others, which can also be demonstrated by Table 5 in the user study. This phenomenon makes the values less convincing.

2. In the ablation study, most value improvements are introduced by MDNoise; the changes brought by BNoise are not obvious.

3. The paper proposes a method focusing on the fusion of two different objects into another one. In some baselines, such as DreamO and OmniGen, they target a broader and more generalizable application (supporting different prompts with the coexistence of two objects). It would be more intuitive to compare with other papers targeting the same application scale. Besides, it seems this framework is training-free. It could help to compare with other training-free approaches, which might help increase the soundness of the proposed method.

### Questions
1. Evaluation: (a). It would help to validate the comparison if more elaborations or explanations about the evaluation could be provided based on the questions in weakness 1. (b) It's interesting to see how different values given under these metrics for different inputs are semantically good. What are the prompts? Are they all the same? For example, the first column in Figure 6, the results about VMDiff, DreamO, OmniGen, and GPT-4o all seem valid.
2. In the Table 2 ablation part, the comparisons between rows 1 vs. 2 and 3 vs. 4 indicate that the improvements from pure BNoise and $\beta$ search are not obvious. It would help if we had more scores in the setting baseline1 vs baseline 2 vs baseline 1+ BNoise ($\beta$ search) to isolate the improvement caused by BNoise; setting: baseline 1 vs. random noise + MDNoise ($\alpha$ search) to isolate the influence of  MDNoise. It is currently not obvious why the $\epsilon_b$ incorporates more semantic information instead of just Gaussian noise differently sampled.
3. Is this framework commutative? For example, will "a photo of <$T_1$> fused with <$T_2$>" be more similar to $I_1$ while "a photo of <$T_2$> fused with <$T_1$>" is inclined to generate an object similar to $I_2$?
4. How to isolate the improvement proposed by base models? Specifically, the MIP-Adapter is based on sdxl, DreamO is under Flux-1.0-dev, with Krea is more updated. Would some quick validation on these base models using the proposed framework help prove the effectiveness?
5. One possible problem with the current evaluation of concatenation in the second row is that the $\alpha$ changes so small, which may influence a lot under the interpolated space but not be obvious when two latens are separately used. What would happen for other simple values, such as concat($\alpha z_1,(1-\alpha)z_2$)

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents VMDiff, a inference-time optimization framework for two image visual mixing using FLUX-Krea. When attempting visual mixing with existing image generation models, they often struggle with coexistence and bias issues, resulting in conceptually disjoint images. To address these problems, the authors introduce Hybrid Sampling Process (HSP) and Efficient Adaptive Adjustment (EAA). To address these issues, the authors introduce Hybrid Sampling Process (HSP) and Efficient Adaptive Adjustment (EAA). The Hybrid Sampling Process (HSP) includes Blending Noise (BNoise), which concatenates two latents during the inversion process to preserve the original information of both images, and Mixing Denoise (MDeNoise), which interpolates the two latents to fuse the images. Efficient Adaptive Adjustment (EAA) searches the optimal $\alpha$ and then adjusts $\beta_1$, $\beta_2$ based on Similarity Score (SS) between the input images and texts, which are used as parameters in MDeNoise. Experiments demonstrate the effectiveness of the proposed framework for visual mixing on the newly proposed IIOF (Image-Image Object Fusion) dataset, both quantitatively and qualitatively.

### Strengths
1. The paper presents strong qualitative results demonstrating visually compelling outputs. 
2. The training-free approach built upon FLUX-Krea is practical.
3. The experiments show that Blending Noise and Mixing Denoise are sufficient to achieve effective visual concept mixing.
4. A diverse set of experiments provides thorough analysis and validation of the proposed methods.

### Weaknesses
1. The citation format in paragraphs appears inconsistent with the conference’s official style guidelines, which significantly reduces readability.
2. I wonder whether the proposed method is specifically tailored to FLUX-Krea or if it can also be applied to other flow-based models, potentially with different optimal parameters. Demonstrating such generalizability would substantially strengthen the paper and highlight the broader applicability of the approach.
3. The proposed evaluation dataset covers a limited range of categories (40 selected classes), though this limitation is not critical to the overall contribution. 
4. In Lines 79–80, the paper seems to reference the wrong figure example. The sentence “For instance, OmniGen Xiao et al. (2025) generates the doll figurine while entirely neglecting the horse.” likely refers instead to DreamO Mou et al. (2025). 
5. In Line 373, there is a typographical error, an extra comma.

### Questions
1. How much GPU memory is required to mix one pair of images?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents VMDiff, a diffusion-based framework for visual mixing: fusing two input images into a single, coherent hybrid object rather than mere juxtaposition or style transfer. The method has two pillars: (1) a Hybrid Sampling Process (HSP) that first builds BNoise (guided denoising + inversion over a scale-concatenated pair of visual embeddings) and then performs MDeNoise with spherical interpolation of the two image embeddings, and (2) an Efficient Adaptive Adjustment (EAA) that automatically searches fusion parameters 
using a Similarity Score balancing visual and semantic similarity and balance terms. The authors also introduce IIOF, a new benchmark and report strong quantitative metrics, extensive ablations, and user studies showing clear preference for VMDiff over recent multi-concept and mixing/editing baselines.

### Strengths
1. The task is interesting. The paper targets object-level fusion—a regime where many image editing/compositional models struggle (coexistence or bias toward one concept).

2. Well-designed fusion pipeline. The BNoise + MDeNoise split is principled: concatenation before inversion preserves fine details; spherical interpolation during denoising yields seamless integration.

3. The authors conduct a thorough evaluation. A new benchmark (IIOF) + diverse baselines, VQA- and critic-based scores, balance metrics, extensive ablations, and user studies strengthen empirical claims.

4. The authors provide Insightful analysis. The paper’s BNoise vs. MDeNoise discussion and concatenate vs. interpolate design choices are well argued and supported by qualitative/quantitative evidence.

5. The presentation, figures, and algorithm descriptions are clean and easy to follow.

### Weaknesses
I do not have too many concerns. My main concerns are about the computational cost and robustness of the whole pipeline for in-the-wild input.

1. The multi-stage HSP + EAA (multiple generations per search step, occasional resampling) can increase inference latency and compute cost, limiting practical throughput.

2. Robustness and failure analysis are light. The paper lacks a systematic robustness study and only briefly touches on failure cases (e.g., large semantic/structural gaps).

### Questions
Will you release the model and the data? It would be helpful for the image editing community.

### Soundness
3

### Presentation
3

### Contribution
3
