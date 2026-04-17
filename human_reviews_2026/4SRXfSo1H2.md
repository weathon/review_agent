# Inverse Virtual Try-On: Generating Multi-Category Product-Style Images from Clothed Individuals

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Virtual try-on (VTON) has been widely explored for rendering garments onto person images, while its inverse task, virtual try-off (VTOFF), remains largely overlooked. VTOFF aims to recover standardized product images of garments directly from photos of clothed individuals. This capability is of great practical importance for e-commerce platforms, large-scale dataset curation, and the training of foundation models. Unlike VTON, which must handle diverse poses and styles, VTOFF naturally benefits from a consistent output format in the form of flat garment images. However, existing methods face two major limitations: (i) exclusive reliance on visual cues from a single photo often leads to ambiguity, and (ii) generated images usually suffer from loss of fine details, limiting their real-world applicability.
To address these challenges, we introduce TEMU-VTOFF, a Text-Enhanced MUlti-category framework for VTOFF. Our architecture is built on a dual DiT-based backbone equipped with a multimodal attention mechanism that jointly exploits image, text, and mask information to resolve visual ambiguities and enable robust feature learning across garment categories. To explicitly mitigate detail degradation, we further design an alignment module that refines garment structures and textures, ensuring high-quality outputs. Extensive experiments on VITON-HD and Dress Code show that TEMU-VTOFF achieves new state-of-the-art performance, substantially improving both visual realism and consistency with target garments. Code and models are available at: https://temu-vtoff-page.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on the virtual try-off task, which generates product images from clothed human images. The authors propose to use both visual and text cues for a richer feature representation from the clothing. In addition, an alignment loss is introduced to refine garment textures by matching DiT features with DINOv2 features. Results on two benmarks show the proposed method ourperforms existing approches.

### Strengths
1) The motivatio of introducing the alignment loss is clear.

2) The paper is structured well and easy to follow.

3) The authors provide many visualizations of the results.

### Weaknesses
1) The first contribution states that the proposed method does not require category-specific pipelines. However, many similar work including MGT and Any2AnyTryon have achieved this, which makes it less of a valid contribution.

2) In the model design,it is unclear why choosing the eighth Transformer block to match the DINO features. What are the justifications for this design choice? If it's empirical,  can the authors provide ablations since this is a major contribution of the paper? For example, will the generated details benefit from choosing more blocks to match DINO features? Also, since the network predicts noise rather than clean image, will the alignment loss degrades the image quality if the last few transformer blocks are used in the loss function?

3) Evaluation is not sufficient. Common objective metrics like the ones reported in the tables are not very consistent with human perception, especially on evaluating texture details. The authors should provide human study results on the benchmark for a more comprehensive evaluation.

### Questions
1) How important is the first DiT model that is trained on reconstructing human images? The authors mentioned that the features are better aligned because the two DiTs have similar architecture. However, adding noise to the input human image at early stages in the VTOFF inference seems unnecessary and could be hurtful. Will using coarse-to-fine clean feature layers in DINO with simple adaptors achieve similar effects?

2) In Table 1, lower-body clothing shows significantly worse scores than the other two categories. I would like to see more analysis on the reasons behind it.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies the inverse of virtual try-on, i.e., try-off. Given a photo of a person wearing clothes, the goal is to generate a standardized product-style image of the garment. This paper proposes TEMU-VTOFF, a dual DiT architecture with a feature extractor for the dressed person and a generator that attends to image, text, and mask tokens through a multimodal hybrid attention. A garment alignment module encourages high-frequency detail fidelity by aligning internal tokens to a frozen vision backbone. Experiments on Dress Code and VITON-HD report state-of-the-art results with ablations on text, mask, extractor features, and the alignment module.

### Strengths
- The task definition is with practical value. Inverse try-on is useful for catalog data creation and dataset enhancement. Multi-category handling in a single pipeline is appealing.

- The dual DiT setup and multimodal hybrid attention integrate signals from image, text, and mask in a straightforward and scalable way.

- Solid results on Dress Code. The method shows consistent improvements on distributional and perceptual metrics, with ablations that isolate design choices.

### Weaknesses
(1) Mixed gains on VITON-HD. On VITON-HD the improvements are minor or mixed. For example, LPIPS is **22.50** for One Model for All vs **28.44** for TEMU-VTOFF (LPIPS lower is better, so this favors the baseline), while DISTS is **19.20** vs **18.04** (lower is better, so this favors TEMU-VTOFF). This suggests the gains are not uniform across metrics or categories. A deeper per-category analysis is needed.

(2) Metric suitability and tradeoffs. The ablations imply the garment alignment module can trade off visual quality and alignment. Since paired ground truth exists, full-reference metrics like SSIM and LPIPS are directly meaningful. FID measures distributional similarity and can be less diagnostic in a paired setting. Please justify metric choices, add SSIM and PSNR consistently, and explain which metrics correlate with human preference for this task.

(3) Generalization and robustness. No cross-dataset experiments are reported. A simple but informative test is to train on VITON-HD and test on Dress Code, and vice versa, to evaluate robustness and domain shift.

(4) Qualitative artifacts. Several provided examples contain noticeable errors, which should be acknowledged and analyzed:
- Row 1, example 1: sleeves unexpectedly longer; skirt shows an unnatural shadow at the waist.
- Row 1, example 4: garment material looks inconsistent with the source.
- Row 2, example 2: button count and style differ from the source.
- Row 2, example 3: top body length should exceed sleeve length, but the result shows a shorter top.
- Row 2, example 4: trouser color shifts; a bow appears at the waistband that may reflect dataset bias.

A failure-mode analysis with per-category statistics would help.

(5) Broader utility not demonstrated. The paper motivates VTOFF as a tool for data generation, but does not show that the generated product images improve downstream tasks. A small study showing that try-off generated pairs improve try-on training or retrieval would make the case stronger.

(6) Notation and clarity.
Line 189: 𝑓=8 is used but f is never defined. I guess z_t  should be denoted as  (H/f)×(W/f)×3 if the latent has spatial downsampling by 
𝑓. Please fix the symbol table and all affected equations.

### Questions
- Cross-dataset generalization. Can you report train-on-VITON-HD test-on-Dress-Code and the reverse, for both global metrics and per-category breakdowns?

- Metric justification. Given paired ground truth, why prioritize FID or KID? Please add SSIM and PSNR consistently and analyze correlations with human preference. If possible, add a small human study.

- Tradeoff in garment alignment. Can you quantify the quality–alignment tradeoff across a sweep of alignment strengths and report the setting used in the main results?

- Downstream utility. Try using TEMU-VTOFF to synthesize product tiles from arbitrary web or in-the-wild images to create pseudo-pairs for try-on training. Does this improve a standard VTON model’s accuracy or user preference?

- Caption provenance. How are text captions produced and sanitized to avoid leaking color or pattern attributes beyond the intended structure-only template?

### Soundness
3

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
- This paper introduces TEMU-VTOFF, a diffusion-based virtual try-off model that reconstructs standardized product-style garment images from photos of clothed individuals, addressing a task that is largely unexplored compared to traditional virtual try-on.

- This paper uses a dual DiT architecture where one Transformer extracts garment features from the person image and the other generates the clean in-shop garment image, enhanced with multimodal attention using text and masks.

- This paper employs a garment alignment module and novel supervision loss to preserve structure and fine-grained textures, achieving state-of-the-art results on VITON-HD and Dress Code.

### Strengths
- Purpose-built architecture for try-off instead of reversing VTON pipelines, enabling clean reconstruction across multiple garment categories (upper / lower / full-body).

- Multimodal hybrid attention improves disambiguation and detail preservation by combining visual features with textual descriptions.

- High image fidelity and alignment thanks to the garment aligner module, resulting in superior quality and consistency compared to existing methods.

### Weaknesses
- Your attempt to explore a new direction within the VITON domain is impressive. However, while VITON-HD uses full-body datasets, this paper uses datasets without faces. Is this because including faces would cause errors?

- Would VTOFF also work on more limited imagery such as VITON-CROP [1]? Since this work deals with real-world scenarios, I recommend including [1] in the references.

- It would also be helpful if the ablation study section were organized in a more intuitive manner.

[1] Kang, Taewon, et al. "Data augmentation using random image cropping for high-resolution virtual try-on (VITON-CROP)." arXiv preprint arXiv:2111.08270 (2021).

### Questions
Mentioned in the weaknesses.

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
4

### Summary
This paper considers multi-garment Virtual Try Off. The proposed approach leverages a dual DiT architecture based on Stable Diffusion 3, where the first network serves as a feature extractor, and the second diffuses the garment itself.

The feature extractor is trained to diffuse the latents of the model image, and takes as its input the latent $z^t$, the encoded latents of the masked model image and the binary mask of the garment. The diffusion is conditioned on the CLIP embeddings of the original model image scaled and shifted with AdaLN. 

The diffusion network is conditioned on intermediate outputs of the feature extractor and CLIP and T2 textual embeddings of garment captions obtained by Qwen2.5-VL. These are combined in the proposed  Multimodal Hybrid Attention module.

To further promote detail preservation, the intermediate features of the 8th internal block of the diffusion network is aligned with DINOv2 features.

Training is done in two stages: First the feature extractor is trained. In the second stage, only the diffusion network is trained, with the values of the feature extractor for timestep 0 serving as the conditioning for all of the timesteps of the diffusion process.

### Strengths
[S1] Good quantitative and qualitative results

[S2] A good ablation study justifying most of the design choices.

[S3] Well written and easy to follow.

### Weaknesses
[W1] A3 section of the appendix suggests that the garment captions are based on textual descriptions of the e-commerce garment image. This seems like a fundamental flaw as the original garment caption is not going to exist for samples in the wild where the ground truth is not going to be known. This presents information about test directly seeping into the inference process.

[W2] Some unclear implementation details. See questions.

[W3] A couple of additional ablations would be useful. e.g. Velioglu et al. (BMVC, 2025) report better results with SigLip encoding of the conditioning image. Furthermore, why is it necessary to have two different textual embedders applied to the caption and concatenated?

### Questions
[Q1] Which encoder is used to encode the the masked person image in the feature extractor?

[Q2] What is $z_g$ in equation 7? Is the diffusion model in the latent space or the image space?

[Q3] Is garment aligner used during inference?
 
[Q4] How does this model perform on the samples not from the training dataset? How sensitive is it to the errors in the masking of the model picture?

### Soundness
3

### Presentation
4

### Contribution
3
