# SIGMA-Gen: Structure and Identity Guided Multi-Subject Assembly for Image Generation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
We present SIGMA-Gen, a unified framework for multi-identity preserving image generation. Unlike prior approaches, SIGMA-Gen is the first to enable single-pass multi-subject identity-preserved generation guided by both structural and spatial constraints. A key strength of our method is its ability to support user guidance at various levels of precision — from coarse 2D or 3D boxes to pixel-level segmentations and depth — with a single model. To enable this, we introduce SIGMA-Set27K, a novel synthetic dataset that provides identity, structure, and spatial information for over 100k unique subjects across 27k images. Through extensive evaluation we demonstrate that SIGMA-Gen achieves state-of-the-art performance in identity preservation, image generation quality, and speed.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents SIGMA-GEN, a unified framework for multi-subject, identity-preserving image generation under flexible spatial and structural controls. The method allows users to specify both (a) subject identities using exemplar RGB images, and (b) structural arrangements through controls of varying granularity—from 2D bounding boxes to pixel-level depth maps and 3D layouts.

### Strengths
1. The paper elegantly bridges subject personalization and structural control, previously treated as separate problems.

2. The routing–structure control representation is compact and adaptable to multiple input modalities (2D/3D/depth).

3. The authors benchmark across single- and multi-subject settings, include runtime analyses, and evaluate both fidelity and identity preservation.

4. SIGMA-SET27K is systematically generated with aligned modalities, providing a valuable resource for future research.

5. The framework aligns well with creative workflows (scene layout, compositing, virtual try-on), making it broadly relevant beyond academic interest.

### Weaknesses
1. The synthetic dataset may not reflect real-world visual variability; generalization to real photos is untested.

2. All quantitative metrics are automated (DINO, SigLIP, MUSIQ); perceptual user studies would strengthen the claims.

3. Some baselines (e.g., Insert Anything*) are inference-level adaptations, not retrained for fairness—potentially favoring SIGMA-GEN.

4. The paper ablates over control granularity but not over architecture (e.g., without bidirectional compositing or without routing tokens).

### Questions
1. How does the model behave when trained on real datasets with authentic multi-person scenes (e.g., MS-COCO or Visual Genome)?

2. Can the routing–structure image representation generalize to video or temporal settings?

3. Are there failure cases when subject identities overlap or occlude one another heavily?

4. How does the method handle domain shift between synthetic and real identities?

5. Could the authors release data-generation scripts separately from trained models for transparency?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce SIGMA-GEN, a unified framework designed for generating images containing multiple, specific identities. This method is presented as the first to achieve single-pass, multi-subject generation that preserves identity while adhering to both structural and spatial constraints provided by the user. A significant feature is the model's flexibility, allowing user guidance at various levels of precision (from simple bounding boxes to detailed segmentation maps) within one model. To accomplish this, the authors also created a large-scale synthetic dataset, SIGMA-SET27K, providing rich identity and spatial information. The paper claims state-of-the-art performance in identity preservation, image quality, and speed.

### Strengths
- The primary strength is its reported ability to handle multi-subject identity preservation in a single pass while simultaneously respecting complex structural and spatial constraints. This addresses a major limitation in existing generative models.
- The model offers significant practical utility by accepting a wide spectrum of user guidance—from coarse 2D/3D boxes to precise pixel-level maps—within a single, unified framework. This versatility makes it accessible for different use cases without needing specialized models.
- The creation of the SIGMA-SET27K dataset is a valuable contribution in its own right. A large-scale synthetic dataset with comprehensive annotations for identity, structure, and spatial information is a key enabler for this type of complex, multi-modal training and will likely benefit the wider research community.

### Weaknesses
- Lack of Clarity in Methodology and Reporting: The paper's presentation is not as clear as it could be. Specifically, the pipeline for constructing the SIGMA-SET27K dataset is underspecified; critical details about the tools and processing steps used are omitted, making the dataset's creation difficult to reproduce or fully evaluate. 
- Insufficient Experimental Baselines: The experimental comparison, while showing strong results against some methods, is not comprehensive. To truly validate the "state-of-the-art" claim, the evaluation needs to be broadened to include more recent and relevant open-source methods. A comparative analysis against top-tier commercial models (such as those from Google, e.g., gemini-2.5-flash-image-preview, or JiMengAI) is also necessary to properly contextualize the model's performance.

### Questions
I am interested in the model's performance on more fine-grained, real-world customization tasks. Specifically, how effectively does this framework handle high-fidelity, instance-level customization, such as preserving the exact facial identity of a specific, real-world person provided by a user.

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
3

### Summary
This paper presents SIGMA-GEN for multi-subject identity-preserving image generation with structural control at various granularities. The authors introduce SIGMA-SET27K, a synthetic dataset with 27k images containing up to 10 subjects, and demonstrate improvements over baselines especially in multi-subject scenarios.

### Strengths
- Jointly controlling multiple subject identities and spatial layout at various granularities (2D boxes to pixel-level depth) with a single model addresses a practical need in creative workflows.

- Strong quantitative results with 31 points improvement in fidelity and 4x speedup for 5+ subjects, with consistent gains across metrics.

- Well-designed automatic data pipeline and comprehensive ablations provide good insights into component contributions.

- Demonstrated versatility through applications like insertion, reposing, and mixed-granularity control.

### Weaknesses
- Limited technical novelty since the core architecture directly adopts OminiControl's [1] unified attention mechanism without significant modification. The main contribution is dataset engineering rather than methodological innovation.

- Training and evaluating entirely on synthetic data is problematic. You're essentially fitting to outputs from Flux Kontext [2] and other models [3,4], then testing on the same distribution. This doesn't demonstrate real-world generalization, and you should validate on existing benchmarks like DreamBooth [5] with real photographs. Errors from the generation pipeline also propagate into your model.

- Critical technical details are underspecified. How does the model map routing mask intensities (10, 20, 30...) to identity image blocks - is this explicitly supervised or purely learned? The three-stage curriculum suggests fundamental scalability issues rather than unified learning. Why can't you train end-to-end?

- Incomplete baseline comparisons. You cite MultiBooth [6] and other recent multi-subject methods but only compare against MSDiffusion [7]. Also adapting MSDiffusion to use filled masks instead of coordinate embeddings may unfairly disadvantage it. No user studies validate whether your automatic metrics correlate with human perception.

- Training with only subject depths when Table 3 shows full depth improves results (SigLIP-T: 17.73→18.08) seems arbitrary. The bidirectional compositing for occlusions feels like a heuristic workaround rather than principled depth reasoning like InstanceDiffusion [8].

### References

[1] Tan et al., "OminiControl: Minimal and Universal Control for Diffusion Transformer," arXiv 2024

[2] BFLabs et al., "Flux.1 Kontext: Flow Matching for In-Context Image Generation," arXiv 2025

[3] Ren et al., "Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks," arXiv 2024

[4] Wang et al., "MoGe-2: Accurate Monocular Geometry with Metric Scale," arXiv 2025

[5] Ruiz et al., "DreamBooth: Fine Tuning Text-to-Image Diffusion Models," CVPR 2023

[6] Zhu et al., "MultiBooth: Towards Generating All Your Concepts in an Image," AAAI 2025

[7] Wang et al., "MS-Diffusion: Multi-Subject Zero-Shot Image Personalization," arXiv 2024

[8] Wang et al., "InstanceDiffusion: Instance-Level Control for Image Generation," CVPR 2024

### Questions
- Can you evaluate on real-world data with actual photographs as identity images to validate generalization beyond the synthetic training distribution? How does performance compare when the identity images and test scenarios come from real captures rather than model-generated content?

- Please clarify the identity routing mechanism mathematically and show whether the three-stage curriculum is necessary with ablations. Why not compare to other recent multi-subject methods you cite like MultiBooth, and can you provide user studies validating your metrics?

### Soundness
3

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
4

### Summary
The method focuses on the problem of fine-grained control in text-image generation. Existing methods support the function of controlling the single-subject structure or identity, but cannot handle the multi-subject structure and consistent identity with the accurate spatial layout. The proposed SIGMA-GEN proposed several modules to tackle the balance between accuracy and efficiency in multi-subject generation.

### Strengths
Strength:
-	The paper is well-motivated with several technical challenges;
-	The method proposed seems sound and correct.
-	The new evaluation sub-benchmark is proposed with a pipeline for the dataset generation.

### Weaknesses
Weakness: 
-	It seems confusing that which exact module corresponds to tackle the problem of multi-subject structure and identity. It seems that the proposed modules are not unique to this specific challenge. 
-	The novelty of the pipeline is limited. It seems integration of existing modules into a pipeline. Please directly compare with existing baseline methods and show the novelty of each module.

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
2
