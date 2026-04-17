# MOSAIC: Multi-Subject Personalized Generation via Correspondence-Aware Alignment and Disentanglement

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Multi-subject personalized generation presents unique challenges in maintaining identity fidelity and semantic coherence when synthesizing images conditioned on multiple reference subjects. Existing methods often suffer from identity blending and attribute leakage due to inadequate modeling of how different subjects should interact within shared representation spaces. We present MOSAIC, a representation-centric framework that rethinks multi-subject generation through explicit semantic correspondence and orthogonal feature disentanglement. Our key insight is that multi-subject generation requires precise semantic alignment at the representation level—knowing exactly which regions in the generated image should attend to which parts of each reference. To enable this, we introduce SemAlign-MS, a meticulously annotated dataset providing fine-grained semantic correspondences between multiple reference subjects and target images, previously unavailable in this domain. Building on this foundation, we propose the semantic correspondence attention loss to enforce precise point-to-point semantic alignment, ensuring high consistency from each reference to its designated regions. Furthermore, we develop the multi-reference disentanglement loss to push different subjects into orthogonal attention subspaces, preventing feature interference while preserving individual identity characteristics. Extensive experiments demonstrate that MOSAIC achieves SOTA performance on multiple benchmarks. Notably, while existing methods typically degrade beyond 3 subjects, MOSAIC maintains high fidelity with 4+ reference subjects, opening new possibilities for complex multi-subject synthesis applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes MOSAIC, a representation-centric framework for multi-subject image generation that enforces explicit semantic correspondences and orthogonal feature disentanglement to prevent subject mixing. It introduces SemAlign-MS, a dataset with fine-grained, point-to-point correspondences between multiple reference subjects and target regions. Built on this, a semantic correspondence attention loss aligns each generated region to its designated reference parts, improving subject consistency and spatial localization. Besides, it also develops multi-reference disentanglement loss to prevent feature interference. Overall, the work reframes multi-subject generation as a representation-level alignment problem and delivers both the data and objective needed to make precise alignment feasible.

### Strengths
1. New fine-grained dataset: Introduces a meticulously annotated, point-to-point multi-subject correspondence dataset (SemAlign-MS) that fills a clear gap and is poised to catalyze progress in the community.

2. Loss tied to correspondences: Leverages those point-to-point annotations to design a Semantic Correspondence Attention Loss that enforces precise regional alignment, improving subject identity fidelity.

3. Promising quantitative results: Reports strong quantitative gains that indicate the approach’s effectiveness for multi-subject consistency and localization.

### Weaknesses
1. Incomplete baselines: Missing comparisons to strong multi-subject/identity methods such as FastComposer [1] and Face-Diffuser [2], and so on.

2. Limited image-quality evaluation: The paper does not report results on widely used VLM-based metrics (e.g., UnifiedReward and HPSv3), undermining claims about perceptual quality and semantic fidelity.

3. Multi-person degradation: Quality drops when reference images contain multiple people (e.g., Fig. 4, last row), indicating instability under higher human-centric subject density.

4. Dataset analysis gaps: Lacks statistics visualizations for SemAlign-MS (e.g., subject category distribution, subjects count distribution per sample, prompt length), obscuring coverage and potential biases.


[1] "Fastcomposer: Tuning-free multi-subject image generation with localized attention." IJCV (2025): 1175-1194.

[2] "High-fidelity person-centric subject-to-image synthesis." CVPR, 2024.

### Questions
Please see the weakness. 

If the authors can thoroughly address all the listed weaknesses, I would be inclined to raise my score.

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
5

### Summary
This paper proposes the MOSAIC framework for the multi-subject-driven text-to-image generation task. Its core innovations are as follows:
1. Correspondence-Aware Alignment (SCAL): It utilizes a cross-image semantic correspondence mechanism to accurately map the feature spaces of different subjects to the target generation positions, avoiding subject localization and semantic confusion.
2. Multi-subject Disentanglement Layer (MDL): It achieves disentanglement between subjects at the feature layer, thereby enabling precise control over the appearance and position of each subject.
3. It constructs the SEALIGN-MS dataset, ensuring the accuracy of training supervision through geometric consistency and semantic filtering.
In the two benchmark tests of DreamBench and XVerseBench, MOSAIC outperforms existing methods in multiple metrics (CLIP-I, CLIP-T, DINO, DPG, ID-Sim, IP-Sim, AES).
Overall, the paper is relatively comprehensive in terms of data construction, algorithm design, and experimental validation, with the goal of improving the controllability and consistency of multi-subject generation.

### Strengths
1. It constructs a high-quality SEALIGN-MS dataset, improving the quality of supervision signals.
2. It introduces DIFT and GeoAware-SC for refined token-level correspondence, effectively alleviating the subject confusion issue in the multi-subject generation process.
3. Its performance in multiple target generation tasks reaches the state-of-the-art level.
4. The framework design can be extended to other conditional generation tasks, such as multi-person synthesis and television content generation.
5. SCAL is responsible for accurate mapping, and MDL is responsible for isolating interference; their combination significantly improves the quality of multi-subject tasks.

### Weaknesses
1. High computational complexity and lack of lightweight property: The corresponding point matching and cross-attention supervision in SCAL increase additional computational overhead, which may affect the inference speed.
2. The effect of MDL depends on mapping accuracy: If the correspondence of SCAL is inaccurate, the isolation effect of MDL may also be affected, leading to the contamination of subject features.
3. Dataset bias towards specific domains: Whether SEALIGN-MS has domain bias lacks distribution analysis and discussion.
4.The paper provides limited qualitative comparisons but lacks a systematic analysis of MOSAIC’s failure types such as local artifacts, semantic mismatch, and color shift, which limits the assessment of its robustness.

### Questions
1.The paper lacks quantitative analysis of inference efficiency and complexity.
2.In the Evaluation Setting, what are the specific details of the sample size, category distribution, and diversity of text prompts in DreamBench and XVerseBench? Is there any data bias that may affect evaluation fairness?
3.Are the category and scene distributions of SEALIGN-MS balanced?
4.Can step-by-step visualizations of the effects of MDL and SCAL on the generation results be provided in the appendix to facilitate a more intuitive understanding of their roles?
5.What is the basis for setting the LoRA rank to 128? Are there any ablation experiments on the impact of different ranks such as 32, 64, 256 on performance?
6.Can MOSAIC maintain consistent generation quality in multi-subject scenes when one reference image is high-resolution and another is low-resolution, or when some subjects are partially occluded?

### Soundness
3

### Presentation
4

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
The paper introduces MOSAIC, a multi-subject personalization framework that supervises reference to target attention with SCAL and reduces cross-subject leakage by MDL. It also releases SemAlign-MS, a dataset of dense point correspondences enabling effective attention-level supervision. On DreamBench and XVerseBench, MOSAIC achieves SOTA gains, for scenes with 4+ subjects.

### Strengths
1. The paper represent the multi-object personalization as representation optimization problem. Then introduce two different losses for the combined target. The storytelling is clear and straightforward.

2. The paper prove the 4+ object personalization tasks with less degradation. May apply for complex personalization cases.

3. The two losses introduced is well-defined and easy to follow.

### Weaknesses
1. The comparisons between the FLUX-based model to other baseline finetuned from SDXL may not be fair. While the increasing on performance is rather marginal. The method may works for other backbones as well since the main contribution is two of losses. It will be better to see the improvement across conventional backbones.

2. The quality of SemAlign-MS relies on the introduced construction models. It is not clearly how accurate the quality of the dataset itself.

3. Cost of computation complexity has not been clearly discussed.

### Questions
1. Can you report results on a second backbone (e.g., SDXL) to demonstrate novelty of SCAL/MDL?

2. How do performance and training cost scale with the number/quality of correspondence points?

3. How is quality check of the SemAlign-MS dataset?

4. Any failure cases for MOSAIC still struggles?

### Soundness
3

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
4

### Summary
tries to improve multi-subject personalized generation, by both aligning semantics for each subject/identity (knowing which regions in the generated image should attend to which parts of each reference), as well as disentangling each reference by pushing different subjects into orthogonal subspaces to minimize feature interference. They resolve the alignment challenge with a fine-annotated dataset and correspondence loss, and approach the later with a disentanglement loss. While both aspects are been explored and methods and datasets are proposed to resolve the challenges, MOSAIC does repurpose the similar ideas in a novel way.

### Strengths
1. The proposed correspondence loss for alignment is attention-based, which seems principled and interesting to me. 

2. The multi-reference disentanglement loss is based on **orthogonality**, which is a fairly new perspective in multi-subject generation. This perspective is more theoretically grounded and can also inspire more theories and algorithms, the loss formulation can be considered as a novel contribution to how we enforce it in diffusion models.

3. The construction pipeline of SemAlign-MS is sound and inspires future development of new benchmark too.

### Weaknesses
1. SemAlign-MS lacks detailed information, e.g. I don't find number of samples, splits, distributions, etc.? To claim a dataset contribution, it is crucial to include these details, as well as examples with full annotations and comparisons with a few similar datasets.

2. Both the reference alignment and multiple-reference disentanglement have been proposed and studied. To me, the intuition and conceptual novelty is modest, e.g. [1] and its variants proposes similar alignment and [2] leverages orthogonal adaptation for multiple conditioning. I acknowledge that this work has technical novelty though. 

3. Table 3: Can you add the ablation setting with MD and without SCA? This helps me compare which contributes more. 

4. Can MOSAIC works for/generalizes to other datasets/tasks besides SemAlign-MS? The authors claim that MOSAIC achieves SOTA on multiple benchmarks but I don't find such experiment results. Also, I think the correspondence loss relies on fine-grained annotations, making it hard to decouple and apply to other datasets/tasks?

5. While the results look good, there lacks quantitative/qualitative evaluation of generation diversity. Because both additional losses are enhancing fidelity, it inevitably makes the training weighs identity preservation more.

[1] Xiao, Guangxuan, et al. "Fastcomposer: Tuning-free multi-subject image generation with localized attention." International Journal of Computer Vision 133.3 (2025): 1175-1194.

[2] Po, Ryan, et al. "Orthogonal adaptation for modular customization of diffusion models." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2024.

### Questions
Please see details in Weaknesses. I think this work presents new insights and shows technical novelty, but there also exists confusions and concerns regarding overclaims and experiment details.

### Soundness
3

### Presentation
3

### Contribution
2
