# FFHQ-Makeup: Paired Synthetic Makeup Dataset with Facial Consistency Across Multiple Styles

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 4, 2, 2

## Abstract
Paired bare-makeup facial images are essential for a wide range of beauty-related tasks, such as virtual try-on, facial privacy protection, and facial aesthetics analysis. However, collecting high-quality paired makeup datasets remains a significant challenge. Real-world data acquisition is constrained by the difficulty of collecting large-scale paired images, while existing synthetic approaches often suffer from limited realism or inconsistencies between bare and makeup images.
Current synthetic methods typically fall into two categories: warping-based transformations and text-to-image generation. The former often distorts facial geometry and compromises makeup precision, while the latter tends to alter facial identity and expression, undermining consistency.
In this work, we present FFHQ-Makeup, a high-quality synthetic makeup dataset that pairs each identity with multiple makeup styles while preserving facial consistency in both identity and expression. Built upon the diverse FFHQ dataset, our pipeline transfers real-world makeup styles from existing datasets onto 18K identities by introducing an improved makeup transfer method that disentangles identity and makeup. Each identity is paired with 5 different makeup styles, resulting in a total of 90K high-quality bare–makeup image pairs.
We release FFHQ-Makeup as the first large-scale, multi-style, paired bare–makeup dataset, which we expect will serve as a valuable resource for future research in beauty-related tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper constructs a new facial makeup dataset, named FFHQ-Makeup, to support the research in the area of face image beautification and aesthetic analysis. The key idea is to disentangles identity and makeup so multiple makeup styles can be rendered while preserving facial consistency in both identity and expression. The constructed FFHW-Makeup dataset contains 90K (18Kx5) images, which could serve as a valuable resource for future research in face-related vision research.

### Strengths
Originality. I think simultaneous rendering of multiple makeup images has its novelty. To my knowledge, most previous methods only studied the single output scenario. Single-in-Multiple-out has its merit due to the large scale and diversity.
Quality. The reported experimental results as shown in figures and tables generally support the superiority of the proposed method to the baseline/benchmark methods.
Clarity. The paper is easy to follow and understand its contributions.
Significance. The constructed dataset will be a valuable contribution to support facial beauty-related research in computer vision.

### Weaknesses
1. Generally speaking, the technical depth of a paper on dataset construction is shallow. This paper is based on known style transfer techniques and does not develop new algorithms or tools. The novelty can be at most argued at the system or application level.
2. I think the biggest weakness of this work lies in the significance part. It is difficult to advocate for a paper with a relatively narrow technical scope (e.g., face beautification). The ultimate impact along this line of research is limited.
3. Relevance to ICLR. If I were the authors, I would submit this work to CV conferences including biometrics (e.g., FG2025). I don't think face beauty-related work will attract wide interest from the ICLR attendees.
4. Literary presentation. There are several places authors could have polished - e.g., the lack of balance between background and new contribution in abstract write-up, the conciseness of Sec. 3.2 (Method), Fig. 8 appears before Fig. 7, and the shortage of material in Appendix.

### Questions
1. What modification to 3DMM did you adopt for the makeup transfer to work? If any, I think this could be some contribution you can claim and elaborate on.
2. Can this line of research be extended into other style transfer of face images than makeup (e.g., aging, expression, race, and gender)?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a high-quality synthetic makeup dataset FFHQ-Makeup, which pairs each identity with multiple makeup styles while preserving facial consistency in both identity and expression. In order to achieve that, they introduce an improved makeup transfer method that disentangles identity and makeup and transfers real-world makeup styles from existing datasets onto 18K identities upon the diverse FFHQ dataset. Evaluations show that FFHQ-Makeup outperforms existing datasets in both visual quality and facial consistency.

### Strengths
Dataset contribution. This work onstructs a large-scale high-quality and multi-style paired makeup dataset, which would benefit a wide range of future makeup-related research and applications.

### Weaknesses
1. Limited technical novelty. The pipeline mainly relies on the existing model Stable-Makeup. The data construction pipeline appears to merely process existing data using off-the-shelf models, without addressing any substantive technical challenges.
2. Insufficient motivation and lack of interpretability. The ablation study focus on two variants of feature extraction: makeup residual and sampling and re-rendering augmentation. This appears to be only a minor modification of the module, which seems more like an engineering adjustment, and there seems to be no explanation in the methods or experiments section regarding the motivation or justification for this change.
3. Insufficient dataset evaluation. Relying solely on large models for dataset evaluation lacks stability. Furthermore, the evaluation prompts are overly simplistic and fail to provide the models with clear scoring criteria for assessing makeup realism and facial consistency, resulting in low reliability of the evaluation results.
4. Lack of quantitative comparison in ablation study.
5. The document layout does not conform to the required formatting guidelines; it must be set in a single-column format.

### Questions
1. What's the motivation of the improvements in both facial structure control and makeup feature extraction?
2. How is the score for Facial Consistency on the FFHQ-Makeup dataset? Has it been compared with other datasets?
3. Could any other evaluation metrics beyond large models be provided to assess the dataset quality?
4. Is there quantitative comparison results for ablation study？
5. For the two key improvements in both facial structure control and makeup feature extraction, it seems that only the ablation results of makeup feature extraction are provided. Is there ablation results of facial structure control?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces FFHQ-Makeup, a large-scale synthetic paired makeup dataset built upon FFHQ. The authors propose a 3D Morphable Model (3DMM)-guided pipeline that disentangles facial structure from makeup appearance. The dataset aims to provide high facial consistency while maintaining makeup realism and style diversity.

### Strengths
- The dataset construction pipeline is well-structured and combines multiple techniques to improve facial consistency.
- The paper provides thorough ablation studies and qualitative comparisons against existing synthetic datasets, showing clearer visual fidelity and identity preservation.
- The public release of such a large paired dataset could be beneficial for downstream research in makeup transfer and facial analysis.

### Weaknesses
- Limited novelty. The work primarily extends existing diffusion-based makeup transfer pipelines with 3DMM-based residual computation. While this combination is technically reasonable, it appears more as an incremental improvement rather than a conceptual breakthrough. The paper could better clarify what is fundamentally novel about the method compared to previous synthetic data generation approaches.
- In addition, insufficient validation on downstream tasks. The dataset is evaluated mainly on perceptual metrics and user preference studies, but there is no demonstration of how using FFHQ-Makeup actually improves performance on downstream tasks such as makeup transfer, face recognition, or virtual try-on.
- Structural distortion in synthetic faces. Although the paper emphasizes facial consistency, examples show that facial geometry can subtly change after generation. These deformations may be inherent to the diffusion-based synthesis pipeline, but they raise concerns about whether such synthetic pairs truly reflect consistent identity and structure. The paper acknowledges these issues but does not quantify their impact or provide mitigation analysis.
- Unclear handling of partial makeup in FFHQ. Since many faces in FFHQ likely contain light or partial makeup, it is questionable whether applying synthetic makeup on top of already makeup-bearing faces introduces unintended compounding artifacts. The paper briefly mentions manual filtering (Appendix Fig. 10) but does not clarify how consistently this issue is addressed or how many such cases remain.

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper releases FFHQ‑Makeup, a synthetic paired bare–makeup dataset built by transferring makeup from real sources onto FFHQ faces. The pipeline builds on Stable‑Makeup/FreeUV with three ingredients: reconstruct a bare face from a makeup image using 3DMM fitting; compute a 3DMM‑based “makeup residual” and re‑render it on target geometry for style disentanglement; and apply mask‑guided background blending plus extensive manual filtering. The resulting dataset contains 18K identities, each paired with five makeup styles. Quality is assessed via identity/semantic similarity metrics (ArcFace, DINO‑I, SSIM) on pairs, ablations of residual/augmentation, and an automated visual preference study using VLMs (GPT‑4o, Gemini 2.5, Claude) on a small subset. The paper claims better facial consistency and comparable realism to prior synthetic resources (e.g., LADN‑Syn, BeautyBank) and positions the dataset as a general resource for makeup transfer, VTO, and related tasks.

### Strengths
- Scale and structure: reasonably large, paired, multi‑style dataset; pairs are useful for supervised training and controlled evaluation.
- Clear construction pipeline with pragmatic engineering (3DMM‑based residual, re‑rendering augmentation, background blending) and documented manual cleaning.
- The paper is clearly written and acknowledges several remaining limitations (e.g., bias toward daily styles, 3DMM/segmentation artifacts).

### Weaknesses
- Utility not convincingly demonstrated. A dataset paper should show that training models on the new data substantially improves downstream tasks (e.g., makeup transfer, virtual try‑on, recognition under makeup) against strong baselines and across public test sets. The paper lacks such end‑task training/evaluation; results are mostly pairwise similarity and small‑scale preference checks, which do not establish practical value.
- No human evaluation. All “preference” judgments use VLMs on ~50 groups, which are not a substitute for human raters and can be biased by prompts or model idiosyncrasies. A user study assessing realism, identity/expression preservation, and artifact rate is essential for a perceptual domain like makeup.
- Limited fairness and coverage analysis. FFHQ provides diversity, but the paper does not quantify demographic distributions and performance disaggregations (skin tone, age, gender presentation). For a face dataset, absence of such analysis is a major gap.
- Modest novelty. The pipeline is an incremental engineering combination over Stable‑Makeup/FreeUV/ControlNet with 3DMM fitting and residual re‑rendering; as a dataset contribution, this is fine, but then the burden of proof shifts to rigorous evidence of utility (missing here).
- Evaluation scope and rigor. Identity/semantic similarity metrics are reported, but there is no cross‑dataset generalization (train on FFHQ‑Makeup, test on real‑world sets), no comparisons under challenging conditions (extreme styles, occlusions), and no stress/failure analysis beyond a few visuals. The VLM preference study uses unfiltered outputs and a small sample; sensitivity to prompt/model choice is not examined.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1
