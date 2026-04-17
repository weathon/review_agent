# Visual Geometry Transformer in the Wild: Distractor-Free 3D Reconstruction

- Decision: Reject
- Scores: 2, 6, 6, 6, 6

## Abstract
Current end-to-end multi-view 3D reconstruction methods achieve impressive results, but are built on a restrictive assumption: the scene is entirely static with dense correspondence.
This reliance on idealized inputs causes even the most advanced methods to fail in real-world settings, where transient distractors and occlusions present. To address this, we propose \emph{Visual Geometry Transformer in the Wild} (VGTW), an end-to-end framework for robust reconstruction from inconsistent views. At its core, we isolate and suppress distractor-affected regions while preserving the consistent components across views. Specifically, we introduce a distractor-aware training strategy that separates clean features from distractor-contaminated ones in the attention mechanism while enforcing feature consistency across images. To enable this, we train the model with an auxiliary mask prediction head, using supervision from a new dataset we collected with pixel-level distractor masks. The resulting VGTW model is a feed-forward network that directly outputs clean, distractor-free point clouds. Remarkably, it requires no additional 3D supervision, remains computationally efficient, and is compatible with existing pipelines.
Extensive experiments validate our approach, demonstrating state-of-the-art performance and robust generalization in diverse, real-world scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents *VGTW (Visual Geometry Transformer in the Wild)*, a feed-forward transformer-based pipeline that aims to produce distractor-free 3D reconstructions from multi-view images. The authors fine-tune existing feed-forward 3D models (VGGT, π3) with LoRA and introduce two losses (Distractor Suppression and Cross-View Consistency) plus a distractor mask prediction head. They claim improved robustness to transient distractors and strong results on NeRF-on-the-go and RobustNeRF.

While the paper has some strengths, such as insightful analysis of attention leaking to distractors, intuitive loss formulations, there are several concerns. The claims about prior feed-forward methods appear overstated, discussion about several highly related work is missing, and the evaluation is limited and sometimes unclear.

Overall, I would recommend reject as the contribution is incremental, the evaluation has gaps, and some claims are overstated.

### Strengths
1. **Insightful analysis of attention and loss design.** The observation that attention leaks to distractors is both insightful and clearly illustrated in Figure. The design of the two losses (Distractor Suppression and Cross-View Consistency) demonstrates benefits for achieving distractor-free predictions.  

2. **Clarity and presentation.** The paper is generally well-written, with clear figures and intuitive formulations for the proposed losses, and the methodology is easy to follow.  

3. **Strong experimental results.** The method achieves improved metrics on the RobustNeRF and NeRF-on-the-go datasets, showing its effectiveness in handling scenes with distractors.

### Weaknesses
1. **Overstated claim.** The paper repeatedly suggests that VGGT/π3 conceptually cannot handle dynamic scenes. This is not true, as existing feed-forward architectures are already designed to be robust to non-static inputs (especially π3). Therefore, the premise of the paper appears overstated.  

2. **Missing related work.** The paper overlooks directly relevant studies that adapt DUSt3R-style frameworks for dynamic scenes, such as Monst3R [a] and Easi3R [b]. In particular, Easi3R discusses attention mechanisms for handling dynamic objects in reconstruction. This omission makes the contribution appear less novel.  

3. **Limited and questionable evaluation.**  
- Dataset mismatch: Both datasets were originally designed for "distractor-free" NeRF-based **novel view synthesis** rather than feed-forward reconstruction evaluation. The paper mentions that ground truth is "generated using pretrained π³ on distractor-free images" but does not clarify:  
  a. During inference, how are "distractor-free" point clouds obtained, are they using confidence filtering?  
  b. If so, what confidence threshold is used (0.5 in L358 in training section?), and is there a specific reason for choosing that threshold?  
  c. The methods still output distractor point clouds without explicit occlusion area generation. Would simply removing the distractor regions  and comparing it with "GT" point cloulds be a fair evaluation setting?
- Small scale: Only a few cases/scenes are used for evaluation, which seems insufficient for feed-forward methods that claim generalization.

---

[a] Zhang, J., Herrmann, C., Hur, J., Jampani, V., Darrell, T., Cole, F., Sun, D., & Yang, M.-H. (2025). *MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion*. In ICLR 2025.  
[b] Chen, X., Chen, Y., Xiu, Y., Geiger, A., & Chen, A. (2025). *Easi3R: Estimating Disentangled Motion from DUSt3R Without Training*. In ICCV 2025.

### Questions
1. The idea of generating distractor mask is interesting. Could the authors provide qualitative results for the mask? Additionally, I am interested in seeing how using only the mask head (without distractor-aware training, or just use the mask for pretrained π3 predictions) would affect the results.  

2. Regarding training, could the authors specify the computational resources used (e.g., number of GPUs, total training time, and learning rate)? It also appears that the fine-tuning is performed without 3D supervision. Could the authors clarify how much this affects the 3D reconstruction performance?

3. Could the authors explain the evaluation procedure in more detail, as mentioned in *Weakness 3*?  

4. For *Tables 1–3*, the authors might consider adding average scores to facilitate easier comparison.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes to improve the robustness of feed-forward 3D reconstruction models to view-inconsistent distractors such as moving objects and occluders. To this end, the authors first investigate the attention mechanism under distractors: given ground-truth distractor masks, they set the attention logits of the corresponding regions to negative infinity, which markedly improves existing models when distractors are present. Building on this observation, they introduce Distractor-Aware Training (DAT) that fine-tunes attention layers via LoRA and adds distractor-suppression and cross-view consistency losses. To demonstrate effectiveness, the authors annotate a new dataset with distractor masks, and the resulting models outperform strong baselines on real scenes with distractors.

### Strengths
* Originality:

  * To the best of my knowledge, this paper is the first to systematically investigate distractors in attention for feed-forward 3D reconstruction and to propose a corresponding solution that directly addresses the identified failure mode.

* Quality:

  * The experiments are comprehensive, covering different types of distractors and demonstrating effectiveness across multiple strong baselines.

* Clarity:

  * The paper is well organized: it states assumptions, conducts oracle experiments with distractor masks to validate them, and then proposes a practical method informed by these findings.
  * The method is clearly introduced with sufficient technical detail to understand and reproduce.

* Significance:

  * The problem addressed is complementary to existing feed-forward 3D reconstruction efforts; by improving robustness to distractors, the approach has the potential to become a general building block for this family of models.
  * The annotated RobustNeRF-Mask dataset can further catalyze follow-up research on making feed-forward 3D reconstruction methods more robust to distractors.

### Weaknesses
* The notion of “distractor” for attention needs a more precise quantitative definition. Unlike NeRF-style, per-scene optimization on a single sequence, this paper targets large-scale, feed-forward transformers with attention. In such settings, dynamic objects with continuous motion can still induce high cross-view correlation and be incorporated into reconstruction. Therefore, I would recommend to quantify distractor “lifetime” (number of frames or views), apparent speed (pixels/frame or m/s under known intrinsics/poses), and spatial extent, then analyze performance as these variables change. 

* From the qualitative results, improvements in static regions appear marginal. To make the gains more pronounced and measurable, I suggest: (1) also evaluating camera pose accuracy (ATE/RPE) to show benefits beyond point/depth; (2) reporting metrics on static-only regions by masking out distractor areas per frame, so improvements to clean content are not diluted by scene-wide averaging.

* The robustness gains rely on additional supervision from annotated masks, but the current dataset scale is limited; generalization needs stronger support:

  * Assess whether fine-tuning degrades performance on distractor-free data (report deltas on clean subsets).
  * Evaluate generalization to “atypical” distractors absent or rare in training (hold-out categories, cross-domain scenes), and document any regressions.

* It remains unclear why the distractor mask head is not explicitly injected into attention, akin to the “oracle” masking in Section 3. At present, correlation between distractor and static regions is reduced via feature-space objectives alone, which may struggle when distractors share similar appearance or texture with static content. Actionable: add an inference-time attention gating variant that consumes the predicted mask (soft or hard), compare against DAT-only, and analyze cases with look-alike distractors to test whether explicit gating closes the gap to the oracle.

### Questions
- First, please see my weaknesses above.
- Second, Figure 3 is confusing. Since the paper uses LoRA, the intra-frame and inter-frame VGGT transformer blocks should be frozen; however, in Figure 3 these blocks are labeled as trainable. Please clarify.
- Third, how does the method perform on video data with continuously persistent moving objects, e.g., the DAVIS dataset? The experiments on real dataset is important to clarify some aforementioned weaknessness.

### Soundness
3

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
4

### Summary
This paper introduces Visual Geometry Transformer in the Wild (VGTW), a feed-forward multi-view 3D reconstruction framework designed to handle real-world scenarios with transient distractors such as moving people or vehicles.
Built upon prior transformer-based methods like VGGT and π³, VGTW adds:

a Distractor-Aware Training (DAT) strategy that fine-tunes attention via LoRA to suppress dynamic regions;

two novel loss functions—Distractor Suppression Loss and Cross-View Consistency Loss;

an auxiliary mask prediction head for identifying dynamic regions;

a new RobustNeRF-Mask dataset constructed using SAM2 segmentation and optical-flow consistency to generate pixel-level distractor annotations.

The resulting model can directly output clean, distractor-free 3D point clouds and camera poses without requiring any 3D supervision, showing strong performance and robustness across multiple benchmarks.

### Strengths
Addresses a real-world gap: VGTW is the first feed-forward 3D reconstruction framework that explicitly handles transient distractors, a limitation of prior models like VGGT or DUSt3R.

Conceptually simple yet effective: By introducing DAT and LoRA-based fine-tuning, the authors enhance robustness without altering the backbone structure.

No 3D ground-truth supervision: The model only relies on 2D distractor masks, keeping the training lightweight and practical.

Solid empirical results: Experiments show consistent improvements over strong feed-forward baselines, particularly in scenes with heavy occlusions or dynamic content.

### Weaknesses
Dependence on mask supervision:
The method heavily relies on the new RobustNeRF-Mask dataset, where distractor masks are generated using SAM2 and optical-flow consistency, followed by partial manual refinement.
This external dependence limits generalization and raises concerns about the method’s robustness if mask quality degrades or manual curation is reduced.

Limited and potentially unfair comparisons:

Feed-forward baselines (VGGT, MASt3R, DUSt3R, etc.) do not use any distractor supervision, so performance gains might stem from additional training signals rather than methodological novelty.

The paper does not compare against optimization-based dynamic reconstruction methods (e.g., NeRF-W, WildGaussians, SpotLessSplats). While paradigms differ, such a comparison would contextualize VGTW’s efficiency–quality trade-off.

Ablation incompleteness:
While DAT and mask head ablations are shown, there is no study on varying mask quality, missing masks, or cross-domain generalization, making it unclear how resilient the model is to imperfect annotations.

### Questions
Could the authors provide details on the extent of manual correction performed during RobustNeRF-Mask construction? How significant was human intervention relative to SAM2 + optical flow auto-labeling?

To ensure fairness, have the authors tested a VGGT or π³ baseline trained with the same distractor masks (e.g., applying identical supervision but without DAT) to isolate the contribution of the proposed training strategy?

Have the authors compared VGTW against dynamic-object-robust NeRF or Gaussian Splatting methods (e.g., NeRF-W, WildGaussians)? Even if paradigms differ, a runtime–quality comparison would help clarify the practical advantage of feed-forward reconstruction.

How robust is VGTW to noisy or imperfect masks? Have the authors quantified how performance drops when mask quality decreases or when no masks are available for fine-tuning?

Since the model is designed for “in-the-wild” data, have the authors tested it on completely unseen domains (e.g., night scenes, handheld videos, or fast-moving dynamic scenes) to validate generalization?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces VGTW (Visual Geometry Transformer in the Wild), an end-to-end feed-forward 3D reconstruction framework for handling inconsistent multi-view images with transient distractors.
It employs a Distractor-Aware Training (DAT) strategy with two tailored losses and an auxiliary mask head for distractor suppression, trained on the proposed RobustNeRF-Mask, a dataset with pixel-level distractor annotations that enables distractor-free 3D reconstruction without 3D supervision.

### Strengths
1. The paper is well written, clearly structured, and easy to follow.

2. The motivation is clear, focusing on the challenge of transient distractors in real-world multi-view 3D reconstruction.

3. The technical design is reasonable, combining distractor-aware attention, consistency losses, and a mask head in a lightweight manner.

4. The experiments demonstrate consistent improvements over baseline methods, yielding cleaner and more reliable 3D reconstructions.

### Weaknesses
1. **Insufficient evaluation on standard benchmarks**  
The method is mainly evaluated on the dataset introduced in the paper for distractor-free 3D reconstruction. It remains unclear whether the proposed approach maintains comparable performance when input images contain no distractors (i.e., fully static scenes). An additional evaluation on standard benchmarks such as DTU [1] or ETH3D [2] would help verify generalization.

2. **Unclear dataset annotation and segmentation process**  
The paper (around lines 187–188) mentions that “This may include dynamic objects, occluders, and non-rigid deformations.” as part of its distractor definition, but it does not clearly explain how the corresponding motion masks or annotations are obtained. Although the paper indicates the use of SAM for segmentation, it remains unclear how complex interactions or partial deformations are handled. For example, when a standing person opens a refrigerator, is the entire refrigerator segmented as dynamic, or only the moving door? Similarly, is the whole person segmented as dynamic, or only the moving hand? Clarification is needed on whether the segmentation results were manually refined, automatically filtered, or heuristically selected to ensure labeling correctness.

3. **Lack of comparison with dynamic reconstruction baselines**  
The paper omits discussion and comparison with dynamic scene reconstruction methods such as MegaSAM [3] and MonST3R [4], which process full video sequences rather than image collections. Including at least one comparison or discussion with these approaches would strengthen the empirical analysis and better clarify the positioning of the proposed method.

4. **Potential bias in evaluation setup**  
The ground truth is labeled using π³ [5], which is also included as one of the baselines. This raises potential fairness concerns in the evaluation process. One possible solution is to follow PAGE-4D [6] and perform additional evaluation on an independent dataset such as DyCheck [7] to ensure unbiased comparison.

5. **Citation formatting issue**  
The citation in line 125 appears to be incorrect and contains unresolved reference symbols (“;?”). 

**Reference**

[1] Large-Scale Multi-View Stereopsis Evaluation.  
[2] A Multi-View Stereo Benchmark with High-Resolution Images and Multi-Camera Videos.  
[3] MegaSAM: Accurate, Fast, and Robust Structure and Motion from Casual Dynamic Videos.  
[4] MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion.  
[5] π³: Permutation-Equivariant Visual Geometry Learning.  
[6] Monocular Dynamic View Synthesis: A Reality Check.

### Questions
1. The method based on π³ fails to improve the NC metric on the evaluation datasets. Could the authors explain the reason behind this?

2. Although the method is described as lightweight, inference time comparisons are not reported. Could the authors provide quantitative runtime results to support this claim?

3. Could the authors provide more detailed w/o experiments to isolate the effect of each proposed component and demonstrate how each contributes to the overall performance improvement?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a fine-tuning approach based on the state-of-the-art geometric foundation models (e.g., VGGT), to tackle the distrator regions without point correspondence across multi-view input. Author first demonstrate the visualization on the attention map of original inference between using the distrator mask and w/o the mask. Then a couple of loss supervisions are proposed to enhance the feature similarty on the real matched points while suppressing the distractors. Besides, a dataset containing distractor mask is built and released for better problem setup. Extensive experiments showcase the effectiveness of the method.

### Strengths
(1) The paper is well motivated and handling the non-matched regions is an important yet challenging problem in multi-view geometry in a long run. The visualization of the cross attention map for those regions are admirable, to make readers better elaborate the challenge. 

(2) The introduced loss functions are techncially sound to enhance the feature similarity on the true positive matched regions, while suppressing the false positive ones. The loss functions are simple yet effective to be designed and implemented. 

(3) The experiments have demonstrated the validness of the design.

### Weaknesses
(1) The scale and diversity of constructed dataset with the distractor mask is limited (1000 annotated images on based on a single RobustNeRF) dataset, making the problem hard to be scaled up and extended. I was wondering whether using some pretrained optical flow network or dense point matching network, or simply SAM2, can scale up the annotation dataset effectively. By training on diverse dataset, the proposed method could demonstrate the generalizability over wild images.

(2) The method is only evaluated on NeRF-on-the-go dataset, which is hard to measure the generalizability and zero-shot capacity of the proposed method. Besides, authors train and evaluate the framework on the same dataset (RobustNeRF), which makes the technical contribution less convincing since all the baseline method are not trained on this dataset. I would suggest authors to do multiple cross-dataset evaluation to validate the generalizability of the method.

### Questions
Overall the motivation of the paper is encouraging and the technical contribution is clear. However, I still have concerns on the scalability and evaluation protocol as presented in the 'weakness' part. I hope authors could supplement more experiments to further demonstrate the generlizability of the method.

### Soundness
3

### Presentation
3

### Contribution
3
