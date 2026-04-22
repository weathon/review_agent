# PEAR: Pixel-aligned Expressive humAn mesh Recovery

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 2, 4

## Abstract
Reconstructing a human mesh from a single in-the-wild image has long been a central research direction in computer vision. Existing approaches often provide only coarse reconstructions of the overall human structure, while still exhibiting noticeable misalignments in fine-grained regions such as the face and hands. Such subtle deviations may be progressively amplified in downstream tasks, leading to significant errors in the final outcomes. To address this issue, we propose PEAR—a unified framework for human mesh recovery and rendering. PEAR explicitly tackles two major limitations of current methods: inaccurate localization of fine-grained human pose details and insufficient photometric supervision for self-reconstruction.
Specifically, we train a transformer-based model to recover expressive 3D human geometry from a single 2D image, and integrate it with a neural renderer to jointly optimize geometry and appearance. This synergy substantially improves the accuracy of fine-grained human geometry while yielding higher-quality rendering results. In addition, we construct a large-scale dataset of images and videos with human annotations to support model training. Extensive experiments on multiple benchmark datasets demonstrate that the proposed approach achieves significant improvements in both geometric reconstruction accuracy and rendering quality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes PEAR (Pixel-Aligned Expressive Human Mesh Recovery), a unified framework that recovers expressive 3D human meshes from a single image. Unlike prior SMPL-X based methods, PEAR integrates the FLAME head into SMPL-X (forming the Expressive Human Model, EHM), improving facial expressiveness while preserving full-body modeling. The method further introduces a two-stage training pipeline:

* A ViT-based regressor predicts EHM parameters (body, face, hands, camera).

* A neural renderer is used with photometric loss for pixel-level enhancement.

Experiments on datasets such as UBody, 3DPW, EHF, and LSP-Extended show improvements in facial detail, hand accuracy, and overall pixel alignment. The method also demonstrates real-time avatar reconstruction (0.05s inference) and generalizes to downstream animation tasks. Ablation studies validate the benefit of two-stage training.

### Strengths
1. Pixel-level photometric supervision: The second-stage neural rendering significantly improves fine-grained alignment beyond joint/parameter losses.

### Weaknesses
1. FLAME head pose integration:
The paper states the proposed system estimates both SMPL-X and FLAME parameters, but it is unclear how global head orientation is consistently maintained. For example, when the body is rotated or facing away, naïvely replacing the head could cause inconsistencies between the body and face orientation. Clarification is needed on how alignment between body root pose and FLAME global pose is enforced.

2. Stage-2 reliance on upper-body datasets:
The second-stage training leverages upper-body 3DGS datasets (e.g., UBody) for pixel-level enhancement. This raises two concerns:

* Lower-body accuracy may degrade since pixel-level supervision is not applied consistently to the legs/feet.

* Upper-body datasets typically lack diverse whole-body poses, potentially limiting generalization.
Related to this, Table 5 shows ablation across UBody, 3DPW, EHF, and UBody-intra, but it is not clearly explained how the proposed method addressed the above two concerns.

3. Limited full-body evaluations:
While the paper reports on 3DPW, EHF, and UBody, it lacks broader validation on challenging full-body datasets like AGORA or BEDLAM, which test both pixel alignment and kinematic reconstruction under more diverse settings. Without these, the claim of strong generalization to complex whole-body scenarios is not fully convincing.

### Questions
Please refer to Weaknesses

### Soundness
3

### Presentation
3

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
The paper introduces a Pixel-Aligned Expressive Human Mesh Recovery framework that aims to enhance fine-grained human mesh alignment and facial expressiveness from a single image. It jointly estimates SMPLX and FLAME parameters to form an Expressive Human Model (EHM), following GUAVA (ICCV 2025), addressing the common misalignment and expression limitations of existing human motion estimation works.

### Strengths
- Paper is well-written and easy to understand.
- By combining SMPLX and FLAME within the Expressive Human Model and using photometric supervision, the framework effectively captures subtle facial expressions and hand detail, outperforming recent works on human mesh recovery.

### Weaknesses
- The core design of this work that combines an EHM (SMPLX + FLAME) with a neural renderer for pixel-level photometric supervision,  closely follows the formulation of GUAVA (Zhang et al., ICCV 2025). While PEAR extends GUAVA’s upper-body focus to full-body reconstruction and adopts a two-stage training pipeline instead of optimization-based parameter tracking, the overall architecture and objective remain conceptually similar. The paper should clarify the key algorithmic differences or technical contributions beyond scaling the framework to full-body meshes.
- In Eq. (6), the reconstruction parameter set $\Phi = [\theta_t^b, \beta_s^b, \theta_t^h, \beta_s^h, \phi_t^h, \pi_t]$
 mixes body and facial parameters from different frames (source and target), specifically, shape parameters ($\beta$) from the source and other parameters from the target. No ablation or comparison is presented to validate that using source $\beta$ yields better alignment or rendering quality than other alternatives (e.g., all parameters from the target). Ablation study would be necessary to support this design.
- Although the “Human Appearance Reconstruction” part on Related work discusses neural renderers and Gaussian-based methods, it omits a large body of mesh-based literature that directly tackles high-fidelity human surface recovery from single or multiple images. Incorporating these works would better improve the quality of this paper:
    - [1] Saito et al., *“PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization,”* ICCV 2019.
    - [2] Shin et al., *“CanonicalFusion: Generating Drivable 3D Human Avatars from Multiple Images,”* ECCV 2024.
    - [3] Xiu et al., *“ICON: Implicit Clothed Humans Obtained from Normals,”* CVPR 2022.
    - [4] Xiu et al., *“ECON: Explicit Clothed Humans Optimized via Normal Integration,”* CVPR 2023.
    - [5] Liao et al., *“High-Fidelity Clothed Avatar Reconstruction from a Single Image,”* CVPR 2023.
    - [6] Ho et al., *“SITH: Single-view Textured Human Reconstruction with Image-Conditioned Diffusion,”* CVPR 2024.

I will reconsider score when all my concerns are handled well.

### Questions
How does the model perform under challenging conditions such as occlusions, extreme lighting, or side-view inputs? Including visual examples of such cases would help readers understand the framework’s limitations.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed PEAR, a framework for single-image human mesh recovery that (i) regresses an Expressive Human Model (EHM) combining SMPL-X for body and FLAME for head, and (ii) jointly trains with a neural renderer to add pixel-level photometric supervision. The proposed method reduces mesh–image misalignment in fine regions (face, hands), improves facial expressiveness, and enables fast (≈0.05 s) parameter estimation for downstream animation. Quantitatively, PEAR improves head/hand/body reconstruction accuracy across multiple benchmarks and improves rendering metrics (PSNR/SSIM/LPIPS) via joint training.

### Strengths
- The method is straightforward and easy to understand; the manuscript is clearly written.

- The proposed method could perform full-body 3D modeling without the need for cropping.

- A large-scale human mesh dataset is annotated and slated for open release.

### Weaknesses
- Technical novelty. The proposed pipeline closely follows GUAVA:
(a) it adopts the enhanced human parametric model EHM (introduced by GUAVA);
(b) in Stage-1, EHM parameters are trained using pseudo ground truth generated by GUAVA;
(c) in Stage-2, the neural renderer reuses GUAVA’s pipeline.
Hence, compared with GUAVA, the difference is that instead of tracking-based EHM parameter estimation, this paper swaps in an HMR2-based parameter estimator and then jointly optimizes EHM estimation and the neural renderer, further fine-tuning on an extra annotated dataset. While this is potentially useful (pending stronger empirical validation that I have some concerns below), I remain concerned that the technical contribution may be insufficient for the ICLR bar.

- Experiment Results. I feel the current results do not fully substantiate the claims of “a more expressive human model” and “avoiding severe misalignments commonly observed in prior methods.” See questions for my specific concerns.

### Questions
- Table 1. What exactly are the two comparison methods? The paper never identifies them (e.g., optimization-based vs. learning-based? which body/head models are used?). Without these details, it’s difficult to draw meaningful conclusions from Table 1.

- Table 3. As shown, HMR2 outperforms the proposed method on all datasets and metrics. Given HMR2 uses SMPL while the proposed method uses the enhanced EHM, and the network architecture follows HMR2’s implementation, the statement in L318 (“Our approach achieves performance comparable to specialized body pose estimation methods such as HSMR and HMR2”) is hard for me to agree. At least, the paper should discuss this discrepancy.

- Table 3 is missing an important entry. As shown, on COCO and PoseTrack, among all SMPL-X methods, PyMAF-X appears to be the closest baseline to the proposed approach, yet its result on LSP is missing. Please clarify.

- Table 4: As reported, 2 minutes is required for reconstruction and 0.18s for rendering in Config-A (Tracking + GUAVA Renderer which is the GUAVA baseline). These numbers differ dramatically from GUAVA’s reported performance in their paper (≈52.2 FPS and ≈0.1s reconstruction in Tables 1 and 2 of [1]). Why is there such a large discrepancy? Please explain experimental setups and metrics used in this paper.

- Table 5 vs. Table 1. The MLE reported on UBody in Table 5 is inconsistent with Table 1—possibly by an order of magnitude. Please double-check and correct if needed.

References:
[1] Zhang, Dongbin, et al. "GUAVA: Generalizable Upper Body 3D Gaussian Avatar." arXiv preprint arXiv:2505.03351 (2025).

### Soundness
1

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
The proposed method aims to make human mesh recovery more expressive, faster, and better aligned with real pixels. The integration of neural rendering for supervision within a feed-forward transformer pipeline is a clear step forward in practical human modeling However, its conceptual novelty is moderate, being primarily an engineering synthesis rather than a fundamental model innovation. The dependency on pseudo labels and lack of deeper analysis (domain generalization, temporal consistency, and robustness to in-the-wild conditions) slightly weaken the scientific depth.

### Strengths
1. By jointly regressing SMPLX (body) and FLAME (head) parameters under the Expressive Human Model (EHM), the presented method unifies coarse pose estimation with fine-grained facial expressiveness, which is more practical than the SMPLX-only methods.
2. Real-time inference (0.05 seconds per frame) from a single 256×192 image, without cropping or high-resolution input, is practically valuable for downstream animation tasks and interactive applications.
3. The construction of a large-scale dataset with body–face–hand pseudo ground-truth (SMPLX + FLAME parameters) is a valuable community resource.

### Weaknesses
1. From the article, especially the contribution of the introduction part, it is unclear how the method achieves the promising results. Much of the framework builds upon GUAVA and HMR2, with the main innovation being the introduction of pixel-level supervision. Despite of integrating known components, what fundamentally new representational or algorithmic insight does PEAR introduce? Is the gain mainly from adding photometric loss?
2. The paper focuses on alignment but does not address the limitations in clothing diversity or interactions. Can PEAR handle loose clothing, hair, or accessories not modeled by EHM?
3. The improvement from stage 2 is shown but lacks deeper analysis. How does performance vary if the renderer is frozen vs. jointly trained? Does the photometric loss risk overfitting to appearance rather than geometry?
4. Although tested on public benchmarks, most datasets are lab-style or curated internet data. How well does PEAR generalize to truly in-the-wild inputs (e.g., occlusions, extreme expressions, motion blur)? Is there any evaluation on real-time video streams?

### Questions
1. It's necessary to clarify the main technical novelty of the proposed approach.

2.  Can PEAR handle loose clothing, hair, or accessories not modeled by EHM?

3.  How does performance vary if the renderer is frozen vs. jointly trained? Does the photometric loss risk overfitting to appearance rather than geometry?

4.  How well does PEAR generalize to truly in-the-wild inputs (e.g., occlusions, extreme expressions, motion blur)? Is there any evaluation on real-time video streams?

### Soundness
3

### Presentation
3

### Contribution
2
