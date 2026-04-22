# Prioritizing Faithfulness: Efficient Zero-Shot Novel View Synthesis with Adaptive Latent Modulation

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 8, 6, 4

## Abstract
The challenge of camera-controlled novel view synthesis (NVS) lies in balancing high visual fidelity with strict faithfulness to the source scene. We argue that current dominant approaches, which rely on finetuning large-scale diffusion models, often over-emphasize fidelity while struggling with faithfulness due to their generative nature. To address this, we propose a zero-shot NVS pipeline that prioritizes faithfulness and efficiency. Our method introduces two key contributions applied during inference: (1) Test-time Latent Homography Deformation, an on-the-fly homography optimization to deform latents for global motion consistency, and (2) Spatially Adaptive RePaint (SA-RePaint), an extension to RePaint that achieves both structural consistency and texture fidelity by introducing a mathematically-grounded, region-wise balancing of these two objectives. Our evaluations demonstrate substantial improvements in faithfulness and camera accuracy with competitive perceptual scores, highlighting a successful integration of faithfulness, quality, and efficiency. This work offers a promising direction for NVS that rebalances the focus towards greater authenticity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a training-free method for generative novel view synthesis. The proposed method is based on a video diffusion model and introduces two new components compared to prior art: A inference-time optimization for homographies that are used to warp latents of different video frames, as well as a spatially varying noise level used in the inpainting process. The introduced changes are motivated by apparent 'spurious motion', as well as 'drifting synthesis' artefacts in related methods, where the former refers to subjects not remaining multi-view consistent even though having been observed from a novel view, while the latter refers to image (part) generations not following the camera motion.

### Strengths
The paper introduces a few relatively simple fixes that seem to be nonetheless effective for the task of generative inpainting. 
The proposed spatially-adaptive RePaint variant seems novel and is well-motivated. Using the cross-attention of the video diffusion model as guidance signal to provide cross-frame correspondences is a neat trick in the absence of a clean, generated RGB video.

The paper is overall well-structured and provides an extensive amount of analysis and details on the proposed components, which facilitates reproducing the results.

### Weaknesses
**Evaluation and Comparisons to Prior Work**

(1)
The paper lacks comparison against multiple relevant related works. For example: InvisibleStitch [1], a method that additionally uses depth inpainting, which also resolves the issue of the proposed homography warping being not depth-aware or Stable Virtual Camera [2]. This makes me question whether all relevant related work has been cited and compared against.

(2)
The generated novel views shown in the qualitative results are in general less convincing than the prior work Trajectory Crafter, which is also reflected in the quantitative analysis. However, the reported "Faithfulness" scores are better for the proposed method, which makes me wonder how these scores were computed. A more thorough explanation on "valid" regions used for these metrics would be commendable.

(3)
While the paper provides a comparison to prior works wrt. geometric consistency using TSED, it would be commendable to include MEt3R [3], which is a more robust metric.


**Homography Estimation**

(4)
The proposed homography estimation can only be computed on the masked co-visible image parts. However, for these parts, the estimated homography should generally be close to Identity, as the initial latent that is noised and subsequently denoised is the depth-warped input image. Related to that, Figure 3b is misleading. The $z_{0|t}$ is denoised from the (already) depth-warped $y$ (as explained in Sec. 4), not the non-warped input image.

**Limited Applicability to More Recent Video Diffusion Models**

The proposed method does not directly extend to more recent video diffusion models that employ a VAE which introduces temporal compression, as the inpainting masks can then not be directly applied to the latents anymore.


**Paper Writing and Figures**

The paper writing is often not easy to follow. E.g., the description of different RePaint variants, which is an important pre-requisite for the remainder of the paper, is not easy to parse, especially the indexing used.

The figure quality is generally not great, often pixelated. I would recommend the authors to include figures with increased resolution in a revised version. Also, the section labels in Fig. 3b do not match with the paper sections.

[1] Engstler, Paul, et al. "Invisible stitch: Generating smooth 3d scenes with depth inpainting." 2025 International Conference on 3D Vision (3DV). IEEE, 2025.

[2] Zhou, Jensen, et al. "Stable virtual camera: Generative view synthesis with diffusion models." arXiv preprint arXiv:2503.14489 (2025).

[3] Asim, Mohammad, et al. "Met3r: Measuring multi-view consistency in generated images." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

### Questions
Following up on the above mentioned weaknesses, I would like to see clarifications regarding the following points:

(1) 
Is the diffusion model backbone used in the different implementations of related works comparable to the used one in this work? Same goes for the depth estimation model used to compute the initial 3D information.

Could you please include comparisons of your method against all relevant, openly available SOTA methods? E.g., InvisibleStitch or Stable Virtual Camera.

(2)
Please detail how the faithfulness scores were computed. Which region was considered "valid"?

(3)
How does the geometric consistency of generated novel views compare to prior art when evaluated using MEt3R?

(4)
How strong is the warping that is usually introduced through the computed homographies? Could be shown through some proxy metric like a histogram of the introduced area change / IoU of warped and original image space.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Based on the video diffusion model, the authors propose a training-free novel view synthesis method. Compared to previous state-of-the-art work, they introduce homography optimization and Spatially Adaptive RePaint, demonstrating their effectiveness on datasets.

### Strengths
1. It is a training-free approach that leverages pre-trained large-scale video diffusion models, promising potential for improved performance as video generation techniques advance.
2. The generated videos exhibit high visual fidelity and competitive quality compared to other methods.

### Weaknesses
1. There is a lack of comparison with Gaussian Splatting.

### Questions
Although the primary contribution of the paper is in fidelity comparison, the background in the images of Figure 6 appears overly blurred. It would be better to replace these images with ones that better validate the experimental results' competitive fidelity.  (The figures in the appendix would be much better)

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
3

### Summary
This paper introduces a training-free pipeline for NVS task from a single source image, which focuses on the faithfulness to the input while achieving efficiency. The method follows a render-then-inpaint manner: it first lifts the source image to a 3D point cloud, renders views along a specified camera trajectory. Then it inpaints disoccluded regions with SVD via a modified RePaint technique. 

The key contributions include: (1) test-time latent homography deformation, an optimization that aligns latent predictions with rendered images to prevent drifting synthesis and ensure motion coherence, and (2) SA-RePaint, which derives a per-pixel noise map to balance structural consistency and texture fidelity by matching local variances.

The authors claim that the proposed method outperforms existing methods in faithfulness and camera accuracy with competitive visual quality.

### Strengths
The paper proses to rebalance priorities in NVS of faithfulness and efficiency, which is often sidelined in favor of fidelity in generative approaches. The technical details are clearly derived (e.g., the closed-form solution for $\Sigma$ in Theorem 1) and illustrative figures effectively convey the pipeline and trade-offs. 

Overall, this paper is well-written. This work also presents a new perspective for generative NVS scenario where faithfulness is critical and efficiency is also demanded.

### Weaknesses
Given this is a training-free method for generative NVS, one weakness could be the assumption of the method.
The core method *Test-time Latent Homography Deformation* assumes largely planar or global motions (homography deformation), which may not handle complex parallax, non-rigid, and large-motion scenes well.

### Questions
1. How does the method perform on inputs with significant depth variations or dynamic elements, given homography's global nature? Although a brief discussion of the limitation in given in the appendix, some examples of failure cases would help evaluate the limitations of current method.

2. Since the proposed method relies on latent-space manipulations during inference and treats the video diffusion model as a black box, it appears naturally extendable to DiT architectures. Have the authors experimented with applying it to more recent DiT-based video models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work introduces a training-free NVS method aimed at improving faithfulness and geometric consistency in diffusion-based novel view synthesis. It introduces a Latent Homography Deformation module to enforce content coherence and a Spatially Adaptive RePaint mechanism to address the structure-texture trade-off. The proposed method achieves competitive results in both quantitative and qualitative evaluations on several benchmark datasets.

### Strengths
* This work addresses the challenging faithfulness issues that typically exist in NVS methods relying on diffusion models and manages to mitigate them through a training-free solution.


* The idea of applying homography warping in the latent space to address “drifting synthesis” issues is interesting and sound.


* The concept of applying per-pixel noise levels is mathematically well illustrated.


* The manuscript is well structured and easy to follow.

### Weaknesses
* **The ablation study seems confusing.** As reported in Tab. 2, compared to the baseline, the introduced components bring only minor improvements, or even worse results, in terms of faithfulness. This contradicts the motivation of “prioritizing faithfulness.” A more thorough analysis would help clarify this issue.


* **The improvements over other comparison models seem to mainly stem from the enhanced baseline.** As shown in Tab. 1, the proposed method performs significantly better than other state-of-the-art models. However, Tab. 2 shows that the final model performs even worse than the baseline. In this case, the superiority of the method may purely come from an improved baseline. It would be more convincing to apply the proposed modules to other baselines, such as Trajectory Crafter, to verify whether these introduced components genuinely contribute to the performance gains.


* **The effectiveness of the “prefill” module is not analyzed.** As mentioned in L208, the re-projected image is first prefilled with a classical inpainting method before being fed into the VAE. It would be helpful to show how important this step is. Moreover, applying the same prefill step to other comparison methods, such as Trajectory Crafter, could help ensure a fairer comparison.

### Questions
For the comparison figures (e.g., Fig. 6 and Fig. 7), it would be better to include the Ground Truth images to confirm whether the proposed method indeed maintains faithfulness better than others.

### Soundness
3

### Presentation
3

### Contribution
3
