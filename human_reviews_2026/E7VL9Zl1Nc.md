# FastGHA: Generalized Few-Shot 3D Gaussian Head Avatars with Real-Time Animation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 8, 4

## Abstract
Despite recent progress in 3D Gaussian-based head avatar modeling, efficiently generating high fidelity avatars remains a challenge. Current methods typically rely on extensive multi-view capture setups or monocular videos with per-identity optimization during inference, limiting their scalability and ease of use on unseen subjects. To overcome these efficiency drawbacks, we propose FastGHA, a feed-forward method to generate high-quality Gaussian head avatars from only a few input images while supporting real-time animation. Our approach directly learns a per-pixel Gaussian representation from the input images, and aggregates multi-view information using a transformer-based encoder that fuses image features from both DINOv3 and Stable Diffusion VAE. For real-time animation, we extend the explicit Gaussian representations with per-Gaussian features and introduce a lightweight MLP-based dynamic network to predict 3D Gaussian deformations from expression codes. Furthermore, to enhance geometric smoothness of the 3D head, we employ point maps from a pre-trained large reconstruction model as geometry supervision. Experiments show that our approach significantly outperforms existing methods in both rendering quality and inference efficiency, while supporting real-time dynamic avatar animation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a feed-forward framework for reconstructing animatable 3D Gaussian head avatars from only a few input images. The model first reconstructs a canonical Gaussian head using multi-view features extracted from pre-trained DINOv3 and Stable Diffusion VAE encoders, fused by a transformer-based module. Then, a lightweight MLP conditioned on FLAME expression codes predicts per-Gaussian deformations for real-time facial animation. Additionally, a geometry prior from the VGGT is employed as regularization to improve 3D consistency.

### Strengths
- Clear two-stage design that separates canonical reconstruction and dynamic animation.

- Efficient inference: <1s reconstruction and >60 FPS animation.

- Use of VGGT geometry prior enhances structural consistency.

- Outperforms Avat3r quantitatively in controlled datasets.

- Technically elegant combination of DINOv3, diffusion VAE, and Gaussian Splatting.

### Weaknesses
-  Methodological innovation over Avat3r is relatively incremental, primarily combining known modules (DINOv3, VAE, VGGT) rather than introducing a fundamentally new approach.
- Strong dependency on accurate camera poses, which significantly limits real-world usability. The method cannot operate from uncalibrated or monocular video, unlike many recent “in-the-wild” approaches, even though the paper acknowledges it in the limitations.

- In-the-wild results are weak and underexplored. Only a few examples are shown, and they exhibit degraded appearance and noticeable artifacts. The paper should provide more analysis on why generalization drops and how to mitigate it.

- Mouth region artifacts are frequent (see Figs. 2–3 and video results), with incomplete geometry and texture distortions, suggesting the method struggles with open-mouth or speaking expressions.

- Most visualizations show static reconstructions, not novel-view or novel-expression results, which are more important in avatar applications.

- The paper lacks convincing results for large-viewpoint novel renderings; side-view artifacts are visible in the supplementary video, contradicting the claim of a fully 3D avatar.

### Questions
- How does the model behave if camera calibration is slightly inaccurate or unavailable? Could the method be extended to work with monocular or unposed multi-view inputs?

- Can diffusion-based priors help improve in-the-wild generalization?

- Please provide more quantitative metrics for novel-view (side/back-view) and novel-expression performance.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces FastGHA, a novel feed-forward method designed to overcome the challenges of efficiently generating high-fidelity 3D Gaussian-based head avatars by utilizing only a few input images.

This approach notably supports real-time animation and achieves high fidelity through a two-stage pipeline: first reconstructing a canonical Gaussian head representation, which disentangles identity from the input expressions, followed by a deformation process. The system learns a per-pixel Gaussian representation, aggregating multi-view information using a transformer-based encoder that fuses semantic features from DINOv3 and color features from a Stable Diffusion VAE. For animation, the explicit Gaussian representation is augmented with learnable per-Gaussian features, and a lightweight MLP-based dynamic network is introduced to predict fine-scale 3D Gaussian deformations using descriptive expression codes, such as FLAME parameters. To improve the geometric consistency and robustness of the resulting 3D head, the method employs point maps derived from a pre-trained large reconstruction model (VGGT) as geometry supervision in the form of a regularization loss during training.

Extensive experiments confirm that FastGHA significantly outperforms existing state-of-the-art methods in both rendering quality and inference efficiency, enabling avatar reconstruction in less than one second.

### Strengths
- The paper presents a clear methodology and is well-structured.
- FastGHA employs a feed-forward approach to directly predict a per-pixel Gaussian representation. This design is explicitly chosen to enable instantaneous reconstruction of unseen subjects, avoiding lengthy per-identity optimization or template Gaussians rigged to 3DMMs required by many prior methods. Previous approaches struggled to handle fluffy elements like hair due to the rigging constraint.
- The method significantly addresses efficiency drawbacks by supporting real-time dynamic avatar animation. This speed is achieved through a lightweight MLP-based dynamic network that predicts fine-scale Gaussian deformations from expression codes, contrasting with computationally slower methods like those using cross-attention blocks. For a standard four-image input, the avatar reconstruction takes less than one second, and the animation speed remains high (e.g., 62 FPS for four inputs).
- FastGHA demonstrably outperforms existing state-of-the-art methods (including Avat3r, InvertAvatar, and GPAvatar) in reconstruction fidelity, showing a large gap in quantitative metrics such as PSNR, SSIM, LPIPS, and identity preservation.

### Weaknesses
- The paper identifies Avat3r as the "most related" state-of-the-art method. However, the crucial quantitative comparison table (Table 1) lacks the results for Avat3r on the Nersemble dataset. This prevents a full comparison, especially given that FastGHA's best performance is achieved when training on both Ava-256 and Nersemble ("Ours (both)").
- The animation pipeline depends on the accuracy of FLAME expression codes ($z_{exp}$) obtained using "off-the-shelf head tracking tools". Errors generated by these trackers when processing complex or extreme expressions (e.g., a wide open mouth or squinting, as shown in the input examples) could compromise the fidelity of the downstream deformation network $D$.

### Questions
- In the qualitative comparisons (as shown in video), I still observe flicker in the Gaussian primitives. Could the authors explain the cause of these artifacts? Would increasing the number of input views alleviate the issue? Including curves of performance versus number of views in the final version would help analyze the impact of few-shot inputs.

- The method reconstructs a canonical Gaussian Head even when the input images contain expressions, and the authors state that the canonical Gaussian is unsupervised. I am curious why this representation can be learned. Could the authors offer a deeper discussion? Is this related to using neutral faces during training, enabling the model to learn that, when the expression code is absent, it should output a canonical Gaussian?

- Could the authors discuss whether the pre-processing of VGGT—specifically the alignment of its reconstructed point cloud using 2D landmarks and corresponding 3D keypoints from a tracked mesh—introduces additional errors or complexities? In particular, how do inaccuracies in landmark detection and mesh tracking affect the reliability of the geometry prior and the regularization loss $L_{geo}$, given that the VGGT point cloud quality is reported as "not high enough" and "inconsistent" (Line 849-850)?

### Soundness
3

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
4

### Summary
The paper addresses the task of reconstructing an animatable 3DGS-based avatar from 4 posed input images.

Following the recent success of LRM-style approached, like Avat3r,, the method employs a multi-view transformer which use features from a stable diffusion VAE, DINOv2 and plucker ray embeddings to predict per-pixel Gaussian attributes. 

Importantly, the set of gaussian attributes is extended to contain per-Gaussian latent features, similar to GHA and NPGA, which are then used to condition an MLP which animates the Gaussian based on FLAME expression code conditoning. Compared to Avat3r, the MLP is more light weight than the cross-attention based architecture, which is relevant for real-time applications.

Adding supervision against aligned VGGT depth predictions helps to produce more 3D consistent avatars.

The method is thoroughly evaluated against comparable sota baselines, and additionally shows examples on real-world images.

### Strengths
- The task of accessable avatar creation from a few selfie images finds importance in many down-stream applications. Therefore, the improved visual quality and faster animation speed immediatley become more significant. 
- Smart and simple usege of pretrained VAE weights for encoder/decoder, which is also ablated to be beneficial. The same can be said for the VGGT supervision.
- Overall the architecture seems to be slightly simplified comapred to Avat3r, which seems quite helful for future improvements that build upon the presented method.
- The paper is well-written and easy to follow.
- The method is thoroughly evaluated against to most relevant baselines, and on two different datasets, leaving little room for doubt about the evaluation. Furthermore, the SOTA baselines are significantly outperformed.
-

### Weaknesses
- The main weakness of the paper that I am still seeing is the limited novelty, since it mainly introduces some technical changes compares to Avat3r. However, the quality and animation speed are improved, and the paper is well evaluted. Therefore, we can be sure that the method solidly advances the field on such a highly relevant task. Therefore, I don't mind the limited novelty.
- Currently, the method is limited by FLAME expression codes. However, anything else would likely be out-of-scope for this project, and it would make animation from 2D videos arguably harder, bc. obtaining FLAME codes is so easy.
- Somewhow the number in the ablation study don't match the main results. Is there an explanation for this? Otherwise, this gives the impression that the ablations were sloppily exectued.

### Questions
- The text states that the 4 input views have different facial expressions during training. However, all results show identitcal expressions, or quite similar expressions. Does the model still perform well with very different expressions?
- Fig. 5: would be interesting to see how much the canonical point cloud changes with different inputs, e.g. the first example has the mouth slightly open in canonical space, which might be caused by the input images.
-  Is the VGGT superivsion competed offline? How good do the recoinstructions look? In my experience they can be rather blurry for faces and exhibit many stiching artifacts. Could this have been done with e.g. COLMAP as well?

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
4

### Summary
This paper proposes FastGHA, which efficiently generates high-quality, real-time animated 3D Gaussian head avatars with few-shot inputs. The method fuses multi-view features through a Transformer encoder and utilizes Gaussian features and a lightweight MLP network to achieve real-time animation based on facial expression encoding. Simultaneously, geometric supervision is introduced to improve the geometric smoothness of the 3D head.

### Strengths
* The method demonstrates satisfactory few-shot reconstruction quality and visualization effects.
* The method's reconstruction design is reasonable, effectively integrating information from different views, and can be well driven by new expressions.

### Weaknesses
* This method may struggle with "unreasonable" user input and the paper does not show these results. Furthermore, reasonable input is difficult to define. For example, if user input from a particular perspective is missing, the result is currently unknown. 
* This method may struggle to handle an arbitrary number of input viewpoints. Because it relies on VxHxW self-attention, too many views exponentially increase the computational cost of this part.
* Due to the linear increase in the number of Gaussians and the presence of the driving MLP, the driving speed decreases linearly with increasing viewpoint. Merging and controlling the total number of reconstructed Gaussians might be a good solution.

### Questions
* Why does the MLP need to modify the color of the Gaussians during animation? What would happen if it didn’t?
* The paper conducts ablation experiments on removing the VAE feature. What would be the result if only the VAE feature is kept and the DINO v3/Sapiens feature is removed?
* Removing the per-Gaussian feature shows a significant difference in results. What would happen with other dimension choice of the per-Gaussian feature, for example, 16 or 64?
* This method seems to have a hidden limitation: the requirement for a reasonable distribution of input viewpoints. Does this mean that the method lacks or has limited reconstruction ability when certain viewpoints are not covered? Structurally, the method can reduce the input to just one image. What is the performance when only a single frontal image is provided, or what if one frontal image and one side image, with the other side / up side missing?
* As a few-shot method, it is also reasonable and beneficial to discuss and compare with some of the latest one-shot methods, which can highlight the unique advantages of few-shot approaches. The authors could discuss and compare with methods such as GAGAvatar and LAM.
* Since the method is completely unsupervised in the canonical space, the statement “By design, the canonical head
is devoid of expression and thus does not correspond to any of the input expressions, allowing animation to happen downstream. ” is too arbitrary. It is possible for the model to learn a canonical space with an open mouth, and in fact, a canonical space with an open mouth may make it easier to model teeth than the closed-mouth shown in the Figure 1. What causes the model converges to a closed-mouth canonical space?
* There is no explanation for C_d and C_e in Equation 3. After reading, it seems refer to the DINO feature and encoder feature, but this should be explicitly stated.
* The additional Gaussian features used in animation MLP should not be claimed as "learnable", because it is not parameters of the model, but rather the result of inference. They are essentially the same thing as other inferred Gaussian parameters.

### Soundness
3

### Presentation
3

### Contribution
3
