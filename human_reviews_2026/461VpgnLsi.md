# ReSplat: Degradation-agnostic Feed-forward Gaussian Splatting via Self-guided Residual Diffusion

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Recent advances in novel view synthesis (NVS) have predominantly focused on ideal, clear input settings, limiting their applicability in real-world environments with common degradations such as blur, low-light, haze, rain, and snow. While some approaches address NVS under specific degradation types, they are often tailored to narrow cases, lacking the generalizability needed for broader scenarios. To address this issue, we propose Restoration-based feed-forward Gaussian Splatting, named ReSplat, a novel framework capable of handling degraded multi-view inputs. Our model jointly estimates restored images and gaussians to represent the clear scene for NVS. We enable multi-view consistent universal image restoration by utilizing the 3d gaussians generated during the diffusion sampling process as self-guidance. This results in sharper and more reliable novel views. Notably, our framework adapts to various degradations without prior knowledge of their specific types. Extensive experiments demonstrate that ReSplat significantly outperforms existing methods across challenging conditions, including blur, low-light, haze, rain, and snow, delivering superior visual quality and robust NVS performance. Code is available at https://github.com/yh-yoon/ReSplat.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
1. **Originality-wise**: The core idea of creating a synergistic loop between a universal image restoration model and a feed-forward Gaussian Splatting model is highly novel. Specifically, using the intermediate 3D geometry from the GS model to enforce multi-view consistency in a diffusion-based restoration process is a clever and previously unexplored mechanism for this problem.
2. **Quality-wise**: The claims are strongly supported by comprehensive experiments across a wide array of synthetic, mixed, and real-world degradations. The method consistently achieves state-of-the-art results in both novel view synthesis and image restoration tasks, demonstrating the framework's robustness and effectiveness. 
3. **Clarity-wise**: The paper is well-structured and clearly articulates a complex problem and its solution. The overall framework is well-illustrated with diagrams that effectively convey the interplay between the restoration and synthesis modules. The motivation, methodology, and results are presented logically and are easy to follow.

### Strengths
1. **Solves a Novel and Practical Problem**: The work addresses degradation-agnostic novel view synthesis, a topic of high practical importance for real-world applications that has been underexplored compared to synthesis from clean images.

2. **Innovative Synergistic Framework**: The core contribution is a novel framework where an image restoration model and a Gaussian Splatting model work in tandem. The use of 3D geometry from the GS model to guide the UIR model and ensure multi-view consistency is a key innovation.

3. **SOTA Performance and Thorough Validation**: The method demonstrates superior performance on both novel view synthesis and image restoration tasks across a comprehensive set of experiments, including single, mixed, and real-world degradations.

### Weaknesses
1. Ambiguous Mechanism and Potential for Detail Suppression in the Pre-filtering Module: The paper proposes a pre-filtering module to suppress artifacts but provides insufficient insight into its inner workings. The mechanism, which uses self-attention on both the corrupted and restored images, is a black box. It is unclear whether the module learns to identify specific "restoration artifacts" or if it simply learns to penalize any high-frequency regions that differ significantly from the degraded input.

2. the complete absence of failure cases is a significant omission for a paper claiming such robust, "agnostic" capabilities. A rigorous scientific contribution requires a transparent discussion of a method's limitations. 
---
I have listed my concerns, and the score will be adjusted based on the author's response.

### Questions
Please refer to Weaknesses part.

### Soundness
3

### Presentation
2

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
The paper introduces ReSplat, a novel framework for degradation-agnostic NVS. Unlike prior methods that either assume clean inputs or focus on specific degradation types, ReSplat integrates 3D Gaussian Splatting with a residual diffusion-based universal image restoration module. The approach jointly estimates restored multi-view images and explicit scene geometry through Gaussian splats, enabling multi-view consistent restoration and sharper novel view generation.

### Strengths
1. Combining Gaussian splatting with a residual diffusion model is conceptually new and addresses degradation-agnostic NVS in a unified way. And the proposed models have strong performance than competitive baselines (DiffUIR, GAURA) across both synthetic and real-world degradations, with consistent improvements in PSNR/SSIM/LPIPS.
2. Evaluation covers multiple degradation types (blur, haze, low-light, snow, rain) and mixed-degradation scenarios, showcasing robustness.
3. The method maintains practical inference speed (under one second for three views), which is important for deployment.
Ablation study: Clearly shows the effect of multi-view alignment and pre-filtering, validating each module’s importance.

### Weaknesses
1. Real-world evaluations are restricted to blur, haze, and low-light datasets. Claims of degradation-agnostic performance would be stronger with more diverse real-world tests (e.g., snow/rain in the wild).

2. While inference is efficient, the paper does not provide sufficient detail about training cost (e.g., GPU hours, memory usage). For diffusion-based models, this is important.

3. Although the model claims to be degradation-agnostic, it is unclear how well it generalizes to unseen or compound degradations not present in training.

4. The core innovation lies more in integration than in fundamentally new algorithms. Some may find the contribution incremental.

5. The ablation studies are weak, the model are constructed by the SOTA restoration network and NVS network, how the framework performs when changing these parts with other weaker restoration and NVS networks?

6. Some important NVS in low-quality scene [1,2] and image unified image restoration [3,4,5,6] papers are lacking.

[1] HQGS: High-Quality Novel View Synthesis with Gaussian Splatting in Degraded Scenes. 
[2] Robustgs: Unified boosting of feedforward 3d gaussian splatting under low-quality conditions.
[3] Adair: Adaptive all-in-one image restoration via frequency mining and modulation. 
[4] Perceive-ir: Learning to perceive degradation better for all-in-one image restoration
[5] Multi-task image restoration guided by robust DINO features.
[5] Restore Anything with Masks: Leveraging Mask Image Modeling for Blind All-in-One Image Restoration.

### Questions
1. The real-world evaluations are limited to blur, haze, and low-light datasets. How would the method perform under more diverse real-world degradations such as snow or rain in the wild?
2. The paper reports efficient inference but does not discuss training cost. What are the training resources (e.g., GPU hours, memory usage) required, and how do they compare with baselines like DiffUIR?
3. The method claims to be degradation-agnostic. How well does ReSplat generalize to unseen or compound degradations that were not included in training?
4. The contribution seems more about integration than proposing fundamentally new algorithms. Can the authors clarify which parts of the framework they view as the key novel technical contributions, beyond combining restoration and Gaussian splatting?
5. The ablation studies only vary internal modules but still rely on strong SOTA backbones for restoration and NVS. How would the framework perform if weaker restoration networks or weaker NVS models were used in place of the chosen SOTA baselines?
6. Several important recent works on NVS under degraded scenes [1,2] and unified image restoration [3–6] are not discussed. 
[1] HQGS: High-Quality Novel View Synthesis with Gaussian Splatting in Degraded Scenes.
[2] RobustGS: Unified boosting of feedforward 3D Gaussian Splatting under low-quality conditions.
[3] Adair: Adaptive all-in-one image restoration via frequency mining and modulation.
[4] Perceive-IR: Learning to perceive degradation better for all-in-one image restoration.
[5] Multi-task image restoration guided by robust DINO features.
[6] Restore Anything with Masks: Leveraging Mask Image Modeling for Blind All-in-One Image Restoration.

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
This paper proposes novel framework that can synthesize clean novel-view images given by the corrupted multi-view images. The model can construct clean 3D representation from images with diverse types of degradation by combining diffusion-based UIR model and feed-forward 3DGS.

### Strengths
* The motivation of this paper is explained clearly in the introduction part. Most of the previous literatures in the *Novel View Synthesis with Degradations* are limited to certain types of degradation and can't be generalized to corruptions that are net seen in the training dataset. GAURA attempted to mitigate this limitation by constructing degradation-aware generalizable NeRF but didn't leverage the prior knowledge of pretrained 2D UIR model. ReSplat points out those limitations which are important enough for the practicality of 3DGS.
* ReSplat can be regarded as combination of DiffUIR and MVSGaussian. However, paper also proposes two novel modules in Sec. 3.3 and Sec. 3.4 to improve the aggregation of information from multi-view images. I believe Sec. 3.3 is more notable contribution where they explicitly fuse the features from matching points across multi-view images, thereby aggregating the multi-view information effectively. This is common practice in feed-forward 3DGS [c], but there is novelty to use this technique for 3D-aware denoising process.
* Experiments are conducted on both synthetic and real-world datasets with multiple types of degradation. ReSplat consistently shows the performance improvement compared to the baselines.

---

### References

[c] Charatan, David, et al. "pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2024.

### Weaknesses
### Weaknesses

There are some parts that technical details are missing.

* In the preliminary, the author explain about the original 3DGS. However, I believe author should explain the brief summary for feed-forward 3DGS (like MVSGaussian) instead of standard 3DGS since they used the feed-forward 3D model in this work. This explanation of the feed-forward model will also aid understanding in the later sections, such as Sec. 3.4. It seems that the writing assumes readers are already familiar with generalizable 3DGS models, such as MVSGaussian and skips a lot of technical details in the main paper.

* In the Algorithm. 1, the novel view image $I_{nv}$ is used in line 7 but there is no explanation about this part in the main paper. Is this rendering loss conducted between rendered image and ground-truth clean image? Furthermore, detailed explanations about training with objective terms are missing in the main text where all of them are briefly summarized into Algorithm. 1.

* It is unclear that how the *Pre-filtering with warped features* operates. How does the outputted $[W^i_{pre}]^N_{I=1}$ contribute to the final weights map? Furthermore, the motivation of this module is also hard to understand. In the L85-86, the paper explained that this module can assist to achieve *artifact-free* novel view synthesis. However, it is hard to grasp the relationship between the terms of 'artifact-free' and the operations of this module. 

* What does the operation 'IR → NV' in Tab. 1 mean? How are the single-image IR methods such as DiffUIR evaluated on novel view synthesis? According to L372-374, it seems that author adopted additional adapter to transfer single-image IR models into multi-view settings. Is it correct?

* Author can diversify the baselines used in the comparison. For example, the most naive solution of achieving 3D reconstruction with UIR is: 1) Restore the corrupted multi-view images with pretrained 2D UIR method. 2) Reconstruct the clean 3DGS by using the restored multi-view images and pretrained feed-forward 3DGS. Author should compare the performance of ReSplat with this naive two-stage framework.

* How many input views are used during the evaluation? How many views can ReSplat handle? Please specify the evaluation details.

* There are previous literatures [a, b] that tried to use diffusion prior to construct clean 3D representation from the degraded images. However, all of them are limited to certain types of degradation (low-resolution or motion blur). It is better to cite those papers in the *Related works* since they are closely related with this paper in terms of using diffusion model for the 3D-IR task.

---

### References

[a] Lee, Seungjun, and Gim Hee Lee. "DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

[b] Lee, Jie Long, Chen Li, and Gim Hee Lee. "Disr-nerf: Diffusion-guided view-consistent super-resolution nerf." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

### Questions
Refer to the *Weaknesses* section.

### Soundness
3

### Presentation
2

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
The paper introduces a method, ReSplat, for feedforward NVS when input images are degraded in various ways, ReSplat leverages the feedforward 3D Gaussian Splatting method MVSplat[1] and an universal residual diffusion model DiffUIR[2] to solve the problem. The authors propose using Gaussian points information in the Diffusion encoder to promote 3D consistency and a pre-filtering technique to eliminate residual artifacts.

### Strengths
1. The topic of NVS with degraded inputs the paper focuses on is important and interesting.
2. ReSplat outperforms other methods in both quantitative and qualitative results.
3. The experiments are comprehensive, evaluating on both synthetic and real-world image degradations.

### Weaknesses
1. My main concern is the effectiveness of the proposed 3D alignment module and pre-filtering technique, as the improvements in Table 4 ablation study are moderate.
2. No qualitative figure in ablation study is provided. 
3. A limitations section should be discussed and added.

### Questions
1. I don't understand how exactly Gaussian points information are used in the diffusion model. After per-point features obtained in Sec 3.3, are they using as conditional features in the diffusion model? I suggest outlining the attention and diffusion encoder equations. 
2. What are the memory consumption and training time of ReSplat?

### Soundness
2

### Presentation
2

### Contribution
2
