# NVS-Solver: Video Diffusion Model as Zero-Shot Novel View Synthesizer

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6, 6

## Abstract
By harnessing the potent generative capabilities of pre-trained large video diffusion models, we propose a new novel view synthesis paradigm that operates without the need for training. The proposed method adaptively modulates the diffusion sampling process with the given views to enable the creation of visually pleasing results from single or multiple views of static scenes or monocular videos of dynamic scenes. Specifically, built upon our theoretical modeling, we iteratively modulate the score function with the given scene priors represented with warped input views to control the video diffusion process. Moreover, by theoretically exploring the boundary of the estimation error, we achieve the modulation in an adaptive fashion according to the view pose and the number of diffusion steps. Extensive evaluations on both static and dynamic scenes substantiate the significant superiority of our method over state-of-the-art methods both quantitatively and qualitatively. The source code can be found on https://github.com/ZHU-Zhiyu/NVS_Solver.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a training-free sampling method that generates image sequences conditioned on input camera trajectories. The input view is warped onto the target views, serving as pseudo-GT variables. To condition the reverse process, the paper re-computes the predicted mean of a reverse step with an interpolation between the predicted mean of the current noisy latent and the warped samples. It further explores calculating the optimal interpolation weight and two guidance methods (replacement and gradient). Both quantitative result and qualitative results show the proposed method can generate smooth, high-fidelity image sequences.

### Strengths
1. The paper presents a training-free method that enables a pre-trained diffusion model to generate image sequences based on given camera trajectories.
2. The qualitative videos demonstrate that the proposed method generates plausible outputs.

### Weaknesses
1. The paper relies on an off-the-shelf depth estimation network to warp the input image(s), which may be prone to scale inconsistencies across frames. Since the reverse process is largely dependent on the warped images, depth estimation errors could propagate to the final outputs.
2. The paper does not explain how occluded regions are handled in depth-warping. Handling the occluded regions can be particularly challenging when the camera pose variation is large, as $\tilde{\mathbf{\mu}}_{t, \mathbf{p}_i}$ in Eq.12 may produce degenerate outputs. 
3. The camera trajectories in most qualitative results are limited to relatively small variations or 360-degree circular poses, where scale ambiguity from the depth estimation network can largely be ignored. This raises robustness of the proposed method when the camera variations are large. The paper could benefit from including more examples with larger camera trajectories.
4. The toy experiment in Fig.2 (a) shows that the $\mathcal{E}_D$ decreases with the diffusion reverse process. This brings another question on how the predicted means converge to the GT image (loss=0) where it is very unlikely to have sampled $\mathbf{X}_t$ that would have led to the desired GT. I believe more details of the experiment can help understanding the paper better.
5. The paper makes some assumptions to compute the optimal interpolation weight $\lambda(t, \mathbf{p}_i)$ in Sec.4.2. However, the paper does not present ablation study on the choice of the weight (except when $\lambda(t, \mathbf{p}_i) = \infty$, which also shows comparable results). To validate the choice of the weight schedule, the paper could present comparisons to other naive techniques (e.g., linear, constant, exponential). Additionally, the interpolation weight depends on a set of hyperparameters $\{ v_1, v_2, v_3 \}$ which may require engineering effort to tune on new scenes. 
6. While the paper shows promising quantitative and qualitative results, the number of scenes used for evaluation is insufficient to validate the method's effectiveness and differs from previous state-of-the-art methods.

### Questions
1. The proposed method sets the number of inference step to 100, which is quite large. Do other diffusion-based baselines also use the same number of inference step?
2. The paper mentions the camera trajectory is estimated (L.360). Could you provide more details on how the pose metrics (ATE, RPE) are computed?

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
4

### Summary
The authors propose a training free novel view synthesis paradigm based on video diffusion model. Specificcally, warped depth maps are utilized and certain sampling methods are proposed to ensure high quality NVS.

### Strengths
1. The work is complete and results seem to be good quantitively.

### Weaknesses
1. The work is complete, but the novelty and the contribution are not strong enough to meet the quality of ICLR. A good training-free is not that appealing after previous works like MotionCtrl, and it highly relys on the capability of SVD. It is hard to know about it generalizability around various video generation methods. 
2. It is recommended to show "Directly guided sampling" and "Posterior sampling" clearly in the view of practical implementation. Some illustration would be appreciated.
3. Some equations should be displayed more clearly. For instance in eq.8, I(P_0) should not be related to the pose index i, but in eq.9 it is related to i. Maybe in eq.9 the I(P_0) should be revised as I(P_i)?

### Questions
Please refer to Weaknesses.

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
3

### Summary
1. This paper proposes a novel view synthesis pipeline without any training. The pipeline can take single or multiple views of static scenes or monocular videos of dynamic scenes as input.
2. This paper modulates the score function with the warped input views to control the video diffusion process and generate visually pleasing results. They achieve the modulation in an adaptive fashion based on the view pose and the number of diffusion steps.
3. They conduct extensive results on both static and dynamic scenes and show promising results with both evaluation numbers and visualizations.

### Strengths
1. The proposed adaptive modulation of the score function in the diffusion process is novel.
2. The proposed method achieves better results in various scenarios compared to baselines.
3. The authors provide the code with an anonymous link, ensuring the applicability of the results.

### Weaknesses
My primary concerns are with the references and experimental details:

1. Some key references on diffusion-based NVS are missing [1,2,3,4,5,6,7]. Among these, [3] specifically focuses on scenes and has released its code. Is there a particular reason it was not included in the comparison?
2. How is the synthesized view pose calculated in this paper? In Line 364, it states that 'current depth estimation algorithms struggle to derive absolute depth from a single view or monocular video, resulting in a scale gap between the synthesized and ground truth images.' When calculating pose error, does the proposed method account for this scale gap?

**Minor Points:**

1. In Table 1, it is stated that MotionCtrl [Wang et al., 2023b] and 3D-aware [Xiang et al., 2023] do not require training. However, as I understand, they do require fine-tuning.

[1] Wu, Rundi, et al. "Reconfusion: 3d reconstruction with diffusion priors." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2024.

[2] Watson, Daniel, et al. "Novel View Synthesis with Diffusion Models." *The Eleventh International Conference on Learning Representations*.

[3] Yu, Jason J., et al. "Long-term photometric consistent novel view synthesis with diffusion models." *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2023.

[4] Cai, Shengqu, et al. "Diffdreamer: Towards consistent unsupervised single-view scene extrapolation with conditional diffusion models." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

[5] Tseng, Hung-Yu, et al. "Consistent view synthesis with pose-guided diffusion models." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2023.

[6] Chan, Eric R., et al. "Generative novel view synthesis with 3d-aware diffusion models." *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2023.

### Questions
Line 415 mentions that the proposed method can achieve 360-degree NVS. Would it be possible to include a comparison with ZeroNVS [7] to better demonstrate its effectiveness?


[7] Sargent, Kyle, et al. "ZeroNVS: Zero-Shot 360-Degree View Synthesis from a Single Image." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2024.

### Soundness
3

### Presentation
2

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
This paper introduces a NVS method that leverages large pre-trained video diffusion models without additional training. The approach adaptively modulates the diffusion sampling process using input views to produce high-quality results from single or multiple views of static scenes or dynamic videos. Theoretical modeling is used to iteratively adjust the score function based on scene priors, enhancing control over the diffusion process. The modulation adapts to view pose and diffusion steps.

### Strengths
1. The proposed approach is entirely training-free, meaning it directly leverages pre-trained large video diffusion models without requiring additional fine-tuning or retraining. This feature not only reduces computational demands but also makes it adaptable to a wide range of applications where time or resources for training may be limited. The flexibility of using pre-trained models enhances its practicality, allowing users to apply this method to various scenes and tasks with minimal setup.

2. The generated videos maintain high visual fidelity, delivering smooth results. This quality stems from the adaptive modulation of the diffusion process, which effectively incorporates scene details and structures from the given views, ensuring that outputs are kind of realistic.

3. The method is underpinned by theoretical modeling, which guides its adaptive modulation strategy. By iteratively adjusting the score function with scene priors and analyzing estimation error boundaries, the approach achieves both controlled and adaptive modulation of the diffusion process.

### Weaknesses
1. The comparison between this method and NeRF-based methods is fundamentally imbalanced. NeRF techniques incorporate an underlying 3D structure, enabling them to render any view with predictable performance, as the 3D structure informs which views are feasible and which are not. In contrast, the proposed method lacks an explicit 3D representation, limiting its view synthesis capabilities to specific views with no guarantee of consistent performance. This distinction is significant, as NeRF's inherent 3D information allows interpretable, reliable results across views, whereas this method’s output reliability is less predictable and may vary based on input views.

2. To achieve a fair comparison between the proposed method and NeRF-based techniques, the authors should first reconstruct a 3D model from the output video of this method and then re-render the scene from that reconstructed model. This process would allow for a direct assessment of both methods’ rendering consistency and quality, ensuring that comparisons consider the 3D structure NeRF inherently leverages. Also, reconstruction error like PSNR should be reported.

3. The video results of the proposed method exhibit visible flickering artifacts, which could substantially affect reconstruction quality and consistency. A deeper analysis is needed to assess how these artifacts impact overall reconstruction accuracy and to identify potential mitigation strategies. This might include tuning reconstruction parameters to minimize flickering, which would help improve the method’s output stability and robustness, especially for applications sensitive to temporal consistency.

4. A major contribution is the derivation of the parameter $\lambda$ in Section 4.2, which aims to minimize the estimation error upper bound in Equation 15. However, a gap remains between this upper bound and the actual estimation error represented in the left side of Equation 15. To strengthen the theoretical foundation, the authors should provide a more comprehensive analysis of how reducing the upper bound affects the actual estimation error. This could be achieved through statistical analysis and empirical evidence showing how well the method reduces estimation error in practice, thereby validating the theoretical assumptions.

### Questions
For dynamic scene comparison, it is said "For monocular video-based NVS, we downloaded nine videos from YouTube, each comprising 
frames and capturing complex scenes in both urban and natural settings." 

Why not just following the dynamic nerf settings? They have well aligned ground-truth for measuring the reconstruction performance.
Generation metrics like FID are not that reliable.

Soma example datasets are HyperNerf, DyCheck (https://hangg7.com/dycheck/)

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
2

### Summary
This work introduces a diffusion model-based approach to achieve novel view synthesis. In particular, it leverages the depth-warped views as guidance to achieve adaptative modulation. Experiments on single-view images, multi-view images, and monocular video input-based novel view synthesis showcase the efficacy of the introduced methods.

### Strengths
* The idea of using depth-warped images as guidance for novel view synthesis is reasonable.
* It is interesting to see that the temporal consistent video diffusion model can be effectively reformulated to achieve geometrical consistent NVS in a training-free manner.
* Experiments on several challenging settings, including 360-degree NVS from a single view, verify the significance of the introduced method.

### Weaknesses
* Accessing the geometry accuracy. For the 360-degree case, e.g., the truck, it would be better to apply mesh reconstruction on the rendered views, similar to Fig. 5(b) in latentSplat [Wewer et al. ECCV 2024]. The reconstructed mesh will provide a clearer understanding of how well the rendered views maintain correct geometry.

* Pixel-aligned metrics. For the NVS task, it would be better to report comparisons with state-of-the-part methods regarding pixel-aligned metrics, e.g., PSNR and SSIM. 

* Discussion with feed-forward 3DGS models. It might be interesting to see comparisons with detailed analysis between the introduced methods and those feed-forward 3DGS models, e.g., pixelSplat [Charatan et al., CVPR 2024], MVSplat [Chen et al., ECCV 2024]. And it would be better to consider adding these methods to the related work for better coverage of recent NVS works.

### Questions
Kindly refer to [Weaknesses].

### Soundness
3

### Presentation
2

### Contribution
3
