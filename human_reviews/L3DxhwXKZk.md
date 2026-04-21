# ExpanDyNeRF: Expanding the Viewpoint of Dynamic Scenes beyond Constrained Camera Motions

- Avg Score: 4.60
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 6, 5, 6

## Abstract
In the domain of dynamic Neural Radiance Fields (NeRF) for novel view synthesis, current state-of-the-art (SOTA) techniques struggle when the camera's pose deviates significantly from the primary viewpoint, resulting in unstable and unrealistic outcomes. This paper introduces Expanded Dynamic NeRF (ExpanDyNeRF), a monocular NeRF method that integrates a Gaussian splatting prior to tackle novel view synthesis with large-angle rotations. ExpanDyNeRF employs a pseudo ground truth technique to optimize density and color features, which enables the generation of realistic scene reconstructions from challenging viewpoints. Additionally, we present the Synthetic Dynamic Multiview (SynDM) dataset, the first GTA V-based dynamic multiview dataset designed specifically for evaluating robust dynamic reconstruction from significantly shifted views. We evaluate our method quantitatively and qualitatively on both the SynDM dataset and the widely recognized NVIDIA dataset, comparing it against other SOTA methods for dynamic scene reconstruction. Our evaluation results demonstrate that our method achieves superior performance.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors proposed a new way to improve the performance of novel view synthesis (NVS) in dynamic scenes. They leverage the existing work $D^{4}NeRF$[1] for their scene optimization part. Besides, they utilize the DreamGaussian[2] to generate the 3D priors and then use the primary camera poses are supervision to fit the transformation between cameras and generated objects. In this way, novel views with novel cameras of the object are generated. However, I do think this paper is more likely to be a technical work instead of a research paper, as it lacks contributions in the method part. The good thing is they proposed a new dataset that can help the research in this field. Also, I think this paper should include more experiments with proper baselines and more public datasets to prove their effectiveness.

[1] Zhang, Boyu, Wenbo Xu, Zheng Zhu, and Guan Huang. "Detachable novel views synthesis of dynamic scenes using distribution-driven neural radiance fields." arXiv preprint arXiv:2301.00411 (2023).

[2] Tang, Jiaxiang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. "Dreamgaussian: Generative gaussian splatting for efficient 3d content creation." arXiv preprint arXiv:2309.16653 (2023).

### Strengths
1. The authors proposed a new dataset for the NVS task.
2. The authors conducted enough ablation studies to prove the contribution of each component of their model.
3. The author combine the GS system and NeRF system together.

### Weaknesses
1. The presentation of Algorithm 1 is really bad. From my perspective, the authors should explain the content in the Algorithm table instead of only 'Details can be found in ...', if they want to put this big table in the main paper.
2. I think the content in Sec 3.1 should be a high-level explanation of the existing work the authors followed.
3. I think the 'primary view' should be some prior camera knowledge, which means that the authors indeed feed prior cameras into their model and generate more novel views serving as more supervision. However, the baselines to which the authors compare their method were not given camera priors like RoDynRF,... 
4. This paper is lacking of experiments to prove their effectiveness. I would like to see more comparisons on more public datasets (NeRF-DS[1], DAVIS[2], Hypernerf[3], iPhone[4], ...) with more recent works (spacetimeGS[5],... )

[1] Yan, Zhiwen, Chen Li, and Gim Hee Lee. "Nerf-ds: Neural radiance fields for dynamic specular objects." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8285-8295. 2023.

[2] Pont-Tuset, Jordi, Federico Perazzi, Sergi Caelles, Pablo Arbeláez, Alex Sorkine-Hornung, and Luc Van Gool. "The 2017 davis challenge on video object segmentation." arXiv preprint arXiv:1704.00675 (2017).

[3] Park, Keunhong, Utkarsh Sinha, Peter Hedman, Jonathan T. Barron, Sofien Bouaziz, Dan B. Goldman, Ricardo Martin-Brualla, and Steven M. Seitz. "Hypernerf: A higher-dimensional representation for topologically varying neural radiance fields." arXiv preprint arXiv:2106.13228 (2021).

[4] Gao, Hang, Ruilong Li, Shubham Tulsiani, Bryan Russell, and Angjoo Kanazawa. "Monocular dynamic view synthesis: A reality check." Advances in Neural Information Processing Systems 35 (2022): 33768-33780.
[5] Li, Zhan, Zhang Chen, Zhong Li, and Yi Xu. "Spacetime gaussian feature splatting for real-time dynamic view synthesis." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8508-8520. 2024.

### Questions
1. How do you decompose the static background and the dynamic foreground? Do you have G.T. motion masks?
2. Which type of distribution-based encoding network did you use? If from existing works, please cite them, if not, please claim clearer that you propose it.
3. Why do the authors leverage the Gaussian priors, but then map them to the NeRF system? Why not use the Gaussian system for all?
4. How long does the proposed method require per scene?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The work proposes ExpanDyNeRF, a method for dynamic scene reconstruction and novel view synthesis, which handles significant camera rotational changes. Such robustness is achieved due to the integration of a Gaussian Splatting prior based on a generative image-to-3D model. In addition, a dynamic multi-view dataset based on the GTA-V video game SynDM is proposed to evaluate the rendering quality of dynamic subjects from significant viewing angle deviations from the primary camera and demonstrate the superior performance of ExpanDyNeRF in comparison to dynamic NVS baselines.

### Strengths
1. The paper is well-written and easy-to-follow.
2. The effort behind creating a multi-view GTA-V-based benchmark with a latency decrease from 16.7 ms to as low as 0.2 ms is impressive.
3. Ablation studies show the effectiveness of introduced novel view loss at largely deviated perspectives.

### Weaknesses
1. The proposed supervision requires training a Gaussian Splatting (GS) for every timestamp along with the dynamic NeRF model, which appears computationally suboptimal. With this, it would be important to emphasize the advantages of the proposed method in comparison to existing alternatives, such as DreamGaussian4D [1], with additional background reconstruction using a GS model.
2. Experimental results are rather limited (two datasets including a proposed benchmark) and do not demonstrate a clear advantage of the proposed method compared to other dynamic reconstruction methods. Moreover, none of the chosen baselines utilizes a 3D prior, which seems unfair as the current solution leverages one.
3. Operational range [-45$^\circ$, 45$^\circ$] is rather limiting, especially for 360$^\circ$ object-centric dynamic reconstruction for creation of avatars and 4D assets. Addressing the issue appears important for the practical usage of this work.

[1] Ren et al. "Dreamgaussian4d: Generative 4d gaussian splatting.", 2023.

### Questions
1. The concept of parallax described in Section 3.2 and motion perception are also addressed in [2] by leveraging semantic understanding. It would be interesting to include it as a baseline and test the method on the proposed significant rotational changes.
2. Can translational changes be addressed in the work, especially in the generation of novel deviated views? What about changes in the distance to the object (e.g. zoom-ins)?
3. Another potential benchmark could be ActorsHQ [3] or Multiface [4] for rendering dynamic humans and faces from various angles.


[2] Liu et al. "Gear-NeRF: Free-Viewpoint Rendering and Tracking with Motion-aware Spatio-Temporal Sampling.", 2024.
[3] Işık et al. "Humanrf: High-fidelity neural radiance fields for humans in motion.", 2023.
[4] Wuu et al. "Multiface: A dataset for neural face rendering.", 2022.

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
This paper aims to improve the modelling of dynamic scenes, particularly in terms of their quality when sampled from novel views. The authors introduce an Expanded Dynamic NeRF (ExpanDyNeRF) method for improved dynamic 3D scene reconstruction. Different from previous dynamic NeRFs, this method incorporates a 3D Gaussian Splatting prior into the pipeline, using it to supervise the training of a dynamic NeRF at each frame. The authors also curate a new synthetic dataset (SynDM) that has multiple viewpoints with dynamic camera motion, while also providing corresponding side views for evaluation purposes. On the proposed SynDM dataset. the proposed method achieves significantly better results as compared to existing dynamic scene reconstruction methods.

### Strengths
This paper tackles an important problem. Existing works that attempt to model dynamic scenes tend to perform badly at novel views. Thus, the tackled problem is meaningful.

This paper not only proposes a method for better modelling dynamic scenes, it also proposes a new synthetic dataset that may greatly facilitate the training of better models. In my opinion, the lack of a suitable dataset seems to be a fundamental issue blocking the progress of improvements in the area. This dataset may help to contribute to the long-term developments in the field.

The reported results on the proposed SynDM dataset, show that the proposed ExpanDyNeRF significantly surpasses the current SOTAs.

### Weaknesses
Although the proposed ExpanDyNeRF attains good performance, most of the improvements are largely engineering-based contributions without too many additional insights. For e.g., using a 3D gaussian prior and extracting novel views as supervision signals. Thus, since the technical novelty of the paper is not the best, and it seems there are no huge insights on why exactly the pipeline works well, the novelty of the paper may be limited.

The proposed method does not show gains on the existing NVIDIA dynamic scenes dataset. Even though the existing NVIDIA dataset only uses stationary cameras, and are not exactly the target of the proposed method, but the proposed method also uses a lot more components and computations, and thus should not offer lower performance than existing methods in my opinion. 

Other specific concerns and questions have been placed under the “Questions” section.

### Questions
Firstly, it seems that the super-resolution loss (and the use of a pre-trained super-resolution model) is overly important. Specifically, from Table 2, it seems that most of the performance gains comes from the addition of the super-resolution loss L_sr. Furthermore, previous works mostly do not use a super-resolution loss or technique. Thus, this raises the possibility that the good performance and metrics of the method are mostly due to this pre-trained super-resolution model providing good guidance signals, instead of the other proposed designs (e.g., sampling from a 3D prior). Could the authors 1) provide more qualitative/quantitative results without the use of a pre-trained super-resolution method, or 2) provide experiment results on baseline methods, while adding the super-resolution loss to them. The results of either of these may show the actual extent of importance of the super-resolution loss, as well as better show the contributions of the other proposed designs.

Another concern lies in the optimization time. It seems like there are two large and potentially expensive components that have been added to the pipeline as compared to previous dynamic NeRF works: 1) Fitting of a 3D Gaussian Splatting prior at every frame; 2) Usage of a potentially large pre-trained super-resolution model for losses at every iteration. How much additional costs do these components add to the overall optimization time?


There are some claims that PSNR is not that good a metric for evaluating predictions with large deviation, as PSNR may not be sensitive to certain distortions. Although some qualitative evidence is provided in Fig 5 to show this, the evidence still seems a little lacking to dismiss the metric. For instance, it could be that D4NeRF may only have certain blurred frames (that have been shown), but otherwise performs quite well. Could there be further explanations or insights into why PSNR might fail here? Otherwise, could there be more qualitative visualizations to convince the reader? Since PSNR is often used to measure blurs and image quality, there needs to be more evidence or insights to make this claim convincing.


The authors should provide further clarifications on the dataset. From Section 3.3, it seems that there are 9 scenes, with 22 cameras each. How many of these cameras are moving? Also, how are these cameras moving? Are they all moving in the same way? At the same time, for each scene, is the scene recorded multiple times to feature different dynamic entities, e.g., humans? Otherwise, is each of the 9 scenes recorded with only one type of entity? Overall, how many scene/entity combinations are there?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper introduces ExpanDyNeRF, a method that enhances dynamic Neural Radiance Field (NeRF) models for rendering 3D scenes from viewpoints significantly deviated from the primary camera. Traditional dynamic NeRFs struggle with these novel views, often producing artifacts or blurring. ExpanDyNeRF incorporates a Gaussian splatting prior and a pseudo ground truth approach, which help refine density and color features, enabling realistic scene reconstructions from various perspectives. The paper also presents SynDM, a new GTA V-based multiview dataset specifically designed for dynamic scene reconstruction with diverse angles. Through quantitative and qualitative evaluations on SynDM and the NVIDIA dataset, ExpanDyNeRF shows superior performance in rendering stability and fidelity compared to other state-of-the-art models, addressing limitations in existing NeRF methods.

### Strengths
1. A SynDM dataset, a novel GTA V-based dynamic Multiview dataset. This dataset provides a valuable resource for evaluating dynamic 3D reconstructions from varying viewpoints, with ground truth that supports assessment of model generalization to wide-angle novel views.

2. To enhance dynamic Neural Radiance Field (NeRF) models for rendering 3D scenes from viewpoints significantly deviated from the primary camera.

### Weaknesses
1. The process for generating pseudo-ground truth is somewhat unclear. The way to use Gaussian splatting prior is not clear. Further explanation on how this method connects would clarify the methodology.

2. The integration of Gaussian splatting and its specific role in improving novel view synthesis lacks sufficient clarity. Some recent research such as [1] deploy diffusion as prior has achieved great results. Additional details on how Gaussian splatting affects density and color optimization at deviated angles would strengthen the understanding of its contribution to model performance.

3. The experiments only demonstrate performance on the SynDM and NVIDIA datasets. I suggest conducting experiments on datasets more widely used in dynamic NeRF, such as the DAVIS and iPhone datasets.

[1] Chen, H., Loy, CC, & Pan, X. (2024). MVIP-NeRF: Multi-view 3D Inpainting on NeRF Scenes via Diffusion Prior. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 5344-5353.

### Questions
In the discussion about the GTA-V platform, the author refers “semi-freezing" of the game’s graphic state. How does it differ from a complete freeze and how does it enable smooth transitions across multiple camera views without introducing significant latency?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces ExpanDyNeRF, an innovative approach to dynamic NeRFs for novel view synthesis that tackles large-angle viewpoint deviations. By leveraging Gaussian splatting and pseudo-ground truth optimization, the method achieves notable stability and realism in dynamic scenes, even from challenging perspectives. A key contribution is the introduction of the Synthetic Dynamic Multiview (SynDM) dataset, built from GTA V, specifically designed to evaluate dynamic, multi-view scenes—filling a critical gap in the current dataset landscape. ExpanDyNeRF demonstrates significant improvements over state-of-the-art methods, with substantial gains in FID, LPIPS, and PSNR metrics, particularly excelling in color and shape fidelity.

### Strengths
Comparisons with state-of-the-art dynamic multi-view synthesis methods show clear improvements both quantitatively and qualitatively, especially in larger viewpoint settings. The proposed pseudo-ground truth optimization with Gaussian priors enhances scene details, with each component thoroughly evaluated in ablation studies. The SynDM dataset, offering multi-view dynamic data with ground-truth side views for benchmarking, is a valuable addition to the community, enabling reproducible evaluations in dynamic NeRF research. Efforts to reduce latency are also commendable, and making the source code public would further benefit the field.

### Weaknesses
While benchmarking on synthetic data is understandable given dataset constraints, additional evaluations on real-world data would more thoroughly demonstrate the method’s robustness. Handling larger viewpoint deviations remains challenging, as both the baseline and proposed methods show limitations. A discussion of promising future directions—particularly whether this approach opens up possibilities for substantial gains in this area—would be valuable. Furthermore, an analysis of runtime and computational resource usage would help assess the scalability and deployment potential of the method.

### Questions
1. Have you considered testing ExpanDyNeRF on real-world dynamic datasets? How do you anticipate the model would perform on non-synthetic data, and are there plans to include such evaluations in future work?

2. Both the baseline and proposed methods encounter limitations with larger viewpoint deviations. Do you foresee any specific improvements or techniques that could help address these challenges? Does ExpanDyNeRF open up any new avenues for significant performance gains in this area? 

3. Could you provide more details on the runtime performance and computational resources required for both training and inference? Understanding the method’s efficiency and memory usage would help assess its scalability and real-world deployment feasibility.

4. Given the engineering efforts noted in reducing latency, do you plan to release the source code for ExpanDyNeRF? This would be valuable for the community and help facilitate reproducible research.

### Soundness
3

### Presentation
3

### Contribution
3
