# Learning Text-driven 3D Human Motion Generation from 3D-free Web Videos

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Text-driven 3D human motion generation has gained attention for synthesizing complex movements from textual descriptions. Traditional approaches depend on expensive 3D motion capture, which restricts motion diversity, whereas 2D human videos provide abundant and accessible data. However, the absence of large-scale annotated 2D motion datasets and the challenge of generating 3D motion from 2D data remain unresolved. To address this, we introduce MotionWeb, a dataset comprising over 100k motion clips, 17 million frames, and 160 hours of data, with 2D keypoints extracted using state-of-the-art pose estimation models, significantly reducing annotation costs. We further propose Keypoint To Motion (K2M), an efficient framework for text-driven 3D motion generation leveraging 2D supervision without requiring 3D annotations. Experiments show that our method efficiently generates realistic 3D motion with improved both quality and diversity using large-scale 2D supervision.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents the MotionWeb dataset, which collects motion clips from the web, covering a diverse set of indoor and outdoor activities annotated with 2D keypoints and corresponding textual descriptions. Additionally, the authors propose the Keypoint-to-Motion method, which leverages 2D keypoint data during training to enable text-driven 3D human motion generation. Experiments are conducted on latest benchmark and the proposed dataset.

### Strengths
1. The method relies on 3D-free supervision, significantly reducing the cost and complexity associated with generating 3D annotations for motion generation tasks.
2. The paper is well-written, clearly structured, and easy to follow.

### Weaknesses
1. The use of the VQ-VAE generator component is not novel, as it has been widely adopted in previous 3D motion generation work. The proposed method appears to rely heavily on combining existing techniques, which may be seen as more of an engineering integration rather than a methodological innovation.
2. It is unclear whether directly generating SMPL parameters offers a clear advantage over conventional methods. An ablation study comparing this approach to alternatives would help validate its effectiveness.
3. The model incorporates numerous regularization terms to ensure natural and robust 3D motion generation. However, the balance between these regularization terms plays a critical role in model performance. Is there any ablation study on the choice and sensitivity of these hyperparameters? Without such analysis, there is concern that the reported results may be selectively optimized.

### Questions
1. Is there any particular reason for directly generating SMPL parameters offers a clear advantage over conventional methods?
2. The proposed method includes multiple regularization terms in model training. How sensitivity of the model to the hyper parameter setting?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The method focuses on the problem of lack of diversity in 3D mocap datasets. The manuscript proposed a method to generate high-quality 3D motions via large scale 2D videos without the need of accurate 3D annotations. They also bring a new dataset termed as MOTION WEB. They also propose keypoint 2 motion (K2M), a framework for text-driven 3D motion generation, leveraging the 2D supervision.

### Strengths
Strength:
-	The paper is well-motivated with several technical challenges;
-	The method proposed seems sound and correct.
-	The pipeline for the dataset generation shows the effectiveness.

### Weaknesses
Weakness: 
-	The architecture and design is normal. In fact, the core of SMPL and idea of SMPlify in 3D leverages the idea of 3D projection and using 2D information as the supervision. This makes the novelty limited. 
-	The idea of injecting the text features to the motion VQ embedding seems with limited novelty.
-	The evaluation seems limited, and maybe incorporate more baseline methods are necessary.

### Questions
See  weakness

### Soundness
3

### Presentation
2

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
This work examines the problem of learning a 3D text-to-motion model using large-scale web data and supervision from 2D pose/motion detection algorithms. The key insight is that 2D supervision can be used to train a 3D model by learning a VQ-VAE model which quantizes 2D poses to discrete latents that decode into 3D SMPL parameters. A masked generative model is used to predict VQ-VAE tokens which are decoded into 3D SMPL meshes and projected to 2D for supervision during the training stage. Experiments show that the proposed method can follow a broader range of text descriptions than prior works.

### Strengths
* The overall approach is interesting and promising. 3D motion capture data is quite sparse, and 3D pose/motion estimation from video is not yet mature enough to provide high quality data. 2D pose/motion models are quite robust and accurate, and using this kind of data to learn 3D models has the potential to expand the scope of data that can be used to train human motion models.
* The experimental results provide evidence that the model can follow more diverse text instructions than existing models.

### Weaknesses
* Since no 3D data is used, the method is susceptible to errors caused by ambiguity between 2D and 3D poses. This is especially true for cases where limbs are pointing away from the camera or towards the camera. While the method provides a nice way to learn 3D models with 2D data only, this approach has built-in limitations.
* In the video examples provided in the supplementary material, the results from the proposed method could follow text prompts that previous methods could not. However, previous methods have significantly more natural motions and the proposed method often has limbs and joints at odd or contorted angles. I am curious if the less natural pose in the proposed method is a result of errors from 2D estimation, 2D to 3D conversion, or both.
* The comparison is Table 2 is not entirely fair because the current method was trained with SMPL parameters while the other methods are converted to SMPL parameters, which could lead to quality loss. This is not a fault of the authors or a major weakness.

### Questions
* To what extent do ambiguities between 2D and 3D poses affect the model? Can you provide analysis of typical failure cases?

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
5

### Summary
The paper addresses text-driven 3D human motion generation by introducing MotionWeb, a large-scale dataset with over 100k motion clips and 160 hours of 2D keypoints extracted from videos to overcome the limitations of expensive 3D motion capture. It proposes Keypoint To Motion (K2M), an efficient framework that generates realistic and diverse 3D motions using only 2D supervision, without requiring 3D annotations.

### Strengths
- The paper is clearly presented with careful ablation studies.
- The writing and structure of the paper are clear and easy to follow.
- The authors construct a new large-scale 2D dataset named MotionWeb.

### Weaknesses
- The core contribution of this paper is learning 3D human motion generation from 2D videos; however, this concept has been explored in prior work [1]. As a result, the novelty appears limited. The authors should more clearly articulate the unique aspects of their approach compared to existing methods.
- Furthermore, experimental comparisons with prior work [1] are insufficient. The authors should evaluate their method against a broader range of existing techniques for 3D human motion generation from 2D videos.
- The paper lacks fair comparisons with state-of-the-art methods on standard benchmarks. The authors should benchmark their approach against recent methods on widely used 3D human motion generation datasets, rather than relying solely on their own collected dataset. Given the claim that learning from 2D videos is more scalable, it is crucial to demonstrate its effectiveness on established benchmarks, not just custom data.
- The baseline methods compared are outdated. The authors should include comparisons with the latest state-of-the-art approaches, such as works [2-6], in 3D human motion generation. The current comparisons fail to convincingly show the proposed method's superiority.
- Based on the visualization samples provided by the authors, the proposed model does not outperform existing methods, and the generated motions are noticeably inferior to those from prior approaches.


[1]: Ren Z, Huang S, Li X. Realistic human motion generation with cross-diffusion models[C]//European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024: 345-362.

[2]: Meng Z, Xie Y, Peng X, et al. Rethinking diffusion for text-driven human motion generation[J]. arXiv preprint arXiv:2411.16575, 2024.

[3]: Zhang J, Fan H, Yang Y. Energymogen: Compositional human motion generation with energy-based diffusion model in latent space[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 17592-17602.

[4]: Yuan W, He Y, Shen W, et al. Mogents: Motion generation based on spatial-temporal joint modeling[J]. Advances in Neural Information Processing Systems, 2024, 37: 130739-130763.

[5]: Zhang Z, Kong B, Liu Q, et al. Towards robust and controllable text-to-motion via masked autoregressive diffusion[C]//Proceedings of the 33rd ACM International Conference on Multimedia. 2025: 9326-9335.

[6]: Guo C, Hwang I, Wang J, et al. SnapMoGen: Human Motion Generation from Expressive Texts[J]. arXiv preprint arXiv:2507.09122, 2025.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
