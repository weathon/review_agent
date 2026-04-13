## Human Reviewer 1

### Summary
The paper presents FreeSplatter, a scalable feed-forward framework for reconstructing 3D Gaussian representations from uncalibrated sparse-view images while simultaneously recovering camera parameters. To cover both object-centric and scene-level settings, the authors train two variant models on a large amount of 3D dataset. The proposed method addresses challenges associated with fast pose-free sparse-view reconstruction and accurate camera pose estimation.

### Strengths
1. FreeSplatter addresses the problem of pose-free Gaussian Splatting for sparse-view reconstruction in a feed-forward and scalable manner.
2. The idea of integrating camera parameter recovery into the feed-forward framework with 3DGS reconstruction showcases is insightful.
3. Using a streamlined transformer architecture is straightforward and useful as the experiments suggest.
4. The method’s scalability is advantageous, making it suitable for large-scale real-world applications.

### Weaknesses
1. The pipeline contribution is limited as the architecture design is quite similar to GS-LRM [1] without specific designs for accurate camera pose estimation. The camera pose is estimated after generating Gaussian Maps which is independent in the feed-forward framework and more likely to a post-processing stage with the PnP algorithm. This makes the paper paradigm like just A (GS-LRM architecture) plus B (PnP algorithm).
2. A more elegant design may incorporate camera pose estimation into the feed-forward framework (e.g., generating image tokens with camera tokens simultaneously).
3. The author cannot address the occluded areas in a scene-level setting and the strategy in object-centric is quite similar to the Splatter Image [2]. This also hurt the contribution of the paper. Maybe the insight in Flash3D [3] can be incorporated into the feed-forward architecture.



[1] Zhang K, Bi S, Tan H, et al. GS-LRM: Large reconstruction model for 3d gaussian splatting; ECCV 2024.

[2] Szymanowicz S, Rupprecht C, Vedaldi A. Splatter image: Ultra-fast single-view 3d reconstruction; CVPR 2024.

[3] Szymanowicz S, Insafutdinov E, Zheng C, et al. Flash3D: Feed-Forward Generalisable 3D Scene Reconstruction from a Single Image; arXiv 2024.

### Questions
1. How is the generalizability of the FreeSplatter? As the author mentioned the pipeline is scalable and can be trained on large data, I want to see more discussion about the generalizability of various datasets (e.g., the difference of angles of input views or indoor scene generalize to outdoor scene).
2. Have you considered to train object-level and scene-level data in a unified framework? What is the key challenge to realize the goal?
3. As you mentioned the scalability of the FreeSplatter, why do you only use two input views to train the framework in a scene-level setting (just same as GS-LRM)? How is the performance under more views?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
5

---

## Human Reviewer 2

### Summary
This paper proposes a feed-forward method, FreeSplatter, for reconstructing 3D Gaussian Splatting from pose-free sparse images. The proposed method concatenates input images, which are then processed by a Vision Transformer network that outputs 3D Gaussians in a reference input view. The predicted Gaussians can be used for novel view synthesis and to estimate camera pose through PnP. FreeSplatter is trained on both object-level and scene-level datasets, and the results demonstrate SOTA performance across on both object-level and scene-level datasets.

### Strengths
- The proposed method demonstrates impressive results for novel view synthesis with unposed inputs.
- The pose-free pipeline is promising and produces strong results.
- The network architecture of the proposed method is simple but effetive.
- The paper is well-written and easy to understand.

### Weaknesses
- Lack of fair quantitative comparisons with methods such as pixelSplat and MVSplat. Although Figure 8 presents qualitative results on RE10K, quantitative results are absent. Even though the proposed method requires ground-truth depth information for training, it can still be trained on RE10K by initially training on ScanNet++ and then fine-tuning on RE10K.
- Some comparisons with baseline methods may be unfair. For instance, the patch size of the ViT backbone is critical in pixel-aligned Gaussian prediction methods, as it determines the number of Gaussians. The proposed method, following GS-LRM, uses a patch size of 8, whereas pixelSplat and MVSplat use a patch size of 16. How does performance compare if both methods use a patch size of 8?
- In Line 428, the authors claim that the proposed method outperforms pose-dependent methods. However, this claim lacks a fair comparison against pose-dependent methods under similar conditions as discussed before. An ablation study could compare the proposed method to a pose-dependent approach using the same backbone, such as the GS-LRM pipeline.
- Limited comparison with other SOTA pose-estimation methods, such as RoMa [1] for scene-level pose estimation and Cameras as Rays [2] for object-level pose estimation.
- No ablation study on the effectiveness of adding reference and source embeddings.

[1] Edstedt, Johan, et al. "RoMa: Robust dense feature matching." CVPR. 2024.
[2] Zhang, Jason Y., et al. "Cameras as rays: Pose estimation via ray diffusion." ICLR. 2024.

### Questions
- As noted in the weaknesses, while the proposed pipeline is promising and demonstrates strong performance, a more rigorous comparison and additional experimental analysis are needed to substantiate the claims and more clearly showcase the method’s effectiveness.
- The results are promising but require significant GPU resources for training, which may make reproduction difficult for the community. Will the code be made available?

### Soundness
1

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
5

---

## Human Reviewer 3

### Summary
This work presents a new pose-free 3DGS-based generalizable sparse-view scene reconstruction method, FreeSplatter, which leverages a transformer-based feed-forward model to predict the pixel-wise Gaussian maps under the same coordinate system. Similar to DUSt3r, unknown camera parameters can be obtained through existing PnP solvers. The experimental results demonstrate the superior reconstruction quality compared to prior state-of-the-art, even ones with ground truth poses.

### Strengths
1. This work wisely distills the recent advances in 3D reconstruction such as DUSt3R, 3DGS, and LRM, into one unified framework. Unlike prior works directly integrating multiple pre-trained foundation models, this framework/model design is neat and effective. 
2. The method considers both object-level and scene-level reconstructions, albeit in two separate pre-trained models, which again shows the effectiveness of the proposed pose-free reconstruction strategy. This is another advantage compared to other works.
3. The quantitative improvements are significant. Assuming the reported numbers are accurate and reliable, this would be a great achievement.

### Weaknesses
1. Since the proposed method focuses on sparse-view reconstruction. It would be better if authors could also include the comparison with other optimization-based sparse-view reconstruction methods, for example, ReconFusion, GaussianObject, InstantSplat, etc.
2. L305-311 discusses the occlusion issue with Gaussian maps. Even though the authors propose a strategy to alleviate this problem, I assume this method could still suffer from missing Gaussian points in occluded areas.

### Questions
1. The maximum number of views can be used. The paper shows the results using up to 6 views. Is it possible to add more views for more complex 3D scenes? 
2. What are the inference compute and time costs w.r.t different number of input views?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper follows prior work in large-scale reconstruction models, focusing specifically on unposed sparse-view reconstruction. Given a set of unposed sparse-view input images, it first divides the images into patches and applies linear layers to convert them into patch tokens, which are then processed by transformer layers. The patch tokens are subsequently converted back to the original resolution through linear layers and unpathify, resulting in Gaussian maps. These Gaussian locations are specified in the coordinate frame of a reference view. The camera poses can be computed by inputting the Gaussian positions into traditional solvers. The authors trained two versions of the model: one for scenes and one for objects.

### Strengths
1. It's interesting to learn that "camera poses may not be essential for training high-quality and scalable large reconstruction models."

2. The proposed method is technically sound and elegantly designed.

3. The authors demonstrate that the proposed method outperforms baseline methods.

### Weaknesses
1. The idea of outputting point maps relative to the main (or first) view is not novel in either pose estimation [1] or unposed sparse-view reconstruction [2].

2. Extending LRM to an unposed sparse-view setting is also not new. PF-LRM seems highly similar to the proposed method, although there are some differences: (a) the proposed method predicts point maps in the coordinate frame of a reference view rather than in the object/world frame; (b) it uses Gaussian Splatting instead of NeRF; (c) PF-LRM includes an additional differentiable PnP loss; (d) the proposed method is trained for two versions, for both scene and object data, while PF-LRM focuses mainly on objects; and (e) the proposed method leverages an alignment loss.

I am unable to identify further major differences. If there are others, please remind me. Regarding (b) and (d), I feel these are natural variations and may not be considered as major contributions. For other differences, such as (a) and (c), I believe the current paper lacks sufficient analysis and experiments to demonstrate whether these differences are essential or if they lead to improved performance or degraded performance.

Although PF-LRM has not released code, it has been evaluated on standard datasets, and the results are reported in its paper. Given the high similarity between PF-LRM and the proposed method, I strongly recommend the authors compare their approach with PF-LRM on both reconstruction and pose estimation tasks. For instance, they could follow PF-LRM's evaluation protocol, consult PF-LRM’s authors for guidance, and even request assistance with testing examples.

3. I find the discussion on lines 523-525 particularly interesting. In 3D AIGC, predicting perfectly 3D-consistent multi-view images from 2D diffusion models is extremely challenging. It would be promising if the proposed method could address 3D reconstruction from inconsistent multi-view images effectively. However, the current paper only provides a few qualitative examples of the proposed method, which lack sufficient information. I would be very interested to see both qualitative and quantitative comparisons with recent SoTA feed-forward reconstruction methods when handling predicted inconsistent multi-view images.

4. Comparing MASt3R on the object-level dataset is unfair, as it was not trained on object datasets. A more reasonable comparison would be between FreeSplatter-S and MASt3R on object datasets, given that both were trained on scene datasets. Another option would be to fine-tune MASt3R on object datasets.

5. For sparse-view reconstruction, only rendering metrics are compared. It would be beneficial to also include 3D native metrics, which better reflect the geometric quality of the methods.

6. A natural alternative to "pixel-alignment loss" is enforcing alignment along the ray direction by predicting only the depth value. This variant should also be considered in the ablation study.

7. The ablation study is somewhat limited in scope.

[1] Wang, Shuzhe, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. "Dust3r: Geometric 3D Vision Made Easy." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20697–20709. 2024.

[2] Xu, Chao, Ang Li, Linghao Chen, Yulin Liu, Ruoxi Shi, Hao Su, and Minghua Liu. "Sparp: Fast 3D Object Reconstruction and Pose Estimation from Sparse Views." arXiv preprint arXiv:2408.10195 (2024).

### Questions
1. Do you use estimated poses or ground-truth poses for calculating rendering losses during training?

2. What are the biggest challenges in training a unified model for both objects and scenes?

3. How do you sample poses for the training data? Are there any assumptions during pose sampling, such as normalized objects, fixed intrinsics, or a specific look-up direction? How do you ensure that the camera poses during training can cover all scenarios encountered in the real world? Can you handle cases where, for a single object, some input images are taken in landscape orientation, some in portrait, and some even upside down?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 5

### Summary
The paper introduces FreeSplatter, a framework for pose-free 3D reconstruction from sparse-view images without known camera parameters. The model uses a transformer architecture to convert multi-view image tokens into 3D Gaussian maps, which serve both high-fidelity 3D modeling and fast camera pose estimation. It outperforms traditional methods in both reconstruction quality and pose estimation accuracy.

### Strengths
- This paper proposes feed-forward pipeline for pose-free 3D reconstruction with sparse input images.
- It shows the state-of-the-art performance in view synthesis, and comparable performance in pose estimation.
- The results are well-presented.

### Weaknesses
- Method section lacks detail and detailed explanations, regarding how the suggested method is aligned with the motivation of the paper, and how it is intended to improve performance.
- Ablation studies are weak. Influence of the number of input views is subsidiary, considering the motivation and methodology of this paper. Only quantitative result via plug-in-plug-out styled experiment on pixel-alignment loss is naive.
- Additional application examples are not aligned with the paper's methodology. Both zero123++ and MVDream output result images with pre-defined camera poses. The paper mentions that 'we have to figure out how the camera poses of the multi-view diffusion model are defined', 'this tedious steps are not required any longer', but it is difficult to agree when the pose is already known. Maybe some additional results supporting 'our method can render significantly more clear views from the images than previous pose-dependent LRMs' would be more persuasive.

### Questions
- In section 3.2.Image Tokenization, how do you retrieve 'view embeddings'? What kind of information would it behold? Please provide any explanation, supporting visualization or analysis, if no additional training is required.
- In section 3.2.Camera Pose Estimation, the paper claims that its method is 'feed-forward manner without global alignment'. Correct me if I'm wrong, but isn't PnP-RANSAC is a 'global alignment' process? please clarify.
- Is PnP operated every time the feed-forward happens? Result of PnP will be dependent on gaussian map result, and the gaussian map result will be dependent on PnP result. Is PnP differentiable? If not, how is the gaussian map trained during training stage? Does it assume that the PnP pose result from imperfect gaussian maps is correct?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
3

### Confidence
4