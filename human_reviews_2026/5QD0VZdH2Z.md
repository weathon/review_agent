# Implicit Surface Reconstruction from Sparse and Noisy Poses with Large Motions

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Recent advances in implicit surface reconstruction have significantly improved 3D reconstruction techniques. However, challenges persist, particularly when dealing with sparse and noisy poses. Traditional methods attempted to address these challenges through photometric and geometric consistency, but they often struggled as camera baselines increased. This difficulty arises due to incorrect guidance caused by occlusions during the learning of neural implicit representation. To overcome this issue, we propose an approach that incorporates uncertainty-aware guidance for multi-view consistency, allowing for better adaptation to scenarios with sparse and noisy inputs. Additionally, to facilitate the learning of surface geometry in a challenging setup, we propose a geometric smoothing termed progressive SDF loss. Through empirical studies on occlusion handling and geometric smoothing, our method achieved state-of-the-art performance, significantly enhancing both the refinement of noisy camera poses and surface reconstruction quality. This advancement strengthens the robustness and flexibility of implicit surface reconstruction in challenging conditions, paving the way for more effective applications in computer vision and 3D scene understanding.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper improves neural implicit surface reconstruction from sparse and noisy poses by using uncertainty-aware guided multi-view consistency. The method models the uncertainty by computing the unprojected errors of correspondences based on rendered depths. Then, the method incorporates this uncertainty into patch-wise NCC and  unprojection errors. In addition, the method introduces progressive SDF loss to smooth implicit surfaces. The experiments show that the method can achieve competitive performance on DTU under narrow and wide baselines.

### Strengths
1. The method leverages uncertainty to focus on high confident photometric and geometric information to refine noisy poses.
2. The method follows iSDF and uses coarse geoemtry in early training to impose progressive SDF loss.
3. The method is simple and easy to follow.

### Weaknesses
1. The ablation study is only conducted on one sene, DTU 110 scan. This experiment is not convincing.
2. In fact, the method uses patch-wise NCC to help surface reconstruction. Then, the baseline used in this paper, SPARF + NeuS is unfair as NeuS does not use patch-wise NCC. A fair baseline should be SPARF + Geo-NeuS.
3. In ablation study, it seems that the better sufaces with progressive SDF loss cannot help pose refinement.

### Questions
1. To better validate the effiectiveness of the introduced strategies, it is better to conduct ablation study on all DTU scans.
2. As SPARF can provide competitive or better pose estimation and the method uses patch-wise NCC, the baseline should be repalced with SPARF + Geo-NeuS, which combines NeuS with patch-wise NCC to verify the surface reconstruction performance.
3. With progressive SDF loss, the surface reconstruction becomes better, I wonder if this can help pose refinement.
4. For uncertainty modeling, there exist many NeRF works model uncertainty with different stragtegies, such as ActiveNeRF [1] and AR-NeRF [2]. I wonder if these strategies can replace the uncertainty modeling in this work.   
[1] Pan et al. ActiveNeRF: Learning where to See with Uncertainty Estimation, ECCV 2022.   
[2] Xu et al. Few-shot NeRF by Adaptive Rendering Loss Regularization. ECCV 2024.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper tackles sparse-view implicit surface reconstruction when camera poses are noisy and viewpoints are widely separated. It proposes a joint optimization framework that (1) injects geometry-based uncertainty into pose refinement and geometric learning; and (2) introduces a pSDF regularization term for smoothing surfaces without over-blurring details. Experiments on DTU and BlendedMVS demonstrate state-of-the-art results in both pose and surface accuracy and geometry.

### Strengths
1. Satisfactory experimental results. The method conduct experiments on DTU, BlendedMVS datasets in both narrow and wide settings, and show clear advantages in pose estimation and surface reconstruction.

2. The writing is clear and the methodology is easy to reproduce.

3. The use of multi-view depth discrepancies of matched pairs to quantify uncertainty is insightful.

### Weaknesses
1. Limited technical novelty. In the Occlusion Handling part, the main contribution is using the projected depth discrepancy of 2D match pairs as an uncertainty estimate. The NCC loss and geometric consistency constraints are adaptations of existing works (e.g., NeuralWarp, PGSR). Additionally, the surface smoothing strategy is not substantially different from the constraint in iSDF. It mainly swaps the supervision source from LiDAR points to estimated surface points.

2. Insufficient comparison with recent work. The experimental baselines stop at methods from 2024 or earlier, lacking evaluations against the latest approaches.

### Questions
In iSDF, the pseudo SDF at a query point is defined by its distance to sparse sensor points, which are typically reliable, making the supervision well-founded. In this paper’s setting, however, the reference surface is coarse and may deviate from the true geometry. Using this as the SDF target could introduce bias and drive the optimization toward incorrect minima. The authors should discuss this risk or provide empirical evidence and sensitivity analysis.

### Soundness
2

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
Summary: This paper proposes a method for robust surface reconstruction from spare and noisy camera poses with wide camera baselines. The core of the approach is two regularization strategies which help in generating faithful surfaces in sparse scenarios. 

The first one is enforcing uncertainty aware geometric and photometric consistency as a loss to handle occlusions, where the uncertainty is pre-computed using pixel correspondences obtained from a pre-trained model. With this precomputed uncertainty the original photometric consistency loss proposed in [1] and geometric loss proposed in [2] are scaled appropriately to handle occlusions.

Secondly, a surface smoothness constraint is enforced, so that the representation is able to reach a solution even in sparse and degenerate scenarios. 

Comparisons of both surface reconstruction quality and camera pose error on DTU and BlendedMVS dataset show that this method can robustly reconstruct the surface along with optimizing camera poses under sparse images and noisy pose as input.

### Strengths
* The motivation of the paper is clear, and the paper is well-written with clear presentation and the proposed method is easy to follow with adequate visual illustrations at the required places (Fig. 3, 4, 5).

* The paper is solving a relevant problem. Reconstructing a high fidelity surface from sparse and noisy input in a per-scene optimization framework is still a relevant problem as most of the recent methods focus on training a prior model on a dataset, because of which they are generally limited by the generalization ability compared to per-scene optimization techniques.

* The SDF smoothness constraint and the occlusion handling approach seems interesting and its effectiveness is validated with adequate ablations. Particularly the training strategies used to overcome local minima (over-smooth surfaces) is interesting.

### Weaknesses
* Although the strategy to robustly optimize for surfaces under the proposed constraints is interesting, the constraints itself are not new. Basically NCC loss and the geometric constraint is pretty common and all this technique does is scaling the loss with the precomputed uncertainty obtained from the correspondence network. Also the SDF regularization technique seems to be already proposed in iSDF [3] in which the authors propose a variant in which the optimization starts from a coarse geometry obtained after certain iterations.

* The paper uses a correspondence prediction network PDCNet [4] which seems to be pretty old as many advanced models like Dust3r [5] exist, which the authors themselves have mentioned in the Conclusion section (Sec. 5). Have they tried their approach with more latest models? I am asking this question, because I have a feeling that the ability of this method to handle wide baseline cameras is generally limited by the correspondence network’s ability for the same. An experiment to what extent this method can handle “wide” baselines is required to understand the robustness of this approach.

* Although the proposed approach for SDF regularization seems effective, have the authors tried by incorporating simple smoothness priors like depth from a foundation model into their framework? An ablation with such prior will be helpful to better understand the effectiveness of the smoothness constraint.

### Questions
Please refer to weakness. 

Minor Questions: 

B in eq. 6 is the batch size of what? Rays? 

The paper should discuss some prior based methods which handle such wide baselines camera for reconstruction in the related work for e.g. papers like [6] and [7] and many more exists which solve similar problem with a prior based approach.

References: 

[1] Shi-Sheng Huang, Zixin Zou, Yichi Zhang, Yan-Pei Cao, and Ying Shan. Sc-neus: Consistent neural surface reconstruction from sparse and noisy views. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pp. 2357–2365, 2024c.

[2] Prune Truong, Marie-Julie Rakotosaona, Fabian Manhardt, and Federico Tombari. Sparf: Neural radiance fields from sparse and noisy poses. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4190–4200, 2023.

[3] Joseph Ortiz, Alexander Clegg, Jing Dong, Edgar Sucar, David Novotny, Michael Zollhoefer, and Mustafa Mukadam. isdf: Real-time neural signed distance fields for robot perception. arXiv preprint arXiv:2204.02296, 2022.

[4] Prune Truong, Martin Danelljan, Luc Van Gool, and Radu Timofte. Learning accurate dense correspondence and when to trust them. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5714–5724, 2021.

[5] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20697–20709, 2024.

[6] Vora, Aditya, Akshay Gadi Patil, and Hao Zhang. "Divinet: 3d reconstruction from disparate views using neural template regularization." Advances in Neural Information Processing Systems 36 (2023): 66768-66781.

[7] Du, Yilun, et al. "Learning to render novel views from wide-baseline stereo pairs." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

### Soundness
3

### Presentation
3

### Contribution
3
