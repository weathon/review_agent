# UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 4

## Abstract
Dense 4D reconstruction from unposed images remains a critical challenge, with current methods relying on slow test-time optimization or fragmented, task-specific feedforward models. We introduce UFO-4D, a unified feedforward framework to reconstruct a dense, explicit 4D representation from just a pair of unposed images. UFO-4D directly estimates dynamic 3D Gaussian Splats, enabling the joint and consistent estimation of 3D geometry, 3D motion, and camera pose in a feedforward manner. Our core insight is that differentiably rendering multiple signals from a single Dynamic 3D Gaussian representation offers major training advantages. This approach enables a self-supervised image synthesis loss while tightly coupling appearance, depth, and motion. Since all modalities share the same geometric primitives, supervising one inherently regularizes and improves the others. This synergy overcomes data scarcity, allowing UFO-4D to outperform prior work by up to 3 times in joint geometry, motion, and camera pose estimation. Our representation also enables high-fidelity 4D interpolation across novel views and time. Please visit our project page for visual results: https://ufo-4d.github.io/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces UFO-4D, a feedforward model that takes two unposed images (with intrinsics) and predicts a dynamic 3D Gaussian Splat representation in a canonical frame plus the relative camera pose. A single differentiable 3DGS rasterizer is used to render RGB, pointmaps, and scene flow at arbitrary times and viewpoints by swapping rendered attributes through the same rasterizer. Training mixes supervised losses with photometric and edge-aware smoothness losses on rendered outputs. Strong results are reported on Stereo4D, Bonn, and KITTI, including 4D spatio-temporal interpolation from just two frames.

### Strengths
1. Unified explicit 4D design. Casting image/geometry/flow supervision through one renderer on 3D Gaussians without any additional regression heads is elegant and practical, aligning with the efficiency of 3DGS splatting while extending it to 4D prediction.
2. Quantitative gains. Clear improvements vs Zero-MSF and DynaDUSt3R on multiple datasets, suggesting the rendering supervision is beneficial for motion estimation. 
3. Good ablation studies on training data, including single-dataset and mixed-dataset configurations, clarifying domain-transfer trade-offs.

### Weaknesses
1. Pose metrics are missing. Pose is central but quantitative metrics aren’t reported.
2. Lack of evaluations. Additional comparisons would strengthen the paper, particularly to dynamic reconstruction settings like **MonST3R [1]**, methods that jointly estimate pointmap and scene flow such as **Dynamic Point Maps [2]**, **St4RTrack [3]**, **POMATO [4]**, and classic scene-flow estimators like **RAFT-3D [6]**.
3. As noted in the paper’s limitations, the method is designed and evaluated for two views; there is no demonstrated extension to multi-view or longer sequences. This is problematic for dynamic scenes where motion reasoning benefits strongly from multi-view temporal context. Compared with several works [2,3,4,5] that do scale to multi-view settings, restricting to only two views appears limiting for long-range tracking, occlusion handling, and temporal consistency.
4. Lack of large-baseline or large-motion evaluation. The examples and datasets used in the paper mostly appear to involve small baselines and moderate motion. It would strengthen the work to include results on large-baseline or high-motion scenarios, or to evaluate on datasets such as MPI-Sintel that feature stronger motion variation. Demonstrating robustness in these challenging regimes would better validate the method’s generalization and motion reasoning capabilities.
5. Figure issue (Fig. 7): In the last example, the image and depth outputs appear swapped.

[1] Zhang, Junyi, et al. "Monst3r: A simple approach for estimating geometry in the presence of motion." ICLR’24.

[2] Sucar, Edgar, et al. "Dynamic Point Maps: A Versatile Representation for Dynamic 3D Reconstruction." ICCV’25.

[3] Feng, Haiwen, et al. "St4rtrack: Simultaneous 4d reconstruction and tracking in the world." ICCV’25.

[4] Zhang, Songyan, et al. "POMATO: Marrying Pointmap Matching with Temporal Motion for Dynamic 3D Reconstruction." ICCV’25.

[5] Han, Jisang, et al. "D^ 2USt3R: Enhancing 3D Reconstruction with 4D Pointmaps for Dynamic Scenes." NeurIPS’25.

### Questions
1. The motion and depth maps in the qualitative figures appear blurrier than those from ZeroMSF. Is this softness an inherent effect of the Gaussian representation and splatting, or does it result from the rendering resolution or regularization losses used during training?
2. Does the proposed method still perform reliable 4D interpolation when the two input views have a large baseline or significant motion?

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
This paper proposes a method to predict appearance and geometry (in the form of gaussian splats, motion and cameras) from two views. This is done in a feedforward manner and the main novelty is in connecting 2D motion/geometry supervision (in the form of point maps and scene flow) to the 3D outputs of the network, which is done by reusing the gaussian splats rendering aparatus for non-RGB rendering.

### Strengths
Although there are several papers tackling the general problem of 4D estimation (many correctly mentioned in the related work), this contribution stands out as a simple idea that is very well executed. Namely, the idea of rasterizing geometric and motion information into 2D to supervise it with scene flow and point maps (i.e. LIDAR) from real datasets seems somewhat novel (if a little obvious in hindsight, like all good ideas). This differs from approaches that demand 3D supervision, which often limits them to synthetic data, and approaches based purely on photometric supervision, which are plagued with degenerate solutions in the case of motion data (due to the inherent ambiguity of camera vs object motion).

The paper is written very clearly and the experiments are extensive, with significant gains. These are slightly tempered by the pretraining being based on recent SOTA weights, but the method brings a clear advantage.

### Weaknesses
For being based on NoPoSplat and MASt3r, a direct comparison with those methods where appropriate is needed. This would quantify the actual gains the method brings. However, it is understood that broadening the applicability of the original moodels can be a sufficient justification for the proposal.

As a minor aside on clarity, the smoothness loss should be written explicitly to make the paper self-contained, since it is an important factor for performance.

### Questions
Mostly I would like to know that a comparison is underway or that preliminary experiments indicated the gains are worth it, on a apples-to-apples comparison. This would turn the rating into a strong accept.

As the contribution is predicated on a simple idea, if other reviewers bring up (published) papers that call its priority into question, I would like to see a proper distinction between the methods.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces UFO-4D, a unified feedforward architecture for reconstructing dense 4D scenes from two images without known camera poses. The method simultaneously predicts dynamic 3D Gaussian splats and the relative camera transformation in one forward pass, creating an explicit 4D scene representation that jointly recovers 3D structure, motion fields, and camera pose while enabling spatio-temporal interpolation. By employing differentiable rasterization, the framework integrates both self-supervised photometric constraints and supervised signals for geometry and motion, establishing a strong coupling between depth and motion prediction. Comprehensive evaluations show state-of-the-art results across multiple geometry and motion benchmarks, supported by qualitative and quantitative analysis, ablation studies, and novel 4D interpolation use cases.

### Strengths
- **Unified feedforward architecture:** Unlike previous works that obtain geometry and motion estimation with separate steps, the proposed UFO-4D offers an unified 4D representation capable of solving multiple downstream perception tasks jointly which has been found to be beneficial in other domains.
- **Leveraging 4DGS representation:** By leveraging 4D Gaussian Splatting as the representation, this allows the model to be trained with additional photometric supervision, and also allow other estimations such as motion vectors to be easily rasterized allowing additional self-supervised signals, and allowing all signals to train a joint representation.
- The overall architecture is simple and straightforward, with the details of the architecture being explicitly mentioned, making it easy for re-implementation of the method.

### Weaknesses
- **Lack of direct comparison:** Although I agree that leveraging a representation that tightly couples both pointmap and motion estimation will further boost performance, it is difficult to understand the benefits that come from adopting this new representation itself as the proposed method also utilizes a very recently open-sourced dataset Stereo4D for training. It would be nice to include a comparison of baseline method trained with the same dataset recipe to better highlight the advantages of the proposed representation.
- **Small baselines:** The qualitative figures shown in the paper seem to take two frames with small baselines as input. As a result, it is difficult to understand how well the method is estimation motions when given large baselines (two frames that are far away) as input.
- **Limited evaluation:** Adding more evaluation results in dynamic datasets such as Sintel[1] or TUM-Dynamics[2] could better highlight the strength of the paper. Also adding evaluation of the estimated camera pose for all datasets would be needed to emphasize the importance of adopting the proposed representation.

### References

---

[1] Butler, Daniel J., et al. "A naturalistic open source movie for optical flow evaluation." European conference on computer vision. Berlin, Heidelberg: Springer Berlin Heidelberg, 2012.

[2] Sturm, Jürgen, et al. "A benchmark for the evaluation of RGB-D SLAM systems." *2012 IEEE/RSJ international conference on intelligent robots and systems*. IEEE, 2012.

### Questions
Q1. How does UFO4D estimate relative camera pose? How does the performance of camera pose estimation compare with prior works?

Q2. As mentioned in the discussion of the Stereo4D dataset, there seem to be pixels in both images where the tracking and correspondence information is not provided in the dataset. For these pixels, what does the velocity vector in the predicted Gaussian learn? In addition, how sparse are these signals during training? Can other supervision from dense losses compensate for this sparsity?

Q3. How robust is UFO4D to large baseline images?

Q4. Why does UFO4D initialize the weights from Mast3r instead of learning directly from the finetuned version MonST3R? It seems that training could be further boosted and become more efficient when initialized from MonST3R.

With several questions and unclear parts remaining, I would like to recommend border reject as my initial rating. However, with my questions being resolved, I am eager to raise my score.

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
This paper focuses on a dynamic Gaussian reconstruction framework that infers 4D scenes from two images. The reconstructed 4D scenes enables motion extraction and novel view synthesis. The experiments demonstrate the effectiveness of the proposed method.

### Strengths
1. The paper focuses on a cutting-edge task that directly predicts Gaussian attributes from unposed images. The task is applicable of many downstream applications.
2. This paper implement a standard and simple network with pure ViTs, which can be potentially scaled up to large datasets and parameters.

### Weaknesses
1. The results are not convincing. This paper does not show enough visualization on the rendering quality of the reconstructed 3DGS at different novel views. And no videos are provided as the supplementary for a clear visualization on the reconstruction quality. So I am not sure whether the generated 3DGS are of high-quality. Given the current figures, I do not see significant performance gain compared to only predict colored points, which weakens the insight of predicting Gaussian primitives.
2. A video showing novel view synthesis results are required.
3. I found that the input image pairs are often of very similar viewpoints. I wonder if the method can handle pairs of images that contains large offsets in camera poses? 
4. A comparison with 4DGT is recommended since 4DGT also focuses on predicting dynamic Gaussians. Since 4DGT requires camera poses as input, you can simply apply an off-the-shelf camera pose estimation model first.

### Questions
Please refer to the weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
3
