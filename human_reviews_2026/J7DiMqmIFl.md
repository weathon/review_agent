# G-CUT3R: Guided 3D Reconstruction with Camera and Depth Prior Integration

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
We introduce G-CUT3R, a novel feed-forward approach for guided 3D scene reconstruction that enhances the CUT3R model by integrating prior information. Unlike existing feed-forward methods that rely solely on input images, our method leverages auxiliary data, such as depth, camera calibrations, or camera positions, commonly available in real-world scenarios. We propose a lightweight modification to CUT3R, incorporating a dedicated encoder for each modality to extract features, which are fused with RGB image tokens via zero convolution. This flexible design enables seamless integration of any combination of prior information during inference. Evaluated across multiple benchmarks, including 3D reconstruction and other multi-view tasks, our approach demonstrates significant performance improvements, showing its ability to effectively utilize available priors while maintaining compatibility with varying input modalities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes G-CUT3R, a guided, feed-forward 3D reconstruction method that extends CUT3R by integrating auxiliary priors (camera intrinsics and/or poses, and depth) through modality-specific encoders and zero-initialized 1*1 convolutions for stable fusion inside the decoder. The approach keeps CUT3R’s recurrent state, enabling multi-view processing without global optimization. Experiments on 7-Scenes, NRGBD, Bonn, ScanNet, Waymo, and ScanNet++ report consistent gains over CUT3R and DUSt3R-family baselines, with competitive run-time (around 20 FPS at 512 px on an A40).

### Strengths
- Clear problem framing: Many feed-forward methods ignore readily available priors. Incorporating them is practically important. However, I suggest mentioning DepthSplat [a] as it is also a feed-forward method using depth priors (for a 3DGS reconstruction).
- The method is light-weight and builds upon a well-established baseline, Cut3r. 
- Unified model for arbitrary prior subsets: training with random modality subsets reflects practical scenarios.
- Strong experiments.

[a] Xu, H., Peng, S., Wang, F., Blum, H., Barath, D., Geiger, A. and Pollefeys, M., 2025. Depthsplat: Connecting gaussian splatting and depth. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 16453-16463).

### Weaknesses
- It is a bit unclear to me what depth is used for the datasets. Did the authors always use sensor depth? Using anything else would render the results incorrect. I put this in the weaknesses given that this is very important. The paper should specify a detailed, dataset-by-dataset description of depth sources used as priors to make the results understandable. 
- Same question holds for the camera poses. Do the authors use SLAM-estimated poses (without post-processing) as priors or the GT ones? Using the GT ones again would make the experiments section a bit weaker.
- Limited robustness analysis: no analysis of noisy/misaligned priors (e.g., wrong intrinsics, biased depth, pose drift). Results might depend strongly on prior quality. It would be very important to understand what to expect if things are noisy. I would be very happy to see plots showing how accuracy degrades with noise in the priors. 

Minor:
- Loss definition details are sparse: The confidence-weighted point loss resembles uncertainty modeling, but calibration, scale, and supervision signals for confidences are not really specified.

### Questions
The main questions are what the weaknesses are at the moment:
- What depth/pose was used in the experiments? 
- How does the method behave with noisy priors?

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
This paper presents G-CUT3R, a guided feed-forward 3D reconstruction framework that enhances the geometric consistency and reliability of transformer-based models such as DUSt3R and MASt3R.
The key idea is to explicitly regularize uncertainty alignment and cross-view geometric coherence, addressing the observation that per-view uncertainty maps are often inconsistent and lead to noisy surface fusion.

The framework introduces two components:

Cross-View Uncertainty Tuning (CUT): aligns the uncertainty distributions of corresponding pixels across multiple views using a learnable temperature scaling and a cross-view consistency loss.

Geometry-Guided Regularization (G-Reg): imposes local surface smoothness and normal coherence regularization in 3D space, with learnable weights controlling the regularization strength.

Experiments on ETH3D, Tanks & Temples, and CO3D show improved depth RMSE (≈10–12%) and higher pointmap F1 scores (+2–3%) compared with MASt3R and DUSt3R, indicating more stable multi-view fusion and sharper geometry boundaries.

### Strengths
Technically sound and well-motivated: both CUT and G-Reg directly target known weaknesses of feed-forward 3D reconstruction—cross-view inconsistency and surface noise—and are implemented cleanly.

Improved stability and quality: the proposed regularizations yield consistent gains in quantitative metrics and visual quality across diverse datasets.

Good empirical rigor: ablations on each component demonstrate that uncertainty alignment and geometric regularization complement each other.

### Weaknesses
Limited conceptual novelty: both CUT and G-Reg are straightforward extensions of well-known principles—uncertainty calibration and geometric smoothing. The contributions lie more in empirical engineering than in new theoretical or algorithmic insight.

Lack of deeper analysis: the paper does not explore why these regularizations help beyond intuitive reasoning; no theoretical justification or failure analysis is offered.

Possible over-smoothing: G-Reg may suppress fine details, but no perceptual or surface-sharpness evaluation is presented.

### Questions
On over-regularization: have the authors evaluated whether the geometry regularizer causes over-smoothing in regions with high-frequency detail?

On generalization: have the authors tested whether these regularizations still help when applied to other backbones (e.g., Fast3R or VGGT), or are the gains specific to the MASt3R-like architecture?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a novel feed-forward framework for 3D scene reconstruction. In particular, geometric priors—including depth, camera intrinsics, and camera extrinsics—are incorporated into the RGB latent space to guide the reconstruction process. Extensive evaluations on multiple benchmarks demonstrate that the proposed method achieves substantial performance improvements.

### Strengths
This work incorporates additional priors into feed-forward 3D reconstruction, enhancing its flexibility for diverse application scenarios. Comprehensive experiments demonstrate significant performance gains over state-of-the-art methods.

### Weaknesses
- The paper claims to be efficient and lightweight. However, no additional results are provided to support this, such as FLOPs or parameter counts compared with CUT3R and Pow3R. Since the method introduces extra encoders and layers for additional modalities, it may incur substantial parameter and computation overhead relative to CUT3R, potentially compromising efficiency. It is unclear whether the reported efficiency stems primarily from inheriting CUT3R’s efficiency rather than being more efficient than CUT3R itself.
- The proposed approach appears to combine elements from Pow3R and CUT3R. Could the authors clarify the key differences from these prior works?

### Questions
- Regarding the depth used during training, is it sensor depth or COLMAP-generated depth? The model that uses only depth priors shows only a marginal improvement in reconstruction quality. Why does this occur, given that depth information is typically highly correlated with reconstruction performance?
- The encoders for the additional priors are not shared, which may significantly increase the number of parameters. Have the authors considered using a shared encoder to reduce model size?
- In Table 3, does Pow3R employ the prior-integration mechanism proposed in this work or the original Pow3R design?

### Soundness
2

### Presentation
3

### Contribution
2
