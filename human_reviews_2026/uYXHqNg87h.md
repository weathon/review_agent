# MultiViewPano: Training-Free 360° Panorama Generation via Multi-View Diffusion and Pose-Aware Stitching

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 2

## Abstract
We propose MultiViewPano, a training-free framework for generating 360° panoramas from one or more arbitrarily positioned input images with varying fields of view. Existing panorama generation methods are limited by requirements for fixed 90° field-of-view inputs, single viewpoint assumptions, or extensive task-specific fine-tuning. Our approach addresses these limitations by leveraging a pretrained multi-view diffusion model (SEVA) to synthesize overlapping novel views along strategically sampled camera trajectories, followed by a novel pose-aware stitching algorithm that exploits known camera geometry for seamless fusion. Unlike traditional feature-based stitching, our pose-aware approach directly utilizes camera poses to determine optimal seams and blending regions. Experiments on standard benchmarks demonstrate that our method achieves competitive performance with state-of-the-art approaches (FID of 12.82 on Laval Indoor, 22.99 on SUN360) while supporting arbitrary input configurations without requiring retraining or fine-tuning. \href{https://anonymous.4open.science/r/MultiViewPano-71D7/README.md}{Anonymous Repository link.}

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a novel, training-free method for generating 360° panoramas from pinhole camera images. The approach leverages the existing large-scale video model, SEVA, to synthesize a series of pinhole images along a defined camera trajectory. These images are then seamlessly combined into a final panorama using a custom-designed "Pose-based Stitcher" module.
A key advantage of this method is its flexibility, supporting both single- and multi-view inputs. The authors note that using multiple inputs can improve output quality. This framework overcomes critical limitations of prior work, such as the rigid requirement for 90° field-of-view inputs, the need for dataset-specific fine-tuning, and the inability to handle multiple input images. The method demonstrates strong, competitive results on the Laval Indoor and SUN360 datasets.

### Strengths
The paper proposes a training-free method that requires no additional training, offering strong generalization. Its performance on the Laval Indoor and SUN360 datasets even surpasses some methods that require dataset-specific fine-tuning.

The proposed method is highly flexible. Unlike prior work, it does not require input images to have a precise 90° field-of-view (FoV) or be captured from a fixed position. Instead, it supports any pinhole camera images as input, as long as their intrinsics and extrinsics (poses) are known.

The paper is well-structured, with clear logic that is easy to follow.

### Weaknesses
1. Lack of 360° Wraparound Consistency: The paper claims superior realism and semantic consistency. However, I observed a significant lack of 360° wraparound consistency at the horizontal edges (left and right) of the panoramas. When attempting to "wrap" the panorama by stitching the far-right edge to the far-left, the seams do not align seamlessly. This issue is evident in the first example of Figure 5, as well as in the second and fourth (top-to-bottom) examples in Figure 6. I question if this is a failure of the blending/feathering trick at the boundary, as this issue is notably absent in the CubeDiff results. This left-to-right consistency is crucial for panoramic images.


2. Critique of the 2D Stitching Method: I have reservations about the proposed stitching method. The approach, which relies on dynamic programming based on 2D pixel-wise L2 costs, seems to prioritize 2D pixel similarity. A more geometrically-sound approach should perhaps be considered. For instance, in overlapping regions, priority should be given to points closer to the camera to correctly handle 3D occlusion, rather than preserving content based solely on surrounding pixel consistency.

3. Confounding Variable (Clarity AI Upscaler): The use of the powerful "Clarity AI Upscaler" as a post-processing step introduces a confounding variable. It becomes difficult to determine whether the final seamless quality is achieved by the proposed dynamic programming stitcher or by the upscaler (which itself performs seamless tiling).

4. Unfair Comparison: Related to the point above, it seems unfair to compare their final, post-processed (upscaled) results against the non-upscaled outputs of the baseline methods.

5. Missing Discussion on Computational Cost: The framework relies on multiple large-scale models (SEVA, Clarity AI Upscaler). This likely results in significant computational overhead (e.g., resource consumption, inference latency) compared to other methods, yet this critical aspect is not discussed or benchmarked.

6. Insufficient Validation of Multi-View Claims: The claim to handle any number and pose of inputs is not sufficiently validated.

（1）The experiments are missing a quantitative ablation study on how the number of input images or the nature of their trajectories impacts the final output quality.

（2）Additionally, the methodological choice of using the camera pose centroid as the "scene center" for multi-view inputs seems arbitrary. This is not the "true" scene center, and it is questionable whether this is the most robust or geometrically reasonable choice.

7. Missing Related Work: The related work section is missing comparisons to other recent and highly relevant methods that also generate panoramas from pinhole inputs, such as PanoDecouple[1] (CVPR'25). While DA2: Depth Anything in Any Direction[2] might be too recent, a discussion and comparison against PanoDecouple seem warranted.

8. Missing LLM Usage Disclosure: A required section detailing the use of Large Language Models (LLMs) appears to be missing. I thoroughly reviewed the main paper and the supplementary materials, but could not find the LLM usage disclosure as mandated by ICLR.

### Questions
1. On Geometric Consistency and 3DGS Potential: The paper highlights improved consistency. How robust is this consistency in practice? Specifically, are the final stitched panoramas geometrically accurate enough to be used for downstream 3D scene reconstruction tasks, such as 3D Gaussian Splatting (3DGS)? If this method can serve as a reliable "pseudo-camera" rig for 3DGS, I would consider this a significant contribution and would be inclined to raise my score.

2. Request for Deeper Multi-View and Failure Analysis:

（1） The quantitative experiments are conducted entirely in the single-image-input regime. Could the authors provide more qualitative examples and discussion on both success and failure cases specifically for the multi-view input scenario?

（2） Could the authors also analyze how well the synthesized panorama preserves the content and fidelity of the original input pinhole images when multiple inputs are provided?

### Soundness
2

### Presentation
2

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
The paper introduces MultiViewPano, a training-free framework for generating 360° panoramas from one or more arbitrarily posed images. The method leverages a pre-trained multi-view diffusion model (SEVA) to synthesize novel views and employs a pose-aware stitching algorithm for seamless panorama assembly. The approach is evaluated on standard image-to-pano benchmarks and claims competitive results with state-of-the-art methods, while supporting flexible input configurations without retraining.

### Strengths
- The pipeline is well-thought out, modular, allowing for future improvements (e.g. compatibility with improved multi-view models)
- The training-free nature and support for arbitrary camera poses and fields of view are practical
- The pose-aware stitching algorithm is a useful addition
- Quantitative and qualitative results on standard benchmarks (Laval Indoor, SUN360) are competitive

### Weaknesses
- The method does not address coverage near polar regions, which is a common and prominent source of artifact in panorama generation. While the authors address this, it remains a weakness.
- The approach is heavily dependent on SEVA’s consistency; performance degrades with extreme viewpoint changes, and the method inherits SEVA’s weaknesses with reflective and transparent surfaces.
- Despite being designed for multi-view input, the evaluation is almost entirely in the single-image-to-panorama regime due to the lack of multi-view benchmarks. Even a small, manually curated dataset could greatly improve the work. Claimed advantages for multi-view scenarios are not rigorously validated. (Datasets like UDIS [1] could be adapted to this task, e.g. stitching using conventional algorithms to get a panorama, and then sampling various views)
- The need for post-processing (Clarity AI Upscaler) to address diffusion model artifacts suggests that the raw output quality is not always sufficient, and introduces a large computational overhead.

[1] Nie, Lang, et al. "Parallax-tolerant unsupervised deep image stitching." Proceedings of the IEEE/CVF international conference on computer vision. 2023.

### Questions
- How robust is the method to inaccurate or noisy camera pose estimates, which are common in real-world multi-camera setups?
- Can the approach be extended to support vertical camera trajectories, or is this fundamentally limited by the current model and pipeline?
- Are there plans to establish or contribute to a multi-view panorama benchmark to better validate the method’s intended use case?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces MultiViewPano, a training-free method for generating panoramas from multiple input images with known camera poses and fields of view. The framework is built upon existing pretrained models, achieving reasonable results and demonstrating some design contributions. However, the overall novelty is limited, as most components are adaptations of prior work.

### Strengths
1.The method flexibly leverages existing pretrained models to achieve panoramic generation, and the overall architectural design contributes to the integration of multi-view information.
2.The proposed Pose-Aware Stitching algorithm is a notable addition, introducing a new approach to image alignment and blending.

### Weaknesses
1.Unfair comparison due to known camera poses:
oPrevious works such as PanoDiffusion and CubeDiff do not assume access to camera poses, whereas the proposed method explicitly uses them as priors (Line 196). This raises fairness concerns in comparisons.
oMoreover, if the input camera poses are unknown or inaccurate — which often happens in real-world scenarios — the proposed pipeline may not function properly. The authors should discuss the robustness of their method under pose estimation errors.
2.Strong dependence on SEVA outputs:
oThe method relies heavily on SEVA to produce the initial results. It is unclear how MultiViewPano performs when SEVA outputs poor-quality results. The authors are encouraged to provide visual examples or quantitative results in cases where SEVA fails, to demonstrate the robustness of their method.
3.Incomplete ablation study:
oThe ablation analysis is insufficient. The paper applies multiple quality enhancement steps, making it unclear which improvements stem from the proposed modules. The authors should visualize intermediate results at each stage to show the contribution of each component — e.g., whether seam removal is due to Pose-Aware Stitching or seamless tiling.
4.Potential distortions and artifacts introduced by enhancement steps:
oThe pipeline includes several enhancement stages, each of which may introduce geometric distortions or noise. The authors should explain how these issues are mitigated or controlled throughout the refinement process.
5.Limited performance improvement:
oIn Table 1, the proposed method does not outperform CubeDiff, particularly in indoor scenes. The authors should provide a detailed analysis of why the method fails to show consistent improvements across different settings.
6.Formatting issue:
oIn Table 1, the leading scores across metrics should be boldfaced for clarity and consistency.

### Questions
1.How fair is the comparison with prior works that do not assume access to camera poses? How does MultiViewPano perform when camera poses are inaccurate or unavailable?
2.How robust is the proposed method to poor-quality SEVA outputs? Can the authors provide visualizations of such cases?
3.Could the authors expand the ablation study and visualize intermediate outputs to clarify which modules contribute to which improvements?
4.How are geometric distortions and artifacts handled during repeated enhancement steps?
5.Why does the proposed method not outperform CubeDiff, especially for indoor scenes?
6.Could the authors ensure that all best-performing metrics in Table 1 are boldfaced for readability?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
MultiViewPano presents a training-free framework for generating full 360° panoramas from one or more arbitrarily posed images with varying field-of-view (FoV). Unlike prior arts that require fixed 90° FoV, single-center capture, or task-specific fine-tuning, the method leverages a pre-trained multi-view diffusion model (SEVA) to synthesize overlapping novel views along a designed camera trajectory, and then fuses them via a pose-aware stitching algorithm that directly exploits known camera geometry instead of fragile feature matching.

Key Contributions

1） Training-free pipeline: Supports any number of images, any pose, any FoV without retraining or task-specific optimization.

2） Pose-aware stitching: A geometric fusion module that projects views to a common spherical canvas.

### Strengths
Overall, the key strength of this work lies in its practical applicability: 

1）Training-free and practical framework.
The paper presents a panorama generation pipeline that requires no fine-tuning or re-training, greatly improving versatility and deployment efficiency—especially in real-world scenarios where large-scale panoramic datasets are unavailable.

2） Multi-view input support broadens applicability.
Unlike most prior methods that rely on a single image or fixed-view inputs, MultiViewPano accepts any number of images with arbitrary poses, making it far more adaptable to practical capture conditions.

### Weaknesses
My main concern regarding this work is its limited novelty. 

1） Essentially, it combines the SEVA model with traditional stitching methods, offering marginal technical contribution.

2） The robustness of the approach remains unclear when SEVA fails to produce consistent novel views, especially in challenging scenarios with large viewpoint changes or repetitive textures.

3） The experimental comparison is insufficient. Notably, recent methods such as Diffusion360, PanFusion and HunyuanWorld are not included in the evaluation, which undermines the comprehensiveness of the benchmark.

### Questions
1.3D consistency failures in SEVA

In my understanding, SEVA does not always guarantee 3D-consistent multi-view outputs.
Under such conditions, would a reasonable panorama still be produced?

2.Incomplete baseline coverage

The experimental comparison omits several recent strong baselines.

### Soundness
3

### Presentation
3

### Contribution
2
