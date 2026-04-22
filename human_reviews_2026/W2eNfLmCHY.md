# DeLiVR: Differential Spatiotemporal Lie Bias for Efficient Video Deraining

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 2, 6

## Abstract
Videos captured in the wild often suffer from rain streaks, blur, and noise. In addition, even slight changes in camera pose can amplify cross-frame mismatches and temporal artifacts. Existing methods rely on optical flow or heuristic alignment, which are computationally expensive and less robust. To address these challenges, Lie groups provide a principled way to represent continuous geometric transformations, making them well-suited for enforcing spatial and temporal consistency in video modeling. Building on this insight, we propose DeLiVR, an efficient video deraining method that injects spatiotemporal Lie-group differential biases directly into attention scores of the network. Specifically, the method introduces two complementary components. First, a rotation-bounded Lie relative bias predicts the in-plane  angle of each frame using a compact prediction module, which normalized coordinates are rotated and compared with base coordinates to achieve geometry-consistent alignment before feature aggregation. Second, a differential group displacement computes angular differences between adjacent frames to estimate a velocity. These biases are combined with temporal decay and a banded attention mask to emphasize short-range reliable relations while suppressing long-range noise. DeLiVR achieves sharper details, fewer rain remnants, and stronger temporal coherence on both synthetic and real rainy benchmarks. The code is publicly available at https://github.com/Shuning0312/ICLR-DeLiVR.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes DeLiVR, a video deraining framework that leverages spatiotemporal Lie-group differential biases to enforce geometric consistency across frames. By embedding Lie-group formulations directly into the attention mechanism, the method effectively models continuous transformations, reducing reliance on optical flow or heuristic alignment. It introduces two key components: (1) a rotation-bounded Lie relative bias for geometry-consistent spatial alignment, and (2) a differential group displacement that captures inter-frame angular velocity to guide temporal attention. Experimental results on multiple benchmarks validate the approach’s superior performance and efficiency compared to existing methods.

### Strengths
1. The paper introduces Lie group theory into video deraining for the first time, providing a principled geometric prior that replaces traditional optical flow with continuous transformation modeling, leading to more robust spatial–temporal consistency.
2. The proposed differential spatiotemporal Lie Bias, composed of the rotation-bounded Lie relative bias and differential group displacement, is elegantly integrated into the attention mechanism. This design explicitly encodes rotational and velocity information, effectively capturing dynamic rain streak variations in real-world videos.
3. The paper is clearly written with solid motivation, detailed methodological exposition, and comprehensive experimental validation, making the technical contributions easy to follow and well-justified.

### Weaknesses
1. While the paper introduces Lie group theory as a prior for modeling continuous transformations, the theoretical connection between the proposed Lie biases and actual motion manifolds is not fully discussed. It would be helpful to clarify how the chosen Lie algebra parameterization ensures stability or accuracy under complex non-rigid motion.
2. The paper claims that VDMamba (CVPR 2025) achieves slightly higher PSNR (36.29 vs. 34.06) and SSIM (0.973 vs. 0.952) but worse perceptual quality (LPIPS 0.010 vs. 0.039). However, the LPIPS metric indicates that VDMamba actually achieves better perceptual results. Therefore, the corresponding discussion in the paper is not rigorous and should be revised to ensure the accuracy of experimental interpretation.
3. The downstream application evaluation is limited to the Rain-Syn-Complex dataset, which is synthetic. Including evaluations on real captured tasks would make the results more convincing.

### Questions
In Table 3, both the S2VD score (26.78) and the proposed method’s score (82.52) are highlighted in bold. It is recommended to use consistent formatting or clarify the highlighting rule to avoid confusion.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes DeLiVR, a method that injects spatiotemporal Lie-group differential biases into Transformer attention scores to achieve geometry-consistent alignment without relying on optical flow. The core contributions include a rotation-bounded Lie relative bias for spatial alignment and a differential group displacement for temporal consistency. Experiments on synthetic and real-world benchmarks demonstrate improved rain removal, detail preservation, and temporal consistency compared to existing methods.

### Strengths
-This work addresses a salient problem in video restoration caused by camera rotation, namely spatial and temporal misalignment. The proposed Lie group based in-plane rotation estimation directly targets this challenge, and the method is well motivated with a clear and coherent rationale.

-The proposed method achieves state of the art overall performance on public benchmarks and has an advantage in inference speed.

### Weaknesses
-The in-plane rotation estimation proposed in this paper appears to target only the camera itself. In principle, the scheme applies to any video restoration task and has no obvious relevance to deraining, that is, the authors did not design for the tilt angle of rain. This makes the paper somewhat tenuous as a dedicated video deraining approach.

-Camera shake is inherently 3 dimensional, raising doubts that a in-plane model can resolve the stated issue. The evidence presented in the paper is limited and does not convincingly support claims of mitigating cross frame misalignment and temporal artifacts. Since the method explicitly targets rotation, the evaluation should isolate rotation induced errors and demonstrate corresponding gains; modest and unstable aggregate improvements are not sufficient.

-The authors deny the robustness of optical‑flow approaches and claim that unreliable optical flow will harm such video restoration models. This view should be supported by more solid comparative experiments, given that optical‑flow‑based methods are indeed mainstream in this field. In addition, since the authors introduce physical guidance by embedding a bias for the rotation angle, they should compare with other approaches that use bias modulation. Most straightforwardly, I believe there should be a comparison under the same baseline and modulation scheme between using optical flow for modulation and using the predicted rotation angle, which would directly show that the proposed rotation estimation is more reliable than optical flow.

-The organization and presentation of this paper are not yet at a publishable level. For example, most of Section 3 consists of existing formulas used as background, with few formulas of substantive value that are newly derived by the authors, which makes it hard to grasp the key points. The figures are rather rough (in Fig. 3 many panels do not clearly show differences in details), and some figures do not match their captions (Fig. 4).

### Questions
-What is the specific connection between the proposed method and rain? Is the predicted rotation that of the camera or the tilt angle of the rain streaks? Overall, what I most want to know is whether the method can transfer to video restoration tasks under other degradations, what its scope of applicability is, and whether it can handle three‑dimensional camera shake.

-The results of this paper require more interpretability validation. I would like to see examples where the proposed method specifically addresses cross frame misalignment and temporal artifacts, rather than only showing improvements in overall metrics.

-Since the authors claim that optical‑flow‑based methods are not robust, they should conduct controlled experiments under the same baseline to demonstrate that the proposed rotation prediction provides more reliable guidance than optical flow. At present, the paper discusses this conclusion only in general terms and lacks strong experimental evidence.

-The paper requires more proofreading and fine‑grained refinement in details, and both the method illustrations and the figures need a more polished presentation.

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
3

### Summary
The paper introduces DeLiVR, a video deraining method that injects spatiotemporal Lie group-based biases into Transformer attention layers, enabling geometry-consistent feature alignment and motion-aware modeling. It achieves strong performance on synthetic and real-world benchmarks, with enhanced efficiency and robustness over prior methods.

### Strengths
Novel Use of Lie Groups in Attention: Introduces a principled and efficient method to incorporate SO(2) Lie algebra into Transformer attention, improving spatiotemporal consistency without optical flow.

Strong Generalization: Achieves SOTA or competitive results on synthetic datasets and real-world benchmarks like WeatherBench, with improved downstream performance.

Elegant Design and Efficiency: Uses lightweight geometric predictors instead of full equivariant models, maintaining inference speed and model compactness (2.64M params vs 12–58M in baselines).

Thorough Ablation and Theoretical Justification: Detailed experiments, regularization analysis, and Lie group derivations demonstrate rigorous understanding and contribution clarity.

### Weaknesses
SO(2)-only Limitation: The model restricts itself to in-plane rotation (SO(2)) and may underperform on more complex real-world motion (e.g., 3D translations, SE(3)), as briefly acknowledged.

Over-reliance on Geometry: The spatiotemporal bias fusion mechanism may not sufficiently adapt when motion is irregular or scene semantics dominate (e.g., occlusions, deformable objects).

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3
