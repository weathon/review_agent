# Sharp Monocular View Synthesis in Less Than a Second

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
We present SHARP, an approach to photorealistic view synthesis from a single image. Given a single photograph, SHARP regresses the parameters of a 3D Gaussian representation of the depicted scene. This is done in less than a second on a standard GPU via a single feedforward pass through a neural network. The 3D Gaussian representation produced by SHARP can then be rendered in real time, yielding high-resolution photorealistic images for nearby views. The representation is metric, with absolute scale, supporting metric camera movements. Experimental results demonstrate that SHARP delivers robust zero-shot generalization across datasets. It sets a new state of the art on multiple datasets, reducing LPIPS by 25–34% and DISTS by 21–43% versus the best prior model, while lowering the synthesis time by three orders of magnitude. Code and weights are provided at https://github.com/apple/ml-sharp.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces SHARP (Single-image High-Accuracy Real-time Parallax), a method for photorealistic view synthesis from a single image. SHARP directly regresses a 3D Gaussian representation through a single feedforward pass, achieving real-time high-resolution rendering with metric scale. It combines a Depth Pro encoder, a modified DPT depth decoder, a U-Net-based depth adjustment module, and a Gaussian decoder, trained in two stages—synthetic pretraining and self-supervised finetuning. Experiments on multiple datasets demonstrate strong zero-shot generalization and significant improvements over prior methods in both quality and efficiency. The main contributions include efficient single-image 3D Gaussian regression, depth ambiguity resolution, real-time rendering capability for AR/VR, and establishing a new benchmark for monocular view synthesis.

### Strengths
Originality: The primary novel contribution is the formulation of a single, feed-forward network that can directly regress the parameters of a complete, high-resolution 3D Gaussian representation from a single image. This moves beyond prior works that might generate simpler representations or require slow, per-scene optimization. Furthermore, the introduction of a learned depth adjustment module, inspired by Conditional Variational Autoencoders (C-VAEs), is a clever adaptation to handle the inherent ambiguity in monocular depth estimation, optimizing the depth map specifically for the end-goal of high-fidelity view synthesis rather than just metric accuracy. This demonstrates a nuanced understanding of the problem's core challenges.

Quality: The work is technically solid and experimentally comprehensive. Technically, it employs a well-structured architecture that includes a Depth Pro monodepth backbone (with selective unfreezing for task adaptation), a custom Gaussian decoder, and a detailed loss formulation combining color, perceptual (LPIPS), and alpha losses with regularizers for floaters, smoothness, and Gaussian variance to mitigate common 3D reconstruction artifacts. Experimentally, SHARP is evaluated on several datasets (Middlebury, ScanNet++, WildRGBD, etc.) and compared with recent state-of-the-art approaches, including diffusion-based methods. The ablation studies systematically examine key components—such as perceptual loss, depth adjustment, self-supervised fine-tuning, and backbone unfreezing—and the use of perceptual metrics (LPIPS, DISTS) instead of traditional ones (PSNR, SSIM) is appropriately justified.

Clarity: The paper is clearly written and well-structured. The introduction defines the goals—fast synthesis, real-time rendering, and metric scale—while figures effectively illustrate key achievements, such as reduced latency and high-fidelity outputs. The methodology is logically organized, with clear system diagrams and concise mathematical formulations.

Significance: SHARP addresses the challenge of synthesis latency, reducing generation time from minutes to under a second and making real-time single-image view synthesis more practical for interactive applications such as AR/VR. It also sets a strong baseline for feed-forward methods, demonstrating that a carefully designed regression-based approach can achieve high-fidelity nearby views efficiently. These results may guide future research on combining quality and speed in view synthesis.

### Weaknesses
-1. In the ablation study for SSFT (Table 11), the quantitative metrics for the model with and without SSFT are very close, with some metrics even slightly worse after fine-tuning.

-2. The model's Stage 1 training relies on a large-scale, in-house synthetic dataset. While the quality of this dataset is likely a key contributor to the model's success, it poses a challenge for reproducibility and fair comparison.

### Questions
-1. The paper notes that SHARP is designed for “nearby views.” However, since the term “nearby” can be subjective and application-dependent, could the authors clarify the model’s effective operational range? For instance, the authors could consider plotting key perceptual metrics (e.g., DISTS or LPIPS) against the camera’s baseline distance or angular deviation from the source view, or providing a similar form of analysis to illustrate the model’s effective operational range.

-2. Have the authors encountered cases where the Depth Pro still produces inaccurate estimates even after the depth adjustment module? If so, how does such inaccuracy affect the quality of the generated results, and are the reconstructions still satisfactory under these conditions?

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
The paper introduces SHARP, a feed-forward method for real-time view synthesis from a single image.
It predicts a 3D Gaussian representation via one forward pass (<1 s on an A100), enabling photorealistic nearby view rendering at over 100 FPS.
SHARP combines a Depth Pro backbone, a learned depth-adjustment module, and a Gaussian decoder trained end-to-end.
Experiments show strong improvements (25–40% LPIPS/DISTS gains) over prior feed-forward and diffusion-based methods. 
However the visualization only shows 1-to-1 view synthesis.

### Strengths
Novel combination of monocular depth inference and 3D Gaussian Splatting with impressive speed and fidelity.

Clear architecture and training pipeline; loss design and curriculum are well justified.

Extensive comparisons across datasets and perceptual metrics.

### Weaknesses
- Novel-view range unclear. 

The paper does not specify how far target views are from the input. Report actual displacement (e.g., angle, translation) and analyze performance versus view distance.

- View-to-view consistency. 

Since only one novel view for each scene is reported, temporal stability across continuous camera motion is unknown. Evaluating flickering with continuous multiple novel view renderings for frame-to-frame consistency is desired.

- Multi-view generalization. 

Can SHARP handle multi-view-to-multi-view synthesis, or is it strictly single-image input due to the monocular depth backbone? Clarify applicability to general NVS pipelines. As most baselines are designed for multiple to multiple NVS, it is a bit unfair setting to directly compare.

- 3DGS necessity. 

The model predicts ~1.2 M Gaussians even for small parallax. This appears over-parameterized; justify why a full 3DGS is required. A comparison with pure pixel output like LVSM  under the same training setup would clarify whether 3DGS truly improves quality. And whether 3DGS really ensures 3D consistency under this setting is unclear.

- Reproducibility. 

Synthetic data generation and rendering details are not publicly available.

### Questions
Please see the weakness.

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
4

### Summary
This paper introduces SHARP, a method for photorealistic view synthesis from a single image. It predicts a 3D Gaussian representation of the scene in under one second using a single feedforward neural pass. The resulting representation allows real-time, high-resolution rendering of nearby viewpoints with accurate metric scaling. SHARP achieves state-of-the-art image fidelity, outperforming diffusion-based and feedforward baselines.

### Strengths
1. The proposed method is fast and efficient, while achieving high-quality results. 

2. The experimental results demonstrate strong performance across multiple datasets and metrics. 

3. The writing is clear and the engineering contributions are solid.

### Weaknesses
1. The work is more like a system engineering paper rather than a novel research contribution. The scientific novelty is limited. The authors should better highlight the key innovations. 

2. It's better that the authors can provide video results to showcase the real-time rendering capabilities. 

3. The font used in the paper seems to be non-standard.

### Questions
Please refer to the weakness section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on the task of predicting Gaussian primitives from a given image for novel view synthesis from neighbor viewpoints. The key insight lies in a depth guided framework that predicts Gaussian attribute refinement to a initialized 3DGS. The results outperforms current video-based generative models.

### Strengths
1. The task is a promising way for VR/AR applications. This paper focuses on a cutting-edge field.
2. The results are convincing which perform many video based methods that require costly inference.

### Weaknesses
1. The authors should provide a video in the supplementary for a more clear comparison with SOTA methods. Since the method outputs a 3DGS, it is more convincing to attach a video showing novel view synthesis results of the 3DGS.
2. How large offset range can the model handle? For the regions that are not visible in the current image, does the model has the capability to generatively infer the occlusions and scene extensions? 
3. More recent works like See3D should be compared.

My most concern lies in a lack of direct video showing the quality of GS rendering.

### Questions
Please refer to the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3
