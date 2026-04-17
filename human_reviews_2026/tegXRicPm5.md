# $\textbf{SDPose}$: Exploiting Diffusion Priors for Out-of-Domain and Robust Pose Estimation

- Decision: Reject
- Scores: 2, 2, 8, 4

## Abstract
Pre-trained diffusion models provide rich multi-scale latent features and are emerging as powerful vision backbones. While recent works such as Marigold and Lotus adapt diffusion priors for dense prediction with strong cross-domain generalization, their potential for structured outputs (e.g., human pose estimation) remains underexplored. In this paper, we propose \textbf{SDPose}, a fine-tuning framework built upon Stable Diffusion to fully exploit pre-trained diffusion priors for human pose estimation. 
First, rather than modifying cross-attention modules or introducing learnable embeddings, we directly predict keypoint heatmaps in the SD U-Net’s image latent space to preserve the original generative priors. 
Second, we map these latent features into keypoint heatmaps through a lightweight convolutional pose head, which avoids disrupting the pre-trained backbone. 
Finally, to prevent overfitting and enhance out-of-distribution robustness, we incorporate an auxiliary RGB reconstruction branch that preserves domain-transferable generative semantics. To evaluate robustness under domain shift, we further construct \textbf{COCO-OOD}, a style-transferred variant of COCO with preserved annotations. With just one-fifth of the training schedule used by Sapiens on COCO, SDPose attains parity with Sapiens-1B/2B on the COCO validation set and establishes a new state of the art on the cross-domain benchmarks HumanArt and COCO-OOD. Extensive ablations highlight the importance of diffusion priors, RGB reconstruction, and multi-scale SD U-Net features for cross-domain generalization, and t-SNE analyses further explain SD’s domain-invariant latent structure. We also show that SDPose serves as an effective zero-shot pose annotator for controllable image and video generation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
SDPose is a fine-tuning framework on Stable Diffusion that repurposes pre-trained diffusion priors for human pose estimation.
While diffusion models offer rich multi-scale features and strong cross-domain robustness, their use for structured outputs like pose estimation is underexplored, and existing pose methods often degrade under domain shift and demand heavy fine-tuning.
SDPose asks whether SD U-Net latent features alone can produce reliable heatmaps.
Adopting the x₀-prediction design and the Lotus “Detail Preserver” strategy preserves fine detail and avoids overfitting, enabling efficient adaptation with minimal architectural changes.
With one-fifth of Sapiens’s training schedule, SDPose matches Sapiens-1B/2B on COCO, sets new SOTA on HumanArt and COCO-OOD.

### Strengths
1. By fine-tuning pre-trained diffusion models, the method achieves state-of-the-art performance in OOD pose estimation benchmark with significantly fewer training steps.

2. It also demonstrates that the intermediate features of pre-trained diffusion models are highly beneficial for generalized pose estimation.

### Weaknesses
1. Although prior works, as cited in the paper, have shown that the last and penultimate layers of the SD U-Net provide strong features, this study focuses on exploring features specifically for pose estimation, so conducting ablation only on these two layers seems insufficient.

2. The architecture appears to be largely borrowed from the Lotus paper, suggesting a lack of novelty.

3. For COCO-OOD generation, applying only Monet-style paintings seems limited. While Monet’s style is indeed out-of-domain, it would have been more convincing if other artistic styles were also used for style transfer.

4. The paper’s readability needs improvement.

### Questions
1. Much of the design seems to be inspired by the Lotus paper[1]. Could the key differences or novel contributions be clarified beyond the task difference?

2. Since it seems necessary to find an appropriate training epoch that balances pose estimation performance and generalization from the pretrained model, are there any experiments showing performance variation across training epochs?

[1] He, Jing, et al. "Lotus: Diffusion-based visual foundation model for high-quality dense prediction." arXiv preprint arXiv:2409.18124 (2024).

### Soundness
2

### Presentation
1

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
This paper introduces SDPose, a novel framework that adapts a pre-trained Stable Diffusion (SD) model for human pose estimation. The core idea is to leverage the rich, general-purpose visual features learned by the SD model's U-Net to achieve superior robustness, especially on out-of-domain (OOD) data like artistic images. Instead of modifying the cross-attention mechanisms or adding new embeddings, SDPose makes minimal changes: it adds a lightweight convolutional head to predict keypoint heatmaps directly from the U-Net's intermediate features and uses an auxiliary RGB reconstruction task to prevent overfitting and preserve the model's generative priors.

### Strengths
- It targets the OOD task and finds a simple method to use the pretrained model for the OOD task. It  shows the application of SDPose as a zero-shot pose annotator for ControlNet image generation and video generation provides tangible evidence of its superior qualitative performance over baselines like DWPose
- The paper is well organized and evaluations are comprehensive.

### Weaknesses
- My major concern is the novelty of this work. The novelty of this paper is quite marginal. It just uses the pretrained model for a pose estimation task. Nothing seems special or challenging. 
- This paper just did the experiments but not provide some insights. It would be better to provide some explanations about why the proposed method works better.

### Questions
Refer to strengths and weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents SDPose, a fine-tuning framework that leverages Stable Diffusion (SD) as a vision backbone for human pose estimation. Instead of modifying cross-attention layers or using learned embeddings, SDPose directly predicts keypoint heatmaps in the SD U-Net’s latent space, preserving the generative priors of diffusion models. The authors also construct COCO-OOD, a style-transferred variant of COCO for systematic robustness evaluation. SDPose matches or surpasses Sapiens-1B/2B on COCO with only one-fifth of the training cost and achieves new state-of-the-art performance on HumanArt and COCO-OOD. The model further demonstrates zero-shot pose annotation capabilities for pose-guided image and video generation, establishing SDPose as an efficient, generalizable, and versatile framework for structured prediction using diffusion priors

### Strengths
- Novel approach: Exploits Stable Diffusion’s latent features directly, preserving generative priors instead of introducing ad-hoc conditioning modules.
- Strong performance: Matches SoTA on COCO and achieves new records on HumanArt and COCO-OOD, with significant training efficiency (1/5 of Sapiens training time)
- Benchmark contribution: Introduces COCO-OOD, enabling standardized OOD robustness evaluation for pose models.

### Weaknesses
- Analysis of computational cost: The paper highlights training efficiency but provides limited discussion of inference-time latency and memory overhead compared to conventional backbones like ViTPose.
- Ablation breadth: Some architectural design choices (e.g., selection of diffusion timestep or use of SD-v1.5 vs v2) could benefit from more justification or quantitative analysis.

### Questions
- How would SDPose perform under multi-person scenarios or crowded scenes, where top-down cropping might limit context?
- Would fine-tuning from more recent foundation diffusion models (e.g., SDXL) further improve generalization, or is the performance already saturated?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes SDPose, a fine-tuning framework built upon Stable Diffusion (SD) to adapt pre-trained diffusion priors for 2D human pose estimation, particularly focusing on out-of-domain robustness. SDPose operates entirely in the latent space of the SD U-Net without modifying attention modules. The authors add a lightweight heatmap head to map multi-scale latent features to keypoint heatmaps and an auxiliary RGB reconstruction branch for generative regularization, preserving domain-transferable visual semantics. Additionally, the authors introduce COCO-OOD, a style-transferred version of COCO (via CycleGAN), to benchmark robustness under domain shift. Experiments show SDPose achieves SOTA performance on COCO, HumanArt, and COCO-OOD. It also demonstrates zero-shot usability as a pose annotator for controllable image/video generation.

### Strengths
1. The paper, unlike prior adaptations (e.g., fine-tuned cross-attention or learned condition embeddings), leverages pre-trained diffusion features with minimal architectural change, preserving SD’s representational power. The auxiliary RGB reconstruction task is a simple yet effective regularization strategy for maintaining generative semantics during fine-tuning.
2. The authors show SD Pose almost matches Sapiens-1B/2B on COCO while requiring 1/5 training epochs and a smaller backbone compared to the 2B backbone. The paper also shows SOTA performance on other in-the-wild datasets - HumanArt and COCO-OOD. Ablations validating the contribution of diffusion priors and auxiliary reconstruction are helpful.
3. Experimental tables are well-organized and reproducibility details (hardware, hyperparameters, dataset splits) are appreciable.

### Weaknesses
1. The core idea of repurposing SD U-Net features for downstream vision is shared with prior works like Marigold, Lotus, and GenLoc. The proposed approach’s novelty lies mainly in its architectural restraint (no attention tuning) and addition of a reconstruction branch.
2. The paper does not deeply analyze why diffusion priors confer robustness, e.g., what specific latent feature properties (multi-scale, texture invariance, semantic richness) contribute most.
3. OOD tests (HumanArt, COCO-OOD) are style-based and geometric or camera-domain shifts (e.g., occlusion, lighting, or viewpoint) are not considered.

### Questions
1. Would SDPose generalize to other OOD types such as blur, occlusion or synthetic-to-real transfer? Could the authors test or discuss applicability in non-artistic domain shifts?
2. Did the authors explore multi-scale fusion from different U-Net layers, rather than selecting a single feature map?

### Soundness
3

### Presentation
3

### Contribution
3
