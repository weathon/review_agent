# UniISP: A Unified ISP Framework for Both Human and Machine Vision

- Decision: Reject
- Scores: 6, 4, 6

## Abstract
Compared to RGB images, raw sensor data provides a richer representation of information, which is crucial for accurate recognition, particularly under challenging conditions such as low-light environments. The traditional Image Signal Processing (ISP) pipeline generates visually pleasing RGB images for human perception through a series of steps, but some of these operations may adversely impact the information integrity by introducing compression and loss. Furthermore, in computer vision tasks that directly utilize raw camera data, most existing methods integrate minimal ISP processing with downstream networks, yet the resulting images are often difficult to visualize or do not align with human aesthetic preferences. This paper proposes UniISP, a novel ISP framework designed to simultaneously meet the requirements of both human visual perception and computer vision applications. By incorporating a carefully designed Hybrid Attention Module (HAM) and employing supervised learning, the proposed method ensures that the generated images are visually appealing. Additionally, a Feature Adapter module is introduced to effectively propagate informative features from the ISP stage to subsequent downstream networks. Extensive experiments demonstrate that our approach achieves state-of-the-art performance across various scenarios and multiple datasets, proving its generalizability and effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes UniISP, a unified image signal processing framework that serves both human perception (high-quality RAW→RGB rendering) and machine vision (detection/segmentation) in a single, jointly trained system. 

The core design consists of two parts: a lightweight Hybrid Attention Module (HAM) integrated within a U-Net–style ISP to efficiently capture global context for perceptual reconstruction, and a Feature Adapter that fuses multi-scale frequency information from the ISP stage into downstream RGB-pretrained backbones. Training uses an alignment pipeline (global color mapping plus optical-flow consistency mask) and an adaptive loss that balances human-vision and task losses via EMA-weighted λ. 

Claimed contributions are the unified ISP formulation, HAM, the feature-adapter bridge, and state-of-the-art results over multiple tasks/datasets.

### Strengths
The paper’s strengths are multifaceted. In terms of originality, it tackles a practical yet underserved goal. One ISP that simultaneously satisfies human perception and machine vision by coupling a channel-wise Hybrid Attention Module (HAM) with a frequency-aware Feature Adapter that injects RAW-stage features into RGB-pretrained backbones. This architecture moves beyond task-specific ISPs toward a unified formulation. 

On quality, the evaluation is broad and methodical: RAW→RGB on ZRR with PSNR/SSIM/LPIPS, detection on PASCAL RAW and LOD, and segmentation on ADE20K RAW, with consistent gains and competitive efficiency (e.g., UniISP surpasses prior ISPs in Table 2, improves mAP across lighting in Table 3, and raises mIoU across backbones/illumination in Table 4). 

The training mechanics, including GCM plus optical-flow alignment for supervision and an adaptive EMA-weighted balance between human and machine losses, are concretely specified with equations and implementation details, aiding reproducibility. 

Clarity is generally strong: the system diagram and HAM/adapter schematics make the pipeline legible, and the supplementary ablations further dissect module contributions, particularly for HAM and the flow-consistency mask. 

Significance lies in demonstrating that a lightweight ISP can enhance both perceptual fidelity and downstream recognition across normal, dark, and over-exposed regimes, indicating real deployment potential where visualization and autonomy coexist (e.g., mobile or automotive).

### Weaknesses
While the paper is well engineered, several weaknesses limit its contribution and practical impact. 
First, the novelty is incremental relative to recent RAW-aware pipelines that already co-optimize ISP and perception (e.g., Dirty-Pixel, RAW-Adapter, AdaptiveISP, DynamicISP). Your gains hinge on a specific HAM and a frequency-based feature adapter, rather than a fundamentally new training principle. The paper would benefit from a head-to-head reimplementation of these closest methods under identical budgets and data, to isolate the architectural value beyond joint optimization itself. 

Second, the core training assumption, availability of paired RAW-RGB supervision, is fragile and acknowledged as a limitation; large-scale, diverse real RAW-RGB pairs are scarce and synthesized RAW diverges in distribution, which questions scalability and robustness. Specifically, add semi-/self-supervised variants that eliminate the need for pairs and report cross-sensor generalization (trained on ZRR, tested on unseen sensors or CFAs). 

Third, the alignment route (GCM plus optical-flow warping with a forward–backward mask) is brittle in occlusions and texture-poor regions; it exhibits quantitative sensitivity to mask thresholds, flow errors, and ablates the alignment stage compared to learned alignment, verifying that improvements are not artifacts of supervision drift. 

Fourth, the adaptive loss employs a heuristic EMA-weighted λ; it provides ablations versus fixed λ, curriculum schedules, and alternative multi-objective methods (e.g., GradNorm, PCGrad), along with stability plots to justify the choice. 

Fifth, evaluation breadth is narrow in terms of distribution shift, as it does not include cross-dataset tests (e.g., training on PASCAL RAW normal and evaluating on LOD dark) or robustness to exposure extremes, HDR scenes, motion blur, and burst/video sequences, where alignment and frequency fusion could behave differently.

### Questions
1. Beyond joint optimization, what specific capability does HAM + frequency-based Feature Adapter provide that RAW-Adapter, AdaptiveISP, DynamicISP, or Dirty-Pixel cannot?
   
2. How does the method perform without paired supervision or with noisy/sparse pairs?
   
4. How sensitive are results to mask thresholds, flow errors, and scene content (repetitive textures, occlusions, low texture)?
   
5. Why does EMA-based λ outperform fixed λ, curricula, or gradient-balancing approaches?
   
6. What features are being amplified/suppressed by HAM and the frequency fusion at different stages?

### Soundness
2

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
5

### Summary
This paper introduces UniISP, a unified image signal processing framework that balances human visual quality and machine vision performance. It employs a Hybrid Attention Module (HAM) for visually pleasing image generation and a Feature Adapter to transfer informative features to downstream tasks. Experiments show that UniISP achieves good results across multiple datasets and scenarios, demonstrating its' generalization and effectiveness.

### Strengths
The paper explores both human-vision and machine-vision tasks based on camera RAW data. The authors conducted extensive experiments and provided a detailed ablation study in the supplementary material. The proposed method achieves state-of-the-art performance on most datasets.

### Weaknesses
1. Regarding the image restoration and high-level detection/segmentation parts, the authors still need to train them separately. Therefore, the proposed approach is not a truly unified ISP model, and the title somewhat overstates its scope, which also reduces the perceived level of innovation to some extent.

2. Regarding the image restoration part, the authors only conducted experiments on standard RAW-to-sRGB datasets. This does not fully demonstrate the true generalization capability of the model. The authors are encouraged to further evaluate their method on RAW-to-sRGB datasets under extreme lighting conditions, such as the See-in-the-Dark (SID) dataset.

3. For the semantic segmentation dataset, the improvement on ADE20K-raw does not seem to be very large compared with RAW-Adapter, additionally, the proposed method contains more network parameters.

3. It’s better to capitalize the all fig.x (e.g., fig.2 in line 161) to Fig.x

### Questions
For Table.2, the LiteISP, AWNet... are almost old methods (3~4 years ago), could you compare with some recently SOTA ISP methods ?

As weakness.2, what is the performance if the proposed method evaluate on the low-light RAW dataset like See-in-the-Dark (SID) dataset.

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
This paper proposes a novel deep learning framework for Image Signal Processing (ISP), UniISP, which simultaneously optimizes the requirements of both human perception and machine vision: generating visually friendly RGB images for human observation while retaining data information helpful for downstream visual tasks (such as detection and segmentation).

UniISP is based on the U-Net architecture, introducing a Hybrid Attention Module (HAM) for feature extraction and designing a Feature Adapter to enhance feature transfer capabilities in the ISP stage.

Extensive experiments show that UniISP outperforms existing methods on multiple tasks and datasets, including raw-to-RGB mapping, object detection, and semantic segmentation, demonstrating strong generalization and effectiveness.

### Strengths
Unified Architecture: For the first time, UniISP systematically solves the problem of simultaneously addressing the needs of both human vision and machine vision within a single ISP, offering a novel approach with strong practical significance.

Ingenious Design: The introduction of HAM combines convolution and self-attention, balancing local and global feature modeling while preserving high-resolution processing capabilities.

Feature Adapter: Efficiently embeds multi-scale features from the ISP stage into the downstream network, effectively improving task performance, especially significantly outperforming competitors under extreme conditions such as low light.

Extensive Empirical Validation: Detailed experiments were conducted on multiple tasks and datasets, comparing with various mainstream methods, demonstrating clear superiority and thorough experimentation.

Lightweight and Efficient: Fewer parameters and less computation than most competitors, facilitating practical deployment.

### Weaknesses
While the calculation of the end-to-end adaptive parameter λ is novel, its actual tuning and convergence stability have not been thoroughly explored, and its generalization ability under different tasks/data requires further detailed analysis.

Main experiments focused on public datasets and synthetic/low-light scenes. Further validation of its generalization ability in real-world complex scenes (such as differences in ISPs between different mobile phone camera brands) is expected.

Limited improvement in some quantitative metrics under extreme scenes such as low light/overexposure; further supplementation with subjective evaluation (human eye-friendliness) is needed.

Model training relies on various pre-trained networks (such as VGG/ResNet), which may limit the deployment environment during inference.

### Questions
an the core modules of UniISP (such as HAM and Feature Adapter) be used independently for other types of tasks, such as special perception scenarios like medical imaging?

How can we ensure that the adaptive adjustment of parameter λ does not overfit a particular task when encountering extreme data such as completely dark/completely bright scenes? Can manual constraints or dynamic allocation at different stages be introduced?

Have the performance, power consumption, etc., been tested during inference on real mobile devices? Have overquantization or compression optimizations (such as INT8/FP16 schemes) been considered?

Can this method be extended to video ISP (denoising/color correction under temporal consistency constraints, etc.)? Future consideration could include combining it with temporal models or Transformer enhancements.

### Soundness
2

### Presentation
2

### Contribution
2
