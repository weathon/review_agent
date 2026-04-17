# Improving Sparse-View 3DGS Generalization via Flat Minima Optimization

- Decision: Reject
- Scores: 2, 2, 4, 8

## Abstract
Recent advances in neural rendering have established 3D Gaussian Splatting (3DGS) as a highly efficient representation for novel view synthesis, enabling real-time training and rendering with strong fidelity. However, when supervision is limited to a sparse set of input views, 3DGS tends to overfit to the observed images, resulting in poor generalization to unseen viewpoints. We approach this challenge from the perspective of flat minima (FM) optimization, which seeks solutions that remain stable under small parameter perturbations and are thus more robust. Viewing Gaussian parameters as trainable weights, we adapt FM principles to the geometric and dynamic nature of 3DGS by introducing several key techniques. First, we propose a Scale-Adaptive Perturbation (SAP) scheme that scales perturbation magnitude according to each Gaussian’s anisotropy, preserving fine details while promoting robustness. Second, we adopt stochastic perturbation where each Gaussian is probabilistically perturbed or left unchanged, allowing perturbations while preventing oversmoothing of scene details. Third, we schedule the perturbation magnitude to increase gradually during training, avoiding excessive noise before Gaussians capture stable structure. Finally, we incorporate periodic reinitialization of non-positional parameters such as scale, rotation, and opacity, and Spherical Harmonics (SH) coefficients. preventing degenerate cases like elongated Gaussians and maintaining well-conditioned primitives throughout optimization. Together, these techniques form a lightweight framework that integrates seamlessly into existing 3DGS pipelines without architectural changes. Experiments on LLFF and Mip-NeRF360 demonstrate that our method consistently improves both quantitative metrics and perceptual quality under sparse-view supervision, producing reconstructions that are sharper, more stable, and better generalized to novel viewpoints.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a flat-minima optimization framework for sparse-view 3DGS that injects scale-adaptive perturbations into Gaussian positions during training to reduce overfitting. It further improves novel view rendering qualities by mixing perturbed and unperturbed objectives, perturbation magnitude scheduling, and periodic re-initialization. It significantly improved the rendering quality of vanilla 3DGS under sparse-view settings.

### Strengths
- The integration of Flat Minima Optimization into vanilla 3DGS under sparse view settings to avoid overfitting is well motivated.
- The method is clearly explained and easy to reproduce.
- The ablation studies clearly demonstrate the effectiveness of each of the proposed components.

### Weaknesses
- Limited quantitative gain: the main issue of this paper is that it's not convincing that the proposed method really outperforms the baseline method DropGaussian. In most experiments, it only achieves very marginal quantitative improvements. In some occations (e.g., 12-view NVS), the PSNR of DropGaussian is even higher.
- Missing baselines: two strong baselines for sparse-view NVS are missing: MAtCha Gaussians (CVPR 2025) and Difix3D+ (CVPR 2025). They leverage monocular depth priors and diffusion priors to achieve high-quality NVS results, respectively. Comparing to these additional two baselines would provide a more comprehensive assessment of the effectiveness of the proposed method.

### Questions
I recommend the authors to add qualitative/quantitative comparisons to the missing baselines MAtCha Gaussians and Difix3D+. I also recommend the authors to provide a side-by-side video comparison with all baselines to help better assess the rendering qualities.

Apart from the mentioned weaknesses, I have an additional question:
- Can the proposed method (at least the probablistic Scale-Adaptive Perturbation part) be implemented as a general plug-in applicable to a wide range of sparse-view Gaussian Splatting pipelines? For instance, if incorporated into baselines such as DropGaussian, would it achieve a comparable relative quantitative improvement over those methods as it does when applied to vanilla 3DGS? Demonstrating consistent relative gains across diverse pipelines would make the contribution significantly more convincing, highlighting the method’s plug-and-play versatility and broad applicability.

I would consider raising my score if the author could add the missing baselines and show a consistent improvement by plugging in the proposed approach into more baseline methods.

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
4

### Summary
This paper addresses the limited generalization of 3D Gaussian Splatting (3DGS) under sparse-view supervision.
 The authors reinterpret 3DGS optimization as a form of weight learning and propose a set of flat-minima-inspired perturbation strategies to promote robustness.
 Specifically, the method perturbs Gaussian parameters with magnitudes scaled by their anisotropy, applies stochastic perturbations to a random subset of Gaussians, schedules the perturbation strength to increase during training, and periodically reinitializes non-positional parameters to avoid degeneracy.
 Experiments on LLFF and Mip-NeRF360 show consistent improvements over existing baselines such as DropGaussian and CoR-GS.

### Strengths
1. Framing sparse-view overfitting as a sharp-minima problem provides a fresh and intuitive perspective.
2. The approach is simple to implement, adds minimal computational cost, and integrates easily into existing 3DGS pipelines.
3. Ablation studies and comparisons are clearly presented, isolating the contributions of each component.

### Weaknesses
1. Despite the flat-minima motivation, the method essentially performs Gaussian noise injection—a well-known regularization technique. Similar stochastic or Bayesian formulations (e.g., 3DGS-MCMC) are not cited or compared.
2. No analysis (e.g., curvature or sharpness metrics) is provided to demonstrate that the optimization indeed converges to flatter minima.
3. Reported improvements are small and may fall within expected training variance, raising questions about the actual impact of the proposed components.

### Questions
1. Can you provide evidence that the proposed method achieves flatter minima, such as curvature or Hessian-based analysis?
2. How sensitive is performance to perturbation probability, noise scale, and reinitialization frequency?

### Soundness
2

### Presentation
2

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
This paper tackles sparse-view 3D Gaussian Splatting (3DGS) overfitting by importing flat-minima (FM) optimization ideas and tailoring them to the geometry of Gaussians. The core method (1) perturbs Gaussian positions with a Scale-Adaptive Perturbation (SAP) whose noise matches each Gaussian’s anisotropy and size, (2) applies those perturbations stochastically (some Gaussians may have perturbations and some may not) per-Gaussian to avoid oversmoothing, (3) linearly schedules perturbation strength over training, and (4) periodically reinitializes non-positional parameters (scale, rotation, opacity, SH) to prevent degenerate elongated Gaussians. Notably, the method is lightweight and does not require any architectural changes to 3DGS. The method improves PSNR/SSIM and lowers LPIPS versus baselines. Ablations confirm position noise with anisotropic scaling, stochastic application, scheduling, and reinitialization each contribute to the gains. Qualitatively, results show sharper details and better geometric consistency in under-constrained regions.

### Strengths
1. Well-motivated approach: The connection between flat minima optimization and sparse-view generalization in 3DGS is intuitive and well-articulated. Viewing overfitting as convergence to sharp minima is a reasonable perspective.
2. Comprehensive ablations: The paper includes thorough ablation studies examining each component (perturbation design, parameter choices, stochastic strategy, scheduling, reinitialization), providing evidence for design decisions.
3. Practical and lightweight: The method integrates seamlessly into existing 3DGS pipelines without architectural changes or significant computational overhead.
4. Consistent improvements: Results show steady gains across multiple datasets, view settings, and metrics (PSNR, SSIM, LPIPS), demonstrating robustness.

### Weaknesses
1. The core ideas (perturbation-based optimization, parameter reinitialization) are well-established techniques. While the adaptation to 3DGS is reasonable, the conceptual contribution feels incremental.
2. While consistent, the quantitative gains are relatively small. Some qualitative differences (eg. in Figure 2) are subtle.
3. The paper doesn't provide rigorous analysis of why position perturbations specifically lead to flatter minima or how the proposed techniques relate to formal FM theory. No measurement of actual loss landscape sharpness before/after.
4. The method introduces several hyperparameters. Limited discussion of sensitivity to these choices or guidance on setting them for new scenarios.

### Questions
1. How does computational cost compare to baselines? While you mention it's lightweight, actual training time and memory comparisons would be helpful.
2. Why perturb only positions and not jointly optimize with other parameters? Table 4 shows position is best, but have you tried learned perturbation schedules for different parameter types?
3. Can this approach be combined with other sparse-view methods (depth priors, semantic features) for further improvements?
4. What happens in extremely sparse settings (e.g., 2 views)? Are there failure cases where your method doesn't help?


Additionally, Figure 1 doesnt seem to be rendering on PDF viewers.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a novel optimization framework for improving the generalization of 3D Gaussian Splatting (3DGS) in sparse-view novel view synthesis. Traditional 3DGS methods tend to overfit when trained on limited input images, leading to poor reconstruction quality in unseen viewpoints. The authors propose incorporating Flat Minima (FM) optimization—a concept from neural network training—into 3DGS to mitigate this issue. Specifically, they introduce a Scale-Adaptive Perturbation (SAP) scheme that applies probabilistic, geometry-aware perturbations to Gaussian positions, along with perturbation scheduling and periodic parameter reinitialization to improve robustness and prevent overfitting. Experiments on LLFF and Mip-NeRF360 datasets demonstrate that this method achieves consistent improvements in PSNR, SSIM, and LPIPS compared to existing 3DGS baselines (DropGaussian, CoR-GS, and DNGaussian). Ablation studies validate the importance of each proposed module, showing that the FM-inspired perturbations improve both fidelity and generalization, particularly under sparse-view settings.

### Strengths
1. While this paper builds on existing FM optimization techniques for improving model generalization, extending parameter perturbation from traditional neural network weight training to gaussian splatting field fitting is entirely novel. The Scale-Adaptive Perturbation method, in particular, represents a unique application of perturbation.

2. The paper clearly outlines each design decision in its perturbation scheme, with every section providing information relevant to understanding the methodology and results.

3. The paper offers strong evidence for its conclusions through quantitative tables and rigorous ablative studies that motivate each design decision.

4. The significance to the field of sparse view novel synthesis is clear: the paper delivers a gaussian splatting field fitting pipeline that improves both quantitative and qualitative results while opening a new direction for future methods—the incorporation of FM optimization techniques and ideas.

### Weaknesses
1. Regarding presentation, Equation 4 does not clearly show how sampling covariance depends on the scale and rotation of a given Gaussian kernel, though this relationship is illustrated in the Scale Adaptive Perturbation section of Figure 1.
2. Additionally, while the tables covering the ablation studies are informative, including visual ablative results similar to those in Figure 2 would strengthen the argument.
3. In terms of soundness, one aspect requiring further investigation is the scope of experiments on perturbing different parameters. Although evidence suggests that perturbing position alone yields the best overall performance, the experiments and accompanying discussion are insufficient to prove this conclusively. Notably absent are perturbation experiments on elements such as spherical harmonic coefficients. The paper would benefit from expanding these experiments or acknowledging that further exploration may be necessary for future work.

### Questions
My questions are primarily related to the concerns I had about your “Effect of Perturbing Different Parameters” section. I feel the section/ablative study indicates that applying perturbation to other parameters of gaussian splatting fields will not improve performance. It is not quite clear if this is a claim you intend to make or if you encourage further exploration in this area.

### Soundness
3

### Presentation
3

### Contribution
3
