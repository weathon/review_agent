# Residual Pyramid Atrous Filtering Network with the Error Low-Rank Respresentation

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Image filtering aims to eliminate perturbations and textures while preserving dominant structures, serving a pivotal role in various image processing tasks. More recently, significant advances in filtering techniques have been developed. However, existing approaches typically suffer from oversmoothing edges, gradient reversal, and halos. Such issues originate from the difficulty in striking an optimal trade-off between filtering multi-scale textures and preserving edges. Furthermore, deep learning-based filtering frameworks lack modules designed to capture features of different long-range dependence textures. Consequently, the task of filtering textures while maintaining edge integrity remains a significant challenge. To address these issues, we propose a novel residual pyramid atrous filtering network (RPAFNet) that utilizes the error low-rank representation. Specifically, we introduce a lightweight dilated spatial convolution (LDSC) module for effectively extracting multi-scale texture features. To boost the reconstruction feature space, we propose a difference residual layer (DRL) module for connecting the encoder and decoder. Additionally, by employing low-rank approximation, we introduce a new non-convex optimization model, termed gradient error low-rank representation model (GELR), which effectively suppresses textures and preserves edges. This paper provides complete theoretical derivations for solving GELR and its convergence. Extensive experiments demonstrate that the proposed approach outperforms previous techniques in attaining an equilibrium between texture filtering and edge retention, as validated by both visual comparison and quantitative evaluation across various smoothing and downstream applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes RPAFNet with two structure modifications (LDSC and DRL) and one theoretical model (GELR) to address the challenge in image filtering, that is, 1) balancing multi-scale texture suppression and 2) structural edge preservation. Based on the LDSC and DRL, the RPAFNet is capable of establishing multi-range correlation and preserving edge information. Additionally, the GELR uses $||\nabla u||_1$  regularization to avoid edge blurring and suppress over-smoothing and $\beta||\nabla u-\nabla x|| $ to precisely reduce textures through low-rank approximation. Overall, the RPAFNet achieves remarkable results in several experiments.
​

### Strengths
- The motivation is clear, and the overall story is sound. The RPAFNet is based on both theoretical (GELR) and engineering optimization (LDSC, DRL).
- The theoretical derivation is solid.
- The experimental results are stable to show the effectiveness of the proposed model.

### Weaknesses
- Despite the proposed method being able to extend to multiple downstream tasks,  the comparison is conducted on generic methods like Deepwls and WTL1. Since it is an NN-based model without performance advantages, it is important to show the performance gap with the task-specific model.
- The inference performance and comparison are not included, and the training overhead for introducing GELR.
- The theoretical analysis is clear; however, it might be more persuasive to provide a visual comparison for the proposed GELR.
- The LSDC and DRL are not novel modules but a modification of existing methods. The design of dilation convolution (large kernel convolution) is widely used and explored.

### Questions
See weakness.

### Soundness
3

### Presentation
2

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
This paper presents a network called RPAFNet (Residual Pyramid Atrous Filtering Network) for image smoothing, aiming to suppress textures while preserving fine structural details. The model adopts a typical U-shaped encoder–decoder architecture, enhanced with two modules: LDSC (Large-Dilation Spatial Convolution) to expand the receptive field, and DRL (Difference Residual Layer) to strengthen feature reconstruction. In addition, the authors propose a Gradient Error Low-Rank Representation (GELR) that combines Total Variation (TV) with a low-rank constraint. Using an ADMM-based optimization strategy, the GELR term is incorporated as an additional loss to further improve structural preservation during smoothing.

### Strengths
1.Well-organized and technically sound

The paper is clearly structured, with a well-defined model design and mathematically consistent derivations. The experiments are reasonably comprehensive, showing good implementation quality and logical consistency.

2.Potential for further improvement and extension

Although the novelty is limited, the work provides a clear and systematic framework for traditional image smoothing models. It also leaves room for future extensions, such as incorporating differentiable optimization or generative modeling methods (e.g., DPO or diffusion/flow-based models).

### Weaknesses
1.Limited architectural and module-level innovation

RPAFNet adopts a conventional U-shaped encoder–decoder framework without introducing substantial structural novelty. The LDSC module relies on atrous convolution to expand the receptive field, a technique that was thoroughly explored in the DeepLab series (2017). In contrast, modern approaches typically employ Transformer or hybrid attention mechanisms to handle long-range dependencies, making the proposed design appear outdated. Meanwhile, the DRL module performs structure compensation merely through feature differencing, showing strong similarity to standard residual or skip connections. It lacks clear theoretical justification and empirical validation of its independent contribution. Overall, the design appears to be an integration of existing techniques rather than a breakthrough in architectural paradigm.

2.Conservative loss design and potential optimization conflicts.

The loss formulation mainly follows the traditional “TV + low-rank” joint regularization framework, which has been extensively used in earlier image restoration and smoothing tasks. The paper introduces no novel constraint or optimization mechanism. Furthermore, the simultaneous use of structural loss L1, perceptual loss L2, and an SSIM term in both training and evaluation may cause gradient conflicts and metric bias. Since SSIM is explicitly optimized during training, its improvement in evaluation may partially result from direct loss fitting rather than genuine structural enhancement.

3.Lack of explanation for experimental settings

The paper employs bilinear interpolation with a down sampling ratio of 0.8, which is an unconventional choice. However, no clear motivation, empirical justification, or performance sensitivity analysis is provided, limiting the interpretability and reproducibility of the results.

### Questions
1.On the architectural design and innovation

The paper repeatedly highlights the “limited receptive field” as a key bottleneck, but this issue has already been addressed in recent architectures such as Transformer or hybrid attention networks. Could the authors clarify how the LDSC module provides advantages over standard atrous convolution, multi-scale structures, or Transformer-based architectures in terms of receptive field and feature representation? It would be helpful to include quantitative or qualitative comparisons to demonstrate its unique contribution.
Furthermore, the DRL module performs structure compensation mainly through feature differencing, which appears conceptually similar to residual or skip connections. Could the authors elaborate on the motivation and theoretical foundation of DRL, and provide ablation results to justify its necessity and effectiveness compared with simpler alternatives such as residual fusion or attention-based refinement?

2.On loss design and optimization objectives

The current “TV + Low-rank” joint regularization follows a traditional formulation without introducing new constraints or optimization mechanisms. Moreover, combining (L_1) and (L_2) losses may introduce conflicting gradient directions and affect structure preservation, have the authors observed such instability?
Additionally, since the SSIM term is used in both training and evaluation, could this cause metric bias and artificially inflate performance? It would strengthen the validity of the results if the authors could report outcomes of a version trained without SSIM loss, or analyze its impact empirically.


3.On experimental settings

The paper uses bilinear interpolation with a down sampling ratio of 0.8, which is quite unusual. The authors should clarify during the rebuttal why this value was chosen — for example, to ensure gradient smoothness, multi-scale feature consistency, or based on empirical tuning — and analyze its sensitivity to this parameter.

4.On methodological extension and potential research value

While the paper focuses on image smoothing, it conceptually overlaps with texture suppression and structure enhancement problems often addressed by generative models. Have the authors considered applying the proposed framework in generative or diffusion-based architectures (e.g., Stable Diffusion, Flow Matching) to evaluate its scalability and potential cross-domain benefit? Such discussion would help clarify the method’s broader applicability and originality.

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
The paper proposes an image filtering network called RPAFNet, which aims to remove multi-scale textures while preserving edges. The network consists of two novel core modules:  Lightweight Dilated Spatial Convolution (LDSC) module and Difference Residual Layer (DRL). Lightweight dilated convolution is used to extract multi-scale texture features, and the difference between encoded features is used as a skip connection, emphasizing high-frequency information. An additional Gradient Error Low Rank (GELR) model is introduced to calculate a non convex optimization term based on low rank approximation at the loss end to further suppress texture and preserve edges.

### Strengths
1. The optimization is novel. Monotonically decreasing iterative closed form solutions for the GELR objective function is provided and convergence to a limit point is proved, efficiently suppressing texture and overcoming oversmoothing issues.
    
2. Integrating "dilated pyramid feature extraction+differential residual skip connection+low rank texture suppression" into an end-to-end framework, the three complement each other and have a clear idea.
    
3. The method is demonstrated with rich experiments. The quantitative indicators are comprehensively leading and the qualitative results are impressive.

### Weaknesses
1. LDSC uses multiple dilation combinations, whick might induce gridding artifacts. It’s suggested that using FFT graph to verify whether this risk exists in RPAFNet.
    
2. Theoretical assumption is too strong. GELR convergence proof relies on the assumption of "Lipschitz continuity and bounded gradient", and whether the actual network feature map distribution satisfies this assumption has not been verified. Suggest providing statistics on the gradient/Lipschitz constant during the training process.
    
3. Sensitivity curves were not performed for the combination of dilation rate and low rank rank value r.

### Questions
1. What’s $T^k$ in Equation 13?
    
2. Has the combination of different dilation rates been systematically searched? Is there a better "receptive field scheduling" strategy (such as dynamic dilation)?
    
3. Is the design friendly to high-resolution images (such as 4K)? Does the computational complexity increase with the square of resolution?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
1. Proposed RPAFNet, a residual pyramid atrous filtering network for image smoothing.
2. Introduced LDSC module to extract multi-scale texture features using dilated convolutions.
3. Designed DRL module to enhance feature space via difference residual connections.
4. Developed CTUM module to fuse local and global features for better reconstruction.
5. Formulated a novel GELR model using gradient error low-rank representation for texture suppression.
6. Provided complete theoretical derivation and convergence proof for the optimization algorithm.
7. Demonstrated superior performance over state-of-the-art methods across multiple datasets and applications.

### Strengths
1. The paper demonstrates high originality by formulating a novel non-convex optimization model (GELR) that creatively combines a classical total variation term with a low-rank constraint on the gradient error.
2. It presents a significant architectural innovation with its RPAFNet, which integrates purpose-built modules like LDSC and DRL specifically designed to handle multi-scale textures and enrich the feature space.
3. The work is of exceptional quality due to its theoretical rigor, providing complete derivations for the non-convex optimization and a comprehensive convergence analysis for the proposed ADMM algorithm.
4. The experimental quality is thorough and convincing, featuring extensive comparisons against 14 state-of-the-art methods across multiple datasets and meaningful downstream applications.

### Weaknesses
1. The GELR model's reliance on ground-truth gradients (∇x) during training severely limits its real-world applicability, restricting it to synthetic datasets and preventing use on natural images where ideal smoothed targets are unavailable.
2. Computational efficiency is completely unanalyzed, with no reporting of inference speed, model size, or comparison to efficient alternatives, making practical utility impossible to assess.
3. The central claim of superior multi-scale texture handling lacks quantitative validation, relying solely on visual examples rather than objective metrics measuring texture-scale uniformity or structural preservation.
4. Critical ablation studies are incomplete, with the CTUM module's impact shown only numerically without visual proof, and the hyperparameter selection process lacking principled justification for balancing texture removal versus edge preservation.
5. Specific artifact analysis is insufficient, failing to provide direct visual evidence of improvement on stated problems like gradient reversal and offering only superficial treatment of the acknowledged low-contrast texture limitation.

### Questions
1. The GELR model's reliance on ground-truth gradients (∇x) during training severely limits its real-world applicability, restricting it to synthetic datasets and preventing use on natural images where ideal smoothed targets are unavailable.
2. Computational efficiency is completely unanalyzed, with no reporting of inference speed, model size, or comparison to efficient alternatives, making practical utility impossible to assess.
3. The central claim of superior multi-scale texture handling lacks quantitative validation, relying solely on visual examples rather than objective metrics measuring texture-scale uniformity or structural preservation.
4. Critical ablation studies are incomplete, with the CTUM module's impact shown only numerically without visual proof, and the hyperparameter selection process lacking principled justification for balancing texture removal versus edge preservation.
5. Specific artifact analysis is insufficient, failing to provide direct visual evidence of improvement on stated problems like gradient reversal and offering only superficial treatment of the acknowledged low-contrast texture limitation.

### Soundness
3

### Presentation
3

### Contribution
3
