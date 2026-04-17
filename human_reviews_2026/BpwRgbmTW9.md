# DRGSplat: Depth-Regularized 3D Gaussian Splatting

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
3D Gaussian Splatting has rapidly become a leading method for photorealistic novel view synthesis. However, its geometric accuracy often lags behind its visual fidelity. Existing methods to improve geometry typically constrain the 3D Gaussians directly, compromising their volumetric nature. We introduce DRGSplat, a novel depth-regularization approach for 3D Gaussian Splatting that enhances geometric accuracy without direct modifications of the Gaussian primitives. DRGSplat regularizes the rendered depth maps during training with three key losses: a monocular depth loss enforcing global consistency, a surface normal loss refining local detail, and a new uncertainty-aware curvature loss that selectively penalizes high-gradient regions while avoiding the gradient instabilities common to direct curvature constraints. Experiments on standard benchmarks show that DRGSplat keeps the strong photometric quality of Gaussian Splatting while substantially improving geometric accuracy and outperforming state-of-the-art geometry-focused methods. On the ETH3D dataset, DRGSplat improves reconstruction accuracy, completeness, and F1 scores of 3DGS by 15, 25, and 17 percentage points, respectively. The source code will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes DRGSplat, a depth-regularized 3D Gaussian Splatting framework that improves geometric fidelity without explicitly manipulating Gaussian primitives. The method enforces geometry through image-space regularization on rendered depth, normals, and a curvature term with uncertainty modeling. Extensive experiments on ETH3D, DTU, and Mip-NeRF360 demonstrate improved surface accuracy and high-quality novel view synthesis. The idea of maintaining the volumetric nature of Gaussians while introducing geometry priors is conceptually sound and technically well detailed.

### Strengths
1. Presents a principled depth-regularization strategy for 3DGS without altering Gaussian structure.

2. Combines monocular depth, surface normals, and uncertainty-aware curvature constraints elegantly.

3. Maintains or slightly improves photometric performance, suggesting no trade-off between geometry and appearance.

4. Implementation details and ablation studies are thorough, demonstrating the individual contribution of each loss term.

### Weaknesses
1. Lack of evaluation on Tanks and Temples (TNT), a standard and critical benchmark for 3D reconstruction, particularly in Gaussian-based scene modeling. TNT is widely used to validate scalability, scene complexity handling, and real-world 3D consistency; therefore, the absence of TNT evaluation weakens claims of general effectiveness.

2. Missing comparisons with key depth-supervised Gaussian baselines. The method relies heavily on monocular geometric priors, yet does not include direct comparisons against contemporary depth-guided 3DGS approaches such as

- DN-Splatter

- PGSR

- VCR-Gaussian

   These works also leverage depth or normals and represent strong baselines in geometry-enhanced Gaussian Splatting. Excluding them raises concerns about the completeness of the empirical evaluation.

3. Although the paper claims strong generalization, evaluations remain centered on indoor-centric datasets; outdoor and large-scene performance is less demonstrated compared to established benchmarks.

Overall, while the contributions are meaningful, the experimental coverage is not yet sufficient for full validation.

### Questions
1. How does DRGSplat perform on the Tanks and Temples benchmark? Can the authors provide results or plan to include them?

2. Compare with DN-Splatter, PGSR, and VCR-GauS. These are natural baselines for depth-supervised Gaussian splatting.

3. Does the method scale effectively to large-scale outdoor scenes where monocular priors may be noisy and COLMAP alignment sparse?

4. How robust is the approach to errors in monocular depth estimation, particularly in reflective or low-texture areas?

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
The paper introduces DRGSplat, a depth-regularized approach for 3D Gaussian Splatting that aims to improve the geometric accuracy of scene reconstruction while maintaining photorealistic rendering for novel view synthesis. DRGSplat leverages monocular depth, normal, and a curvature loss to regularize 3D Gaussian Splatting at the rendered depth level, which enables improving geometric consistency without directly altering the underlying Gaussian primitives as done in previous works. Comprehensive experiments on ETH3D, DTU, and Mip-NeRF360 benchmarks demonstrate notable improvements in geometric metrics compared to baseline and state-of-the-art methods, while preserving or improving photometric quality.

### Strengths
- The overall paper is well-written and easy to understand, where the motivation of implementation of each of the regularizations : depth, normal, and curvature regularization is explained in detail.
- The pipeline of converting the trained 3DGS to mesh is also explicitly mentioned, allowing easy re-implementation of the proposed method.
- The proposed method is simple and versatile, allowing the method to be applied to variations of the original 3DGS algorithm.
- The proposed method is simple yet effective, where it outperforms previous approaches (2D Gaussian Splatting, Sugar) tailored for better surface reconstruction.

### Weaknesses
### Major Weakness

- **Technical Novelty:** Although I appreciate the simplicity and effectiveness of the proposed method, I am largely concerned about the technical novelty of this method. As shown in the ablation of Table 5, the most important regularization of the proposed method is the monodepth loss. However, there has been numerous prior works which apply depth regularization during training, such as in [1,2,3] (see references listed below). As the proposed monodepth regularization follows the pipeline of computing scale and shift values also done in [1,2,3], it seems that the performance improvement mainly comes from the strong off-the-shelf geometry estimator (Metric3Dv2) rather than a technical contribution done in this work. Based on the lack of novelty of the monocular depth regularization loss alone, I think it is important to emphasize the importance of the two other proposed regularizations (normal loss, curvature loss), where I did not find enough emphasis in the current version of the manuscript.

### Minor Weakness

- **Reliance to off-the-shelf network:** As shown in Table 6, the proposed method largely depends on the performance of the off-the-shelf network’s estimation quality. Although the authors leverage the confidence value from the off-the-shelf module,  it would be better to analyze more on how robust the proposed method is when the off-the-shelf estimates are highly noisy.
- **Analysis on computation overhead:** As this model requires the usage of large foundation models for geometry estimation, it would be nice to analyze the required computation for this method.
- **Analysis on overall optimization time:** When comparing with previous methods such as 2D Gaussian Splatting and Sugar, it would be nice to compare the overall training time to reconstruct the scenes to better highlight the advantage or contribution of the proposed method.

### References

---

[1] Zhang, Qilin, et al. "CDGS: Confidence-Aware Depth Regularization for 3D Gaussian Splatting." arXiv preprint arXiv:2502.14684 (2025).

[2] Huang, Zexu, Min Xu, and Stuart Perry. "DET-GS: Depth-and Edge-Aware Regularization for High-Fidelity 3D Gaussian Splatting." arXiv preprint arXiv:2508.04099 (2025).

[3] Chung, Jaeyoung, Jeongtaek Oh, and Kyoung Mu Lee. "Depth-regularized optimization for 3d gaussian splatting in few-shot images." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

### Questions
Q1. Could the authors provide insights into how the proposed method differs from previous monocular depth regularization methods? 

Q2. Could the authors provide additional analysis of the newly proposed normal loss and curvature loss on the final performance? Other than the quantitative comparison in Table 5, it would be better to add some qualitative comparisons after adding each regularization. Also, could the authors provide quantitative tables of the benefits of the newly formulated normal loss, which is directly computed on top of the predicted depth map, compared to prior works that utilize normal regularizations by directly calculating the normal from the 3D Gaussians?

Q3. The original 3DGS training pipeline has multiple hyperparameters that decide how the Gaussians will be cloned or split during optimization, where the norm of the gradient is one of the critical hyperparameters. Does adding additional loss signals require modifying these hyperparameters?

Q4. How is the monocular depth variance defined for the curvature loss? I could not find a proper definition for this value.

Q5. Could the authors explain the overall computation requirement and training time?

### Soundness
3

### Presentation
3

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
The paper introduces DRGSplat, which focuses in enhancing the geometry in gaussian-based methods with depth regularization. The depth regularization consists of three losses, with a standard depth loss, normal loss, and the proposed curvature-based loss. Notably, the normal and the curvature is obtained from approximating the derivatives of the depth loss, which allows all signals to be propagated through the rendered depth. DRGSplat shows that it can significantly boost the geometry of 3DGS-based methods by introducing the three losses during training.

### Strengths
1. The core idea of the paper of introducing geometric prior through the rendered depth map is quite convincing, given its results and comparison to other methods with "explicit constraints".

2. The method for obtaining normals and curvatures is sensible, which bases on finite difference approximation on the rendered depth map. This well-differentiates the work from other depth regularization works, where the normal is often obtained from the gaussian primitives, rather than the rendered outputs.

3. The method is based on additional losses, and can be implemented without altering rendering pipelines for existing methods. The modularity of the method, given how it could improve 3DGS and GOF, suggests that the method can be also applied to future works with a stronger baseline representation.

### Weaknesses
1. The method seems to be heavily influenced by the pre-trained monocular depth model. Although the paper includes an ablation of this, it seems like there are more to be explored.
- Since the method utilizes both the predicted depth and its uncertainty from a pre-trained depth model, it would be interesting to see whether the method is benefitting from high-quality depth predictions, or better uncertainty prediction. A more detailed case study, where each of the losses are ablated for different depth models, would provide a better understanding.
- Furthermore, an additional ablation with compared methods, such as DN-Splatter, with identical depth backbone seems to be required to rule out that the model is simply benefitting from a stronger depth model.

2. The paper is missing some details and explanation.
- Although the rendered depth is the most important component, the equation for obtaining it is not defined in the manuscript. The reviewer is aware of how depth is normally obtained in 3DGS, but it still seems crucial to explicitly have this as there are some works that alter the depth rendering strategy for depth regularization.
- How is $C\_{predict}$ obtained? Does it go through the same process as $C(\\cdot)$?

3. The paper has some noticeable mistakes.
- The citation format does not seem correct. Please check the formatting guidelines. 
- (Nit) The images in the Fig. 3 is not correctly aligned.

### Questions
Please refer to the questions in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
