# GR-Gaussian: Graph-Based Radiative Gaussian Splatting for Sparse-View Tomographic Reconstruction

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
Computed tomography (CT) reconstruction under sparse-view settings remains highly challenging due to severe artifacts. Recently, 3D Gaussian Splatting (3DGS) has shown promise for this task, but existing methods often rely on view-averaged gradient magnitudes, which easily cause needle-like artifacts in sparse views. To overcome this limitation, we propose GR-Gaussian, a graph-based 3DGS framework. It explicitly leverages a CT-specific prior, where regions of the same tissue or material have similar attenuation coefficients, forming a natural structural relationship among neighboring points. This structure motivates a graph-based representation, which guides gradient refinement to suppress needle-like artifacts. To exploit this structure, GR-Gaussian introduces (1) a Denoised Point Cloud Initialization strategy that mitigates initialization errors, and (2) a Pixel-Graph-Aware Gradient strategy that leverages graph-based density differences to refine gradient computation, improving splitting accuracy and density representation. Experiments on X-3D and real-world datasets validate the effectiveness of GR-Gaussian, achieving PSNR improvements of 0.67 dB and 0.92 dB, and SSIM gains of 0.011 and 0.021. These results highlight the importance of embedding domain-specific structural priors for accurate CT reconstruction under challenging sparse-view conditions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents GR-Gaussian, a graph-based framework for sparse-view tomographic reconstruction that extends the principles of 3D Gaussian Splatting (3DGS) to the CT domain. The key insight is that regions representing the same material or tissue exhibit similar attenuation properties, which can be naturally modeled as a graph structure connecting neighboring Gaussian kernels.

### Strengths
Its main strengths lie in the integration of structural priors through graph modeling, practical effectiveness under sparse sampling.
Clarity of implementation and reproducibility.

### Weaknesses
1. What are needle-like artifacts in sparse views, and what are their causes? The author should present the needle-like artifacts and provide a detailed analysis of their causes.
2. Table 1 is hard to read. What is the configuration? What are the "HO BS AO Average RD"?
3. The proposed method and related results exhibit noticeable secondary artifacts, especially in the chest and teapot cases in Figure 6.
4. The comparison of reconstruction time with other methods is missing.
5. Although the proposed graph-based formulation is well-motivated, the mathematical justification for how graph Laplacian regularization directly impacts convergence stability and artifact suppression is weak. The connection between structural priors and gradient refinement could be strengthened with ablation or convergence plots illustrating how the graph term influences training dynamics.
6. The evaluation relies solely on PSNR and SSIM for quality assessment. While these metrics reflect reconstruction fidelity, they do not capture perceptual or diagnostic quality — particularly important in medical CT contexts.
7. The proposed PGA strategy modifies gradient magnitudes based on local density differences (Eq. 12), but this adjustment is heuristically designed with limited theoretical or empirical justification for parameter λg. It is unclear how this linear combination between pixel-level and graph-level gradients balances stability versus over-smoothing. The lack of sensitivity analysis or ablation on λg prevents understanding of the robustness of the method across datasets and sparsity levels.
8. The graph-based operations (KNN search, Laplacian regularization, voxel weighting) add significant computational overhead compared to vanilla 3D Gaussian Splatting. Although the authors mention efficient CUDA implementation, the algorithmic complexity is not analyzed. There is no discussion of scalability when the number of Gaussians (M = 50,000) increases or when applied to higher-resolution CT volumes. A complexity analysis or profiling would clarify whether the method truly achieves real-time or near-real-time performance.

### Questions
Please see Weaknesses.

### Soundness
2

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
5

### Summary
The paper proposes a Graph-Based Radiative Gaussian Splatting reconstruction algorithm for sparse-view CT. The proposed method achieves the improvement in reconstruction quality by integrating denoised point cloud initialization and pixel-graph-aware gradient strategies into the 3DGS.

### Strengths
The paper addresses the sparse-view CT reconstruction task by considering the role of point cloud initialization and regularization methods when employing Gaussian splatting for scene modeling. Specifically, it proposes the use of a denoised point cloud initialization approach and a pixel-graph-aware gradient strategy based on graph Laplacian regularization to enhance the contribution of large kernels with small gradients during the densification process.

### Weaknesses
Given that Gaussian splatting has already been successfully applied in tomographic imaging (Zha et al., NeurIPS 2024), the present work builds upon this foundation by considering the impact of point cloud initialization and weighted gradient estimation methods, yet its innovation remains limited. In particular, numerical results indicate that the performance difference between the method proposed in this work and that of (Zha et al., NeurIPS 2024) is not significant. Furthermore, this work closely resembles a preprint on arXiv (https://arxiv.org/abs/2508.02408). If both submissions originate from the same research team, would this constitute a violation of the double-blind review policy?

### Questions
1. The improvement in reconstruction accuracy compared to R²-GS is not significant. I recommend the authors validate the advantage of their method under extremely sparse scenarios.

2.The ablation studies in the paper are insufficiently comprehensive. The authors only individually discuss the effects of graph Laplacian regularization, Denoised Point Cloud Initialization, and the Pixel-Graph-Aware Gradient strategy on the baseline model, but lack analysis of their combined effects. Given that the final model's performance improvement does not equal the sum of the three individual strategies' contributions, the effectiveness of their integration remains unverified.

3. The paper lacks essential introductory elements: for instance, it neither explains nor cites the R²GS method. The ablation studies provide no visualizations, which makes it impossible to evaluate the reconstruction effects. Furthermore, the comparative methods do not include specialized approaches designed for sparse-view reconstruction.

4.The paper lacks a dedicated discussion regarding the model's effectiveness and limitations.

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
This paper targets sparse-view CT reconstruction with a 3D Gaussian representation. The key ideas are a Denoised Point Cloud Initialization (De-Init) that smooths an FDK volume before sampling Gaussians, and a Pixel-Graph-Aware Gradient (PGA) that boosts densification gradients near structural boundaries using local density differences on a KNN graph. The authors report consistent gains over classical (FBP/FDK, SART/ASD-POCS), NeRF-like, and 3DGS baselines on X-3D and a real-data set (pine, seashell, walnut) under 25 views.

### Strengths
1. The paper explicitly analyzes why pixel-only gradient heuristics under-split flattened, long-axis Gaussians in sparse views, motivating a structure-aware fix. The failure case and formulation in Fig. 3(b) are on point.
2. There are consistent gains with ablations and parameter sweeps. The study covers De-Init/PGA/Reg toggles and sensitivity on k (neighbors) and $\sigma$d (denoise strength), as well as early-stopping via PSNR monitoring to avoid overfit. Results peak at k≈6 and $\sigma$d≈3, with qualitative improvements in needles/edges. The simulated X-3D (organs/biological/objects) and noisy real data are both considered; quantitative tables plus visual examples are provided.
3. The video demonstration in supplementary materials clearly shows the visualization strength of the proposed method.

### Weaknesses
1. The novelty is incremental relative to known capacity-/structure-aware schemes. The KNN graph and density-weighted gradients are sensible, but close in spirit to existing neighborhood-aware regularization and boundary-focused densification used in level-of-detail 3DGS or Laplacian smoothing.
2. The core SOTA baseline appears to be R$^2$-Gaussian and SAX-NeRF; it would be helpful to include more recent, optimized radiative-Gaussian implementations or stronger iterative CT baselines under identical geometry/noise, and to report statistical significance across more cases.
3. Only 25-view results are reported in the main text. Although Appendix include a few comparisons on other different view count, the detail of each view's experiment is not reported. I suggest the authors adding more detailed results on other alternative views to demonstrate a comprehensive experiment.

### Questions
Please refer to the weaknesses part, especially weakness 1&2.

### Soundness
2

### Presentation
3

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
This paper proposes a graph-based gaussian splatting method for CT reconstruction, outperforming previous methods or other INR methods. It is self-supervised and does not require large amount of training data.

### Strengths
1. I appreciate the idea of representing the CT images as a graph of Gaussians since it has the potential of combining with other segmentation masks or other methodology and facilitate doctor's understanding.
2. The performance looks good.

### Weaknesses
1. The paper is not well written. The training part is quite obscure. I do not understand how the Gaussians are trained, what datasets used, what is the network architecture.
2. Is the data learning any distribution level information? Or is it overfitting to a single patient. I encourage authors to incorporate distribution priors in the training since there exists a great amount of CT scan images and many paper report the scalability of model performance with more training data.
3. The visualization in figure.6 is concerning, the 25 projection reconstruction looks containing several artifacts, as mostly common seen in NeRF-related reconstruction methods. Please discuss how to mitigate those artifacts.
4. Authors should also compare their methods with diffusion or flow-based methods for solving inverse problems, such as Score-SDE, MCG or so on.

### Questions
Is the data learning any distribution level information? Or is it overfitting to a single patient. I encourage authors to incorporate distribution priors in the training. Is it possible to train a latent diffusion model that generate Gaussian Splats and use the method proposed by the authors?

### Soundness
3

### Presentation
3

### Contribution
2
