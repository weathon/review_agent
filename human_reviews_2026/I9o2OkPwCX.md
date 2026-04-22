# Large Depth Completion Model from Sparse Observations

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
This work presents the Large Depth Completion Model (LDCM), a simple, effective, and robust framework for single-view metric depth estimation with sparse observations. Without relying on complex architectural designs, LDCM generates metric-accurate dense depth maps use a transformer. It outperforms existing approaches across diverse datasets and sparse observations. We achieve this from two key perspectives: (1) leveraging existing monocular foundation models to improve the quality of sparse depth inputs, and (2) reformulating training objectives to better capture geometric structure and metric consistency. Specifically, a Poisson-based depth initialization strategy is firstly introduced to generate a uniform coarse dense depth map from diverse sparse observations, providing a strong structural prior for the network. Regarding the training objective, we replace the conventional depth head with a point map head that regresses per-pixel 3D coordinates in camera space, enabling the model to directly learn the underlying 3D scene structure instead of performing pixel-wise depth map restoration. Moreover, this design eliminates the need for camera intrinsic parameters, allowing LDCM to naturally produce metric-scaled 3D point maps. Extensive experiments demonstrate that LDCM consistently outperforms state-of-the-art methods across multiple benchmarks and varying sparsity levels in both depth completion and point map estimation, showcasing its effectiveness and strong generalization to unseen data distributions. Code and models are publicly available at \href{https://pkqbajng.github.io/ldcm/}{pkqbajng.github.io/ldcm/}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes the Large Depth Completion Model (LDCM) to address the problem of estimating dense and metrically accurate depth maps from sparse depth observations. It mainly involves a Poisson-based coarse depth initialization and a point map regression head for 3D structure prediction. The authors performs comprehensive evaluations across multiple benchmark datasets under varying sparsity of inputs, demonstrating state-of-the-art results.

### Strengths
1. Robust Results. The extensive experiments on diverse benchmarks (including KITTI, ETH3D, DIODE, etc.) demonstrate that LDCM consistently outperforms existing methods, especially in zero-shot settings and under sparse input conditions. This indicates a high level of generalization and robustness of the proposed model.

2. Extensive Training and Comprehensive Evaluation. The authors have invested considerable effort in training LDCM on a large number of datasets. The extensive evaluation spans a wide range of sparsity levels, providing a comprehensive comparison. 

3. Effective Preprocessing Method. The Poisson-based depth initialization strategy that incorporates geometric priors from monocular depth models to construct initial coarse depth is interesting.

### Weaknesses
1. Inference Time. While LDCM demonstrates excellent performance across multiple benchmarks, the model’s inference time has not been explicitly discussed. It would be useful to include an analysis of the model's runtime efficiency.

2. More Comparison Methods. Pow3R introduces sparse priors to predict pointmaps, a direct comparison with the pow3r model is necessary.

3. Visualization Results. Although the paper provides extensive quantitative evaluations, additional visualizations of the depth completion and point map estimation results would enhance the reader's understanding of the model’s performance.

[1] CVPR 2025, Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors

### Questions
The sparse points shown in the head figure are difficult to visualize clearly. To improve clarity and provide a better qualitative analysis, it would be beneficial to plot the sparse points directly on the RGB images.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces LDCM, a transformer-based model to estimate dense metric depth from a single image and a sparse depth. It proposed a Poisson-based preprocessing strategy that leverages a pretrained depth foundation model (Depth Anything V2) to produce an initial coarse depth map and replaced the conventional depth head with a pointmap head. Trained on a large collection of high-quality datasets, the proposed model demonstrates superior performance across various benchmarks under different sparse levels, outperforming previous state-of-the-art methods.

### Strengths
1. The proposed model is simple but effective. It outperforms a recent SOTA baseline, such as PriorDA and PromptDA.
2. Compared to previous coarse alignment method, the proposed Poisson-based strategy is effective, providing a strong initialization.
3. The authors perform a detailed set of zero-shot experiments on multiple datasets, demonstrating the effectiveness of LDCM in both depth completion and point map estimation. The model consistently outperforms prior methods in a variety of settings.
4. This paper is well written and the implementation details are comprehensive and easy to follow.

### Weaknesses
1. The main concern comes from the limited qualitative comparison. The manuscript mainly provides quantitative results, where the proposed LDCM ranks first on most of the benchmarks. The authors may consider adding more visualized results to better illustrate the performance differences and provide more intuitive insights into the effectiveness of the proposed LDCM.
2. The inference time of each component should be mentioned to provide a comprehensive evaluation for the proposed model.
3. Recently, there are lots of 3D reconstruction models which introduce additional depth priors (e.g., WorldMirror, MapAnything, Pow3R). To may understanding, the sparse prior provides both metric and relative geometry guidance. Although these methods generate relative geometry, the metric scale can be recovered by the proposed Poisson strategy or least square alignment. Comparing performance with these methods would provide a more comprehensive evaluation.

### Questions
Please see "Weaknesses" section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents the Large Depth Completion Model (LDCM), a novel framework for generating dense, metrically accurate depth maps from sparse and irregular depth inputs. The key innovation lies in a two-stage approach: first, it proposes a preprocessing pipeline that uses a pretrained depth foundation model to estimate a gradient field, which is then integrated via a Poisson solver to generate a coarse but geometrically coherent initial depth map. Second, for depth prediction, LDCM introduces the use of a pointmap representation—a 3D point cloud-like structure—to directly model scene geometry, enabling more precise depth completion.

The method achieves state-of-the-art performance across six major benchmark datasets, excelling in both depth completion and pointmap estimation tasks, with consistent top rankings on all evaluated metrics.

### Strengths
1. The paper is well-structured and transparent, offering comprehensive implementation details, including a thorough list of training datasets, which enhances reproducibility. Also, the code in the supplementary material shows the details of the model.
2. The proposed gradient-field-based initialization using a foundation model and Poisson reconstruction effectively preserves geometric structure and metric scale, providing a strong starting point for refinement.
3. This work is the first to introduce the pointmap as a core representation in depth completion, enabling direct 3D scene modeling and improving structural fidelity.
4. Evaluation is comprehensive, covering multiple benchmarks, with detailed quantitative results, ablation studies, and comparisons that convincingly demonstrate the superiority of the proposed method.

### Weaknesses
1. While the quantitative performance is impressive, the paper would benefit significantly from more visual comparisons (e.g., side-by-side depth map visualizations) to intuitively illustrate LDCM’s advantages over baselines, especially in challenging regions like object boundaries or low-sampling areas.
2. Recent advances in diffusion-based models have demonstrated strong performance in depth estimation and completion (e.g., [1][2]), despite their typically longer inference times. While LDCM may prioritize efficiency, including a comparison with these state-of-the-art diffusion methods would provide a more comprehensive benchmarking landscape.
3. The training data includes synthetic and high-quality real datasets. However, it is crucial to discuss the impact of introducing more real-world data.
4. The paper compares with monocular depth estimation models, which are inherently scale-ambiguous. Providing these details of scale alignment is crucial for fair and reproducible comparison.

Overall, I’m inclined to accept this paper, but I encourage the authors to provide more comparison and ablation experiments.


[1] Marigold-DC: Zero-Shot Monocular Depth Completion with Guided Diffusion

[2] DepthLab: From Partial to Complete


 Minor weakness

1. There is a minor inconsistency in Table 8 (e.g., "Unidepthv1" vs. "Unidepth V1") that also appears in subsequent tables. This should be corrected for clarity and consistency.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a two-stage pipeline that produces a metric-consistent dense depth map from a sparse prior + a monocular relative-depth foundation model via a Poisson-style reconstruction, and then feeds that coarse depth and RGB image into the network that regresses a metric 3D point map. The method is evaluated zero-shot across many benchmarks and reports better performance than previous works in both depth completion and point-map estimation tasks.

### Strengths
	The proposed Poisson-based initialization framework is architecturally simple yet highly effective, avoiding complex model design.
	The paper provides extensive empirical evaluation through cross-dataset, zero-shot testing on multiple benchmarks (KITTI, iBims-1, DIODE, ETH3D, etc.), demonstrating strong generalization.

### Weaknesses
	Table 3 evaluates different alignment strategies (Global, LWLR, Poisson), but it would be insightful to include a comparison using Equation (1) only, i.e., a baseline without the global affine correction ($\alpha, \beta$) to show the quantitative gain from incorporating the global affine term.

	Since DepthAnything, VGGT, and MoGe also predict relative depth as the authors mentioned, in Tables 1 and 5, it would further strengthen the paper to show their results after applying the same Poisson-based alignment used in this paper. Demonstrating that the current result of this paper still outperforms these baselines after identical alignment would make the contribution more compelling. Also, in table 2 (point map evaluation), it would further strengthen the evaluation if the authors could include comparisons with recent point map models such as VGGT or DUSt3R, even under RGB-only settings.

	As shown in Appendix Tables 8 and 9, the performance degrades more noticeably than some baselines when depth noise is added. The paper lacks an analysis explaining this sensitivity, e.g., whether it arises from their framework design.

	The inference time comparison is not discussed in the methodology, experimental results, or tables. Including the inference time would help verify the effectiveness and practicality of the proposed method.

I will reconsider the score when all the concerns are handled well.

### Questions
	Could the authors clarify the computational cost of the Poisson reconstruction stage (Eq. 1–4) compared to simpler alignment methods like LWLR?

	The related work ‘monocular geometry estimation’ section could be improved by adding recent and concurrent 3D foundation models such as
- [1] Jiang, Lihan, et al. "AnySplat: Feed-forward 3D Gaussian Splatting from Unconstrained Views." SIGGRAPH Asia 2025
- [2] Keetha, Nikhil, et al. "MapAnything: Universal feed-forward metric 3D reconstruction." arXiv preprint arXiv:2509.13414 (2025).

Including and discussing these would better situate the proposed approach in the broader landscape of geometry foundation models.
Further reference recommendation for depth-completion task:
- [1] Jeong, Chanhwi, et al. "Test-Time Prompt Tuning for Zero-Shot Depth Completion." ICCV 2025.

### Soundness
3

### Presentation
2

### Contribution
3
