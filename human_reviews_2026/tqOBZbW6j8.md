# DiffPBR: Point-Based Rendering via Spatial-Aware Residual Diffusion

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 4

## Abstract
Neural radiance fields and 3D Gaussian splatting (3DGS) have significantly advanced 3D reconstruction and novel view synthesis (NVS). Yet, achieving high-fidelity and view-consistent renderings directly from point clouds---without costly per-scene optimization---remains a core challenge. In this work, we present DiffPBR, a diffusion-based framework that synthesizes coherent, photorealistic renderings from diverse point cloud inputs. We demonstrate that diffusion models, when guided by viewpoint-projected noise explicitly constrained by scene geometry and visibility, naturally enforce geometric consistency across camera motion. To achieve this, we first introduce adaptive CoNo-Splatting, a technique for fast and faithful rasterization that ensures efficient and effective handling of point clouds. Secondly, we integrate residual learning into the neural re-rendering pipeline, which improves convergence, generalization, and visual quality across diverse rendering tasks. Extensive experiments show that our method outperforms existing baselines with an improvement of **3~5dB** in rendered image quality, a reduction from **41 to 8** in GPU hours for training, and an increase from **3.6fps to 10fps** (our one-step variant) in rendering speed frequency.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes DiffPBR, a diffusion-based framework for point-based rendering to produce photorealistic and view-consistent images directly from point clouds. The method introduces two core components: (a) Adaptive CoNo-Splatting, a differentiable point rasterizer that adaptively adjusts point scales using a learnable global regularizer; (b) Spatial-Aware Residual Diffusion (RDDM) – a diffusion model conditioned on geometry-aware, view-consistent noise maps and trained to predict residuals. The two model variants, -Q and -E, have shown large improvements over the baseline method of PFGS.

### Strengths
- The idea of applying noise mechanism in 3D space (rather than 2D) is very interesting to me, and the implementation does achieve satisfactory results on common benchmarks. The residual diffusion formulation is elegant and well-motivated, reducing computational cost while preserving fidelity.
- The pipeline is well-illustrated and the pseudo-code & the appendix demonstrate commendable clarity and reproducibility for readers.
- The proposed CoNo-Splatting has the potential to be developed as a common plug-in for most point-based rendering methods.

### Weaknesses
- The two claimed contributions—Adaptive CoNo-Splatting and Spatial-Aware Residual Diffusion—are somehow independent, with limited conceptual or algorithmic coupling. Each component could function as a standalone contribution, making the overall framework appear as a loose combination rather than a tightly integrated method.
- The evaluation of performance is not sufficient, in terms of both benchmarks and baselines. The three datasets are either bounded (ScanNet) or captured under well-controlled lab environment. More challenging cases such as Tanks and Temples are recommended to see the real potential of these methods. For baselines, I think the authors should at least include 3DGS as it's one of the most known method for point-based rendering.
- The rendering fps is not suitable for real-time applications, though it's not considered as a drawback of this work.

### Questions
- What does the PyTorch3D in the experiments particularly mean? Do the authors mean directly rasterizing points by PyTorch3D? If so, it shouldn't be a baseline method since PyTorch3D is just an implementation.
- What if Adaptive CoNo-Splatting is applied to other methods? I wonder its performance as a plug-and-play module.
- Also it seems that the authors have misused the tex commands of \citep{} and \citet{}. Please correct these citation formats in the modified manuscript accordingly.

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
3

### Summary
This paper proposes DiffPBR, a point-based rendering framework that integrates a diffusion model to enhance the quality and view consistency of images rendered from colored point clouds. The method consists of three main components: 1) An adaptive "CoNo-Splatting" technique that rasterizes both color and a structured 3D noise map from the point cloud; 2) A residual diffusion process that learns to predict the difference between the coarse rendering and the ground truth, conditioned on the rendered noise and a mask; 3) A learnable global scale parameter optimized via a balance of coverage and compactness losses. The authors demonstrate state-of-the-art results on several datasets (ScanNet, DTU, THuman2.0) in terms of PSNR/SSIM/LPIPS, while also reporting significant improvements in training efficiency and inference speed compared to the baseline PFGS.

### Strengths
The overall framework is well-motivated and addresses the core artifacts of point-based rendering (holes, aliasing, view inconsistency) in a unified, learnable manner. The integration of a diffusion model for refinement is a logical and powerful choice.

### Weaknesses
1. Limited Conceptual Novelty: The core components—"residual diffusion" and "3D-consistent noise"—are applications of existing ideas rather than fundamental innovations. Residual learning is standard in image restoration, and 3D-consistent generation is a well-established research direction. The paper does not sufficiently delineate its conceptual advance beyond the specific application to point cloud rendering.
2. Inadequate Comparison to True SOTA: The most significant omission is a direct comparison with 3D Gaussian Splatting (3DGS) and its variants, which are the current undisputed state-of-the-art in terms of rendering quality and speed for scene-specific novel view synthesis. Bypassing this comparison undermines the claim of superior performance.
3. Unconvincing and Unfair Efficiency Claims: The dramatic reduction in training time compared to PFGS is likely attributable to the replacement of PFGS's heavy CNN-based point refinement network with a much simpler rendering front-end. This is an architectural choice, not necessarily a pure algorithmic efficiency gain, making the comparison somewhat misleading.
4. Methodological Ambiguities: The end-to-end joint training of the splatting parameters (β) and the diffusion model creates a non-stationary training target for the diffusion network, the stability and necessity of which are not validated through ablations. The use of the term "adversarial balance" for a simple weighted sum of losses is inaccurate and misleading.

### Questions
1. Why was a direct quantitative and qualitative comparison with 3D Gaussian Splatting (3DGS) omitted? Given that 3DGS represents the current peak performance for this problem (even if scene-specific), how does DiffPBR's rendering quality on a per-scene basis compare?
2. The joint optimization of the splatting stage (via $L_ens$) and the diffusion model (via $L_dm$) is non-standard. Can you provide an ablation study training the diffusion model with a fixed β (e.g., from a pre-trained stage or a heuristic) to demonstrate that the joint training is truly beneficial and not detrimental to convergence?

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
4

### Summary
This paper proposes a model, called DiffPBR, that generates photo-realistic, view-consistent point-cloud renderings. DiffPBR is a two-stage method that combines adaptive, noise-consistent splatting with a diffusion model to render point clouds from calibrated camera view parameters. First, the point cloud is projected into the image domain using point-based splatting with a globally truncated scatter-distance threshold. This step introduces noise and artificial rasterization artifacts, serving as an initial rendering proposal. Then, a DDPM flow-based backbone is trained end to end to produce denoised, photo-realistic renderings for each camera view.

To show that DiffPBR is an end-to-end solution for rendering single-modal point-cloud input, the authors conduct experiments on the ScanNet, DTU, and THuman 2.0 datasets to demonstrate (i) improved rendering quality and (ii) manageable rendering speed (**2**–10 FPS, depending on Quality mode or Efficient mode, which the authors refer to as DiffPBR-Q and DiffPBR-E, respectively).

### Strengths
The paper offers an insightful focus on decoupling the point cloud information from the rendering pipeline’s noise scheduling. Point-based splatting is much lighter than PFGS’s per-point CNN scale predictor and is explicitly tuned to produce good inputs for the diffusion refiner rather than the final image.

The idea of adopting a diffusion model that diffuses from something other than pure Gaussian noise recalls [Cold Diffusion](https://proceedings.neurips.cc/paper_files/paper/2023/file/80fe51a7d8d0c73ff7439c2a2554ed53-Paper-Conference.pdf), which might indicate its efficacy for this particular problem.

The experiments are self-contained, and the ablations are extensive.

### Weaknesses
My biggest concern is the training setup given what is already known about the input. The assumption that the point cloud and calibrated camera views are available is both an advantage and a restriction. Is DiffPBR aware of noise arising from point-cloud registration, and if it is trained per scene, why is it not biased, and how can it transfer or generalize across scenes? I understand Figure 5 tries to address transferability by directly migrating models without fine-tuning, but I am unsure why such a scheme would work. More experiments may be needed to convince the reader that geometry-aware noise patterns are interchangeable.

In my view, recent works like [Vanilla Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [FSGS](https://zehaozhu.github.io/FSGS/), and [RPBG](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02379.pdf) should be included for comparison even though they are NVS methods that rely on (sparse) multi-view input. Comparing to their results would provide supportive evidence for why DiffPBR can outperform and better utilize point-cloud information. My suggestion is to test bilaterally: give one NVS method a full prior of point-cloud information, and give DiffPBR the sparse-view COLMAP registration results in the large-scale scenarios used in those NVS papers. These comparisons would make the paper more compelling.

Some minor concerns, if resolved, would also improve the completeness of the paper, and one could consider raising the score if most concerns are addressed or justified.

* No code is provided for reproducibility.
* L204–206, Eqn. (1): using the median of all average scales seems a very bold guess. What happens if a different percentile (the median is the 50th percentile) is used, and why is the mean not an appropriate statistic?
* In L283 A1 L11, why is the residual ground truth given as a linear interpolant of $I_{\epsilon}^v$ and $I_{r}^v$? Can you explain $\gamma_t$, as it seems undefined.
* Why do DiffPBR-E and DiffPBR-Q require the same training time on average per scene? I think their training pipelines can be interchangeable, and DiffPBR-E may be more generalizable since it uses a relative image-consistency loss rather than comparing with ground truth. Have you considered cross-checking whether DiffPBR-Q training plus one-step inference would work?
* I notice that in Table 2, PFGS is reported to have 3.6 FPS while DiffPBR-Q only gets 2 FPS inference speed. Please clarify the claim in L24 of the abstract.
* Although the ablation is extensive, the choices of $\lambda_{cov}=0.01$ and $\lambda_{cmp}=0.1$ in L864 are not discussed. Do they affect performance?

### Questions
* Can you define the CNSplat function (L246–L248) as a mathematical formula?
* In L49, does FPGS refer to PFGS?
* Can you explain why a graphics-based rendering technique can fail to provide photorealistic renderings on the given datasets? Have you considered rendering the ScanNet or DTU data using Mitsuba rather than PyTorch3D?
* In L714, should $\mathcal{F}$ be $\mathcal{F}_{\theta}$?
* Apart from Figure 6 (noise-covariance patterns across views) and average rendering performance, have you considered other schemes to quantify view consistency of the rendering output? I don’t have an immediate answer in mind, but I would like to hear more about how to safeguard “consistent multi-view synthesis” (L93).

### Soundness
3

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
This work presents a method for rendering 3D point clouds to images. Specifically it aims for photorealistic rendering even from sparse point-clouds. This is achieved by using a customised diffusion model to "tidy up" the sparse rendering. To ensure the diffusion results are approximately multi-view consistent, noise is first sampled supported on the point-cloud, then rasterised to different viewpoints. The model then predicts the residual to map the sparse point cloud rendering to a realistic image. On three datasets of point-clouds paired with ground-truth images, the method is shown to achieve higher-fidelity rendering than selected baselines.

### Strengths
The proposed pipeline is novel – the combination of size-adaptive rasterisation strategy with diffusion-based refinement using spatially (multi-view) consistent noise has not been tried before.

The approach to arranging 3D-consistent diffusion noise is simple and elegant; similarly predicting residuals instead of directly reconstructing images is a sensible choice.

The main experiments on DTU, ScanNet and THuman2 show that the proposed method achieves higher rendering fidelity than Neural Point-Based Graphics, its more-recent extension NPBG++, TriVol (based on density fields), and PFGS (based on gaussian splatting). This holds across all three datasets, according to PSNR, SSIM and LPIPS metrics.

The method is shown to be significantly faster (for training and inference) than PFGS, and also more robust to point density (i.e. it retains higher fidelity even when the point cloud becomes sparser).

There are additional experiments on cross-domain generalisation, using the fairly disparate domains of Thuman2.0 people and the DTU reconstruction dataset. Even between these very different domains, the method achieves respectable visual fidelity when trained on images from one and validated on images from the other.

There is a fairly thorough ablation study measuring the benefit of various design decisions and components of the proposed method. Most notably, the 3D-consistent noise shows a significant improvement in PSNR compared with 2D noise.

The paper is clear, well-structured, and pleasant to read.

### Weaknesses
The technical contribution is fairly small – diffusion with 3D-consistent noise is now fairly well-established in other tasks such as multi-view-consistent novel view synthesis. Beyond this, the task of rendering sparse point-cloud realistically has been studied by several methods (including the baselines used in this work), and the additional insight given by the present work is fairly small. Similarly the "CoNo Splatting" described in Sec 3.1 is a trivial strategy for rendering point-clouds, accompanied by a simple pair of regularisers to ensure the points are a sensible size.

The text implies that the model guarantees 3D consistency across viewpoints; however while there is a strong encouragement to do so by the 3D representation of noise, it is not guaranteed since the diffusion itself occurs in 2D pixel space.

There is no comparison against naive baselines like nearest-neighbour interpolation of colors into empty pixels in 2D image space, or off-the-shelf diffusion-based inpainting after rendering the point-cloud at one pixel per point.

The results on efficiency and robustness wrt point-cloud density both compare only with PFGS, not with the other baselines. These results should be added to give a more complete picture of relative performance.

The work ignores the fundamental issue that point-cloud rendering is ill-posed – unless we know a true physical scale for each point, there is not enough information to know whether a given set of nearby, coplanar points in fact represent a continuous, solid surface rather than some kind of holed, porous structure. Both gaussian splats and NeRFs are more explicitly 'solid' in this sense since they define densities over non-infinitesimal volumes. This point should be discussed in the introduction, which currently is presented as if there is one true

### Questions
See the points discussed in weaknesses above – in particular the missing baselines / ablations.

L60 "diffusion models in image restoration rely on the assumption of pure noise inputs" – this is poorly worded, they typically input both a noisy image (as conditioning for the diffusion process) and latent noise that will be conditionally decoded to the final image; thus they do not receive *only* pure noise. Moreover, it is common in image restoration to start the denoising process at a DDIM-inverted or noised version of the input image, for exactly the reasons discussed in this paragraph.

Please provide LPIPS & SSIM for the ablation and other experiments that only use PSNR – LPIPS in particular is much more reflective of perceived quality.

Many \citet or \cite should become \citep

### Soundness
3

### Presentation
3

### Contribution
1
