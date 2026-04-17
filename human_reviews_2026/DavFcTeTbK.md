# Signal Structure-Aware Gaussian Splatting for Large-Scale Scene Reconstruction

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
3D Gaussian Splatting has demonstrated remarkable potential in novel view synthesis. In contrast to small-scale scenes, large-scale scenes inevitably contain sparsely observed regions with excessively sparse initial points. In this case, supervising Gaussians initialized from low-frequency sparse points with high-frequency images often induces uncontrolled densification and redundant primitives, degrading both efficiency and quality. Intuitively, this issue can be mitigated with scheduling strategies, which can be categorized into two paradigms: modulating target signal frequency via densification and modulating sampling frequency via image resolution. However, previous scheduling strategies are primarily hardcoded, failing to perceive the convergence behavior of the scene frequency. To address this, we reframe scene reconstruction problem from the perspective of signal structure recovery, and propose SIG, a novel scheduler that Synchronizes Image supervision with Gaussian frequencies. Specifically, we derive the average sampling frequency and bandwidth of 3D representations, and then regulate the training image resolution and the Gaussian densification process based on scene frequency convergence. Furthermore, we introduce Sphere-Constrained Gaussians, which leverage the spatial prior of initialized point clouds to control Gaussian optimization. Our framework enables frequency-consistent, geometry-aware, and floater-free training, achieving state-of-the-art performance with a substantial margin in both efficiency and rendering quality in large-scale scenes.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper identifies a key problem in scaling 3D Gaussian Splatting to large-scale scenes: the mismatch between sparse initial 3D points and high-frequency image supervision. This mismatch causes uncontrolled growth of redundant Gaussians, hurting both efficiency and quality. The authors reframe the problem as one of signal frequency synchronization and propose a scheduler to align the 2D image supervision with the 3D Gaussian representation's evolving complexity.

### Strengths
1. The sufficient experiments adequately illustrate the superiority of the method and the effectiveness of each model design. The settings of experiments and baseline implementations are also well explained, making it easier to follow the paper and reproduce results.
2. The efficiency and quality of large-scale scene reconstruction are of significance. This paper introduces a sound and innovative method to push the performance bound over existing methods.

### Weaknesses
1. The order of LPIPS and SSIM in Tab.2 and Tab.3 seems to be wrong.
2. Fig.2 itself is not well self-contained. The formulas and pipelines are, to some extent, confusing, such as the meaning of the blue dashed line. And it should provide a reference to the related sections.

### Questions
See weakness.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a frequency-consistent framework for large-scale 3D scene reconstruction. It mathematically defines the average sampling frequency and signal bandwidth of 3D Gaussian representations, and introduces a scheduler that adaptively adjusts image resolution and Gaussian densification based on scene frequency convergence. Additionally, it proposes Sphere-Constrained Gaussians to enforce geometry-aware optimization and reduce redundancy. The paper achieves state-of-the-art rendering quality and faster training compared to the baselines.

### Strengths
- This paper formalizes when to raise resolution and when to densify using a bandwidth convergence signal, which improves the performances of baselines that rely on predefined schedules
- The proposed method is a plug-and-play component that could be applied to multiple baselines (e.g., 3DGS, CityGS, BlockGS) and leads to consistent performance gains and better efficiency
- The ablations studies clearly show the benefits brought by each of the proposed components

### Weaknesses
- The convergence threshold k, neighbor count K, and scaling factor l (e.g., K=15, l=15, and max_offset is scaled down to 0.7×) are stated but not sufficiently justified. Are these values chosen empirically, or derived from any principled analysis? Moreover, do they require dataset-specific tuning, especially under varying levels of SfM sparsity or reconstruction noise?
- Missing baselines: some relavant works in this domain are not compared to, especially: CityGaussianV2 (ICLR 2025), FlashGS (CVPR 2025)

### Questions
As noted in the weaknesses, I recommend that the authors include ablations on key hyperparameters, or at least provide a discussion on how these values are selected and whether they are kept consistent across all experiments. I also suggest to provide additional qualitative/quantitative comparisons with CityGaussianV2 and FlashGS.

Apart from above, I have an additional question:
- Does the proposed method also improve geometry? Although it's hard to directly evaluate geometries in such large scenes, I would be interested to see a similar geometry evaluation as in CityGaussianV2.

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
This paper aims at solving the frequency inconsistent issue during 3DGS training. It leverages the frequency information to check whether it is necessary to increases the image resolution during training. Moreover, it introduces the Sphere-Constrained Gaussians to constrain its moving offset. Experiments under large-scale dataset validates its effectiveness since it beats SOTA large-scale 3DGS approaches.

### Strengths
(1) The idea is novel, which leverage the frequency information to schedule the resolution of training images.

(2) The writing is good and the paper is easy to understand.

(3) The proposed method is effective, and beats SOTA methods in large-scale datasets.

### Weaknesses
(1) The method is compared to many SOTA methods, but it lacks comparison with a naive baseline: scheduling the image resolutions during training at some hard-coded iterations (e.g. 10,000, 15,000, 20,000), I think some methods did this but it needs to be done in the experiment of this paper to provide a direct comparison.

(2) The experiments part lacks comparison on some standard benchmarks, e.g. the mipnerf360 dataset and the tanks-and-temples dataset.

### Questions
(1) Eq. (5) and Eq.(6) are computed at each training iteration. Will that be time consuming?

(2) At line327, the author mentioned "we use 30,000 training iterations for both the coarse and fine stages". How to define the coarse stage and fine stage? Since the method enlarges the resolution of training images in a coarse-to-fine manner, it is confused to me the 3DGS are trained with a coarse stage and a fine stage?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Signal Structure-Aware Gaussian Splatting (SIG), a framework designed to improve large-scale 3D scene reconstruction using 3D Gaussian Splatting. The authors observe that prior methods supervise Gaussians initialized from sparse low-frequency points using high-frequency image signals, which leads to uncontrolled densification and redundant primitives. To address this, they reformulate scene reconstruction as a signal structure recovery problem and introduce a frequency-based synchronization between image supervision and the Gaussian representation.

Concretely, the paper defines the "average sampling frequency" of images and the "effective scene bandwidth" of Gaussians, derived from a frequency-domain analysis of the opacity field. These metrics guide the adaptive adjustment of training image resolution and Gaussian densification as the scene frequency converges. Additionally, the authors introduce Sphere-Constrained Gaussians (SCG) that restrict each Gaussian’s optimization region to a local sphere determined by its initial point cloud neighborhood, mitigating floaters and structural drift. Experiments on large-scale benchmarks show performance gains in reconstruction quality and training efficiency compared to baselines.

### Strengths
The idea of aligning the Gaussian frequency with the supervision signal frequency is intuitive and well-executed, beginning with a clear definition of Gaussian frequency and followed by thorough verification. Experiments show consistent gains in quality and speed, and the method works as a plug-in for existing frameworks like CityGS and BlockGS.

### Weaknesses
1. It remains unclear how sensitive the method is to the introduced hyperparameters.
2. While the concept of progressive training from low to high resolution is not novel, the paper’s analytical framework for optimally aligning Gaussian frequency with signal frequency is interesting. However, a comparison with the more heuristic use of progressive training is missing.
3. Evaluating the proposed method on conventional NVS benchmarks such as MipNeRF360 would further demonstrate its generalization capability.

### Questions
See the weakness. The LPIPS and SSIM columns appear to be swapped in Tables 2 and 3.

### Soundness
2

### Presentation
2

### Contribution
2
