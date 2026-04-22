# Decomposing Densification in Gaussian Splatting for Faster 3D Scene Reconstruction

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
3D Gaussian Splatting (GS) has emerged as a powerful representation for high-quality scene reconstruction, offering compelling rendering quality. However, the training process of GS often suffers from slow convergence due to inefficient densification and suboptimal spatial distribution of Gaussian primitives. In this work, we present a comprehensive analysis of the split and clone operations during the densification phase, revealing their distinct roles in balancing detail preservation and computational efficiency. Building upon this analysis, we propose a global-to-local densification strategy, which facilitates more efficient growth of Gaussians across the scene space, promoting both global coverage and local refinement. To cooperate with the proposed densification strategy and promote sufficient diffusion of Gaussian primitives in space, we introduce an energy-guided coarse-to-fine multi-resolution training framework, which gradually increases resolution based on energy density in 2D images. Additionally, we dynamically prune unnecessary Gaussian primitives to speed up the training. Extensive experiments on MipNeRF-360, Deep Blending, and Tanks & Temples datasets demonstrate that our approach significantly accelerates training—achieving over 2x speedup with fewer Gaussian primitives and superior reconstruction performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work addresses slow convergence in 3D Gaussian Splatting (GS) by improving the densification process and Gaussian distribution. The proposed global-to-local densification strategy enhances both global coverage and local refinement, while an energy-guided multi-resolution training framework increases resolution based on energy density. This technique brings 0.5-1min acceleration to existing SOTA with minimal performance loss.

### Strengths
1. The finding of different roles of split and clone is interesting and inspiring. 
2. The paper is easy to read.

### Weaknesses
1. The arrow for "Time" in all Tables is set in opposite directions; the authors should be careful about these typos.
2. The technical innovation and contribution are limited. Bring 0.5-1min acceleration to existing SOTA, so what? Is there any case that it can bring significant benefits?
3. Considering the convergence speed as the target, why do you take 3DGS-accel as the baseline instead of the latest SOTA DashGaussian? How about applying the same strategy to it? Would there be further time cost reduction?

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
6

### Rating Number
6

### Confidence
2

### Summary
This paper investigates the optimization inefficiency in 3D Gaussian Splatting (3DGS) and identifies the imbalance between split and clone operations as the key cause of slow convergence and redundant Gaussian primitives.
Through a detailed analysis, the authors demonstrate that split operations dominate global diffusion of Gaussian primitives, while clone operations primarily handle local refinement.
Building on this observation, the paper proposes a global-to-local densification framework that separates these two phases:

- The global phase uses only split operations for fast and efficient spatial coverage;

- The local phase reintroduces clone operations for fine-grained detail reconstruction.

To further improve training efficiency, the authors design an energy-guided coarse-to-fine multi-resolution training scheme, where the image resolution is adaptively increased based on energy density in the frequency domain. Additionally, an adaptive opacity pruning mechanism dynamically removes redundant primitives based on an upper-bounded opacity threshold.

Comprehensive experiments on MipNeRF360, Deep Blending, and Tanks & Temples datasets demonstrate that the proposed method achieves over 2× acceleration in training speed, with comparable or improved reconstruction quality relative to baseline 3DGS methods.

### Strengths
- The paper presents an insightful and well-founded analysis of the split and clone operations in 3DGS densification. The discovery that split governs global diffusion while clone governs local refinement provides a new conceptual understanding of 3DGS optimization dynamics.
Moreover, the proposed global-to-local densification and energy-guided multi-resolution scheduling represent creative and orthogonal improvements over prior acceleration efforts such as DashGaussian and Mini-Splatting.
- Improving the training efficiency of 3DGS without sacrificing quality is a highly relevant problem, especially for real-time and resource-limited applications. The proposed framework provides a practical path toward more scalable and adaptive Gaussian-based scene representations.

### Weaknesses
- While the paper compares against recent 3DGS acceleration methods (e.g., DashGaussian, Mini-Splatting), it lacks evaluation on broader baselines such as fast NeRF variants (e.g., Instant-NGP, Zip-NeRF). A comparison would better position the proposed approach in the broader context of fast radiance field training.
- The proposed energy-guided multi-resolution strategy assumes sufficient frequency-domain energy correlation with spatial detail quality. It remains unclear how this heuristic behaves under challenging lighting conditions or dynamic scenes where energy distribution may not correspond well to geometric detail.
- While the empirical observation that “split = global, clone = local” is well-motivated, the explanation remains empirical rather than theoretically derived. A more formal analysis (e.g., via gradient flow or spatial entropy) would add conceptual depth and broaden the paper’s impact.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the slow training convergence of 3D Gaussian Splatting (3DGS). The authors identify the bottleneck as the inefficient, mixed application of 'split' and 'clone' operations during the densification phase, leading to wasted computational resources, especially in the early stages of training. The core insight is the explicit decoupling and analysis of the distinct roles of the two densification operations. Based on this, the paper proposes a "Global-to-Local" optimization framework. Experiments demonstrate that the method achieves over 2x training acceleration, utilizes approximately 40% fewer Gaussian primitives than the baseline and maintains or even slightly improves the final reconstruction quality.

### Strengths
1. The paper is well-written and easy to follow.
2. While 'split' and 'clone' are established operations, this paper is the first to systematically analyze and expose their distinct functional roles: splitting for global scene coverage and cloning for local feature refinement. This reframing of the problem from merely "how to densify" to "when and why to use each densification type" is inspiring to me.

### Weaknesses
1. Potential Oversimplification of the Role of Cloning. The paper frames cloning as contributing almost exclusively to local refinement and early-stage redundancy. This might be an oversimplification. In certain cases, such as representing very thin structures (e.g., wires, poles, foliage), cloning might play a constructive role in reinforcing the structure's existence and density early on, where splitting could potentially fragment it.
2. The core assumption is that a global spread phase should always precede a local refinement phase. While this is intuitive for large, complex scenes, it may not be optimal for all types of content. For instance, in scenes dominated by a single, highly detailed foreground object (e.g., a product scan for e-commerce), aggressive local refinement via cloning early on might be more beneficial than a prolonged global spread phase.

### Questions
The proposed method hinges on a "hard" transition from a "split-only" phase to a "split+clone" phase. The paper states this transition is guided by the resolution scheduler (at iteration T2), but the sensitivity to this specific point is not analyzed. The effectiveness of the entire approach could be highly dependent on this timing.

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
This paper addresses the slow convergence in Gaussian Splatting by improving the efficiency of the densification process. It analyzes the distinct roles of the split and clone operations and, based on these insights, introduces a global-to-local densification strategy: the global stage uses only splitting to distribute Gaussians across the entire scene, while the local stage employs both splitting and cloning to refine them. Additionally, the paper proposes a coarse-to-fine multi-resolution training scheme and an adaptive Gaussian pruning method to further enhance convergence speed and training efficiency.

### Strengths
- This paper is well-written with clear motivation.
- The paper provides a clear discussion of the two density control strategies, split and clone. Analyzing how these strategies differ offers valuable insights that could benefit future research in this area.
- The proposed global-to-local densification strategy, the core contribution of this paper, is simple, and I believe it can be broadly applied to Gaussian Splatting optimization.

### Weaknesses
- I agree that the split operation helps distribute Gaussians across the entire scene, but I am not fully convinced that the clone operation alone is responsible for the local refinement. Rather, I would argue that the split operation not only spreads the Gaussians but also contributes to refining local details, albeit at the cost of generating significantly more Gaussians compared to the clone operation. Figure 2 in the main text also supports this observation, showing that the reconstruction from the split-only model is visually more pleasing than that from the clone-only model.
- The three main technical contributions of this paper are (1) global-to-local densification, (2) coarse-to-fine densification, and (3) adaptive opacity pruning. However, the coarse-to-fine densification strategy closely resembles that of DashGaussian, though the energy metric used here is slightly different. In addition, the ablation study (Table 4) shows that the coarse-to-fine densification contributes more effectively to both training speed and rendering quality than the global-to-local densification. Moreover, the idea of adaptive pruning has already been explored in EDC (Efficient Density Control). While the details differ, it would strengthen the paper to include a direct comparison with the adaptive pruning used in EDC. These factors somewhat weaken the novelty of the proposed approach.
- Compared to MSv2 (MiniSplatting version 2), it is difficult to claim that the proposed method achieves a clear state of the art. Although both reconstruction quality and training time are improved over MSv2, the gains are marginal (particularly in training time: 3.55 vs. 3.47), and the proposed method produces substantially more Gaussians. Furthermore, additional comparisons against MSv2 on both the T&T and DeepBlending datasets are needed, and such results should be included in the main experiments.
- The three key components of the proposed method—(1) global-to-local densification, (2) coarse-to-fine densification, and (3) adaptive opacity pruning—appear to have overlapping roles. Specifically, both global-to-local densification and adaptive opacity pruning aim to accelerate training by reducing the number of Gaussians. To better demonstrate the effectiveness of the global-to-local densification, it would be helpful to include a C2F + Pruning variant in the ablation study for comparison.

### Questions
- Why is T2 chosen as the boundary between the global and local densification?

### Soundness
2

### Presentation
3

### Contribution
2
